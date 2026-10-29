"""Backfill phase ``clusters``: replay clustering over existing memories.

Under Tier 1 the clustering strategies' embedding guards skip every event,
so no episode or agent case was ever clustered. This phase synthesizes the
``EpisodeExtracted`` / ``AgentCaseExtracted`` event each row's pipeline run
emitted, and replays them through a throw-away engine that runs only the
two clustering strategies.
"""

from __future__ import annotations

__all__ = ["run_clusters_phase"]

from typing import Any
from uuid import uuid4

import anyio
import anyio.to_thread

from everos.core.errors import MemoryRootLockedError, ProviderNotConfiguredError
from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import to_timestamp_ms
from everos.memory.events import AgentCaseExtracted, EpisodeExtracted
from everos.memory.ports import OwnerType, Scope
from everos.memory.strategies import (
    TRIGGER_PROFILE_CLUSTERING_SPEC,
    TRIGGER_SKILL_CLUSTERING_SPEC,
)
from everos.service.ops._backfill import (
    BACKFILL_LOGGER,
    BackfillDeps,
    BackfillPresenter,
    ClustersPhaseResult,
    ReplayEngine,
    progress_due,
)

logger = get_logger(BACKFILL_LOGGER)


def _row_scope(raw: dict[str, Any], owner_type: OwnerType) -> Scope:
    """The ownership scope of one raw index row."""
    return Scope(
        app_id=raw["app_id"],
        project_id=raw["project_id"],
        owner_id=raw["owner_id"],
        owner_type=owner_type,
    )


async def _scan_all_rows(deps: BackfillDeps, table: str) -> list[dict[str, Any]]:
    """Fetch every row of ``table``, no filter.

    Phase 2 doesn't care whether a row already carries a vector — the
    cluster strategies re-embed the row's text themselves (see
    ``trigger_profile_clustering`` / ``trigger_skill_clustering``); it
    just needs every existing episode / agent case so it can
    synthesize the trigger event Tier 1's gated-off strategies never
    emitted (embed-requiring strategies are body-guarded off when no
    embedding provider is configured — see :mod:`everos.memory.strategies`).

    Callers that need a ``parent_type`` filter (e.g. the Episode scan in
    :func:`run_clusters_phase`, which excludes Reflection-merged rows)
    apply it themselves on the returned rows — this helper stays a plain
    unfiltered fetch shared by every business table.
    """
    records = await deps.vector_tables[table].scan()
    return [record.model_dump(mode="python") for record in records]


def _episode_row_to_event(raw: dict[str, Any]) -> EpisodeExtracted:
    """Synthesize the ``EpisodeExtracted(source="pipeline")`` this row's
    original pipeline run never emitted — Tier 1 gated
    ``trigger_profile_clustering`` off before it could fire (its
    per-dispatch body-guard returns early when embedding is
    unavailable).

    ``event_id`` carries a ``backfill_`` prefix so ops can tell a
    synthesized run apart from a real one in logs / run records.
    """
    return EpisodeExtracted(
        event_id=f"backfill_{uuid4().hex}",
        memcell_id=raw["parent_id"],
        episode_entry_id=raw["entry_id"],
        episode_text=raw["episode"],
        episode_timestamp_ms=to_timestamp_ms(raw["timestamp"]),
        owner_id=raw["owner_id"],
        session_id=raw.get("session_id"),
        app_id=raw.get("app_id", "default"),
        project_id=raw.get("project_id", "default"),
        source="pipeline",
    )


def _agent_case_row_to_event(raw: dict[str, Any]) -> AgentCaseExtracted:
    """Synthesize the ``AgentCaseExtracted`` this row's original pipeline
    run never emitted, mirroring :func:`_episode_row_to_event`."""
    return AgentCaseExtracted(
        event_id=f"backfill_{uuid4().hex}",
        memcell_id=raw["parent_id"],
        case_entry_id=raw["entry_id"],
        task_intent=raw["task_intent"],
        quality_score=raw["quality_score"],
        case_timestamp_ms=to_timestamp_ms(raw["timestamp"]),
        agent_id=raw["owner_id"],
        app_id=raw.get("app_id", "default"),
        project_id=raw.get("project_id", "default"),
    )


async def _emit_synthetic_events(
    deps: BackfillDeps,
    engine: ReplayEngine,
    episodes: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    *,
    presenter: BackfillPresenter,
) -> int:
    """Fan every row into ``engine`` as its own synthetic trigger event.

    Skip rows that a prior Phase-2 run already clustered — otherwise a
    rerun (after Ctrl-C, or ``everos cascade backfill --phase all --yes``
    on a partially-completed root) would re-cluster the same rows and
    grow cluster counts spuriously. ``ClusterStore.find_cluster_for_member``
    is O(log N) via the reverse index and is the exact primitive for
    this dedup. ``member_type`` values (``"episode"`` / ``"case"``) match
    what :func:`trigger_profile_clustering` and
    :func:`trigger_skill_clustering` insert on the write path — a mismatch
    here would silently disable the skip and re-open the double-cluster
    window.

    Episodes first, then agent cases — order doesn't affect correctness
    (each event routes to its own strategy independently); it only
    keeps the progress readout monotonic. The progress counter advances
    for skipped rows too so the readout matches the pre-scan estimate;
    ``ClustersPhaseResult.events_emitted`` reflects "rows processed"
    (real emits + already-clustered skips), not "engine.emit calls".
    """
    total = len(episodes) + len(cases)
    emitted = 0
    clusters = deps.clusters
    for raw in episodes:
        # entry_id is only per-owner unique — scope the reverse lookup
        # so a same-day, same-seq episode under a different owner does
        # not falsely match this owner's cluster (or vice versa).
        existing = await clusters.find_cluster_for_member(
            _row_scope(raw, "user"), "episode", raw["entry_id"]
        )
        if existing is None:
            await engine.emit(_episode_row_to_event(raw))
        emitted += 1
        if progress_due(emitted, total):
            presenter.emit_progress(emitted, total)
    for raw in cases:
        existing = await clusters.find_cluster_for_member(
            _row_scope(raw, "agent"), "case", raw["entry_id"]
        )
        if existing is None:
            await engine.emit(_agent_case_row_to_event(raw))
        emitted += 1
        if progress_due(emitted, total):
            presenter.emit_progress(emitted, total)
    return emitted


async def run_clusters_phase(
    deps: BackfillDeps, *, auto_yes: bool, presenter: BackfillPresenter
) -> ClustersPhaseResult:
    """Rebuild clusters for every episode / agent case via synthetic events.

    Scan every episode + agent case row → surface the estimate →
    confirm once → synthesize the ``EpisodeExtracted`` /
    ``AgentCaseExtracted`` event each row's original pipeline run
    would have emitted had the clustering strategies not been gated
    off under Tier 1 (their embed-requiring body-guards short-circuited
    the dispatch) → replay them through a dedicated
    engine that registers only those two (now-eligible, since embed
    is available) strategies → wait for the engine to drain.

    Idempotent: :func:`_emit_synthetic_events` filters each row through
    :meth:`ClusterStore.find_cluster_for_member` before emitting, so
    rerunning the phase (or triggering it from ``--phase all`` after a
    Ctrl-C interruption) skips rows already attached to a cluster.
    Cluster counts stop growing spuriously across reruns.

    Episode rows carrying ``parent_type == "cluster"`` are Reflection's
    merged episodes (``orchestrator._write_merged_episode``), not source
    pipeline events, and are excluded — mirrors the same
    ``parent_type == "memcell"`` filter idiom used by
    ``extract_user_profile._select_via_cluster``. Synthesizing an event
    for one would carry a bogus ``memcell_id`` (actually a cluster id)
    and defeat ``trigger_profile_clustering``'s own
    ``applies_to=lambda e: e.source == "pipeline"`` exclusion of
    Reflection output.
    """
    # Preflight capability + OME lock BEFORE any collection work. Both
    # are cheap checks; either failing makes the run impossible, so
    # doing them before ``_scan_all_rows`` (which is bounded but not
    # free) matches the Phase-1 defense-in-depth invariant. Capability
    # first: it's a permanent config gap, while the OME lock is
    # transient (stop the server and try again).
    if deps.embedding is None:
        err = ProviderNotConfiguredError(provider="embedding", feature="backfill")
        presenter.capability_missing(
            provider="embedding", feature="backfill", message=str(err)
        )
        return ClustersPhaseResult(blocked_by_capability="embedding")

    if not await anyio.to_thread.run_sync(deps.ome_lock.is_free):
        presenter.server_running()
        return ClustersPhaseResult(aborted=True, blocked_by_server=True)

    episodes = [
        row
        for row in await _scan_all_rows(deps, "episode")
        if row.get("parent_type") == "memcell"
    ]
    cases = await _scan_all_rows(deps, "agent_case")
    total = len(episodes) + len(cases)
    if total == 0:
        presenter.nothing_to_backfill(
            "Nothing to backfill. No episodes or agent cases found."
        )
        return ClustersPhaseResult()

    presenter.estimate_clusters(len(episodes), len(cases))
    if not await presenter.confirm(
        f"proceed with {total:,} memories "
        f"({len(episodes):,} episodes, {len(cases):,} agent cases)",
        auto_yes=auto_yes,
    ):
        return ClustersPhaseResult(aborted=True)

    await deps.ensure_schema()
    clusters_before = await deps.clusters.count()

    engine = deps.engine_for(
        (TRIGGER_PROFILE_CLUSTERING_SPEC, TRIGGER_SKILL_CLUSTERING_SPEC)
    )
    try:
        await engine.start()
    except MemoryRootLockedError:
        # Race window: preflight succeeded, but a server started between
        # probe and engine.start(). Same friendly path — no traceback.
        logger.warning("cascade_backfill_clusters_lock_race")
        presenter.server_running()
        return ClustersPhaseResult(aborted=True, blocked_by_server=True)
    try:
        emitted = await _emit_synthetic_events(
            deps, engine, episodes, cases, presenter=presenter
        )
        drained = await engine.wait_idle(timeout=deps.wait_seconds)
        if not drained:
            logger.warning(
                "cascade_backfill_clusters_wait_timeout",
                timeout=deps.wait_seconds,
            )
    finally:
        await engine.stop()

    clusters_after = await deps.clusters.count()
    presenter.phase_2_complete(clusters_before, clusters_after, emitted)
    return ClustersPhaseResult(
        events_emitted=emitted,
        clusters_before=clusters_before,
        clusters_after=clusters_after,
        drained=drained,
    )
