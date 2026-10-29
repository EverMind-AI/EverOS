"""Backfill phase ``skills``: extract agent skills from clustered cases.

The clusters phase's replay engine runs the skill-clustering strategy but
not ``extract_agent_skill``, so every ``SkillClusterUpdated`` it produced
went unheard. This phase finds each agent-case cluster without a skill,
replays the event for its members through a throw-away engine that runs
only ``extract_agent_skill``, and syncs the written ``SKILL.md`` files into
the index.
"""

from __future__ import annotations

__all__ = ["run_skills_phase"]

import dataclasses
from uuid import uuid4

import anyio
import anyio.to_thread

from everos.core.errors import MemoryRootLockedError, ProviderNotConfiguredError
from everos.core.observability.logging import get_logger
from everos.memory.events import SkillClusterUpdated
from everos.memory.strategies import EXTRACT_AGENT_SKILL_SPEC
from everos.service.ops._backfill import (
    BACKFILL_LOGGER,
    BackfillDeps,
    BackfillPresenter,
    SkillsPhaseResult,
    progress_due,
)

logger = get_logger(BACKFILL_LOGGER)


@dataclasses.dataclass(frozen=True)
class _SkillSourceRow:
    """One (case, cluster) pairing Phase 3 replays as a synthetic
    ``SkillClusterUpdated`` event.

    Mirrors one row of the agent-case-kind cluster's membership
    (``ClusterMember`` via :meth:`ClusterStore.list_for_owner`) — the
    same shape :func:`trigger_skill_clustering` itself emits per fresh
    case, just synthesized after the fact for a case Phase 3 finds
    already clustered but not yet skill-extracted.
    """

    case_entry_id: str
    cluster_id: str
    agent_id: str
    app_id: str
    project_id: str


async def _scan_skill_source(deps: BackfillDeps) -> list[_SkillSourceRow]:
    """Enumerate every clustered agent case Phase 3 should (re-)replay.

    ``trigger_skill_clustering`` runs on the ``agent_case`` track
    (``owner_type == "agent"``, ``kind == "agent_case"`` — see
    ``Cluster.kind`` docstring); Phase 2's throwaway engine
    (see :func:`run_clusters_phase`) registers that strategy but not
    ``extract_agent_skill``, so every ``SkillClusterUpdated`` it emitted
    during Phase 2 had no listener and was dropped. This scan walks
    every agent-owned cluster :meth:`ClusterStore.list_distinct_owners`
    has ever seen and fans each cluster's members out into one row —
    Phase 3's job is to replay the event those members never got
    handled for.

    Idempotent: a cluster is skipped when EITHER
    ``AgentSkillIndex.count_in_cluster`` > 0 (LanceDB knows about it)
    OR :meth:`SkillFiles.names_cluster` returns ``True`` (a
    ``SKILL.md`` on disk names this cluster). The disk check closes
    the failure window where ``extract_agent_skill`` wrote the md but
    Phase 3 was interrupted before ``_sync_new_skill_files`` ran — a
    later re-run would otherwise re-extract every such cluster and
    double the skill count.
    """
    rows: list[_SkillSourceRow] = []
    clusters = deps.clusters
    skills = deps.skills
    for owner in await clusters.list_distinct_owners():
        if owner.owner_type != "agent":
            continue
        owner_id, app_id, project_id = owner.owner_id, owner.app_id, owner.project_id
        for cluster in await clusters.list_for_owner(owner, "agent_case"):
            assert cluster.id is not None  # persisted clusters always carry an id
            already_extracted = await skills.count_in_cluster(owner, cluster.id)
            if already_extracted:
                continue
            if await deps.skill_files.names_cluster(owner, cluster.id):
                logger.info(
                    "cascade_backfill_skill_md_present_skip_recluster",
                    cluster_id=cluster.id,
                    agent_id=owner_id,
                )
                continue
            rows.extend(
                _SkillSourceRow(
                    case_entry_id=case_entry_id,
                    cluster_id=cluster.id,
                    agent_id=owner_id,
                    app_id=app_id,
                    project_id=project_id,
                )
                for case_entry_id in cluster.members
            )
    return rows


def _skill_source_to_event(row: _SkillSourceRow) -> SkillClusterUpdated:
    """Synthesize the ``SkillClusterUpdated`` Phase 2's clustering pass
    emitted into a void — see :func:`_scan_skill_source`."""
    return SkillClusterUpdated(
        event_id=f"backfill_{uuid4().hex}",
        case_entry_id=row.case_entry_id,
        cluster_id=row.cluster_id,
        agent_id=row.agent_id,
        app_id=row.app_id,
        project_id=row.project_id,
    )


async def _sync_new_skill_files(deps: BackfillDeps) -> None:
    """Drain the cascade once so the ``SKILL.md`` files just written are indexed.

    The vectors and clusters phases write their own storage and are
    consistent as soon as they return. ``extract_agent_skill`` only writes
    markdown and relies on cascade for the index, so this phase has to ask
    cascade to catch up, or ``skills_after`` would under-report.
    """
    # Scope the sync to ``agent_skill`` only: an unscoped sweep would
    # walk every registered kind including knowledge_document /
    # knowledge_topic, and if the process has embedding but not rerank,
    # the knowledge handlers are gated off — cascade would then mark
    # every unseen knowledge md as permanently failed. A backfill run
    # must not pollute unrelated kinds' queue state.
    await deps.cascade.sync_once(kinds={"agent_skill"})


async def run_skills_phase(
    deps: BackfillDeps, *, auto_yes: bool, presenter: BackfillPresenter
) -> SkillsPhaseResult:
    """Extract agent skills from Phase 2's clustered agent cases.

    Preflight → sync orphan ``SKILL.md`` files → scan every agent-case
    cluster missing a skill → surface the estimate → confirm once →
    replay the ``SkillClusterUpdated`` event each clustered case never
    got handled for (see :func:`_scan_skill_source`) through a
    dedicated engine that registers only ``extract_agent_skill`` →
    wait for it to drain.

    Sync runs BEFORE :func:`_scan_skill_source` — not after the drain —
    for the mid-Phase-3 Ctrl-C recovery path: ``extract_agent_skill``
    writes ``SKILL.md`` first and only reaches LanceDB via cascade
    afterwards, so an interrupted Phase 3 can leave orphan ``SKILL.md``
    files on disk. Group F's on-disk idempotency probe
    (:meth:`SkillFiles.names_cluster`) skips clusters whose
    ``SKILL.md`` already exists, so on a rerun ``_scan_skill_source``
    can legitimately return an empty list even though those md files
    aren't yet indexed. Running sync unconditionally, before the scan,
    guarantees those orphans get picked up regardless of whether the
    scan finds fresh work. :func:`_sync_new_skill_files` is a no-op
    when the cascade queue is empty, so the cost is one
    ``scan_once`` + ``drain_until_empty`` round-trip.
    """
    await deps.ensure_schema()

    # Preflight capability + OME lock BEFORE the (potentially large)
    # cluster + membership scan. Phase 3 depends on Phase 2's
    # clusters, which only exist if embed was available at OME
    # startup — the feature label reflects what Phase 3 actually does
    # (skill extraction) so the toml hint names the right context.
    # Ordering mirrors :func:`run_clusters_phase`.
    if deps.embedding is None:
        err = ProviderNotConfiguredError(
            provider="embedding", feature="skill_extraction_backfill"
        )
        presenter.capability_missing(
            provider="embedding",
            feature="skill_extraction_backfill",
            message=str(err),
        )
        return SkillsPhaseResult(blocked_by_capability="embedding")

    if not await anyio.to_thread.run_sync(deps.ome_lock.is_free):
        presenter.server_running()
        return SkillsPhaseResult(aborted=True, blocked_by_server=True)

    # Recover orphan SKILL.md files a prior interrupted Phase 3 may
    # have left on disk (see docstring). Runs before the scan so the
    # on-disk idempotency probe can't strand those md files unindexed.
    await _sync_new_skill_files(deps)

    source_rows = await _scan_skill_source(deps)
    if not source_rows:
        presenter.nothing_to_backfill(
            "Nothing to backfill. No agent skill clusters found."
        )
        return SkillsPhaseResult()

    cluster_count = len({row.cluster_id for row in source_rows})
    presenter.estimate_skills(len(source_rows), cluster_count)
    if not await presenter.confirm(
        f"proceed with {len(source_rows):,} agent case(s) across "
        f"{cluster_count:,} cluster(s)",
        auto_yes=auto_yes,
    ):
        return SkillsPhaseResult(aborted=True)

    skills_before = await deps.skills.count()

    engine = deps.engine_for((EXTRACT_AGENT_SKILL_SPEC,))
    try:
        await engine.start()
    except MemoryRootLockedError:
        # Race window mirror of the Phase 2 handling in run_clusters_phase.
        logger.warning("cascade_backfill_skills_lock_race")
        presenter.server_running()
        return SkillsPhaseResult(aborted=True, blocked_by_server=True)
    try:
        emitted = 0
        for row in source_rows:
            await engine.emit(_skill_source_to_event(row))
            emitted += 1
            if progress_due(emitted, len(source_rows)):
                presenter.emit_progress(emitted, len(source_rows))
        drained = await engine.wait_idle(timeout=deps.wait_seconds)
        if not drained:
            logger.warning(
                "cascade_backfill_skills_wait_timeout",
                timeout=deps.wait_seconds,
            )
    finally:
        await engine.stop()

    # Sync of orphan SKILL.md files ran before the scan (see docstring);
    # the new files this drain just wrote also need to land in LanceDB
    # before ``skills_after`` is read, or the reported count would
    # under-report until some later, unrelated cascade run happens.
    await _sync_new_skill_files(deps)
    skills_after = await deps.skills.count()

    presenter.phase_3_complete(skills_before, skills_after, emitted)
    return SkillsPhaseResult(
        events_emitted=emitted,
        skills_before=skills_before,
        skills_after=skills_after,
        drained=drained,
    )
