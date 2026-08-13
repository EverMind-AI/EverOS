"""extract_user_profile strategy — synthesise the user's profile from clusters.

Dual-trigger strategy: profile extraction must run whether or not embedding
is configured, so it listens on **two** events and gates itself so exactly
one path fires per memcell (see :func:`_profile_applies`):

- **Cluster path** (Tier 2+, embedding available): fires on
  :class:`ProfileClusterUpdated`, emitted by ``trigger_profile_clustering``
  after it assigns a memcell to a cluster. Pulls the relevant memcells
  across all "fresh" clusters (:func:`_select_via_cluster`).
- **Direct path** (Tier 1, no embedding): fires on :class:`EpisodeExtracted`
  directly. ``trigger_profile_clustering`` is always registered, but its
  per-dispatch body-guard returns early when
  :func:`get_embedding_capability` reports unavailable, so no
  ``ProfileClusterUpdated`` is ever emitted in Tier 1 — this direct
  path is the sole route to profile extraction there. Pulls memcells
  via a scalar LanceDB timestamp filter, no embedding required
  (:func:`_select_via_timestamp`).

Both paths converge on the same LLM extraction + markdown persist tail of
``extract_user_profile`` — only memcell *selection* differs.

Opensource parity (``mem_memorize.py`` Phase 2):

- **Throttle** (both paths): ``total_count % profile_extraction_interval
  == 0``; default interval = 1 (every memcell triggers a re-extraction).
  Applied at strategy entry so cluster and direct paths honor the same
  gate. ``total_count`` is ``sum(c.count for c in user_clusters)`` on the
  cluster path and :meth:`episode_repo.count_by_owner` on the direct
  path — both express "cumulative units of source-memory for this owner"
  and stay ~1:1 in the extract → memcell → episode → cluster pipeline.
- **Target clusters**: every cluster whose ``last_ts`` is newer than the
  user's existing profile timestamp, plus the current cluster (so the
  freshly-arrived memcell is always counted even when its cluster's
  ``last_ts`` is older than the profile baseline).
- **Input shape**: raw chat messages — algo's ``_render_conversation``
  unwraps the items list. The sqlite ``memcell.payload_json`` column is
  the long-term archive that lets us replay this beyond
  ``unprocessed_buffer``'s lifetime.

Single-sender assumption today: ``event.owner_id`` is treated as the
profile subject. Multi-user clusters land their additional sender's
profile in a follow-up turn (each cluster gets re-evaluated on every
``ProfileClusterUpdated`` for any participating user).
"""

from __future__ import annotations

from everalgo.clustering import Cluster as AlgoCluster
from everalgo.types import MemCell as AlgoMemCell
from everalgo.types import Profile as AlgoProfile
from everalgo.user_memory import ProfileExtractor

from everos.component.embedding import get_embedding_capability
from everos.component.llm import get_llm_client
from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import offline_strategy
from everos.infra.ome.events import BaseEvent
from everos.infra.ome.triggers import Immediate
from everos.infra.persistence.lancedb import episode_repo
from everos.infra.persistence.markdown import (
    ProfileReader,
    ProfileWriter,
    UserProfileFrontmatter,
)
from everos.infra.persistence.sqlite import cluster_repo, memcell_repo
from everos.memory._partition_locks import get_partition_lock
from everos.memory.events import EpisodeExtracted, ProfileClusterUpdated

logger = get_logger(__name__)

PROFILE_EXTRACTION_INTERVAL = 1
"""Opensource parity: re-extract on every Nth clustered memcell.
``N=1`` matches the opensource default; tune via :class:`Settings` once
the storage budget for profile re-extractions becomes a concern."""

PROFILE_MIN_MEMCELLS = 1
"""Opensource parity: skip when the candidate cluster set holds fewer
than ``N`` memcells across all selected clusters."""


_writer: ProfileWriter | None = None
_reader: ProfileReader | None = None


def _get_writer() -> ProfileWriter:
    global _writer
    if _writer is None:
        _writer = ProfileWriter(root=MemoryRoot.resolve())
    return _writer


def _get_reader() -> ProfileReader:
    global _reader
    if _reader is None:
        _reader = ProfileReader(root=MemoryRoot.resolve())
    return _reader


def _profile_applies(event: BaseEvent) -> bool:
    """Route exactly one path per memcell to ``extract_user_profile``.

    - :class:`ProfileClusterUpdated` always applies: it is only ever
      emitted by ``trigger_profile_clustering``, whose per-dispatch
      body-guard short-circuits (returns before emit) whenever
      :func:`get_embedding_capability` reports unavailable. Accepting
      this event unconditionally is therefore safe — it can only arrive
      when embedding is available, so it never overlaps with the direct
      path below.
    - :class:`EpisodeExtracted` applies only for pipeline-sourced
      episodes (not Reflection's merged episodes) while embedding is
      unavailable. Once embedding is available, ``trigger_profile_clustering``
      runs to completion and owns the same memcell via
      ``ProfileClusterUpdated`` — so the direct path must stand down
      here to avoid a double extraction. This check is a live capability
      read (not a registration-time snapshot), so a Tier 3→1 downgrade
      mid-process is picked up on the next event.
    """
    if isinstance(event, ProfileClusterUpdated):
        return True
    if isinstance(event, EpisodeExtracted):
        return event.source == "pipeline" and not get_embedding_capability().available
    return False


async def _select_via_cluster(
    event: ProfileClusterUpdated,
    last_profile_ts: int,
    user_clusters: list[AlgoCluster],
) -> list[str]:
    """Cluster path (Tier 2+): resolve memcells via the user's cluster set.

    The caller (:func:`extract_user_profile`) already fetched
    ``user_clusters`` for the strategy-entry throttle; we take it as a
    parameter to avoid a redundant :meth:`cluster_repo.list_for_owner`
    round-trip. Behavior of the selection step itself is unchanged from
    the pre-dual-trigger implementation.
    """
    # Pick clusters fresher than the existing profile (always include
    # the one we just updated).
    target_clusters = [
        c
        for c in user_clusters
        if c.last_ts > last_profile_ts or c.id == event.cluster_id
    ]
    if not target_clusters:
        return []

    # Resolve cluster members (episode entry_ids) → memcell_ids.
    # Cluster members store episode entry_ids. To reach the memcell
    # payloads we look up each episode's parent_id (= memcell_id)
    # in LanceDB, skipping merged episodes (parent_type=cluster)
    # whose source memcells were already processed before Reflection.
    entry_ids = [m for c in target_clusters for m in c.members]
    episodes = await episode_repo.find_by_owner_entries(
        event.owner_id,
        entry_ids,
        app_id=event.app_id,
        project_id=event.project_id,
    )
    return [
        ep.parent_id for ep in episodes if ep.parent_type == "memcell" and ep.parent_id
    ]


async def _select_via_timestamp(
    event: EpisodeExtracted, last_profile_ts: int
) -> list[str]:
    """Direct path (Tier 1): resolve memcells from the event + a LanceDB supplement.

    Always includes the current event's ``memcell_id`` — it is the
    trigger for this run and its memcell is the one whose profile we
    care about. The LanceDB scan is a best-effort supplement that adds
    older memcells still eligible under ``last_profile_ts``; the cascade
    daemon may not have indexed the freshly-arrived row yet (it runs on
    its own schedule), so relying on LanceDB alone would miss the first
    memory of a fresh Tier-1 install entirely. See
    :class:`EpisodeExtracted` for the event-first contract that lets us
    dodge this race — it carries ``memcell_id`` precisely so downstream
    strategies do not need to poll LanceDB until the row appears.

    Mirrors the cluster path's ``c.id == event.cluster_id`` fallback
    (see :func:`_select_via_cluster`): the currently-firing entity is
    always in the candidate set even when its timestamp is not strictly
    fresher than the last profile — this matters for bulk imports where
    every incoming memcell may carry a historical timestamp.

    Order is unstable across runs (the return is a de-duplicated set
    materialised to a list); downstream ``memcell_repo.find_by_ids``
    reorders back to the caller's list order, so this does not
    destabilise later stages.
    """
    memcell_ids: set[str] = {event.memcell_id}
    # Column projection: this selector only needs `parent_id` to seed the
    # memcell fetch — pulling full rows would drag the two 1024-D vector
    # columns across the wire for every historical episode. Raw-dict
    # return is contractual when `columns` is set.
    supplement = await episode_repo.list_by_owner_after_ts(
        owner_id=event.owner_id,
        after_ts=last_profile_ts,
        parent_type="memcell",
        app_id=event.app_id,
        project_id=event.project_id,
        columns=["parent_id"],
    )
    memcell_ids.update(row["parent_id"] for row in supplement if row.get("parent_id"))
    return list(memcell_ids)


@offline_strategy(
    name="extract_user_profile",
    trigger=Immediate(on=[ProfileClusterUpdated, EpisodeExtracted]),
    applies_to=_profile_applies,
    emits=[],
    max_retries=2,
)
async def extract_user_profile(
    event: ProfileClusterUpdated | EpisodeExtracted, ctx: StrategyContext
) -> None:
    # Serialise on owner_id: user.md is a single per-user file and the
    # body is a read → LLM merge → overwrite sequence. Different users
    # run fully in parallel.
    partition = f"{event.app_id}:{event.project_id}:{event.owner_id}"
    async with get_partition_lock("extract_user_profile", partition):
        existing = await _get_reader().read(
            event.owner_id,
            schema=UserProfileFrontmatter,
            app_id=event.app_id,
            project_id=event.project_id,
        )
        last_profile_ts = existing[0].profile_timestamp_ms if existing else 0

        # Unified throttle: both paths gate on "cumulative units of
        # source-memory for this owner". Cluster path sums cluster
        # counts (pre-refactor semantics); direct path counts owner
        # episode rows in LanceDB. Applied here so a bumped interval
        # (e.g. 10x to cap LLM cost) throttles both Tier 1 and Tier 2+
        # equally instead of only the cluster path.
        user_clusters: list[AlgoCluster] | None
        if isinstance(event, ProfileClusterUpdated):
            user_clusters = await cluster_repo.list_for_owner(
                event.owner_id,
                "user_memory",
                app_id=event.app_id,
                project_id=event.project_id,
            )
            total_count = sum(c.count for c in user_clusters)
        else:
            user_clusters = None
            # Scope the counter to parent_type='memcell' so it matches
            # _select_via_timestamp's selector below — otherwise Reflection-
            # merged rows (parent_type='cluster') inflate the throttle count
            # without ever being selectable, firing the gate at the wrong cadence.
            # TODO(profile-counter): reads LanceDB and therefore races the
            # cascade daemon in the same way _select_via_timestamp used to
            # (fresh Tier-1 install → count=0 until cascade catches up).
            # The throttle only needs a monotonic per-owner integer, so a
            # stale-but-monotonic value is acceptable for now; the followup
            # is to source this from a sqlite ``memcell`` count-by-owner
            # query — see PR #361 review finding M4.
            total_count = await episode_repo.count_by_owner(
                event.owner_id,
                app_id=event.app_id,
                project_id=event.project_id,
                parent_type="memcell",
            )

        if (
            PROFILE_EXTRACTION_INTERVAL > 1
            and total_count % PROFILE_EXTRACTION_INTERVAL != 0
        ):
            logger.info(
                "profile_extraction_throttled",
                owner_id=event.owner_id,
                total_count=total_count,
                interval=PROFILE_EXTRACTION_INTERVAL,
                path="cluster"
                if isinstance(event, ProfileClusterUpdated)
                else "direct",
            )
            return

        # Memcell selection is the only part that differs by trigger; the
        # LLM extraction + persist tail below is shared by both paths.
        # user_clusters is always populated on the cluster branch above,
        # so the fallback to [] here is unreachable — it exists purely to
        # satisfy the type checker without an assert.
        if isinstance(event, ProfileClusterUpdated):
            memcell_ids = await _select_via_cluster(
                event, last_profile_ts, user_clusters or []
            )
        else:
            memcell_ids = await _select_via_timestamp(event, last_profile_ts)

        if len(memcell_ids) < PROFILE_MIN_MEMCELLS:
            logger.info(
                "profile_extraction_below_min_memcells",
                owner_id=event.owner_id,
                memcell_count=len(memcell_ids),
                threshold=PROFILE_MIN_MEMCELLS,
            )
            return

        # Pull memcell payloads from SQLite, rehydrate to algo types.
        memcell_rows = await memcell_repo.find_by_ids(memcell_ids)
        algo_memcells = sorted(
            (AlgoMemCell.model_validate_json(r.payload_json) for r in memcell_rows),
            key=lambda mc: mc.timestamp,
        )
        if not algo_memcells:
            return

        # Run the LLM extractor — INIT (no prior) or UPDATE (existing).
        old_profile = _to_algo_profile(existing[0]) if existing else None
        extractor = ProfileExtractor(llm=get_llm_client())
        new_profile = await extractor.aextract(
            algo_memcells, sender_id=event.owner_id, old_profile=old_profile
        )

        # Write the fresh profile back to users/<user_id>/user.md.
        await _persist_profile(
            new_profile,
            owner_id=event.owner_id,
            app_id=event.app_id,
            project_id=event.project_id,
        )
    logger.info(
        "user_profile_extracted",
        owner_id=event.owner_id,
        path="cluster" if isinstance(event, ProfileClusterUpdated) else "direct",
        memcell_count=len(algo_memcells),
        mode="UPDATE" if old_profile is not None else "INIT",
    )


# ── helpers ──────────────────────────────────────────────────────────────


def _to_algo_profile(fm: UserProfileFrontmatter) -> AlgoProfile:
    """Rehydrate an algo :class:`Profile` from the markdown frontmatter."""
    return AlgoProfile.model_validate(
        {
            "owner_id": fm.user_id,
            "summary": fm.summary,
            "timestamp": fm.profile_timestamp_ms,
            "explicit_info": list(fm.explicit_info),
            "implicit_traits": list(fm.implicit_traits),
        }
    )


async def _persist_profile(
    profile: AlgoProfile, *, owner_id: str, app_id: str, project_id: str
) -> None:
    """Write the freshly extracted profile to ``users/<user_id>/user.md``."""
    extras = profile.model_dump(exclude={"owner_id", "summary", "timestamp"})
    explicit_info = extras.get("explicit_info") or []
    implicit_traits = extras.get("implicit_traits") or []
    frontmatter = UserProfileFrontmatter(
        id=f"profile_{owner_id}",
        user_id=owner_id,
        summary=profile.summary,
        explicit_info=list(explicit_info),
        implicit_traits=list(implicit_traits),
        profile_timestamp_ms=profile.timestamp,
    )
    await _get_writer().write(
        owner_id,
        frontmatter=frontmatter,
        body=profile.summary,
        app_id=app_id,
        project_id=project_id,
    )
