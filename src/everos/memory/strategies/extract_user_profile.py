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
  body-guard returns early when no embedding provider is configured, so
  no ``ProfileClusterUpdated`` is ever produced in Tier 1 — this direct
  path is the sole route to profile extraction there. Pulls memcells
  via a scalar episode-index timestamp filter, no embedding required
  (:func:`_select_via_timestamp`).

Both paths converge on the same LLM extraction + markdown persist tail of
``extract_user_profile`` — only memcell *selection* differs.

Opensource parity (``mem_memorize.py`` Phase 2):

- **Throttle** (both paths): ``total_count % profile_extraction_interval
  == 0``; default interval = 1 (every memcell triggers a re-extraction).
  Applied at strategy entry so cluster and direct paths honor the same
  gate. ``total_count`` is ``sum(c.count for c in user_clusters)`` on the
  cluster path and the owner's memcell-episode count in the episode
  index on the direct path — both express "cumulative units of
  source-memory for this owner" and stay ~1:1 in the extract → memcell
  → episode → cluster pipeline.
- **Target clusters**: every cluster whose ``last_ts`` is newer than the
  user's existing profile timestamp, plus the current cluster (so the
  freshly-arrived memcell is always counted even when its cluster's
  ``last_ts`` is older than the profile baseline).
- **Input shape**: raw chat messages — algo's ``_render_conversation``
  unwraps the items list. The memcell store is the long-term archive
  that lets us replay this beyond the session buffer's lifetime.

Single-sender assumption today: ``event.owner_id`` is treated as the
profile subject. Multi-user clusters land their additional sender's
profile in a follow-up turn (each cluster gets re-evaluated on every
``ProfileClusterUpdated`` for any participating user).
"""

from __future__ import annotations

from everalgo.user_memory import ProfileExtractor

from everos.core.observability.logging import get_logger
from everos.memory._partition_locks import get_partition_lock
from everos.memory.deps import MemoryDeps
from everos.memory.events import EpisodeExtracted, ProfileClusterUpdated
from everos.memory.mappers import profile as profile_mapper
from everos.memory.models import Cluster, Profile
from everos.memory.ports import Event, Immediate, Scope, StrategySpec

logger = get_logger(__name__)

PROFILE_EXTRACTION_INTERVAL = 1
"""Opensource parity: re-extract on every Nth clustered memcell.
``N=1`` matches the opensource default; tune via :class:`Settings` once
the storage budget for profile re-extractions becomes a concern."""

PROFILE_MIN_MEMCELLS = 1
"""Opensource parity: skip when the candidate cluster set holds fewer
than ``N`` memcells across all selected clusters."""


def _profile_applies(event: Event, deps: MemoryDeps) -> bool:
    """Route exactly one path per memcell to ``extract_user_profile``.

    - :class:`ProfileClusterUpdated` always applies: only
      ``trigger_profile_clustering`` produces it, and its body-guard
      returns before producing anything when no embedding provider is
      configured — so it never overlaps with the direct path below.
    - :class:`EpisodeExtracted` applies only for pipeline-sourced
      episodes (not Reflection's merged episodes) while no embedding
      provider is configured. With one, ``trigger_profile_clustering``
      owns the same memcell via ``ProfileClusterUpdated``, so the direct
      path must stand down to avoid a double extraction. The provider is
      fixed for the process (a tier change needs a restart), so deciding
      at dispatch time is exact.
    """
    if isinstance(event, ProfileClusterUpdated):
        return True
    if isinstance(event, EpisodeExtracted):
        return event.source == "pipeline" and deps.embedding is None
    return False


async def _select_via_cluster(
    event: ProfileClusterUpdated,
    last_profile_ts: int,
    user_clusters: list[Cluster],
    deps: MemoryDeps,
    scope: Scope,
) -> list[str]:
    """Cluster path (Tier 2+): resolve memcells via the user's cluster set.

    The caller (:func:`extract_user_profile`) already fetched
    ``user_clusters`` for the strategy-entry throttle; we take it as a
    parameter to avoid a redundant cluster-store round-trip.
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
    # in the episode index, skipping merged episodes (parent_type=cluster)
    # whose source memcells were already processed before Reflection.
    entry_ids = [m for c in target_clusters for m in c.members]
    episodes = await deps.indexes.episode.find_by_entry_ids(scope, entry_ids)
    return [
        ep.item.parent_id
        for ep in episodes
        if ep.item.parent_type == "memcell" and ep.item.parent_id
    ]


async def _select_via_timestamp(
    event: EpisodeExtracted, last_profile_ts: int, deps: MemoryDeps, scope: Scope
) -> list[str]:
    """Direct path (Tier 1): resolve memcells from the event + an index supplement.

    Always includes the current event's ``memcell_id`` — it is the
    trigger for this run and its memcell is the one whose profile we
    care about. The index scan is a best-effort supplement that adds
    older memcells still eligible under ``last_profile_ts``; the cascade
    daemon may not have indexed the freshly-arrived row yet (it runs on
    its own schedule), so relying on the index alone would miss the first
    memory of a fresh Tier-1 install entirely. See
    :class:`EpisodeExtracted` for the event-first contract that lets us
    dodge this race — it carries ``memcell_id`` precisely so downstream
    strategies do not need to poll the index until the row appears.

    Mirrors the cluster path's ``c.id == event.cluster_id`` fallback
    (see :func:`_select_via_cluster`): the currently-firing entity is
    always in the candidate set even when its timestamp is not strictly
    fresher than the last profile — this matters for bulk imports where
    every incoming memcell may carry a historical timestamp.

    Order is unstable across runs (the return is a de-duplicated set
    materialised to a list); the memcells are sorted by timestamp after
    they are fetched, so this does not destabilise later stages.
    """
    memcell_ids: set[str] = {event.memcell_id}
    supplement = await deps.indexes.episode.list_after(
        scope, last_profile_ts, parent_type="memcell"
    )
    memcell_ids.update(row.item.parent_id for row in supplement if row.item.parent_id)
    return list(memcell_ids)


def _profile_language(
    existing: Profile | None, event: ProfileClusterUpdated | EpisodeExtracted
) -> str | None:
    """The language to write the profile in, frozen once known.

    A stored profile's ``primary_language`` wins and is never changed
    automatically; changing it is a manual edit. Until one is stored, the
    triggering episode's language is used and then frozen.
    """
    if existing is not None and existing.primary_language:
        return existing.primary_language
    return event.language


async def extract_user_profile(
    event: ProfileClusterUpdated | EpisodeExtracted, deps: MemoryDeps
) -> list[Event]:
    """Re-synthesise the owner's profile from the memcells that changed."""
    scope = Scope(
        app_id=event.app_id,
        project_id=event.project_id,
        owner_id=event.owner_id,
        owner_type="user",
    )
    # Serialise on owner_id: user.md is a single per-user file and the
    # body is a read → LLM merge → overwrite sequence. Different users
    # run fully in parallel.
    partition = f"{event.app_id}:{event.project_id}:{event.owner_id}"
    async with get_partition_lock("extract_user_profile", partition):
        existing = await deps.stores.profile.read(scope)
        last_profile_ts = existing.timestamp if existing else 0

        # Unified throttle: both paths gate on "cumulative units of
        # source-memory for this owner". Cluster path sums cluster
        # counts (pre-refactor semantics); direct path counts owner
        # episode rows in the index. Applied here so a bumped interval
        # (e.g. 10x to cap LLM cost) throttles both Tier 1 and Tier 2+
        # equally instead of only the cluster path.
        user_clusters: list[Cluster] | None
        if isinstance(event, ProfileClusterUpdated):
            user_clusters = await deps.stores.cluster.list_for_owner(
                scope, "user_memory"
            )
            total_count = sum(c.count for c in user_clusters)
        else:
            user_clusters = None
            # Scope the counter to parent_type='memcell' so it matches
            # _select_via_timestamp's selector below — otherwise Reflection-
            # merged rows (parent_type='cluster') inflate the throttle count
            # without ever being selectable, firing the gate at the wrong cadence.
            # TODO(profile-counter): reads the index and therefore races the
            # cascade daemon in the same way _select_via_timestamp used to
            # (fresh Tier-1 install → count=0 until cascade catches up).
            # The throttle only needs a monotonic per-owner integer, so a
            # stale-but-monotonic value is acceptable for now; the followup
            # is to source this from a memcell count-by-owner query — see
            # PR #361 review finding M4.
            total_count = await deps.indexes.episode.count(scope, parent_type="memcell")

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
            return []

        # Memcell selection is the only part that differs by trigger; the
        # LLM extraction + persist tail below is shared by both paths.
        # user_clusters is always populated on the cluster branch above,
        # so the fallback to [] here is unreachable — it exists purely to
        # satisfy the type checker without an assert.
        if isinstance(event, ProfileClusterUpdated):
            memcell_ids = await _select_via_cluster(
                event, last_profile_ts, user_clusters or [], deps, scope
            )
        else:
            memcell_ids = await _select_via_timestamp(
                event, last_profile_ts, deps, scope
            )

        if len(memcell_ids) < PROFILE_MIN_MEMCELLS:
            logger.info(
                "profile_extraction_below_min_memcells",
                owner_id=event.owner_id,
                memcell_count=len(memcell_ids),
                threshold=PROFILE_MIN_MEMCELLS,
            )
            return []

        memcells = sorted(
            await deps.stores.memcell.find_by_ids(memcell_ids),
            key=lambda mc: mc.timestamp,
        )
        if not memcells:
            return []

        # Run the LLM extractor — INIT (no prior) or UPDATE (existing).
        old_profile = profile_mapper.to_algo(existing) if existing else None
        language = _profile_language(existing, event)
        extractor = ProfileExtractor(llm=deps.require_llm())
        new_profile = await extractor.aextract(
            memcells,
            sender_id=event.owner_id,
            old_profile=old_profile,
            custom_instructions=deps.instructions.load("profile_extract"),
            output_language=language,
        )

        # Write the fresh profile back to the owner's profile document.
        await deps.stores.profile.write(
            scope, profile_mapper.from_algo(new_profile, primary_language=language)
        )
    logger.info(
        "user_profile_extracted",
        owner_id=event.owner_id,
        path="cluster" if isinstance(event, ProfileClusterUpdated) else "direct",
        memcell_count=len(memcells),
        mode="UPDATE" if old_profile is not None else "INIT",
    )
    return []


EXTRACT_USER_PROFILE_SPEC = StrategySpec(
    name="extract_user_profile",
    fn=extract_user_profile,
    trigger=Immediate(on=[ProfileClusterUpdated, EpisodeExtracted]),
    applies_to=_profile_applies,
    max_retries=2,
)
