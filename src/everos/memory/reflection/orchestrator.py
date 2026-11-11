"""Reflection — Select -> Merge -> Deprecate, in two steps.

Consolidates fragmented cluster members (memcell-derived episodes) into a
single high-quality merged episode per cluster:

- :class:`ReflectionOrchestrator` selects clusters, merges each one, writes
  the merged episode to md and returns an ``EpisodeExtracted`` carrying a
  :class:`~everos.memory.events.ReflectionContext` — the snapshot the second
  step acts on.
- :class:`ReflectionFinisher`, run by ``finish_reflection`` on that event,
  deprecates the replaced members in md (the cascade carries the marks into
  the index), moves the cluster onto the merged episode and records a
  ``ReflectionReport``.

The second step starts as soon as the merged episode is written; it waits
for nothing else. If it fails, the merged episode stays and the originals
stay live, which a later cycle detects as an orphan.

See ``local/2026-06-14-reflection-everos-design.md`` for the full design.
"""

from __future__ import annotations

import uuid
from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    from everalgo.types import Episode as AlgoEpisode
    from everalgo.user_memory import EpisodeReflector

    from everos.component.embedding import EmbeddingProvider

from everos.core.errors import AppError
from everos.core.locks import LOCKS, owner_key
from everos.core.observability.logging import get_logger
from everos.core.observability.tracing import memory_span
from everos.memory._partition_locks import get_partition_lock, user_cluster_lock
from everos.memory.events import EpisodeExtracted, ReflectedEpisode, ReflectionContext
from everos.memory.models import ClusterKind, Episode, ReflectionReport
from everos.memory.ports import (
    AtomicFactIndex,
    AtomicFactStore,
    ClusterStore,
    EpisodeIndex,
    EpisodeStore,
    Filter,
    MemcellStore,
    ReflectionReportStore,
    Scope,
    SourceCheck,
    Stored,
    all_of,
    eq,
    is_null,
    one_of,
)

logger = get_logger(__name__)

_MAX_CLUSTERS_PER_RUN = 10
_PREVIEW_CHARS = 200

type StoredEpisode = Stored[Episode]
type ReflectionMode = Literal["init", "update"]


class _Joined(NamedTuple):
    """A cluster member that joined while reflection ran."""

    vector: Sequence[float]
    timestamp: int


class ReflectionOrchestrator:
    """Merge the eligible clusters of one owner scope.

    Args:
        cluster_store: Clusters and their membership.
        episode_store: Episode truth (merged episode write).
        episode_index: Episode reads (members, orphans).
        memcell_store: Whether a source episode's memcell still exists.
        report_store: Reflection audit trail (which clusters were merged).
        reflector: Algorithm-side EpisodeReflector (areflect).
        custom_instructions: Appended to the reflector's rules; ``None`` for none.
    """

    def __init__(
        self,
        *,
        cluster_store: ClusterStore,
        episode_store: EpisodeStore,
        episode_index: EpisodeIndex,
        memcell_store: MemcellStore,
        report_store: ReflectionReportStore,
        reflector: EpisodeReflector,
        custom_instructions: str | None = None,
    ) -> None:
        self._cluster_store = cluster_store
        self._episode_store = episode_store
        self._episode_index = episode_index
        self._memcell_store = memcell_store
        self._report_store = report_store
        self._reflector = reflector
        self._custom_instructions = custom_instructions

    async def run(
        self,
        *,
        scope: Scope,
        kind: ClusterKind = "user_memory",
    ) -> list[EpisodeExtracted]:
        """Merge each eligible cluster of ``scope``.

        Args:
            scope: Owner scope whose clusters are reflected.
            kind: Memory kind for cluster lookup.

        Returns:
            One ``EpisodeExtracted`` per merged cluster, carrying the
            context :class:`ReflectionFinisher` acts on.
        """
        candidates = await self._select_candidates(scope=scope, kind=kind)
        logger.info(
            "reflection_candidates_selected",
            owner_id=scope.owner_id,
            candidate_count=len(candidates),
        )
        if not candidates:
            return []

        merged: list[EpisodeExtracted] = []
        for cluster_id in candidates:
            event = await self._process_cluster_safely(
                cluster_id=cluster_id, scope=scope
            )
            if event is not None:
                merged.append(event)

        logger.info(
            "reflection_cycle_completed",
            owner_id=scope.owner_id,
            success_count=len(merged),
            skip_count=len(candidates) - len(merged),
        )
        return merged

    async def _process_cluster_safely(
        self, *, cluster_id: str, scope: Scope
    ) -> EpisodeExtracted | None:
        """Merge one cluster, catching errors to allow the cycle to continue.

        Returns:
            The merged episode's event on success, ``None`` on skip or error.
        """
        try:
            return await self._process_cluster(cluster_id=cluster_id, scope=scope)
        except AppError:
            logger.warning(
                "reflection_cluster_skipped",
                cluster_id=cluster_id,
                exc_info=True,
            )
            return None
        except Exception:
            logger.error(
                "reflection_cluster_unexpected_error",
                cluster_id=cluster_id,
                exc_info=True,
            )
            return None

    # ── SELECT ────────────────────────────────────────────────────────────

    async def _select_candidates(self, *, scope: Scope, kind: ClusterKind) -> list[str]:
        """Two-step DB-agnostic candidate selection.

        Returns:
            Cluster IDs sorted by member count descending, limited
            to ``_MAX_CLUSTERS_PER_RUN``.
        """
        reflected = await self._report_store.reflected_cluster_ids(scope)
        counts = {
            cluster.id: len(cluster.members)
            for cluster in await self._cluster_store.list_for_owner(scope, kind)
            if cluster.id is not None
        }
        candidates = [
            cid
            for cid, count in counts.items()
            if (cid not in reflected and count >= 2) or (cid in reflected and count > 1)
        ]
        candidates.sort(key=lambda cid: counts[cid], reverse=True)
        return candidates[:_MAX_CLUSTERS_PER_RUN]

    # ── Per-cluster processing ────────────────────────────────────────────

    async def _process_cluster(
        self, *, cluster_id: str, scope: Scope
    ) -> EpisodeExtracted | None:
        """Merge one cluster and write the merged episode.

        Returns:
            The merged episode's event, ``None`` on skip.
        """
        members, episodes = await self._load_cluster_episodes(
            cluster_id=cluster_id, scope=scope
        )
        if not members or not episodes:
            return None
        # A session deletion keeps cluster rows, so members can name deleted
        # episodes. A merge needs two new episodes (INIT) or one beside the
        # merged one (UPDATE); with fewer, wait for the cluster to grow.
        merged = sum(1 for e in episodes if e.item.parent_type == "cluster")
        if len(episodes) - merged < (1 if merged else 2):
            logger.info(
                "reflection_skipped_too_few_episodes",
                cluster_id=cluster_id,
                members=len(members),
                live=len(episodes),
            )
            return None
        await self._detect_orphans(cluster_id, scope)

        mode, algo_result = await self._reflect_cluster(
            episodes=episodes, owner_id=scope.owner_id
        )
        if algo_result is None:
            return None

        unchanged = await self._sources_alive(scope=scope, episodes=episodes)
        if unchanged is None:
            logger.info("reflection_merge_skipped_source_gone", cluster_id=cluster_id)
            return None
        last_ts = max(ep.item.timestamp for ep in episodes)
        merged_entry_id = await self._write_merged_episode(
            cluster_id=cluster_id,
            scope=scope,
            algo_result=algo_result,
            last_ts=last_ts,
            only_if=unchanged,
        )
        if merged_entry_id is None:
            return None
        logger.info(
            "reflection_merged",
            cluster_id=cluster_id,
            mode=mode,
            source_count=len(members),
            merged_entry_id=merged_entry_id,
        )
        return EpisodeExtracted(
            memcell_id=merged_entry_id,
            episode_entry_id=merged_entry_id,
            episode_text=algo_result.episode,
            episode_timestamp_ms=last_ts,
            owner_id=scope.owner_id,
            session_id=None,
            app_id=scope.app_id,
            project_id=scope.project_id,
            source="reflection",
            reflection=ReflectionContext(
                cluster_id=cluster_id,
                mode=mode,
                source_members=members,
                episodes=[
                    ReflectedEpisode(
                        entry_id=ep.ref.entry_id,
                        md_path=ep.ref.md_path,
                        parent_id=ep.item.parent_id,
                    )
                    for ep in episodes
                ],
                merged_episode=algo_result.episode,
                last_ts=last_ts,
            ),
        )

    async def _reflect_cluster(
        self, *, episodes: list[StoredEpisode], owner_id: str
    ) -> tuple[ReflectionMode, AlgoEpisode | None]:
        """Determine reflection mode and call the algo reflector.

        Returns:
            ``(mode, algo_result)`` where mode is ``"init"`` or ``"update"``
            and algo_result is ``None`` on failure.
        """
        merged_entry_ids = [
            e.ref.entry_id for e in episodes if e.item.parent_type == "cluster"
        ]
        is_update = bool(merged_entry_ids)
        mode: ReflectionMode = "update" if is_update else "init"
        algo_result = await self._call_reflector(
            episodes=episodes,
            merged_entry_ids=merged_entry_ids,
            is_update=is_update,
            owner_id=owner_id,
        )
        return mode, algo_result

    async def _load_cluster_episodes(
        self, *, cluster_id: str, scope: Scope
    ) -> tuple[list[tuple[str, str]], list[StoredEpisode]]:
        """Read cluster members and fetch their episodes from the index.

        Returns:
            ``(members, episodes)`` tuple; either may be empty on skip.
        """
        members = await self._cluster_store.members(cluster_id)
        if not members:
            return [], []

        episodes = await self._fetch_episodes(
            entry_ids=[mid for mid, _ in members], scope=scope
        )
        return members, episodes

    async def _write_merged_episode(
        self,
        *,
        cluster_id: str,
        scope: Scope,
        algo_result: AlgoEpisode,
        last_ts: int,
        only_if: SourceCheck,
    ) -> str | None:
        """Write the merged episode to markdown; return its entry id.

        ``session_id`` stays empty: an aggregation product has no session.
        ``None`` when the store skipped the write (its ``only_if`` failed).
        """
        merged = Episode(
            owner_id=scope.owner_id,
            episode=algo_result.episode,
            summary=algo_result.summary,
            timestamp=last_ts,
            subject=algo_result.subject,
            parent_id=cluster_id,
            parent_type="cluster",
        )
        refs = await self._episode_store.append_many(scope, [merged], only_if=only_if)
        return refs[0].entry_id if refs else None

    async def _sources_alive(
        self, *, scope: Scope, episodes: list[StoredEpisode]
    ) -> SourceCheck | None:
        """Check that every episode sent to the model still exists.

        Each must still be in md and, if it came from a memcell, that memcell
        must still exist. Returns the check the locked write runs (no
        deletion has touched the owner's files since), or ``None`` when a
        source is already gone.

        The owner's delete counter is read *before* the files: a deletion
        that rewrites them after this read bumps it, so the locked write
        sees the change. Read after the files, the counter would already
        include that deletion and the write would go through.
        """
        key = owner_key(
            scope.app_id, scope.project_id, scope.owner_type, scope.owner_id
        )
        seen = LOCKS.delete_count(key)
        entry_ids = {ep.ref.entry_id for ep in episodes}
        if await self._episode_store.exists(scope, entry_ids) != entry_ids:
            return None
        memcell_ids = sorted(
            {ep.item.parent_id for ep in episodes if ep.item.parent_type == "memcell"}
        )
        found = await self._memcell_store.find_by_ids(memcell_ids)
        if len(found) != len(memcell_ids):
            return None

        async def unchanged() -> bool:
            return LOCKS.delete_count(key) == seen

        return unchanged

    # ── Helpers ───────────────────────────────────────────────────────────

    async def _detect_orphans(self, cluster_id: str, scope: Scope) -> None:
        """Log warning if orphan merged episodes exist for this cluster."""
        where = all_of(
            eq("parent_type", "cluster"),
            eq("parent_id", cluster_id),
            is_null("deprecated_by"),
            _owner_filter(scope),
        )
        orphans = await self._episode_index.orphans(where, limit=10)
        if orphans:
            logger.warning(
                "reflection_orphan_detected",
                cluster_id=cluster_id,
                orphan_entry_ids=[o.ref.entry_id for o in orphans],
            )

    async def _fetch_episodes(
        self, *, entry_ids: list[str], scope: Scope
    ) -> list[StoredEpisode]:
        """Fetch episodes by entry_id, sorted by timestamp ascending."""
        episodes = await self._episode_index.find_by_entry_ids(scope, entry_ids)
        return sorted(episodes, key=lambda e: e.item.timestamp)

    async def _call_reflector(
        self,
        *,
        episodes: list[StoredEpisode],
        merged_entry_ids: list[str],
        is_update: bool,
        owner_id: str,
    ) -> AlgoEpisode | None:
        """Call the algo reflector (INIT or UPDATE mode).

        Args:
            episodes: Source episodes.
            merged_entry_ids: Entry IDs of previously merged episodes
                (parent_type=cluster). Empty for INIT.
            is_update: Whether this is an UPDATE (vs INIT) reflection.
            owner_id: Owner for logging on failure.

        Returns:
            An algo Episode result, or ``None`` on failure.
        """
        algo_episodes = _to_algo_episodes(episodes)
        try:
            with memory_span(
                "everos.reflect.consolidate",
                observation_type="generation",
                metadata={"owner_id": owner_id, "is_update": is_update},
            ):
                # Token usage lands on this span via the LLM client wrapper.
                if is_update:
                    return await self._reflect_update(
                        algo_episodes=algo_episodes,
                        episodes=episodes,
                        merged_entry_ids=merged_entry_ids,
                    )
                return await self._reflector.areflect(
                    algo_episodes, custom_instructions=self._custom_instructions
                )
        except AppError:
            logger.warning(
                "reflection_reflector_failed",
                owner_id=owner_id,
                exc_info=True,
            )
            return None
        except Exception:
            logger.error(
                "reflection_reflector_unexpected_error",
                owner_id=owner_id,
                exc_info=True,
            )
            return None

    async def _reflect_update(
        self,
        *,
        algo_episodes: list[AlgoEpisode],
        episodes: list[StoredEpisode],
        merged_entry_ids: list[str],
    ) -> AlgoEpisode | None:
        """Run UPDATE-mode reflection by splitting old/new episodes.

        Returns:
            An algo Episode result, or ``None`` when no old episodes remain.
        """
        merged_set = set(merged_entry_ids)
        pairs = list(zip(algo_episodes, episodes, strict=True))
        old_algo_eps = [ae for ae, e in pairs if e.ref.entry_id in merged_set]
        new_algo_eps = [ae for ae, e in pairs if e.ref.entry_id not in merged_set]
        if not old_algo_eps:
            return None
        return await self._reflector.areflect(
            new_algo_eps,
            old_episode=old_algo_eps[0],
            custom_instructions=self._custom_instructions,
        )


class ReflectionFinisher:
    """Retire what one merge replaced, once the merged episode is written.

    Args:
        cluster_store: Clusters and their membership.
        episode_store: Episode truth (deprecation).
        atomic_fact_store: Atomic-fact truth (deprecation).
        atomic_fact_index: Atomic-fact reads (facts of deprecated episodes).
        report_store: Reflection audit trail.
        embedder: Embedding provider for centroid recomputation.
    """

    def __init__(
        self,
        *,
        cluster_store: ClusterStore,
        episode_store: EpisodeStore,
        atomic_fact_store: AtomicFactStore,
        atomic_fact_index: AtomicFactIndex,
        report_store: ReflectionReportStore,
        embedder: EmbeddingProvider,
    ) -> None:
        self._cluster_store = cluster_store
        self._episode_store = episode_store
        self._atomic_fact_store = atomic_fact_store
        self._atomic_fact_index = atomic_fact_index
        self._report_store = report_store
        self._embedder = embedder

    async def finish(
        self, *, scope: Scope, merged_entry_id: str, context: ReflectionContext
    ) -> ReflectionReport | None:
        """Deprecate the replaced members and move the cluster onto the merge.

        Runs under a per-cluster partition lock; errors are logged, not
        raised, like every other per-cluster Reflection failure. The merged
        episode and the members that joined since reflection read the
        cluster are embedded first, under no lock: a failed embedding leaves
        everything as it was.

        Returns:
            The report on success, ``None`` on failure or an empty diff.
        """
        partition = f"{scope.app_id}:{scope.project_id}:{context.cluster_id}"
        try:
            centroid = await self._embedder.embed(context.merged_episode)
            joined = await self._embed_joined(
                scope=scope, context=context, merged_entry_id=merged_entry_id
            )
            async with get_partition_lock("reflection_deprecate", partition):
                return await self._execute_deprecation(
                    scope=scope,
                    merged_entry_id=merged_entry_id,
                    context=context,
                    centroid=centroid,
                    joined=joined,
                )
        except AppError:
            logger.warning(
                "reflection_deprecate_failed",
                cluster_id=context.cluster_id,
                exc_info=True,
            )
            return None
        except Exception:
            logger.error(
                "reflection_deprecate_unexpected_error",
                cluster_id=context.cluster_id,
                exc_info=True,
            )
            return None

    async def _execute_deprecation(
        self,
        *,
        scope: Scope,
        merged_entry_id: str,
        context: ReflectionContext,
        centroid: Sequence[float],
        joined: Mapping[str, _Joined],
    ) -> ReflectionReport | None:
        """Run the deprecation steps inside the partition lock.

        Returns:
            A ReflectionReport on success, ``None`` when no members to deprecate.
        """
        to_deprecate = await self._resolve_deprecation_targets(
            cluster_id=context.cluster_id, original_members=context.source_members
        )
        # A source deleted since the merge gets no mark: it is gone, and a
        # mark would point a deleted entry at M.
        sources = {ep.entry_id for ep in context.episodes}
        gone = sources - await self._episode_store.exists(scope, sources)
        to_deprecate -= gone
        if not to_deprecate:
            return None

        async def merged_alive() -> bool:  # a deletion of M must win
            return bool(await self._episode_store.exists(scope, [merged_entry_id]))

        dep_ep = await self._deprecate_episodes(
            scope=scope,
            episodes=[ep for ep in context.episodes if ep.entry_id not in gone],
            to_deprecate=to_deprecate,
            merged_entry_id=merged_entry_id,
            only_if=merged_alive,
        )
        if dep_ep is None:
            logger.info(
                "reflection_finish_skipped_merged_gone",
                cluster_id=context.cluster_id,
                merged_entry_id=merged_entry_id,
            )
            return None
        dep_fact = await self._deprecate_facts(
            scope=scope,
            parent_ids=to_deprecate,
            merged_entry_id=merged_entry_id,
            only_if=merged_alive,
        )
        await self._update_cluster_after_merge(
            scope=scope,
            context=context,
            to_deprecate=to_deprecate,
            merged_entry_id=merged_entry_id,
            centroid=centroid,
            joined=joined,
        )
        report = ReflectionReport(
            id=uuid.uuid4().hex,
            cluster_id=context.cluster_id,
            owner_id=scope.owner_id,
            app_id=scope.app_id,
            project_id=scope.project_id,
            mode=context.mode,
            source_members=[m for m in context.source_members if m[0] in to_deprecate],
            merged_entry_id=merged_entry_id,
            deprecated_fact_count=dep_fact,
        )
        await self._report_store.create(report)
        logger.info(
            "reflection_deprecated",
            cluster_id=context.cluster_id,
            deprecated_episode_count=dep_ep,
            deprecated_fact_count=dep_fact,
        )
        return report

    async def _resolve_deprecation_targets(
        self, *, cluster_id: str, original_members: list[tuple[str, str]]
    ) -> set[str]:
        """Re-read cluster members and intersect with the original snapshot.

        Returns:
            Set of member IDs safe to deprecate (present in both snapshots).
        """
        current_members = await self._cluster_store.members(cluster_id)
        current_ids = {mid for mid, _ in current_members}
        original_ids = {mid for mid, _ in original_members}
        return original_ids & current_ids

    async def _deprecate_episodes(
        self,
        *,
        scope: Scope,
        episodes: list[ReflectedEpisode],
        to_deprecate: set[str],
        merged_entry_id: str,
        only_if: SourceCheck,
    ) -> int | None:
        """Retire the selected episode references through the truth store.

        Returns:
            Number of episodes marked; ``None`` when ``only_if`` failed.
        """
        entries = [
            (ep.md_path, ep.entry_id)
            for ep in episodes
            if ep.md_path
            and (ep.parent_id in to_deprecate or ep.entry_id in to_deprecate)
        ]
        if entries and not await self._episode_store.deprecate(
            scope, entries=entries, replaced_by=merged_entry_id, only_if=only_if
        ):
            return None
        return len(entries)

    async def _deprecate_facts(
        self,
        *,
        scope: Scope,
        parent_ids: Collection[str],
        merged_entry_id: str,
        only_if: SourceCheck,
    ) -> int:
        """Mark the live atomic facts of deprecated episodes in their md files.

        ``parent_id`` is an entry id, and entry ids restart per
        (app, project, day): the scope keeps one project's reflection from
        deprecating another project's facts.

        Returns:
            Number of facts marked.
        """
        if not parent_ids:
            return 0
        hits = await self._atomic_fact_index.find(
            all_of(
                one_of("parent_id", sorted(parent_ids)),
                _owner_filter(scope),
                is_null("deprecated_by"),
            )
        )
        entries = [
            (hit.stored.ref.md_path, hit.stored.ref.entry_id)
            for hit in hits
            if hit.stored.ref.md_path
        ]
        if entries:
            await self._atomic_fact_store.deprecate(
                scope, entries=entries, replaced_by=merged_entry_id, only_if=only_if
            )
        return len(hits)

    async def _update_cluster_after_merge(
        self,
        *,
        scope: Scope,
        context: ReflectionContext,
        to_deprecate: set[str],
        merged_entry_id: str,
        centroid: Sequence[float],
        joined: Mapping[str, _Joined],
    ) -> None:
        """Fold the deprecated members into the merged episode.

        Under the owner's clustering lock -- the one a single-id deletion
        takes to take members out of a cluster -- and only while the merged
        episode still exists: once a deletion removed it, adding it as a
        member would leave the cluster pointing at nothing.

        ``count``, ``centroid`` and ``last_ts`` are restamped from the members
        the cluster has now, not from the ones reflection read before its LLM
        call: profile clustering may have added episodes in between, and they
        stay. ``count`` is exact; a member that joined after ``joined`` was
        embedded counts but is left out of the centroid.
        """
        async with user_cluster_lock(scope.app_id, scope.project_id, scope.owner_id):
            if not await self._episode_store.exists(scope, [merged_entry_id]):
                logger.info(
                    "reflection_finish_skipped_merged_gone",
                    cluster_id=context.cluster_id,
                    merged_entry_id=merged_entry_id,
                )
                return
            kept = [
                mid
                for mid, _ in await self._cluster_store.members(context.cluster_id)
                if mid not in to_deprecate and mid != merged_entry_id
            ]
            known = [joined[mid] for mid in kept if mid in joined]
            vectors = [centroid, *(j.vector for j in known)]
            await self._cluster_store.replace_members(
                context.cluster_id,
                removed=to_deprecate,
                added=merged_entry_id,
                member_type="episode",
                centroid=[
                    sum(col) / len(vectors) for col in zip(*vectors, strict=True)
                ],
                count=len(kept) + 1,
                last_ts=max([context.last_ts, *(j.timestamp for j in known)]),
                preview=[context.merged_episode[:_PREVIEW_CHARS]],
            )

    async def _embed_joined(
        self, *, scope: Scope, context: ReflectionContext, merged_entry_id: str
    ) -> dict[str, _Joined]:
        """Embed the members that joined the cluster after reflection read it.

        Usually none. Done before any lock, like the merged episode's
        embedding: a model call under the owner's clustering lock would hold
        every other writer of the owner's clusters for its duration.
        """
        read = {mid for mid, _ in context.source_members} | {merged_entry_id}
        ids = [
            mid
            for mid, _ in await self._cluster_store.members(context.cluster_id)
            if mid not in read
        ]
        if not ids:
            return {}
        stored = await self._episode_store.read_many(scope, ids)
        if not stored:
            return {}
        vectors = await self._embedder.embed_batch([s.item.episode for s in stored])
        return {
            s.ref.entry_id: _Joined(vector=v, timestamp=s.item.timestamp)
            for s, v in zip(stored, vectors, strict=True)
        }


def _owner_filter(scope: Scope) -> Filter:
    """Owner and space, without owner_type (the stores' historical key)."""
    return all_of(
        eq("owner_id", scope.owner_id),
        eq("app_id", scope.app_id),
        eq("project_id", scope.project_id),
    )


def _to_algo_episodes(episodes: list[StoredEpisode]) -> list[AlgoEpisode]:
    """Project stored episodes onto the algo Episode type the reflector reads."""
    # Deferred: avoid pulling LLM libs at module import time.
    from everalgo.types import Episode as AlgoEpisode

    return [
        AlgoEpisode(
            owner_id=e.item.owner_id,
            episode=e.item.episode,
            subject=e.item.subject,
            summary=e.item.summary,
            timestamp=e.item.timestamp,
        )
        for e in episodes
    ]
