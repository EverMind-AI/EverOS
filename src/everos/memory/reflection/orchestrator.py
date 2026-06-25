"""ReflectionOrchestrator — Select -> Merge -> Re-extract -> Deprecate.

Consolidates fragmented cluster members (memcell-derived episodes) into
a single high-quality merged episode per cluster.  The merged episode is
written to md, re-extracted for atomic facts via ``EpisodeExtracted``,
and the originals are deprecated in both md frontmatter and LanceDB.

See ``local/2026-06-14-reflection-everos-design.md`` for the full design.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from everalgo.types import Episode as AlgoEpisode
    from everalgo.user_memory import EpisodeReflector

    from everos.component.embedding import EmbeddingProvider
    from everos.infra.persistence.markdown import EpisodeWriter

import numpy as np

from everos.component.utils.datetime import from_timestamp, to_iso_format
from everos.core.errors import AppError
from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.infra.ome.context import StrategyContext
from everos.memory._partition_locks import get_partition_lock
from everos.memory.events import EpisodeExtracted

logger = get_logger(__name__)

_MAX_CLUSTERS_PER_RUN = 10
_WAIT_TIMEOUT_SECONDS = 120.0


def _escape_sql(value: str) -> str:
    """Escape single quotes for LanceDB SQL-like ``where`` predicates.

    LanceDB has no parameterised query API; doubling the quote
    (``'`` -> ``''``) is the SQL-standard escape.

    Args:
        value: Raw string to escape.

    Returns:
        Escaped string safe for interpolation into a WHERE clause.
    """
    return value.replace("'", "''")


class ReflectionOrchestrator:
    """Run one Reflection cycle for a single owner scope.

    Consolidates fragmented cluster members into a single merged episode
    per cluster via Select -> Merge -> Re-extract -> Deprecate.

    Args:
        cluster_repo: SQLite cluster repository (member CRUD + queries).
        episode_store: LanceDB episode repository (read + update).
        atomic_fact_store: LanceDB atomic fact repository (update).
        episode_writer: Markdown daily-log writer for episodes.
        report_repo: SQLite reflection report repository.
        reflector: Algorithm-side EpisodeReflector (areflect).
        embedder: Embedding provider for centroid recomputation.
    """

    def __init__(
        self,
        *,
        cluster_repo: Any,
        episode_store: Any,
        atomic_fact_store: Any,
        episode_writer: EpisodeWriter,
        report_repo: Any,
        reflector: EpisodeReflector,
        embedder: EmbeddingProvider,
    ) -> None:
        self._cluster_repo = cluster_repo
        self._episode_store = episode_store
        self._atomic_fact_store = atomic_fact_store
        self._episode_writer = episode_writer
        self._report_repo = report_repo
        self._reflector = reflector
        self._embedder = embedder

    async def run(
        self,
        *,
        ctx: StrategyContext,
        owner_id: str,
        owner_type: str = "user",
        kind: str = "user_memory",
        app_id: str = "default",
        project_id: str = "default",
    ) -> list[object]:
        """Run one Reflection cycle for a single owner scope.

        Args:
            ctx: Runtime context (event bus + wait).
            owner_id: Target owner identifier.
            owner_type: Owner type discriminator.
            kind: Memory kind for cluster lookup.
            app_id: Application scope.
            project_id: Project scope.

        Returns:
            List of successful ReflectionReport rows (typed as object
            because the table class lives in infra).
        """
        candidates = await self._select_candidates(
            owner_id=owner_id,
            kind=kind,
            app_id=app_id,
            project_id=project_id,
        )
        logger.info(
            "reflection_candidates_selected",
            owner_id=owner_id,
            candidate_count=len(candidates),
        )
        if not candidates:
            return []

        reports: list[object] = []
        skip_count = 0
        for cluster_id in candidates:
            report = await self._process_cluster_safely(
                ctx=ctx,
                cluster_id=cluster_id,
                owner_id=owner_id,
                owner_type=owner_type,
                app_id=app_id,
                project_id=project_id,
            )
            if report is not None:
                reports.append(report)
            else:
                skip_count += 1

        logger.info(
            "reflection_cycle_completed",
            owner_id=owner_id,
            success_count=len(reports),
            skip_count=skip_count,
        )
        return reports

    async def _process_cluster_safely(
        self,
        *,
        ctx: StrategyContext,
        cluster_id: str,
        owner_id: str,
        owner_type: str,
        app_id: str,
        project_id: str,
    ) -> object | None:
        """Process one cluster, catching errors to allow the cycle to continue.

        Args:
            ctx: Runtime context (event bus + wait).
            cluster_id: Target cluster identifier.
            owner_id: Target owner identifier.
            owner_type: Owner type discriminator.
            app_id: Application scope.
            project_id: Project scope.

        Returns:
            A ReflectionReport on success, ``None`` on skip or error.
        """
        try:
            return await self._process_cluster(
                ctx=ctx,
                cluster_id=cluster_id,
                owner_id=owner_id,
                owner_type=owner_type,
                app_id=app_id,
                project_id=project_id,
            )
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

    async def _select_candidates(
        self,
        *,
        owner_id: str,
        kind: str,
        app_id: str,
        project_id: str,
    ) -> list[str]:
        """Two-step DB-agnostic candidate selection.

        Args:
            owner_id: Target owner identifier.
            kind: Memory kind for cluster lookup.
            app_id: Application scope.
            project_id: Project scope.

        Returns:
            Cluster IDs sorted by member count descending, limited
            to ``_MAX_CLUSTERS_PER_RUN``.
        """
        reflected = await self._report_repo.list_reflected_cluster_ids(
            owner_id, app_id, project_id
        )
        clusters = await self._cluster_repo.list_ids_and_member_counts(
            owner_id, kind, app_id=app_id, project_id=project_id
        )
        count_map = dict(clusters)
        candidates = [
            cid
            for cid, count in clusters
            if (cid not in reflected and count >= 2) or (cid in reflected and count > 1)
        ]
        candidates.sort(key=lambda cid: count_map[cid], reverse=True)
        return candidates[:_MAX_CLUSTERS_PER_RUN]

    # ── Per-cluster processing ────────────────────────────────────────────

    async def _process_cluster(
        self,
        *,
        ctx: StrategyContext,
        cluster_id: str,
        owner_id: str,
        owner_type: str,
        app_id: str,
        project_id: str,
    ) -> object | None:
        """Full flow for one cluster: merge, write, re-extract, deprecate.

        Args:
            ctx: Runtime context (event bus + wait).
            cluster_id: Target cluster identifier.
            owner_id: Target owner identifier.
            owner_type: Owner type discriminator.
            app_id: Application scope.
            project_id: Project scope.

        Returns:
            A ReflectionReport on success, ``None`` on skip.
        """
        await self._detect_orphans(cluster_id, owner_id, app_id, project_id)

        scope = dict(owner_id=owner_id, app_id=app_id, project_id=project_id)
        members, episodes = await self._load_cluster_episodes(
            cluster_id=cluster_id, **scope
        )
        if not members or not episodes:
            return None

        mode, algo_result = await self._reflect_cluster(
            episodes=episodes,
            owner_id=owner_id,
        )
        if algo_result is None:
            return None

        merged_entry_id = await self._write_and_reextract(
            ctx=ctx,
            cluster_id=cluster_id,
            **scope,
            algo_result=algo_result,
            episodes=episodes,
            mode=mode,
            members=members,
        )
        if merged_entry_id is None:
            return None

        return await self._deprecate(
            ctx=ctx,
            cluster_id=cluster_id,
            owner_type=owner_type,
            **scope,
            original_members=members,
            merged_entry_id=merged_entry_id,
            algo_result=algo_result,
            mode=mode,
            episodes=episodes,
        )

    async def _reflect_cluster(
        self,
        *,
        episodes: list[Any],
        owner_id: str,
    ) -> tuple[str, AlgoEpisode | None]:
        """Determine reflection mode and call the algo reflector.

        Args:
            episodes: Source episode rows from LanceDB.
            owner_id: Owner for logging on failure.

        Returns:
            ``(mode, algo_result)`` where mode is ``"init"`` or ``"update"``
            and algo_result is ``None`` on failure.
        """
        merged_entry_ids = [e.entry_id for e in episodes if e.parent_type == "cluster"]
        is_update = bool(merged_entry_ids)
        mode = "update" if is_update else "init"
        algo_result = await self._call_reflector(
            episodes=episodes,
            merged_entry_ids=merged_entry_ids,
            is_update=is_update,
            owner_id=owner_id,
        )
        return mode, algo_result

    async def _load_cluster_episodes(
        self,
        *,
        cluster_id: str,
        owner_id: str,
        app_id: str,
        project_id: str,
    ) -> tuple[list[tuple[str, str]], list[Any]]:
        """Read cluster members and fetch their episode rows from LanceDB.

        Args:
            cluster_id: Target cluster identifier.
            owner_id: Target owner identifier.
            app_id: Application scope.
            project_id: Project scope.

        Returns:
            ``(members, episodes)`` tuple; either may be empty on skip.
        """
        members = await self._cluster_repo.get_members_with_type(cluster_id)
        if not members:
            return [], []

        member_ids = [mid for mid, _ in members]
        episodes = await self._fetch_episodes(
            entry_ids=member_ids,
            owner_id=owner_id,
            app_id=app_id,
            project_id=project_id,
        )
        return members, episodes

    async def _write_and_reextract(
        self,
        *,
        ctx: StrategyContext,
        cluster_id: str,
        owner_id: str,
        app_id: str,
        project_id: str,
        algo_result: AlgoEpisode,
        episodes: list[Any],
        mode: str,
        members: list[tuple[str, str]],
    ) -> str | None:
        """Write merged episode to md and emit re-extraction event.

        Args:
            ctx: Runtime context (event bus + wait).
            cluster_id: Target cluster identifier.
            owner_id: Target owner identifier.
            app_id: Application scope.
            project_id: Project scope.
            algo_result: Algo reflector output with ``.episode`` / ``.subject``.
            episodes: Source episode rows (for timestamp derivation).
            mode: ``"init"`` or ``"update"``.
            members: Original cluster members ``(member_id, member_type)``.

        Returns:
            The ``merged_entry_id`` on success, ``None`` on extraction timeout.
        """
        last_ts = max(ep.timestamp for ep in episodes)
        merged_entry_id = await self._write_merged_episode(
            cluster_id=cluster_id,
            owner_id=owner_id,
            app_id=app_id,
            project_id=project_id,
            algo_result=algo_result,
            last_ts=last_ts,
        )
        logger.info(
            "reflection_merged",
            cluster_id=cluster_id,
            mode=mode,
            source_count=len(members),
            merged_entry_id=merged_entry_id,
        )
        return await self._emit_and_wait_extraction(
            ctx=ctx,
            cluster_id=cluster_id,
            owner_id=owner_id,
            app_id=app_id,
            project_id=project_id,
            algo_result=algo_result,
            merged_entry_id=merged_entry_id,
            last_ts=last_ts,
        )

    async def _write_merged_episode(
        self,
        *,
        cluster_id: str,
        owner_id: str,
        app_id: str,
        project_id: str,
        algo_result: AlgoEpisode,
        last_ts: object,
    ) -> str:
        """Write the merged episode entry to markdown.

        Args:
            cluster_id: Parent cluster identifier.
            owner_id: Target owner identifier.
            app_id: Application scope.
            project_id: Project scope.
            algo_result: Algo reflector output with ``.episode`` / ``.subject``.
            last_ts: Latest source episode timestamp (datetime or int).

        Returns:
            The formatted ``merged_entry_id``.
        """
        last_ts_iso = to_iso_format(from_timestamp(_ts_to_ms(last_ts)))
        if last_ts_iso is None:
            raise ValueError("to_iso_format returned None for valid timestamp")
        inline, sections = _merged_episode_to_entry_body(
            algo_result, cluster_id, owner_id, last_ts_iso
        )
        entry_ids = await self._episode_writer.append_entries(
            owner_id,
            [(inline, sections)],
            app_id=app_id,
            project_id=project_id,
        )
        return entry_ids[0].format()

    async def _emit_and_wait_extraction(
        self,
        *,
        ctx: StrategyContext,
        cluster_id: str,
        owner_id: str,
        app_id: str,
        project_id: str,
        algo_result: AlgoEpisode,
        merged_entry_id: str,
        last_ts: object,
    ) -> str | None:
        """Emit ``EpisodeExtracted`` and wait for cascade to process it.

        Args:
            ctx: Runtime context (event bus + wait).
            cluster_id: Target cluster identifier (for error logging).
            owner_id: Target owner identifier.
            app_id: Application scope.
            project_id: Project scope.
            algo_result: Algo reflector output (episode text).
            merged_entry_id: Entry ID of the written merged episode.
            last_ts: Latest source episode timestamp (datetime or int).

        Returns:
            The ``merged_entry_id`` on success, ``None`` on timeout.
        """
        event = EpisodeExtracted(
            memcell_id=merged_entry_id,
            episode_entry_id=merged_entry_id,
            episode_text=algo_result.episode,
            episode_timestamp_ms=_ts_to_ms(last_ts),
            owner_id=owner_id,
            session_id=None,
            app_id=app_id,
            project_id=project_id,
            source="reflection",
        )
        await ctx.emit(event)
        try:
            await ctx.wait_for_event(event.event_id, timeout=_WAIT_TIMEOUT_SECONDS)
        except TimeoutError:
            logger.error(
                "reflection_extraction_timeout",
                cluster_id=cluster_id,
                event_id=event.event_id,
                merged_entry_id=merged_entry_id,
            )
            return None
        return merged_entry_id

    # ── Helpers ───────────────────────────────────────────────────────────

    async def _detect_orphans(
        self,
        cluster_id: str,
        owner_id: str,
        app_id: str,
        project_id: str,
    ) -> None:
        """Log warning if orphan merged episodes exist for this cluster.

        Args:
            cluster_id: Target cluster identifier.
            owner_id: Target owner identifier.
            app_id: Application scope.
            project_id: Project scope.
        """
        where = (
            f"parent_type = 'cluster' AND parent_id = '{_escape_sql(cluster_id)}' "
            f"AND deprecated_by IS NULL "
            f"AND owner_id = '{_escape_sql(owner_id)}' "
            f"AND app_id = '{_escape_sql(app_id)}' "
            f"AND project_id = '{_escape_sql(project_id)}'"
        )
        orphans = await self._episode_store.find_where(where, limit=10)
        if orphans:
            logger.warning(
                "reflection_orphan_detected",
                cluster_id=cluster_id,
                orphan_entry_ids=[o.entry_id for o in orphans],
            )

    async def _fetch_episodes(
        self,
        *,
        entry_ids: list[str],
        owner_id: str,
        app_id: str,
        project_id: str,
    ) -> list[Any]:
        """Fetch episodes by entry_id.

        Returns:
            Episode list sorted by timestamp ascending.
        """
        rows = await self._episode_store.find_by_owner_entries(
            owner_id,
            entry_ids,
            app_id=app_id,
            project_id=project_id,
        )
        rows.sort(key=lambda e: e.timestamp)
        return rows

    async def _call_reflector(
        self,
        *,
        episodes: list[Any],
        merged_entry_ids: list[str],
        is_update: bool,
        owner_id: str,
    ) -> AlgoEpisode | None:
        """Call the algo reflector (INIT or UPDATE mode).

        Args:
            episodes: Source episode rows from LanceDB.
            merged_entry_ids: Entry IDs of previously merged episodes
                (parent_type=cluster). Empty for INIT.
            is_update: Whether this is an UPDATE (vs INIT) reflection.
            owner_id: Owner for logging on failure.

        Returns:
            An algo Episode result, or ``None`` on failure.
        """
        algo_episodes = _to_algo_episodes(episodes)
        try:
            if is_update:
                return await self._reflect_update(
                    algo_episodes=algo_episodes,
                    episodes=episodes,
                    merged_entry_ids=merged_entry_ids,
                )
            return await self._reflector.areflect(algo_episodes)
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
        episodes: list[Any],
        merged_entry_ids: list[str],
    ) -> AlgoEpisode | None:
        """Run UPDATE-mode reflection by splitting old/new episodes.

        Args:
            algo_episodes: Converted algo Episode objects (parallel to ``episodes``).
            episodes: Source episode rows from LanceDB.
            merged_entry_ids: Entry IDs of previously merged episodes.

        Returns:
            An algo Episode result, or ``None`` when no old episodes remain.
        """
        merged_set = set(merged_entry_ids)
        old_algo_eps = [
            ae
            for ae, e in zip(algo_episodes, episodes, strict=True)
            if e.entry_id in merged_set
        ]
        new_algo_eps = [
            ae
            for ae, e in zip(algo_episodes, episodes, strict=True)
            if e.entry_id not in merged_set
        ]
        if not old_algo_eps:
            return None
        return await self._reflector.areflect(new_algo_eps, old_episode=old_algo_eps[0])

    # ── Deprecate (orchestrator + sub-steps) ─────────────────────────────

    async def _deprecate(
        self,
        *,
        ctx: StrategyContext,
        cluster_id: str,
        owner_id: str,
        owner_type: str,
        app_id: str,
        project_id: str,
        original_members: list[tuple[str, str]],
        merged_entry_id: str,
        algo_result: AlgoEpisode,
        mode: str,
        episodes: list[Any],
    ) -> object | None:
        """Deprecate originals and update cluster membership.

        Runs under a partition lock for concurrency safety.

        Args:
            ctx: Runtime context (unused here, kept for signature compat).
            cluster_id: Target cluster identifier.
            owner_id: Target owner identifier.
            owner_type: Owner type discriminator.
            app_id: Application scope.
            project_id: Project scope.
            original_members: Snapshot ``(member_id, member_type)`` from selection.
            merged_entry_id: Entry ID of the newly written merged episode.
            algo_result: Algo reflector output (for centroid + report).
            mode: ``"init"`` or ``"update"``.
            episodes: Source episode rows (for md patching + timestamp).

        Returns:
            A ReflectionReport on success, ``None`` on failure or empty diff.
        """
        partition = f"{app_id}:{project_id}:{cluster_id}"
        try:
            async with get_partition_lock("reflection_deprecate", partition):
                return await self._execute_deprecation(
                    cluster_id=cluster_id,
                    owner_id=owner_id,
                    app_id=app_id,
                    project_id=project_id,
                    original_members=original_members,
                    merged_entry_id=merged_entry_id,
                    algo_result=algo_result,
                    mode=mode,
                    episodes=episodes,
                )
        except AppError:
            logger.warning(
                "reflection_deprecate_failed",
                cluster_id=cluster_id,
                exc_info=True,
            )
            return None
        except Exception:
            logger.error(
                "reflection_deprecate_unexpected_error",
                cluster_id=cluster_id,
                exc_info=True,
            )
            return None

    async def _execute_deprecation(
        self,
        *,
        cluster_id: str,
        owner_id: str,
        app_id: str,
        project_id: str,
        original_members: list[tuple[str, str]],
        merged_entry_id: str,
        algo_result: AlgoEpisode,
        mode: str,
        episodes: list[Any],
    ) -> object | None:
        """Run the deprecation steps inside the partition lock.

        Args:
            cluster_id: Target cluster identifier.
            owner_id: Target owner identifier.
            app_id: Application scope.
            project_id: Project scope.
            original_members: Snapshot ``(member_id, member_type)`` from selection.
            merged_entry_id: Entry ID of the newly written merged episode.
            algo_result: Algo reflector output (for centroid + report).
            mode: ``"init"`` or ``"update"``.
            episodes: Source episode rows (for md patching + timestamp).

        Returns:
            A ReflectionReport on success, ``None`` when no members to deprecate.
        """
        to_deprecate = await self._resolve_deprecation_targets(
            cluster_id=cluster_id,
            original_members=original_members,
        )
        if not to_deprecate:
            return None

        dep_ep, dep_fact = await self._apply_deprecation_writes(
            episodes=episodes,
            to_deprecate=to_deprecate,
            owner_id=owner_id,
            app_id=app_id,
            project_id=project_id,
            merged_entry_id=merged_entry_id,
        )
        await self._update_cluster_after_merge(
            cluster_id=cluster_id,
            to_deprecate=to_deprecate,
            merged_entry_id=merged_entry_id,
            algo_result=algo_result,
            episodes=episodes,
        )
        report = await self._create_reflection_report(
            cluster_id=cluster_id,
            owner_id=owner_id,
            app_id=app_id,
            project_id=project_id,
            mode=mode,
            original_members=original_members,
            to_deprecate=to_deprecate,
            merged_entry_id=merged_entry_id,
            deprecated_fact_count=dep_fact,
        )
        logger.info(
            "reflection_deprecated",
            cluster_id=cluster_id,
            deprecated_episode_count=dep_ep,
            deprecated_fact_count=dep_fact,
        )
        return report

    async def _apply_deprecation_writes(
        self,
        *,
        episodes: list[Any],
        to_deprecate: set[str],
        owner_id: str,
        app_id: str,
        project_id: str,
        merged_entry_id: str,
    ) -> tuple[int, int]:
        """Patch md frontmatter and mark episodes/facts deprecated in LanceDB.

        Args:
            episodes: Source episode rows (for md patching).
            to_deprecate: Set of member IDs being deprecated.
            owner_id: Target owner identifier.
            app_id: Application scope.
            project_id: Project scope.
            merged_entry_id: Entry ID of the replacement merged episode.

        Returns:
            ``(deprecated_episode_count, deprecated_fact_count)``.
        """
        await self._patch_md_frontmatter(
            episodes=episodes,
            to_deprecate=to_deprecate,
            merged_entry_id=merged_entry_id,
        )
        deprecated_ep_count = await self._deprecate_lance_episodes(
            entry_ids=to_deprecate,
            owner_id=owner_id,
            app_id=app_id,
            project_id=project_id,
            merged_entry_id=merged_entry_id,
        )
        deprecated_fact_count = await self._deprecate_lance_facts(
            parent_ids=to_deprecate,
            owner_id=owner_id,
            merged_entry_id=merged_entry_id,
        )
        return deprecated_ep_count, deprecated_fact_count

    async def _resolve_deprecation_targets(
        self,
        *,
        cluster_id: str,
        original_members: list[tuple[str, str]],
    ) -> set[str]:
        """Re-read cluster members and intersect with the original snapshot.

        Args:
            cluster_id: Target cluster identifier.
            original_members: Snapshot ``(member_id, member_type)`` from selection.

        Returns:
            Set of member IDs safe to deprecate (present in both snapshots).
        """
        current_members = await self._cluster_repo.get_members_with_type(cluster_id)
        current_ids = {mid for mid, _ in current_members}
        original_ids = {mid for mid, _ in original_members}
        return original_ids & current_ids

    async def _deprecate_lance_episodes(
        self,
        *,
        entry_ids: set[str],
        owner_id: str,
        app_id: str,
        project_id: str,
        merged_entry_id: str,
    ) -> int:
        """Mark deprecated episodes in LanceDB by entry_id.

        Returns:
            Number of LanceDB update calls issued.
        """
        coros: list[Any] = [
            self._episode_store.update(
                {"deprecated_by": merged_entry_id},
                where=(
                    f"entry_id = '{_escape_sql(eid)}' "
                    f"AND owner_id = '{_escape_sql(owner_id)}' "
                    f"AND app_id = '{_escape_sql(app_id)}' "
                    f"AND project_id = '{_escape_sql(project_id)}'"
                ),
            )
            for eid in entry_ids
        ]
        if coros:
            await asyncio.gather(*coros)
        return len(coros)

    async def _deprecate_lance_facts(
        self,
        *,
        parent_ids: set[str],
        owner_id: str,
        merged_entry_id: str,
    ) -> int:
        """Mark deprecated atomic facts in LanceDB.

        Args:
            parent_ids: Parent IDs (memcell or episode) whose facts to deprecate.
            owner_id: Target owner identifier.
            merged_entry_id: Entry ID of the replacement merged episode.

        Returns:
            Total number of LanceDB update calls issued.
        """
        if not parent_ids:
            return 0

        coros = [
            self._atomic_fact_store.update(
                {"deprecated_by": merged_entry_id},
                where=(
                    f"parent_id = '{_escape_sql(pid)}' "
                    f"AND owner_id = '{_escape_sql(owner_id)}' "
                    f"AND deprecated_by IS NULL"
                ),
            )
            for pid in parent_ids
        ]
        await asyncio.gather(*coros)
        return len(coros)

    async def _update_cluster_after_merge(
        self,
        *,
        cluster_id: str,
        to_deprecate: set[str],
        merged_entry_id: str,
        algo_result: AlgoEpisode,
        episodes: list[Any],
    ) -> None:
        """Remove old members, add merged, and recompute centroid.

        Args:
            cluster_id: Target cluster identifier.
            to_deprecate: Member IDs to remove from the cluster.
            merged_entry_id: Entry ID of the newly merged episode.
            algo_result: Algo reflector output (episode text for centroid).
            episodes: Source episode rows (for last timestamp).
        """
        await self._cluster_repo.remove_members(cluster_id, to_deprecate)
        await self._cluster_repo.add_member(cluster_id, merged_entry_id, "episode")

        centroid = await self._embedder.embed(algo_result.episode)
        centroid_blob = np.asarray(centroid, dtype=np.float32).tobytes()
        last_ts_ms = _ts_to_ms(max(ep.timestamp for ep in episodes))
        await self._cluster_repo.update_metadata(
            cluster_id,
            centroid_blob=centroid_blob,
            count=1,
            last_ts_ms=last_ts_ms,
            preview_json=json.dumps([algo_result.episode[:200]], ensure_ascii=False),
        )

    async def _create_reflection_report(
        self,
        *,
        cluster_id: str,
        owner_id: str,
        app_id: str,
        project_id: str,
        mode: str,
        original_members: list[tuple[str, str]],
        to_deprecate: set[str],
        merged_entry_id: str,
        deprecated_fact_count: int,
    ) -> object:
        """Build and persist a ReflectionReport row.

        Args:
            cluster_id: Target cluster identifier.
            owner_id: Target owner identifier.
            app_id: Application scope.
            project_id: Project scope.
            mode: ``"init"`` or ``"update"``.
            original_members: Full member snapshot ``(member_id, member_type)``.
            to_deprecate: Subset of members that were actually deprecated.
            merged_entry_id: Entry ID of the replacement merged episode.
            deprecated_fact_count: Number of atomic fact deprecation calls.

        Returns:
            The persisted ReflectionReport row.
        """
        # Deferred: avoid pulling heavy SQLModel table at module import.
        from everos.infra.persistence.sqlite import ReflectionReport

        source_members_json = json.dumps(
            [
                {"member_id": mid, "member_type": mtype}
                for mid, mtype in original_members
                if mid in to_deprecate
            ],
            ensure_ascii=False,
        )
        report = ReflectionReport(
            id=uuid.uuid4().hex,
            cluster_id=cluster_id,
            owner_id=owner_id,
            app_id=app_id,
            project_id=project_id,
            mode=mode,
            source_members=source_members_json,
            source_count=len(to_deprecate),
            merged_entry_id=merged_entry_id,
            deprecated_fact_count=deprecated_fact_count,
        )
        await self._report_repo.create(report)
        return report

    async def _patch_md_frontmatter(
        self,
        *,
        episodes: list[Any],
        to_deprecate: set[str],
        merged_entry_id: str,
    ) -> None:
        """Patch ``deprecated_entries`` in md frontmatter for affected files.

        Groups deprecated episodes by md_path and issues one
        ``patch_frontmatter`` call per file.

        Args:
            episodes: Source episode rows (must have ``md_path``).
            to_deprecate: Set of member IDs being deprecated.
            merged_entry_id: Entry ID of the replacement merged episode.
        """
        path_to_entries: dict[str, dict[str, str]] = defaultdict(dict)
        for ep in episodes:
            is_deprecated = ep.parent_id in to_deprecate or ep.entry_id in to_deprecate
            if is_deprecated and ep.md_path:
                path_to_entries[ep.md_path][ep.entry_id] = merged_entry_id

        root = MemoryRoot.default().root
        for md_path, deprecated_map in path_to_entries.items():
            await self._episode_writer.patch_frontmatter(
                root / md_path,
                {"deprecated_entries": deprecated_map},
            )


def _to_algo_episodes(episodes: list[Any]) -> list[AlgoEpisode]:
    """Convert LanceDB episode rows to algo Episode objects.

    Args:
        episodes: Source episode rows from LanceDB.

    Returns:
        Parallel list of algo Episode objects.
    """
    # Deferred: avoid pulling LLM libs at module import time.
    from everalgo.types import Episode as AlgoEpisode

    return [
        AlgoEpisode(
            owner_id=e.owner_id,
            episode=e.episode,
            subject=e.subject or "",
            timestamp=_ts_to_ms(e.timestamp),
        )
        for e in episodes
    ]


def _merged_episode_to_entry_body(
    algo_result: AlgoEpisode,
    cluster_id: str,
    owner_id: str,
    timestamp_iso: str,
) -> tuple[dict[str, object], dict[str, str]]:
    """Build ``(inline, sections)`` for a merged episode md entry.

    ``session_id`` is intentionally omitted (aggregation product has no
    session); the cascade handler defaults to ``None``.

    Args:
        algo_result: Algo reflector output with ``.subject`` / ``.episode``.
        cluster_id: Parent cluster identifier.
        owner_id: Target owner identifier.
        timestamp_iso: ISO-formatted timestamp for the entry.

    Returns:
        ``(inline, sections)`` tuple ready for ``append_entries``.
    """
    inline: dict[str, object] = {
        "owner_id": owner_id,
        "timestamp": timestamp_iso,
        "parent_type": "cluster",
        "parent_id": cluster_id,
    }
    sections: dict[str, str] = {
        "Subject": algo_result.subject or "",
        "Content": algo_result.episode,
    }
    return inline, sections


def _ts_to_ms(ts: object) -> int:
    """Coerce a timestamp to milliseconds.

    LanceDB episode rows store ``timestamp`` as a ``datetime`` object;
    the algo Episode type uses ``int`` (milliseconds). This helper
    handles both.

    Args:
        ts: A ``datetime``, ``int``, or ``float`` timestamp.

    Returns:
        Timestamp in milliseconds.

    Raises:
        TypeError: When ``ts`` is not a recognised type.
    """
    if isinstance(ts, _dt.datetime):
        return int(ts.timestamp() * 1000)
    if isinstance(ts, (int, float)):
        return int(ts)
    raise TypeError(f"unexpected timestamp type: {type(ts)}")
