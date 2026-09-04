"""DecisionReflectionOrchestrator — Select -> Merge -> Write -> Deprecate.

Consolidates fragmented sqlite ``kind=decision`` cluster members into
one merged Decision per cluster. The merged entry is written to
markdown (``parent_type=cluster``), ``DecisionExtracted(source=
"reflection")`` is emitted so clustering can skip it, and the originals
are deprecated in md frontmatter and Lance ``deprecated_by``.

Copied from :class:`ReflectionOrchestrator` (episode). Do **not** edit
that file — Decision reflection is a sibling cycle, not a flag on the
episode path. There is no atomic-fact re-extract: Decision is not an
episode, and ``wait_for_event`` would hang because no strategy consumes
``source="reflection"`` (``trigger_decision_clustering`` applies only
to ``source="pipeline"``).
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from everalgo.types import Decision as AlgoDecision
    from everalgo.user_memory import DecisionReflector

    from everos.component.embedding import EmbeddingProvider
    from everos.infra.persistence.markdown import DecisionWriter

import numpy as np

from everos.component.utils.datetime import from_timestamp, to_iso_format
from everos.core.errors import AppError
from everos.core.observability.logging import get_logger
from everos.core.observability.tracing import memory_span
from everos.core.persistence import MemoryRoot
from everos.infra.ome.context import StrategyContext
from everos.memory._partition_locks import get_partition_lock
from everos.memory.events import DecisionExtracted

logger = get_logger(__name__)

_MAX_CLUSTERS_PER_RUN = 10


def _escape_sql(value: str) -> str:
    """Escape single quotes for LanceDB SQL-like ``where`` predicates."""
    return value.replace("'", "''")


class DecisionReflectionOrchestrator:
    """Run one Decision Reflection cycle for a single owner scope.

    Args:
        cluster_repo: SQLite cluster repository.
        decision_store: LanceDB decision repository (read + update).
        decision_writer: Markdown daily-log writer for decisions.
        report_repo: SQLite reflection report repository.
        reflector: Algorithm-side DecisionReflector (areflect).
        embedder: Embedding provider for centroid recomputation.
    """

    def __init__(
        self,
        *,
        cluster_repo: Any,
        decision_store: Any,
        decision_writer: DecisionWriter,
        report_repo: Any,
        reflector: DecisionReflector,
        embedder: EmbeddingProvider,
    ) -> None:
        self._cluster_repo = cluster_repo
        self._decision_store = decision_store
        self._decision_writer = decision_writer
        self._report_repo = report_repo
        self._reflector = reflector
        self._embedder = embedder

    async def run(
        self,
        *,
        ctx: StrategyContext,
        owner_id: str,
        owner_type: str = "user",
        kind: str = "decision",
        app_id: str = "default",
        project_id: str = "default",
    ) -> list[object]:
        """Run one Decision Reflection cycle for a single owner scope."""
        candidates = await self._select_candidates(
            owner_id=owner_id,
            kind=kind,
            app_id=app_id,
            project_id=project_id,
        )
        logger.info(
            "decision_reflection_candidates_selected",
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
            "decision_reflection_cycle_completed",
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
                "decision_reflection_cluster_skipped",
                cluster_id=cluster_id,
                exc_info=True,
            )
            return None
        except Exception:
            logger.error(
                "decision_reflection_cluster_unexpected_error",
                cluster_id=cluster_id,
                exc_info=True,
            )
            return None

    async def _select_candidates(
        self,
        *,
        owner_id: str,
        kind: str,
        app_id: str,
        project_id: str,
    ) -> list[str]:
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
        await self._detect_orphans(cluster_id, owner_id, app_id, project_id)

        scope = dict(owner_id=owner_id, app_id=app_id, project_id=project_id)
        members, decisions = await self._load_cluster_decisions(
            cluster_id=cluster_id, **scope
        )
        if not members or not decisions:
            return None

        mode, algo_result = await self._reflect_cluster(
            decisions=decisions,
            owner_id=owner_id,
        )
        if algo_result is None:
            return None

        merged_entry_id = await self._write_and_emit(
            ctx=ctx,
            cluster_id=cluster_id,
            **scope,
            algo_result=algo_result,
            decisions=decisions,
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
            decisions=decisions,
        )

    async def _reflect_cluster(
        self,
        *,
        decisions: list[Any],
        owner_id: str,
    ) -> tuple[str, AlgoDecision | None]:
        merged_entry_ids = [d.entry_id for d in decisions if d.parent_type == "cluster"]
        is_update = bool(merged_entry_ids)
        mode = "update" if is_update else "init"
        algo_result = await self._call_reflector(
            decisions=decisions,
            merged_entry_ids=merged_entry_ids,
            is_update=is_update,
            owner_id=owner_id,
        )
        return mode, algo_result

    async def _load_cluster_decisions(
        self,
        *,
        cluster_id: str,
        owner_id: str,
        app_id: str,
        project_id: str,
    ) -> tuple[list[tuple[str, str]], list[Any]]:
        members = await self._cluster_repo.get_members_with_type(cluster_id)
        if not members:
            return [], []
        member_ids = [mid for mid, _ in members]
        rows = await self._decision_store.find_by_owner_entries(
            owner_id,
            member_ids,
            app_id=app_id,
            project_id=project_id,
        )
        rows.sort(key=lambda d: d.timestamp)
        return members, rows

    async def _write_and_emit(
        self,
        *,
        ctx: StrategyContext,
        cluster_id: str,
        owner_id: str,
        app_id: str,
        project_id: str,
        algo_result: AlgoDecision,
        decisions: list[Any],
        mode: str,
        members: list[tuple[str, str]],
    ) -> str | None:
        last_ts = max(row.timestamp for row in decisions)
        merged_entry_id = await self._write_merged_decision(
            cluster_id=cluster_id,
            owner_id=owner_id,
            app_id=app_id,
            project_id=project_id,
            algo_result=algo_result,
            last_ts=last_ts,
        )
        logger.info(
            "decision_reflection_merged",
            cluster_id=cluster_id,
            mode=mode,
            source_count=len(members),
            merged_entry_id=merged_entry_id,
        )
        event = DecisionExtracted(
            memcell_id=merged_entry_id,
            decision_entry_id=merged_entry_id,
            title=algo_result.title,
            decision_text=algo_result.decision,
            reason=algo_result.reason,
            impact=algo_result.impact,
            tags=list(algo_result.tags),
            decision_timestamp_ms=_ts_to_ms(last_ts),
            owner_id=owner_id,
            session_id=None,
            app_id=app_id,
            project_id=project_id,
            source="reflection",
        )
        await ctx.emit(event)
        # No wait_for_event: clustering applies_to pipeline only, and
        # there is no atomic-fact re-extract for a merged Decision.
        return merged_entry_id

    async def _write_merged_decision(
        self,
        *,
        cluster_id: str,
        owner_id: str,
        app_id: str,
        project_id: str,
        algo_result: AlgoDecision,
        last_ts: object,
    ) -> str:
        last_ts_iso = to_iso_format(from_timestamp(_ts_to_ms(last_ts)))
        if last_ts_iso is None:
            raise ValueError("to_iso_format returned None for valid timestamp")
        inline, sections = _merged_decision_to_entry_body(
            algo_result, cluster_id, owner_id, last_ts_iso
        )
        entry_ids = await self._decision_writer.append_entries(
            owner_id,
            [(inline, sections)],
            app_id=app_id,
            project_id=project_id,
        )
        return entry_ids[0].format()

    async def _detect_orphans(
        self,
        cluster_id: str,
        owner_id: str,
        app_id: str,
        project_id: str,
    ) -> None:
        where = (
            f"parent_type = 'cluster' AND parent_id = '{_escape_sql(cluster_id)}' "
            f"AND deprecated_by IS NULL "
            f"AND owner_id = '{_escape_sql(owner_id)}' "
            f"AND app_id = '{_escape_sql(app_id)}' "
            f"AND project_id = '{_escape_sql(project_id)}'"
        )
        orphans = await self._decision_store.find_where(where, limit=10)
        if orphans:
            logger.warning(
                "decision_reflection_orphan_detected",
                cluster_id=cluster_id,
                orphan_entry_ids=[o.entry_id for o in orphans],
            )

    async def _call_reflector(
        self,
        *,
        decisions: list[Any],
        merged_entry_ids: list[str],
        is_update: bool,
        owner_id: str,
    ) -> AlgoDecision | None:
        algo_decisions = _to_algo_decisions(decisions)
        try:
            with memory_span(
                "everos.reflect.decision_consolidate",
                observation_type="generation",
                metadata={"owner_id": owner_id, "is_update": is_update},
            ):
                if is_update:
                    return await self._reflect_update(
                        algo_decisions=algo_decisions,
                        decisions=decisions,
                        merged_entry_ids=merged_entry_ids,
                    )
                return await self._reflector.areflect(algo_decisions)
        except AppError:
            logger.warning(
                "decision_reflection_reflector_failed",
                owner_id=owner_id,
                exc_info=True,
            )
            return None
        except Exception:
            logger.error(
                "decision_reflection_reflector_unexpected_error",
                owner_id=owner_id,
                exc_info=True,
            )
            return None

    async def _reflect_update(
        self,
        *,
        algo_decisions: list[AlgoDecision],
        decisions: list[Any],
        merged_entry_ids: list[str],
    ) -> AlgoDecision | None:
        merged_set = set(merged_entry_ids)
        old_algo = [
            ad
            for ad, d in zip(algo_decisions, decisions, strict=True)
            if d.entry_id in merged_set
        ]
        new_algo = [
            ad
            for ad, d in zip(algo_decisions, decisions, strict=True)
            if d.entry_id not in merged_set
        ]
        if not old_algo:
            return None
        return await self._reflector.areflect(new_algo, old_decision=old_algo[0])

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
        algo_result: AlgoDecision,
        mode: str,
        decisions: list[Any],
    ) -> object | None:
        """Deprecate originals and update cluster membership.

        ``ctx`` / ``owner_type`` are unused here (kept for signature
        parity with :class:`ReflectionOrchestrator`).
        """
        partition = f"{app_id}:{project_id}:{cluster_id}"
        try:
            async with get_partition_lock("decision_reflection_deprecate", partition):
                return await self._execute_deprecation(
                    cluster_id=cluster_id,
                    owner_id=owner_id,
                    app_id=app_id,
                    project_id=project_id,
                    original_members=original_members,
                    merged_entry_id=merged_entry_id,
                    algo_result=algo_result,
                    mode=mode,
                    decisions=decisions,
                )
        except AppError:
            logger.warning(
                "decision_reflection_deprecate_failed",
                cluster_id=cluster_id,
                exc_info=True,
            )
            return None
        except Exception:
            logger.error(
                "decision_reflection_deprecate_unexpected_error",
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
        algo_result: AlgoDecision,
        mode: str,
        decisions: list[Any],
    ) -> object | None:
        to_deprecate = await self._resolve_deprecation_targets(
            cluster_id=cluster_id,
            original_members=original_members,
        )
        if not to_deprecate:
            return None

        dep_count = await self._apply_deprecation_writes(
            decisions=decisions,
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
            decisions=decisions,
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
        )
        logger.info(
            "decision_reflection_deprecated",
            cluster_id=cluster_id,
            deprecated_decision_count=dep_count,
        )
        return report

    async def _apply_deprecation_writes(
        self,
        *,
        decisions: list[Any],
        to_deprecate: set[str],
        owner_id: str,
        app_id: str,
        project_id: str,
        merged_entry_id: str,
    ) -> int:
        await self._patch_md_frontmatter(
            decisions=decisions,
            to_deprecate=to_deprecate,
            merged_entry_id=merged_entry_id,
        )
        return await self._deprecate_lance_decisions(
            entry_ids=to_deprecate,
            owner_id=owner_id,
            app_id=app_id,
            project_id=project_id,
            merged_entry_id=merged_entry_id,
        )

    async def _resolve_deprecation_targets(
        self,
        *,
        cluster_id: str,
        original_members: list[tuple[str, str]],
    ) -> set[str]:
        current_members = await self._cluster_repo.get_members_with_type(cluster_id)
        current_ids = {mid for mid, _ in current_members}
        original_ids = {mid for mid, _ in original_members}
        return original_ids & current_ids

    async def _deprecate_lance_decisions(
        self,
        *,
        entry_ids: set[str],
        owner_id: str,
        app_id: str,
        project_id: str,
        merged_entry_id: str,
    ) -> int:
        coros: list[Any] = [
            self._decision_store.update(
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

    async def _update_cluster_after_merge(
        self,
        *,
        cluster_id: str,
        to_deprecate: set[str],
        merged_entry_id: str,
        algo_result: AlgoDecision,
        decisions: list[Any],
    ) -> None:
        await self._cluster_repo.remove_members(cluster_id, to_deprecate)
        await self._cluster_repo.add_member(cluster_id, merged_entry_id, "decision")

        centroid = await self._embedder.embed(algo_result.decision)
        centroid_blob = np.asarray(centroid, dtype=np.float32).tobytes()
        last_ts_ms = _ts_to_ms(max(row.timestamp for row in decisions))
        await self._cluster_repo.update_metadata(
            cluster_id,
            centroid_blob=centroid_blob,
            count=1,
            last_ts_ms=last_ts_ms,
            preview_json=json.dumps([algo_result.decision[:200]], ensure_ascii=False),
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
    ) -> object:
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
            deprecated_fact_count=0,
        )
        await self._report_repo.create(report)
        return report

    async def _patch_md_frontmatter(
        self,
        *,
        decisions: list[Any],
        to_deprecate: set[str],
        merged_entry_id: str,
    ) -> None:
        path_to_entries: dict[str, dict[str, str]] = defaultdict(dict)
        for row in decisions:
            is_deprecated = (
                row.parent_id in to_deprecate or row.entry_id in to_deprecate
            )
            if is_deprecated and row.md_path:
                path_to_entries[row.md_path][row.entry_id] = merged_entry_id

        root = MemoryRoot.resolve().root
        for md_path, deprecated_map in path_to_entries.items():
            await self._decision_writer.patch_frontmatter(
                root / md_path,
                {"deprecated_entries": deprecated_map},
            )


def _to_algo_decisions(rows: list[Any]) -> list[AlgoDecision]:
    from everalgo.types import Decision as AlgoDecision

    return [
        AlgoDecision(
            owner_id=row.owner_id,
            title=row.title,
            decision=row.decision,
            reason=row.reason,
            impact=row.impact,
            tags=list(row.tags),
            timestamp=_ts_to_ms(row.timestamp),
        )
        for row in rows
    ]


def _merged_decision_to_entry_body(
    algo_result: AlgoDecision,
    cluster_id: str,
    owner_id: str,
    timestamp_iso: str,
) -> tuple[dict[str, object], dict[str, str]]:
    """Build ``(inline, sections)`` for a merged decision md entry.

    ``session_id`` is omitted — a cluster merge has no conversation
    session. ``parent_type`` is ``cluster``.
    """
    inline: dict[str, object] = {
        "owner_id": owner_id,
        "timestamp": timestamp_iso,
        "parent_type": "cluster",
        "parent_id": cluster_id,
        "tags": list(algo_result.tags),
    }
    sections: dict[str, str] = {
        "Title": algo_result.title,
        "Decision": algo_result.decision,
        "Reason": algo_result.reason,
    }
    if algo_result.impact:
        sections["Impact"] = algo_result.impact
    return inline, sections


def _ts_to_ms(ts: object) -> int:
    """Coerce a Lance datetime or algo-ms int to milliseconds."""
    if isinstance(ts, _dt.datetime):
        return int(ts.timestamp() * 1000)
    if isinstance(ts, (int, float)):
        return int(ts)
    raise TypeError(f"unexpected timestamp type: {type(ts)}")
