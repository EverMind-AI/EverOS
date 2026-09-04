"""extract_principles strategy — synthesise principles.md from decision clusters.

Fires on :class:`DecisionClusterUpdated` (emitted by
``trigger_decision_clustering`` after a sqlite ``kind=decision`` merge).
There is **no** ``DecisionExtracted`` fallback: without embedding the
cluster strategy never emits, so this strategy never runs. That is
intentional — principles are cluster-level Meta Memory.

``principles.md`` is one file per user. Every dispatch **unions all**
``kind=decision`` clusters for the owner: extracting only the triggering
cluster and rewriting the file would wipe every other cluster's
principles. One LLM call per cluster with loadable members
(:meth:`PrincipleExtractor.aextract`); one ``ProfileWriter.write`` of
the concatenated result. Persist happens only after every cluster
extract succeeds so OME retry is whole-strategy.

Input shape is ``list[tuple[entry_id, Decision]]`` per cluster. The
triggering member is built from the event snapshot (title / body /
reason / impact / tags / timestamp) so we do not race cascade. Other
members load from markdown SoT via :class:`DecisionReader`, with a
Lance ``decision`` row as a last-resort fallback. Members that cannot
be loaded are skipped (debug log), not a whole-run failure.

Empty extractor output for a cluster is success. Ids (``pr_<12hex>``)
are minted at write time — EverAlgo ``Principle`` has no id.
"""

from __future__ import annotations

from everalgo.clustering import Cluster as AlgoCluster
from everalgo.types import Decision as AlgoDecision
from everalgo.types import Principle as AlgoPrinciple
from everalgo.user_memory import PrincipleExtractor

from everos.component.llm import get_llm_client
from everos.component.utils.datetime import from_iso_format, to_timestamp_ms
from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.core.persistence.markdown import StructuredEntry
from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import offline_strategy
from everos.infra.ome.triggers import Immediate
from everos.infra.persistence.lancedb import decision_repo
from everos.infra.persistence.markdown import (
    DecisionReader,
    PrincipleFrontmatter,
    PrincipleItem,
    ProfileWriter,
    mint_principle_id,
    render_principles_body,
)
from everos.infra.persistence.sqlite import cluster_repo
from everos.memory._partition_locks import get_partition_lock
from everos.memory.events import DecisionClusterUpdated

logger = get_logger(__name__)

_writer: ProfileWriter | None = None
_decision_reader: DecisionReader | None = None


def _get_writer() -> ProfileWriter:
    global _writer
    if _writer is None:
        _writer = ProfileWriter(root=MemoryRoot.resolve())
    return _writer


def _get_decision_reader() -> DecisionReader:
    global _decision_reader
    if _decision_reader is None:
        _decision_reader = DecisionReader(root=MemoryRoot.resolve())
    return _decision_reader


@offline_strategy(
    name="extract_principles",
    trigger=Immediate(on=[DecisionClusterUpdated]),
    emits=[],
    max_retries=2,
)
async def extract_principles(
    event: DecisionClusterUpdated, ctx: StrategyContext
) -> None:
    # Serialise on owner: principles.md is a single per-user file and the
    # body is read-all-clusters → LLM → overwrite. Different users run
    # fully in parallel.
    partition = f"{event.app_id}:{event.project_id}:{event.owner_id}"
    async with get_partition_lock("extract_principles", partition):
        clusters = await cluster_repo.list_for_owner(
            event.owner_id,
            "decision",
            app_id=event.app_id,
            project_id=event.project_id,
        )

        cluster_pairs: list[list[tuple[str, AlgoDecision]]] = []
        for cluster in clusters:
            pairs = await _load_pairs(cluster, event)
            if pairs:
                cluster_pairs.append(pairs)

        if not cluster_pairs:
            if not clusters or not any(c.members for c in clusters):
                await _persist_principles(
                    [],
                    owner_id=event.owner_id,
                    app_id=event.app_id,
                    project_id=event.project_id,
                )
                logger.info(
                    "principles_extracted",
                    owner_id=event.owner_id,
                    cluster_count=len(clusters),
                    principle_count=0,
                )
            else:
                logger.debug(
                    "extract_principles_no_loadable_decisions",
                    owner_id=event.owner_id,
                    cluster_count=len(clusters),
                )
            return

        extractor = PrincipleExtractor(llm=get_llm_client())
        collected: list[AlgoPrinciple] = []
        for pairs in cluster_pairs:
            collected.extend(await extractor.aextract(pairs, owner_id=event.owner_id))

        await _persist_principles(
            collected,
            owner_id=event.owner_id,
            app_id=event.app_id,
            project_id=event.project_id,
        )
    logger.info(
        "principles_extracted",
        owner_id=event.owner_id,
        cluster_count=len(cluster_pairs),
        principle_count=len(collected),
    )


async def _load_pairs(
    cluster: AlgoCluster, event: DecisionClusterUpdated
) -> list[tuple[str, AlgoDecision]]:
    """Resolve ``(entry_id, Decision)`` for one cluster; skip unloadable members."""
    pairs: list[tuple[str, AlgoDecision]] = []
    seen: set[str] = set()
    for entry_id in cluster.members:
        eid = entry_id.strip()
        if not eid or eid in seen:
            continue
        seen.add(eid)
        decision = await _resolve_decision(eid, event)
        if decision is None:
            logger.debug(
                "extract_principles_member_unresolved",
                owner_id=event.owner_id,
                entry_id=eid,
                cluster_id=cluster.id,
            )
            continue
        pairs.append((eid, decision))
    return pairs


async def _resolve_decision(
    entry_id: str, event: DecisionClusterUpdated
) -> AlgoDecision | None:
    """Snapshot for the triggering row, then md SoT, then Lance fallback."""
    if entry_id == event.decision_entry_id and event.decision_text.strip():
        return _from_snapshot(event)

    structured = await _try_structured(entry_id, event)
    if structured is not None:
        return structured

    return await _try_lance(entry_id, event)


def _from_snapshot(event: DecisionClusterUpdated) -> AlgoDecision:
    return AlgoDecision(
        owner_id=event.owner_id,
        title=event.title,
        decision=event.decision_text,
        reason=event.reason,
        impact=event.impact,
        tags=list(event.tags),
        timestamp=event.decision_timestamp_ms,
    )


async def _try_structured(
    entry_id: str, event: DecisionClusterUpdated
) -> AlgoDecision | None:
    try:
        structured = await _get_decision_reader().find_structured(
            event.owner_id,
            entry_id,
            app_id=event.app_id,
            project_id=event.project_id,
        )
    except ValueError:
        logger.debug(
            "extract_principles_entry_id_unparseable",
            owner_id=event.owner_id,
            entry_id=entry_id,
        )
        return None
    if structured is None:
        return None
    return _from_structured(structured, owner_id=event.owner_id)


def _from_structured(
    structured: StructuredEntry, *, owner_id: str
) -> AlgoDecision | None:
    decision = (structured.sections.get("Decision") or "").strip()
    if not decision:
        return None
    impact = (structured.sections.get("Impact") or "").strip() or None
    return AlgoDecision(
        owner_id=owner_id,
        title=(structured.sections.get("Title") or "").strip(),
        decision=decision,
        reason=(structured.sections.get("Reason") or "").strip(),
        impact=impact,
        tags=_parse_tags(structured.inline.get("tags") or ""),
        timestamp=_timestamp_ms_from_inline(structured.inline.get("timestamp") or ""),
    )


async def _try_lance(
    entry_id: str, event: DecisionClusterUpdated
) -> AlgoDecision | None:
    row = await decision_repo.find_by_owner_entry(
        event.owner_id,
        entry_id,
        app_id=event.app_id,
        project_id=event.project_id,
    )
    if row is None or not row.decision.strip():
        return None
    return AlgoDecision(
        owner_id=row.owner_id,
        title=row.title,
        decision=row.decision,
        reason=row.reason,
        impact=row.impact,
        tags=list(row.tags),
        timestamp=to_timestamp_ms(row.timestamp),
    )


def _parse_tags(raw: str) -> list[str]:
    """Parse ``"[a, b, c]"`` back into ``["a", "b", "c"]``.

    Mirrors cascade ``parse_inline_list`` without importing the handler
    package from a strategy.
    """
    text = raw.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return []
    body = text[1:-1].strip()
    if not body:
        return []
    return [tok.strip() for tok in body.split(",") if tok.strip()]


def _timestamp_ms_from_inline(raw: str) -> int:
    if not raw.strip():
        return 0
    try:
        return to_timestamp_ms(from_iso_format(raw.strip()))
    except (TypeError, ValueError):
        return 0


async def _persist_principles(
    principles: list[AlgoPrinciple],
    *,
    owner_id: str,
    app_id: str,
    project_id: str,
) -> None:
    """Write the union of extracted principles to ``principles.md``."""
    items = [
        PrincipleItem(
            id=mint_principle_id(),
            title=p.title,
            statement=p.statement,
            source_entry_ids=list(p.source_entry_ids),
            timestamp_ms=p.timestamp,
        )
        for p in principles
    ]
    frontmatter = PrincipleFrontmatter(
        id=f"principle_{owner_id}",
        user_id=owner_id,
        principles=items,
    )
    await _get_writer().write(
        owner_id,
        frontmatter=frontmatter,
        body=render_principles_body(items),
        app_id=app_id,
        project_id=project_id,
    )
