"""extract_agent_case strategy — distil one AgentCase from an agent memcell.

Per-memcell extraction: algo's :class:`AgentCaseExtractor` returns ``[]``
(structurally / heuristically filtered) or ``[case]`` (single output).
Quality score is persisted as-is; no everos-side threshold filter
(opensource parity — let downstream rank / filter handle it).

**Multi-agent fan-out**: the algo prompt's output is third-person
(``the agent did X``, ``a different agent lacking this insight``), so
the same case body is a valid reference experience for **every**
assistant sender that participated in the trajectory. We collect all
distinct assistant-side sender_ids in the memcell, write the same case
body once per agent (each gets its own owner_id-scoped md entry), and
emit one ``AgentCaseExtracted`` per agent so the downstream skill
clustering chain runs in each agent's own scope.

Algo limitation (recorded in
``local/notes/2026-05-18-write-read-loop-status.md``): the LLM only
sees a flat ``"assistant"`` role label — per-agent serialisation +
per-agent output is a future algo-side change. Until then, broadcast
is the right md-layer fan-out (mirrors Episode/atomic_fact).
"""

from __future__ import annotations

from everalgo.agent_memory import AgentCaseExtractor

from everos.component.llm import get_llm_client
from everos.component.utils.datetime import from_timestamp, to_iso_format
from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import offline_strategy
from everos.infra.ome.triggers import Immediate
from everos.infra.persistence.markdown import AgentCaseWriter
from everos.memory.events import AgentCaseExtracted, AgentPipelineStarted
from everos.memory.models import AgentCase, MemCell

logger = get_logger(__name__)

_writer: AgentCaseWriter | None = None


def _get_writer() -> AgentCaseWriter:
    global _writer
    if _writer is None:
        _writer = AgentCaseWriter(root=MemoryRoot.default())
    return _writer


@offline_strategy(
    name="extract_agent_case",
    trigger=Immediate(on=[AgentPipelineStarted]),
    emits=[AgentCaseExtracted],
    max_retries=2,
)
async def extract_agent_case(event: AgentPipelineStarted, ctx: StrategyContext) -> None:
    # 1. Find the distinct assistant senders in this memcell; bail if none.
    agent_ids = _collect_agent_ids(event.memcell)
    if not agent_ids:
        logger.warning(
            "agent_case_skipped_no_assistant",
            memcell_id=event.memcell_id,
            session_id=event.session_id,
        )
        return

    # 2. Run the LLM extractor once; algo returns [] or [single case].
    extractor = AgentCaseExtractor(llm=get_llm_client())
    algo_cases = await extractor.aextract(event.memcell)
    if not algo_cases:
        logger.info(
            "agent_case_skipped_by_algo",
            memcell_id=event.memcell_id,
            session_id=event.session_id,
        )
        return

    # 3. For each agent sender: write the same case body to its md and emit
    #    AgentCaseExtracted → downstream trigger_skill_clustering.
    algo_case = algo_cases[0]
    writer = _get_writer()
    for agent_id in agent_ids:
        case = AgentCase.from_algo(
            algo_case,
            owner_id=agent_id,
            session_id=event.session_id,
            parent_id=event.memcell_id,
        )
        inline, sections = _agent_case_to_entry_body(case)
        eid = await writer.append_entry(
            case.owner_id,
            inline=inline,
            sections=sections,
            app_id=event.app_id,
            project_id=event.project_id,
        )
        await ctx.emit(
            AgentCaseExtracted(
                memcell_id=event.memcell_id,
                case_entry_id=eid.format(),
                task_intent=case.task_intent,
                quality_score=case.quality_score,
                case_timestamp_ms=case.timestamp,
                agent_id=case.owner_id,
                app_id=event.app_id,
                project_id=event.project_id,
            )
        )
    logger.info(
        "agent_case_extracted",
        memcell_id=event.memcell_id,
        session_id=event.session_id,
        owner_ids=agent_ids,
        quality_score=algo_case.quality_score,
        fanout=len(agent_ids),
    )


def _collect_agent_ids(memcell: MemCell) -> list[str]:
    """Distinct assistant-side sender_ids in a cell, preserving first-seen order.

    An "assistant-side" item is any ``role == 'assistant'`` ChatMessage or
    any ``kind == 'tool_call'`` request (also assistant-emitted). The
    deterministic order matches Episode pipeline's
    :func:`_unique_user_senders` so two runs over the same memcell fan
    out identically. Empty list means the cell carries no agent
    trajectory — caller logs + skips.
    """
    seen: list[str] = []
    for item in memcell.items:
        sid = getattr(item, "sender_id", None)
        if not sid or sid in seen:
            continue
        if getattr(item, "role", None) == "assistant":
            seen.append(sid)
            continue
        if getattr(item, "kind", None) == "tool_call":
            seen.append(sid)
    return seen


def _agent_case_to_entry_body(
    case: AgentCase,
) -> tuple[dict[str, object], dict[str, str]]:
    """Split a domain AgentCase into ``(inline, sections)`` for md rendering.

    Mirrors ``_atomic_fact_to_entry_body`` / ``_foresight_to_entry_body`` /
    ``_episode_to_entry_body``. ``quality_score`` rides in inline so cascade
    can hash it (it's part of ``content_change_keys`` on the handler);
    KeyInsight is optional and elided when empty.
    """
    inline: dict[str, object] = {
        "owner_id": case.owner_id,
        "session_id": case.session_id,
        "timestamp": to_iso_format(from_timestamp(case.timestamp)),
        "parent_type": "memcell",
        "parent_id": case.parent_id,
        "quality_score": case.quality_score,
    }
    sections: dict[str, str] = {
        "TaskIntent": case.task_intent,
        "Approach": case.approach,
    }
    if case.key_insight:
        sections["KeyInsight"] = case.key_insight
    return inline, sections
