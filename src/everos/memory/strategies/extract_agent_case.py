"""extract_agent_case strategy — distil one AgentCase from an agent memcell.

Per-memcell extraction: algo's :class:`AgentCaseExtractor` returns no case
together with the gate that rejected the memcell (logged as
``agent_case_skipped`` with its ``reason`` and ``detail``), or a single case.
Quality score is persisted as-is; no everos-side threshold filter
(opensource parity — let downstream rank / filter handle it).

**Correctness gate** (``[agent_memory] case_gate_enabled``, off by default):
with a judge in :attr:`MemoryDeps.case_judge`, everalgo admits a memcell by
the judged probability that its work was completed correctly instead of by
its complexity and value filters, and the admitted case carries the
decision as ``case_gate``.

**Multi-agent fan-out**: the algo prompt's output is third-person
(``the agent did X``, ``a different agent lacking this insight``), so
the same case body is a valid reference experience for **every**
assistant sender that participated in the trajectory. We collect all
distinct assistant-side sender_ids in the memcell, write the same case
body once per agent (each gets its own owner_id-scoped md entry), and
return one ``AgentCaseExtracted`` per agent so the downstream skill
clustering chain runs in each agent's own scope.

Algo limitation (recorded in
``local/notes/2026-05-18-write-read-loop-status.md``): the LLM only
sees a flat ``"assistant"`` role label — per-agent serialisation +
per-agent output is a future algo-side change. Until then, broadcast
is the right md-layer fan-out (mirrors Episode/atomic_fact).
"""

from __future__ import annotations

from everalgo.agent_memory import AgentCaseExtractor, AgentCaseGate
from everalgo.types import AgentCase as AlgoAgentCase

from everos.core.observability.logging import get_logger
from everos.memory.deps import MemoryDeps
from everos.memory.events import AgentCaseExtracted, AgentPipelineStarted
from everos.memory.mappers import agent_case as agent_case_mapper
from everos.memory.models import MemCell
from everos.memory.ports import Event, Immediate, Scope, StrategySpec

logger = get_logger(__name__)


async def extract_agent_case(
    event: AgentPipelineStarted, deps: MemoryDeps
) -> list[Event]:
    """Distil one case from the memcell and file it under every agent sender."""
    # 1. Find the distinct assistant senders in this memcell; bail if none.
    agent_ids = _collect_agent_ids(event.memcell)
    if not agent_ids:
        logger.warning(
            "agent_case_skipped_no_assistant",
            memcell_id=event.memcell_id,
            session_id=event.session_id,
        )
        return []

    # 2. Run the extractor once; algo returns no case with the gate that
    #    rejected the memcell, or a single case.
    extractor = AgentCaseExtractor(llm=deps.require_llm(), gate=_case_gate(deps))
    result = await extractor.aextract_with_reason(event.memcell)
    if not result.cases:
        logger.info(
            "agent_case_skipped",
            memcell_id=event.memcell_id,
            session_id=event.session_id,
            reason=result.reason,
            detail=result.detail,
        )
        return []

    # 3. File the same case under every agent sender and announce each copy
    #    → downstream trigger_skill_clustering.
    (algo_case,) = result.cases
    extracted: list[Event] = [
        await _file_case(algo_case, agent_id, event, deps) for agent_id in agent_ids
    ]
    logger.info(
        "agent_case_extracted",
        memcell_id=event.memcell_id,
        session_id=event.session_id,
        owner_ids=agent_ids,
        quality_score=algo_case.quality_score,
        fanout=len(agent_ids),
    )
    return extracted


def _case_gate(deps: MemoryDeps) -> AgentCaseGate | None:
    """everalgo's correctness gate over the configured judge, if there is one."""
    if deps.case_judge is None:
        return None
    return AgentCaseGate(judge=deps.case_judge)


async def _file_case(
    algo_case: AlgoAgentCase,
    agent_id: str,
    event: AgentPipelineStarted,
    deps: MemoryDeps,
) -> AgentCaseExtracted:
    """Write the case under ``agent_id`` and return its announcement."""
    case = agent_case_mapper.from_algo(
        algo_case,
        owner_id=agent_id,
        session_id=event.session_id,
        parent_id=event.memcell_id,
    )
    scope = Scope(
        app_id=event.app_id,
        project_id=event.project_id,
        owner_id=agent_id,
        owner_type="agent",
    )
    ref = await deps.stores.agent_case.append(scope, case)
    return AgentCaseExtracted(
        memcell_id=event.memcell_id,
        case_entry_id=ref.entry_id,
        task_intent=case.task_intent,
        approach=case.approach,
        key_insight=case.key_insight,
        quality_score=case.quality_score,
        case_timestamp_ms=case.timestamp,
        agent_id=case.owner_id,
        app_id=event.app_id,
        project_id=event.project_id,
        case_gate=case.case_gate,
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


EXTRACT_AGENT_CASE_SPEC = StrategySpec(
    name="extract_agent_case",
    fn=extract_agent_case,
    trigger=Immediate(on=[AgentPipelineStarted]),
    emits=frozenset({AgentCaseExtracted}),
    max_retries=2,
)
