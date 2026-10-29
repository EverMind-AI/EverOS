"""everalgo ``AgentCase`` <-> domain :class:`~everos.memory.models.AgentCase`."""

from __future__ import annotations

__all__ = ["from_algo", "to_algo"]

from everalgo.types import AgentCase as AlgoAgentCase

from everos.memory.models import AgentCase


def from_algo(
    algo: AlgoAgentCase, *, owner_id: str, session_id: str, parent_id: str
) -> AgentCase:
    """Domain case from an algo case plus the context only the caller knows.

    The algo case carries no owner, session or source memcell, so the caller
    supplies them. Its ``id`` is dropped: the case store mints the entry id
    the case is addressed by. An empty ``key_insight`` becomes ``None`` so the
    stored case omits the section. Algo extras other than ``case_gate`` are
    dropped.
    """
    data = algo.model_dump(include=set(AgentCase.model_fields) - {"owner_id"})
    return AgentCase.model_validate(
        {
            **data,
            "owner_id": owner_id,
            "session_id": session_id,
            "parent_id": parent_id,
            "key_insight": algo.key_insight or None,
        }
    )


def to_algo(case: AgentCase, *, entry_id: str) -> AlgoAgentCase:
    """Algo case addressed by the stored case's entry id.

    ``case_gate`` rides along only when the case passed an admission gate;
    everalgo reads its presence as "admitted".
    """
    gate = {"case_gate": case.case_gate} if case.case_gate is not None else {}
    return AlgoAgentCase(
        id=entry_id,
        timestamp=case.timestamp,
        task_intent=case.task_intent,
        approach=case.approach,
        quality_score=case.quality_score,
        key_insight=case.key_insight or "",
        **gate,
    )
