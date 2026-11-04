"""The atomic-fact, foresight and reflect reply models and their LLM callsites.

Each model is the shape its prompt's output skeleton shows. Each callsite makes one call and validates the reply against
its model; there is no retry here, the host decides. The episode and profile packages hold their own reply models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from everos.algo.llm import LLMMessage, StrictOutputModel, parse_reply

if TYPE_CHECKING:
    from everos.algo.llm.protocols import LLMClient


class _AtomicFactsBlock(StrictOutputModel):
    """The inner ``{"atomic_fact"}`` object of the atomic-fact skeleton."""

    atomic_fact: list[str]


class _AtomicFactsReply(StrictOutputModel):
    """Atomic-fact reply: the ``{"atomic_facts": {"atomic_fact"}}`` skeleton of both atomic-fact prompts."""

    atomic_facts: _AtomicFactsBlock


class _ForesightItem(StrictOutputModel):
    """One entry of the foresight skeleton's ``foresights`` array: ``{"content", "evidence", "start_time", ...}``.

    The three time fields are nullable because the extractor fills them in when the model leaves them out: a missing
    ``start_time`` falls back to the memcell's first message, and an ``end_time`` and a ``duration_days`` are each
    computed from the other.
    """

    content: str
    evidence: str
    start_time: str | None
    end_time: str | None
    duration_days: int | None


class _ForesightsReply(StrictOutputModel):
    """Foresight reply: the ``{"foresights": [...]}`` skeleton of the foresight prompt."""

    foresights: list[_ForesightItem]


class _ReflectOutput(StrictOutputModel):
    """Reflect reply: the ``{"content", "summary", "title"}`` skeleton of both reflect prompts, in order.

    ``summary`` is declared after ``content`` deliberately: a schema-constrained reply is generated in schema order,
    so a ``summary`` declared first would be written before the narrative it previews.
    """

    content: str
    summary: str
    title: str


async def _call_llm_for_atomic_facts(
    llm: LLMClient, messages: list[LLMMessage]
) -> _AtomicFactsReply:
    """Request an atomic-fact reply and return it validated.

    Raises:
        LLMOutputError: If the reply is not a valid atomic-fact output.
    """
    text = await llm.achat(messages=messages, response_format=_AtomicFactsReply)
    return parse_reply(_AtomicFactsReply, text, operator="atomic_fact")


async def _call_llm_for_foresights(
    llm: LLMClient, messages: list[LLMMessage], *, temperature: float
) -> _ForesightsReply:
    """Request a foresight reply and return it validated.

    Raises:
        LLMOutputError: If the reply is not a valid foresight output.
    """
    text = await llm.achat(
        messages=messages, response_format=_ForesightsReply, temperature=temperature
    )
    return parse_reply(_ForesightsReply, text, operator="foresight")


async def _call_llm_for_reflect(
    llm: LLMClient, messages: list[LLMMessage]
) -> _ReflectOutput:
    """Request a reflect reply and return it validated.

    Raises:
        LLMOutputError: If the reply is not a valid reflect output.
    """
    text = await llm.achat(messages=messages, response_format=_ReflectOutput)
    return parse_reply(_ReflectOutput, text, operator="reflect")
