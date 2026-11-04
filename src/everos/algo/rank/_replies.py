"""The agentic-retrieval and rerank reply models, the shapes their prompts' skeletons show, and the LLM callsites.

Each callsite makes one call and validates the reply against its model; there is no retry here, the host decides. The temperatures are the ones each call has always used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from everos.algo.llm import LLMMessage, StrictOutputModel, parse_reply

if TYPE_CHECKING:
    from everos.algo.llm.protocols import LLMClient


class _SufficiencyReply(StrictOutputModel):
    """Sufficiency reply: the four-field skeleton of the sufficiency prompt, in order."""

    is_sufficient: bool
    reasoning: str
    key_information_found: list[str]
    missing_information: list[str]


class _MultiQueryReply(StrictOutputModel):
    """Multi-query reply: the ``{"queries", "reasoning"}`` skeleton of the multi-query prompt."""

    queries: list[str]
    reasoning: str


class _RefinedQueryReply(StrictOutputModel):
    """Refined-query reply: the refined query in ``text``, the free-text answer wrapped as llm-contract §6.1 asks."""

    text: str


class _RankedCandidate(StrictOutputModel):
    """One ``{"id", "score"}`` item of the rerank skeleton's ``ranked`` list."""

    id: str
    score: float


class _RerankReply(StrictOutputModel):
    """Rerank reply: the ``{"ranked": [{"id", "score"}]}`` skeleton every rerank prompt shows."""

    ranked: list[_RankedCandidate]


async def _call_llm_for_sufficiency(
    llm: LLMClient, messages: list[LLMMessage]
) -> _SufficiencyReply:
    """Request a sufficiency verdict and return it validated.

    Raises:
        LLMOutputError: If the reply is not a valid sufficiency output.
    """
    text = await llm.achat(
        messages=messages, response_format=_SufficiencyReply, temperature=0.0
    )
    return parse_reply(_SufficiencyReply, text, operator="agentic sufficiency")


async def _call_llm_for_multi_queries(
    llm: LLMClient, messages: list[LLMMessage]
) -> _MultiQueryReply:
    """Request complementary queries and return the reply validated.

    Raises:
        LLMOutputError: If the reply is not a valid multi-query output.
    """
    text = await llm.achat(
        messages=messages, response_format=_MultiQueryReply, temperature=0.4
    )
    return parse_reply(_MultiQueryReply, text, operator="agentic multi_query")


async def _call_llm_for_refined_query(
    llm: LLMClient, messages: list[LLMMessage]
) -> _RefinedQueryReply:
    """Request a refined query and return the reply validated.

    Raises:
        LLMOutputError: If the reply is not a valid refined-query output.
    """
    text = await llm.achat(
        messages=messages, response_format=_RefinedQueryReply, temperature=0.3
    )
    return parse_reply(_RefinedQueryReply, text, operator="agentic refined_query")


async def _call_llm_for_rerank(
    llm: LLMClient, messages: list[LLMMessage]
) -> _RerankReply:
    """Request rerank scores and return the reply validated.

    Raises:
        LLMOutputError: If the reply is not a valid rerank output.
    """
    text = await llm.achat(messages=messages, response_format=_RerankReply)
    return parse_reply(_RerankReply, text, operator="rerank")
