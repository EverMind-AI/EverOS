"""everalgo's agent-case gate question answered by an LLM.

:class:`everalgo.agent_memory.AgentCaseGate` asks one provider-neutral
yes / no question about a conversation and wants the probability of "yes";
the host decides how to get it. :class:`LLMCaseJudge` asks the extraction
LLM for that probability as a strict JSON reply. Range and finiteness are
checked by the gate, not here.
"""

from __future__ import annotations

__all__ = ["LLMCaseJudge"]

from typing import TYPE_CHECKING

from everalgo.llm import LLMClient, LLMMessage, StrictOutputModel, parse_reply

if TYPE_CHECKING:
    from everalgo.agent_memory import CaseGateQuestion

_INSTRUCTIONS = """{instructions}

Answer "true" or "false" about the conversation the user sends.
- true: {true_criterion}
- false: {false_criterion}

Reply with the probability, between 0 and 1, that the answer is "true", as
JSON: {{"probability": <number>}}"""


class _SuccessProbability(StrictOutputModel):
    """The judge's reply: the probability that the answer is "true"."""

    probability: float


class LLMCaseJudge:
    """Answer the agent-case gate question with one LLM call.

    Implements :class:`everalgo.agent_memory.CaseJudge`.

    Args:
        llm: The model asked; its reply must pass ``_SuccessProbability``.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def apredict(self, conversation: str, question: CaseGateQuestion) -> float:
        """Return the model's probability that ``question`` holds.

        Args:
            conversation: OpenAI-wire JSON of the conversation.
            question: The question, with what "true" and "false" mean.

        Returns:
            The probability the model gave, unchecked.

        Raises:
            LLMOutputError: The reply is not a ``{"probability": <number>}``
                object.
        """
        system = _INSTRUCTIONS.format(
            instructions=question.instructions,
            true_criterion=question.true_criterion,
            false_criterion=question.false_criterion,
        )
        reply = await self._llm.achat(
            messages=[
                LLMMessage(role="system", content=system),
                LLMMessage(role="user", content=conversation),
            ],
            response_format=_SuccessProbability,
        )
        return parse_reply(_SuccessProbability, reply, operator="case gate").probability
