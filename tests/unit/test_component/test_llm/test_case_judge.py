"""``LLMCaseJudge`` answers everalgo's agent-case gate question with one call.

The question's instructions and criteria go to the model as the system
message and the conversation as the user message; the reply's probability
comes back unchecked (the gate validates the range), and a reply that is not
a probability object is an ``LLMOutputError``.
"""

from __future__ import annotations

import pytest
from everalgo.agent_memory import CaseGateQuestion
from everalgo.llm import LLMOutputError
from everalgo.testing import FakeLLMClient

from everos.component.llm import LLMCaseJudge
from tests.fakes import UNUSABLE_REPLY, agent_track_llm

_QUESTION = CaseGateQuestion(
    instructions="Was the work done right?",
    true_criterion="it was",
    false_criterion="it was not",
)


async def test_returns_the_probability_the_model_gives() -> None:
    llm = agent_track_llm()
    probability = await LLMCaseJudge(llm).apredict('[{"role": "user"}]', _QUESTION)

    assert probability == 0.9
    (call,) = llm.calls
    system, user = call.messages
    assert "Was the work done right?" in system.content
    assert "- true: it was" in system.content
    assert "- false: it was not" in system.content
    assert user.content == '[{"role": "user"}]'


async def test_an_unusable_reply_is_an_output_error() -> None:
    llm = FakeLLMClient(responses=[UNUSABLE_REPLY])
    with pytest.raises(LLMOutputError):
        await LLMCaseJudge(llm).apredict("[]", _QUESTION)
