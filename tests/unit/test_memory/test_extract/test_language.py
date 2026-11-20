"""``decide_output_language``: one detector call per memcell, never raising.

The detector is everalgo's real ``LanguageDetector``; the model answers from
canned replies. What it names is returned as is (its "Other" comes back as
English), a cell without chat text costs no call, and a failed call -- an
error or an unusable reply -- gives ``None`` so the episode is written as
before. Every outcome logs one line naming the memcell.
"""

from __future__ import annotations

from typing import Any

import pytest
import structlog
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, MemCell
from everalgo.user_memory import LanguageDetector, OutputLanguage

from everos.memory.extract.language import decide_output_language
from tests.fakes import UNUSABLE_REPLY


def _cell(*contents: str) -> MemCell:
    return MemCell(
        items=[
            ChatMessage(
                id=f"m{index}",
                role="user",
                content=content,
                timestamp=1_700_000_000_000,
                sender_id="u1",
            )
            for index, content in enumerate(contents)
        ],
        timestamp=1_700_000_000_000,
    )


def _answering(reply: str) -> FakeLLMClient:
    return FakeLLMClient(handler=lambda *_args, **_kwargs: reply)


async def _decide(llm: FakeLLMClient, cell: MemCell) -> OutputLanguage | None:
    return await decide_output_language(
        LanguageDetector(llm=llm), cell, memcell_id="mc_a"
    )


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Chinese", OutputLanguage.CHINESE),
        ("Japanese", OutputLanguage.JAPANESE),
        ("English", OutputLanguage.ENGLISH),
        ("Other", OutputLanguage.ENGLISH),
    ],
)
async def test_the_named_language_is_returned_and_logged(
    label: str, expected: OutputLanguage
) -> None:
    llm = _answering(f'{{"language": "{label}"}}')

    with structlog.testing.capture_logs() as logs:
        language = await _decide(llm, _cell("帮我订明天去杭州的票"))

    assert language is expected
    assert llm.call_count == 1
    assert _lines(logs) == [
        {
            "event": "episode_language_decided",
            "log_level": "info",
            "memcell_id": "mc_a",
            "language": expected.value,
        }
    ]


async def test_a_cell_without_chat_text_costs_no_call() -> None:
    llm = _answering('{"language": "Chinese"}')

    with structlog.testing.capture_logs() as logs:
        language = await _decide(llm, _cell("   "))

    assert language is None
    assert llm.call_count == 0
    assert _lines(logs) == [
        {
            "event": "episode_language_undecided",
            "log_level": "info",
            "memcell_id": "mc_a",
            "reason": "no_chat",
        }
    ]


@pytest.mark.parametrize(
    ("llm", "error_type"),
    [
        (_answering(UNUSABLE_REPLY), "LLMOutputError"),
        (_answering('{"language": "French"}'), "LLMOutputError"),
        (FakeLLMClient(responses=[TimeoutError("deadline")]), "TimeoutError"),
    ],
    ids=["unusable reply", "label outside the set", "client error"],
)
async def test_a_failed_call_degrades_to_none(
    llm: FakeLLMClient, error_type: str
) -> None:
    """Undecided is how every episode was written before the call existed."""
    with structlog.testing.capture_logs() as logs:
        language = await _decide(llm, _cell("hello there"))

    assert language is None
    (line,) = _lines(logs)
    assert {k: line[k] for k in ("event", "log_level", "memcell_id", "reason")} == {
        "event": "episode_language_undecided",
        "log_level": "warning",
        "memcell_id": "mc_a",
        "reason": "failed",
    }
    assert line["error_type"] == error_type


def _lines(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """This module's lines; everalgo logs through the stdlib, not structlog."""
    return [line for line in logs if line["event"].startswith("episode_language")]
