"""Every adapter's prompts must render with exactly the arguments run.py supplies.

A placeholder added to a prompt without threading its argument through raises KeyError
only once the answer stage runs, which is after ADD has already been paid for.

These tests deliberately make no assumption that the four benchmarks share a prompt
shape. They do not: SubtleMemory's judge reads relation metadata, EverMemBench's answer
prompt carries no date line, and only LoCoMo-derived prompts have a context template.
Asserting a common shape is what let three adapters keep LoCoMo's prompts unnoticed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))

BENCHMARKS = ("locomo", "longmemeval", "evermembench", "subtlememory")


def _adapter(name: str):
    import adapters

    return adapters.get(name)


@pytest.mark.parametrize("name", BENCHMARKS)
@pytest.mark.parametrize("question_date", ["", "2023-05-07"])
def test_answer_prompt_renders(name: str, question_date: str) -> None:
    """The answer prompt takes exactly the three arguments run.py passes it."""
    import run
    from config import BenchmarkConfig

    config = BenchmarkConfig.from_toml(name)
    template = run._prompt(config, "ANSWER_PROMPT")
    rendered = template.format(
        context="CTX",
        current_date_line=f"Current Date: {question_date}\n" if question_date else "",
        question="q?",
    )
    assert "CTX" in rendered and "q?" in rendered
    # The date itself reaches the prompt only when that benchmark's prompt asks for one.
    # Match on the date value, not on the words "Current Date": LongMemEval's protocol
    # block mentions them in prose.
    if question_date and "{current_date_line}" in template:
        assert question_date in rendered
    if question_date and "{current_date_line}" not in template:
        assert question_date not in rendered


@pytest.mark.parametrize("name", BENCHMARKS)
def test_judge_prompt_renders(name: str) -> None:
    """The judge prompt renders from run.py's arguments plus the adapter's fields."""
    import run
    from config import BenchmarkConfig

    config = BenchmarkConfig.from_toml(name)
    adapter = _adapter(name)
    extra = getattr(adapter, "judge_fields", lambda _m, _g: {})({}, "gen")
    rendered = run._prompt(config, "JUDGE_USER_PROMPT").format(
        question="q?", golden_answer="gold", generated_answer="gen", **extra
    )
    assert "q?" in rendered and "gen" in rendered
    assert run._prompt(config, "JUDGE_SYSTEM_PROMPT").strip()


@pytest.mark.parametrize("name", BENCHMARKS)
def test_context_renders(name: str) -> None:
    """Context comes from the adapter's builder, or a template when it has one."""
    import run
    from config import BenchmarkConfig

    config = BenchmarkConfig.from_toml(name)
    adapter = _adapter(name)
    episodes = [{"subject": "S", "episode": "E", "timestamp": "T", "session_id": "sid"}]
    own = getattr(adapter, "build_context", None)
    if own is not None:
        rendered = own(episodes, [])
        assert rendered.strip()
        assert own([], []).strip(), "an empty result set still needs a rendering"
    else:
        rendered = run._prompt(config, "CONTEXT_TEMPLATE").format(
            speaker_a="A", speaker_b="B", episodes="EP"
        )
        assert "EP" in rendered


@pytest.mark.parametrize("name", BENCHMARKS)
def test_judge_meta_keys_are_declared(name: str) -> None:
    """An adapter whose judge needs extra fields must declare which ones to thread."""
    adapter = _adapter(name)
    if getattr(adapter, "judge_fields", None) is None:
        return
    keys = getattr(adapter, "JUDGE_META_KEYS", ())
    assert keys, f"{name} defines judge_fields but declares no JUDGE_META_KEYS"
