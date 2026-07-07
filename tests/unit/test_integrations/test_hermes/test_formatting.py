"""Contract tests for ``integrations/hermes/_formatting.py``.

Pins the Hermes-agnostic formatting helpers:

- ``format_prefetch`` emits the ``## EverOS Memory`` header, sorts episodes
  by ``score`` descending, nests atomic facts under their parent episode,
  truncates at a word boundary appending ``" …"`` when it overflows, and
  returns ``""`` when there are no episodes and no profiles. A profile
  block is rendered as a one-liner when present.
- ``format_tool_result`` serialises via ``json.dumps``.
- ``format_memory_write_message`` builds a ``user``-role ``MessageItem``
  with the required fields.
"""

from __future__ import annotations

import json

from hermes._formatting import (
    format_memory_write_message,
    format_prefetch,
    format_tool_result,
)


def _episode(
    subject: str,
    score: float,
    *,
    episode: str = "",
    facts: list[str] | None = None,
) -> dict:
    return {
        "id": f"ep-{subject}",
        "user_id": "u",
        "app_id": "default",
        "project_id": "default",
        "session_id": "s",
        "timestamp": "2026-01-01T00:00:00Z",
        "sender_ids": ["u"],
        "summary": subject,
        "subject": subject,
        "episode": episode,
        "type": "dialogue",
        "score": score,
        "atomic_facts": [
            {"id": f"f{i}", "content": c, "score": 1.0}
            for i, c in enumerate(facts or [])
        ],
    }


def _profile(name: str, score: float) -> dict:
    return {
        "id": f"p-{name}",
        "user_id": "u",
        "app_id": "default",
        "project_id": "default",
        "profile_data": {"name": name},
        "score": score,
    }


def _search_data(episodes=None, profiles=None) -> dict:
    return {
        "episodes": episodes or [],
        "profiles": profiles or [],
        "agent_cases": [],
        "agent_skills": [],
        "unprocessed_messages": [],
    }


# ── format_prefetch ─────────────────────────────────────────────────────────


def test_format_prefetch_empty_returns_empty_string() -> None:
    assert format_prefetch("q", _search_data()) == ""


def test_format_prefetch_header_and_episode_block() -> None:
    data = _search_data(
        episodes=[
            _episode("alpha", 0.9, episode="alpha body", facts=["fact-a", "fact-b"]),
        ]
    )
    out = format_prefetch("q", data)
    assert out.startswith("## EverOS Memory")
    assert "**Episode**: alpha" in out
    assert "alpha body" in out
    assert "- fact-a" in out
    assert "- fact-b" in out


def test_format_prefetch_sorts_episodes_by_score_desc() -> None:
    data = _search_data(
        episodes=[
            _episode("low", 0.1),
            _episode("high", 0.99),
            _episode("mid", 0.5),
        ]
    )
    out = format_prefetch("q", data)
    high_pos = out.find("**Episode**: high")
    mid_pos = out.find("**Episode**: mid")
    low_pos = out.find("**Episode**: low")
    assert high_pos < mid_pos < low_pos


def test_format_prefetch_atomic_facts_nested_under_parent() -> None:
    data = _search_data(
        episodes=[
            _episode("ep1", 0.5, facts=["a", "b"]),
            _episode("ep2", 0.4, facts=["c"]),
        ]
    )
    out = format_prefetch("q", data)
    # Facts of ep1 appear between ep1 subject and ep2 subject.
    ep1_pos = out.find("**Episode**: ep1")
    ep2_pos = out.find("**Episode**: ep2")
    fact_a = out.find("- a")
    fact_b = out.find("- b")
    fact_c = out.find("- c")
    assert ep1_pos < fact_a < fact_b < ep2_pos < fact_c


def test_format_prefetch_profile_one_liner_when_present() -> None:
    data = _search_data(
        profiles=[_profile("Alice", 0.8)],
    )
    out = format_prefetch("q", data)
    assert out.startswith("## EverOS Memory")
    assert "**Profile**: Alice" in out


def test_format_prefetch_truncates_at_word_boundary_with_ellipsis() -> None:
    long_body = "word " * 5000  # well over any reasonable budget
    data = _search_data(
        episodes=[_episode("ep", 0.5, episode=long_body)],
    )
    out = format_prefetch("q", data, max_chars=120)
    assert out.endswith(" …")
    assert len(out) <= 120


def test_format_prefetch_truncation_keeps_header_when_budget_small() -> None:
    data = _search_data(episodes=[_episode("ep", 0.5, episode="x" * 5000)])
    out = format_prefetch("q", data, max_chars=40)
    # Even truncated, the header is the first section; budget > header len.
    assert "## EverOS Memory" in out
    assert out.endswith(" …")


# ── format_tool_result ──────────────────────────────────────────────────────


def test_format_tool_result_is_json_dumps() -> None:
    payload = {"count": 3, "items": ["a", "b", "c"], "nested": {"k": 1}}
    assert format_tool_result(payload) == json.dumps(payload, ensure_ascii=False)


def test_format_tool_result_preserves_unicode() -> None:
    assert format_tool_result({"k": "café"}) == '{"k": "café"}'


# ── format_memory_write_message ─────────────────────────────────────────────


def test_format_memory_write_message_fields() -> None:
    msg = format_memory_write_message("hello world", "u-1", 12345)
    assert msg["sender_id"] == "u-1"
    assert msg["role"] == "user"
    assert msg["timestamp"] == 12345
    assert msg["content"] == "hello world"
    # Required MessageItem keys are all present.
    assert {"sender_id", "role", "timestamp", "content"} <= set(msg)
