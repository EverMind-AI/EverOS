"""Tests for the audit-form structured entry chassis."""

from __future__ import annotations

import pytest

from everos.core.persistence.markdown import (
    StructuredEntry,
    parse_structured_entry,
    render_structured_entry,
)

# ── render ───────────────────────────────────────────────────────────────


def test_render_with_header_inline_and_sections() -> None:
    out = render_structured_entry(
        header="ep_20260422_001",
        inline={
            "type": "episode",
            "user_id": "u_jason",
            "group_id": "sp_1",
        },
        sections={"Summary": "first line\nsecond line"},
    )
    assert out.startswith("## ep_20260422_001\n\n")
    assert "**type**: episode" in out
    assert "**user_id**: u_jason" in out
    assert "**group_id**: sp_1" in out
    assert "### Summary\nfirst line\nsecond line" in out


def test_render_inline_only_no_header_no_sections() -> None:
    out = render_structured_entry(inline={"k": "v"})
    assert out == "**k**: v"


def test_render_lists_use_bracket_notation() -> None:
    out = render_structured_entry(
        inline={"participants": ["u_jason", "u_sarah"], "tags": ("a", "b")}
    )
    assert "**participants**: [u_jason, u_sarah]" in out
    assert "**tags**: [a, b]" in out


def test_render_none_value_renders_empty() -> None:
    out = render_structured_entry(inline={"optional": None})
    assert out == "**optional**: "


def test_render_scalar_uses_str() -> None:
    out = render_structured_entry(inline={"count": 3, "ratio": 0.5, "active": True})
    assert "**count**: 3" in out
    assert "**ratio**: 0.5" in out
    assert "**active**: True" in out


# ── parse ────────────────────────────────────────────────────────────────


def test_parse_full_round_trip() -> None:
    src = render_structured_entry(
        header="ep_001",
        inline={"type": "episode", "user_id": "u_jason"},
        sections={"Summary": "the summary", "Body": "the body"},
    )
    entry = parse_structured_entry(src)
    assert entry.header == "ep_001"
    assert entry.inline == {"type": "episode", "user_id": "u_jason"}
    assert entry.sections == {"Summary": "the summary", "Body": "the body"}


def test_parse_no_header_yields_none() -> None:
    src = "**k**: v\n\n### Section\nbody"
    entry = parse_structured_entry(src)
    assert entry.header is None
    assert entry.inline == {"k": "v"}
    assert entry.sections == {"Section": "body"}


def test_parse_no_inline() -> None:
    src = "## ep_001\n\n### Body\nonly section"
    entry = parse_structured_entry(src)
    assert entry.header == "ep_001"
    assert entry.inline == {}
    assert entry.sections == {"Body": "only section"}


def test_parse_no_sections() -> None:
    src = "## ep_001\n\n**k**: v"
    entry = parse_structured_entry(src)
    assert entry.header == "ep_001"
    assert entry.inline == {"k": "v"}
    assert entry.sections == {}


def test_parse_inline_value_with_colon_kept_verbatim() -> None:
    src = "**timestamp**: 2026-04-22T10:03:11Z"
    entry = parse_structured_entry(src)
    assert entry.inline == {"timestamp": "2026-04-22T10:03:11Z"}


def test_parse_list_value_kept_as_string() -> None:
    """Type-agnostic by design — bracket notation is preserved as text."""
    src = "**participants**: [u_jason, u_sarah]"
    entry = parse_structured_entry(src)
    assert entry.inline == {"participants": "[u_jason, u_sarah]"}


def test_parse_section_with_multiline_body() -> None:
    src = "### Episode\nline 1\nline 2\nline 3"
    entry = parse_structured_entry(src)
    assert entry.sections == {"Episode": "line 1\nline 2\nline 3"}


def test_parse_section_titles_kept_verbatim() -> None:
    """No Title-casing — titles stay exactly as written."""
    src = "### task_intent\ndoc text"
    entry = parse_structured_entry(src)
    assert "task_intent" in entry.sections


def test_parse_tolerates_stray_text_outside_blocks() -> None:
    """Stray paragraphs in the head become part of nothing — silently dropped."""
    src = (
        "## ep_001\n\nrandom prose paragraph\n"
        "**k**: v\nmore stray text\n\n### Section\nbody"
    )
    entry = parse_structured_entry(src)
    # H2 + inline match anchors; stray prose lines that don't match
    # **key**: ... are simply not captured.
    assert entry.header == "ep_001"
    assert entry.inline == {"k": "v"}
    assert entry.sections == {"Section": "body"}


def test_dataclass_immutable() -> None:
    """``StructuredEntry`` is frozen — accidental mutation raises."""
    entry = StructuredEntry(id="", body="", start=0, end=0, header="x")
    with pytest.raises((AttributeError, TypeError)):
        entry.header = "y"  # type: ignore[misc]


def test_structured_entry_inherits_entry() -> None:
    """``StructuredEntry`` is an :class:`Entry` subclass and carries
    the marker context plus the parsed audit-form fields together."""
    from everos.core.persistence.markdown import Entry

    entry = StructuredEntry(
        id="ep_001",
        body="b",
        start=0,
        end=10,
        header="ep_001",
        inline={"k": "v"},
        sections={"S": "x"},
    )
    assert isinstance(entry, Entry)
    assert entry.id == "ep_001"
    assert entry.header == "ep_001"


def test_entry_as_structured_preserves_marker_context() -> None:
    """``Entry.as_structured`` copies id/start/end and adds parsed fields."""
    from everos.core.persistence.markdown import Entry

    entry = Entry(
        id="ep_001",
        body="## ep_001\n\n**k**: v\n\n### Body\nthe body",
        start=42,
        end=128,
    )
    s = entry.as_structured()
    assert isinstance(s, StructuredEntry)
    assert s.id == "ep_001"
    assert s.start == 42
    assert s.end == 128
    assert s.header == "ep_001"
    assert s.inline == {"k": "v"}
    assert s.sections == {"Body": "the body"}


# ── round-trip with realistic Episode entry ─────────────────────────────


def test_round_trip_episode_shape() -> None:
    """Mirrors the shape from the wiki Memory Types doc."""
    inline = {
        "type": "episode",
        "user_id": "u_jason",
        "group_id": "sp_1",
        "session_id": "sess_abc123",
        "timestamp": "2026-04-22T10:03:11Z",
        "parent_type": "memcell",
        "parent_id": "mc_20260422_001",
        "participants": ["u_jason", "u_sarah"],
        "subject": "weekend planning",
    }
    sections = {
        "Summary": "Jason and Sarah discussed weekend coffee plans.",
        "Episode": "At ten in the morning, while making coffee, Jason told Sarah...",
    }
    rendered = render_structured_entry(
        header="ep_20260422_001",
        inline=inline,
        sections=sections,
    )
    entry = parse_structured_entry(rendered)
    assert entry.header == "ep_20260422_001"
    # Lists become string in audit form.
    assert entry.inline["participants"] == "[u_jason, u_sarah]"
    # Scalars round-trip exactly.
    assert entry.inline["session_id"] == "sess_abc123"
    assert entry.sections == sections
