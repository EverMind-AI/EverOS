"""Unit tests for entry marker parsing."""

from __future__ import annotations

from everos.infra.persistence.markdown import (
    UnreadableEntry,
    find_entry,
    scan_entries,
    split_entries,
)


def test_split_no_entries() -> None:
    assert split_entries("# heading\n\nbody.") == []


def test_split_single_entry() -> None:
    body = (
        "preamble\n"
        "<!-- entry:abc123 -->\n"
        "content here\n"
        "<!-- /entry:abc123 -->\n"
        "trailing\n"
    )
    entries = split_entries(body)
    assert len(entries) == 1
    e = entries[0]
    assert e.id == "abc123"
    assert e.body == "content here"
    # offsets should bracket the markers
    assert body[e.start : e.end].startswith("<!-- entry:abc123 -->")
    assert body[e.start : e.end].endswith("<!-- /entry:abc123 -->")


def test_split_multiple_entries() -> None:
    body = (
        "<!-- entry:e1 -->\nfirst\n<!-- /entry:e1 -->\n"
        "<!-- entry:e2 -->\nsecond\n<!-- /entry:e2 -->\n"
    )
    entries = split_entries(body)
    assert [e.id for e in entries] == ["e1", "e2"]
    assert entries[0].body == "first"
    assert entries[1].body == "second"


def test_split_unmatched_open() -> None:
    """Open without close → scan stops; preceding entries are still returned."""
    body = "<!-- entry:e1 -->\nok\n<!-- /entry:e1 -->\n<!-- entry:e2 -->\nno close\n"
    entries = split_entries(body)
    assert [e.id for e in entries] == ["e1"]


def test_split_mismatched_id() -> None:
    """Open id != close id → no match → the unterminated open is skipped."""
    body = "<!-- entry:e1 -->\ncontent\n<!-- /entry:other -->\n"
    entries = split_entries(body)
    assert entries == []


def test_split_reads_on_past_an_unterminated_entry() -> None:
    """One open marker without its close costs that entry, not the rest."""
    body = (
        "<!-- entry:e1 -->\none\n<!-- /entry:e1 -->\n"
        "<!-- entry:e2 -->\ntwo, its close marker deleted\n"
        "<!-- entry:e3 -->\nthree\n<!-- /entry:e3 -->\n"
    )
    assert [(e.id, e.body) for e in split_entries(body)] == [
        ("e1", "one"),
        ("e3", "three"),
    ]


def test_scan_names_each_unterminated_entry() -> None:
    body = (
        "<!-- entry:e1 -->\none\n"
        "<!-- entry:e2 -->\n**member_id**: u_dan\n"
        "<!-- entry:e3 -->\nthree\n<!-- /entry:e3 -->\n"
    )
    entries, unreadable = scan_entries(body)

    assert [e.id for e in entries] == ["e3"]
    assert unreadable == [
        UnreadableEntry(id="e1", start=0, reason="no closing marker", text="\none\n"),
        UnreadableEntry(
            id="e2",
            start=body.index("<!-- entry:e2"),
            reason="no closing marker",
            text="\n**member_id**: u_dan\n",
        ),
    ]


def test_scan_of_a_well_formed_body_is_split_entries() -> None:
    body = (
        "<!-- entry:a -->\nx\n<!-- /entry:a -->\n<!-- entry:b -->\ny\n<!-- /entry:b -->"
    )
    entries, unreadable = scan_entries(body)
    assert (entries, unreadable) == (split_entries(body), [])


def test_split_id_with_underscore_and_hyphen() -> None:
    body = "<!-- entry:abc_def-123 -->\nx\n<!-- /entry:abc_def-123 -->\n"
    entries = split_entries(body)
    assert len(entries) == 1
    assert entries[0].id == "abc_def-123"


def test_split_offsets_consistent() -> None:
    body = "before\n<!-- entry:e1 -->\nx\n<!-- /entry:e1 -->\nafter\n"
    e = split_entries(body)[0]
    assert body[e.start : e.end] == "<!-- entry:e1 -->\nx\n<!-- /entry:e1 -->"


def test_find_entry_found() -> None:
    body = (
        "<!-- entry:a -->\nfirst\n<!-- /entry:a -->\n"
        "<!-- entry:b -->\nsecond\n<!-- /entry:b -->\n"
    )
    e = find_entry(body, "b")
    assert e is not None
    assert e.id == "b"
    assert e.body == "second"


def test_find_entry_not_found() -> None:
    body = "<!-- entry:a -->\nx\n<!-- /entry:a -->\n"
    assert find_entry(body, "missing") is None


def test_find_entry_open_without_close() -> None:
    body = "<!-- entry:a -->\nx\n"  # no close
    assert find_entry(body, "a") is None


def test_split_entry_body_no_internal_newline_stripping() -> None:
    """Internal blank lines preserved; only the *single* leading/trailing
    newline introduced by formatter is stripped."""
    body = "<!-- entry:e1 -->\nline1\n\nline3\n<!-- /entry:e1 -->\n"
    e = split_entries(body)[0]
    assert e.body == "line1\n\nline3"
