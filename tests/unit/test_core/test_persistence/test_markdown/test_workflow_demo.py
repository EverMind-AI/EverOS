"""Markdown IO toolkit — typical workflow demo.

Doubles as living documentation for how a caller assembles + reads a
day-level markdown file with multiple ``<!-- entry:id -->`` records.

End-to-end story:
    1. Build a body that contains entry markers.
    2. Use ``MarkdownWriter.write_markdown`` to persist frontmatter + body
       atomically (tmp file + fsync + rename, all inside the target dir).
    3. Use ``MarkdownReader.read`` to parse the resulting file back into
       a ``ParsedMarkdown`` (frontmatter dict + raw body + list[Entry]).
    4. Verify each entry's id / body matches what was written.
    5. Look up a single entry by id with ``find_entry``.
    6. Round-trip: dump_frontmatter + body → parse_frontmatter recovers
       the original mapping.
"""

from __future__ import annotations

from pathlib import Path

from everos.core.persistence import (
    MarkdownReader,
    MarkdownWriter,
    MemoryRoot,
    dump_frontmatter,
    find_entry,
    parse_frontmatter,
)


async def test_typical_day_log_write_then_read(tmp_path: Path) -> None:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    writer = MarkdownWriter(mr)

    # 1. Build a body with two entries (typical day-level append log).
    body = (
        "# Day log\n"
        "\n"
        "<!-- entry:ep_001 -->\n"
        "**Title**: Met Alice\n"
        "We discussed the new project layout.\n"
        "<!-- /entry:ep_001 -->\n"
        "\n"
        "<!-- entry:ep_002 -->\n"
        "**Title**: Read paper X\n"
        "Key idea: end-to-end async pipelines.\n"
        "<!-- /entry:ep_002 -->\n"
    )
    frontmatter = {
        "type": "episodic_day_log",
        "date": "2026-04-22",
        "user_id": "u_jason",
        "tags": ["meeting", "research"],
    }

    # 2. Atomic write via the writer.
    target = mr.users_dir() / "u_jason" / "episodic" / "2026-04-22.md"
    written_path = await writer.write_markdown(
        target, frontmatter=frontmatter, body=body
    )
    assert written_path == target
    assert target.is_file()
    # No leftover temp file.
    leftover = list(target.parent.glob(f".{target.name}.tmp.*"))
    assert leftover == []

    # 3. Read back into ParsedMarkdown.
    parsed = await MarkdownReader.read(target)

    # 4. Validate frontmatter + entries.
    assert parsed.frontmatter == frontmatter
    assert [e.id for e in parsed.entries] == ["ep_001", "ep_002"]
    assert "Met Alice" in parsed.entries[0].body
    assert "Read paper X" in parsed.entries[1].body

    # 5. Single-entry lookup.
    e2 = find_entry(parsed.body, "ep_002")
    assert e2 is not None
    assert "async pipelines" in e2.body

    # 6. Round-trip frontmatter parse / dump.
    composed = dump_frontmatter(frontmatter) + body
    re_meta, re_body = parse_frontmatter(composed)
    assert re_meta == frontmatter
    assert re_body == body
