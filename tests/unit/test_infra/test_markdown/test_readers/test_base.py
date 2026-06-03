"""Tests for ``BaseDailyReader`` chassis.

Symmetric to ``test_writers/test_base.py`` — exercises path resolution
+ entry locating + structured-entry upgrading on a dummy schema.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import ClassVar, Literal

import pytest

from everos.core.persistence import (
    EntryId,
    MemoryRoot,
    StructuredEntry,
    UserScopedFrontmatter,
    render_structured_entry,
)
from everos.infra.persistence.markdown.readers import BaseDailyReader
from everos.infra.persistence.markdown.writers import BaseDailyWriter


class _DemoFrontmatter(UserScopedFrontmatter):
    ENTRY_ID_PREFIX: ClassVar[str] = "demo"
    DIR_NAME: ClassVar[str] = "demos"
    FILE_PREFIX: ClassVar[str] = "demo"
    type: Literal["user_demo"] = "user_demo"


class _DemoWriter(BaseDailyWriter):
    schema = _DemoFrontmatter


class _DemoReader(BaseDailyReader):
    schema = _DemoFrontmatter


@pytest.fixture
def root(tmp_path: Path) -> MemoryRoot:
    return MemoryRoot(tmp_path)


# ── construction ────────────────────────────────────────────────────────


def test_reader_rejects_missing_schema(root: MemoryRoot) -> None:
    class _NoSchemaReader(BaseDailyReader):
        pass

    with pytest.raises(TypeError, match="schema"):
        _NoSchemaReader(root)


def test_reader_rejects_schema_missing_classvars(root: MemoryRoot) -> None:
    class _IncompleteFrontmatter(UserScopedFrontmatter):
        # Missing DIR_NAME / FILE_PREFIX.
        type: Literal["incomplete"] = "incomplete"

    class _IncompleteReader(BaseDailyReader):
        schema = _IncompleteFrontmatter

    with pytest.raises(TypeError, match="missing ClassVar"):
        _IncompleteReader(root)


# ── read_for ────────────────────────────────────────────────────────────


async def test_read_for_returns_none_when_file_missing(root: MemoryRoot) -> None:
    reader = _DemoReader(root)
    assert await reader.read_for("u_jason", dt.date(2026, 4, 22)) is None


async def test_read_for_returns_parsed_when_file_exists(
    tmp_path: Path, root: MemoryRoot
) -> None:
    writer = _DemoWriter(root)
    await writer.append("u_jason", "first body", date=dt.date(2026, 4, 22))

    reader = _DemoReader(root)
    parsed = await reader.read_for("u_jason", dt.date(2026, 4, 22))
    assert parsed is not None
    assert len(parsed.entries) == 1
    assert parsed.entries[0].body == "first body"


async def test_read_for_today_default(root: MemoryRoot) -> None:
    """Omitting ``date`` falls back to today_with_timezone()."""
    writer = _DemoWriter(root)
    await writer.append("u_jason", "today body")

    reader = _DemoReader(root)
    parsed = await reader.read_for("u_jason")
    assert parsed is not None
    assert parsed.entries[0].body == "today body"


# ── find_entry ──────────────────────────────────────────────────────────


async def test_find_entry_resolves_file_from_entry_id(root: MemoryRoot) -> None:
    writer = _DemoWriter(root)
    await writer.append("u_jason", "alpha", date=dt.date(2026, 4, 22))
    await writer.append("u_jason", "beta", date=dt.date(2026, 4, 22))

    reader = _DemoReader(root)
    e = await reader.find_entry("u_jason", "demo_20260422_00000002")
    assert e is not None
    assert e.id == "demo_20260422_00000002"
    assert e.body == "beta"


async def test_find_entry_returns_none_when_file_missing(root: MemoryRoot) -> None:
    reader = _DemoReader(root)
    assert await reader.find_entry("u_jason", "demo_20260422_00000001") is None


async def test_find_entry_returns_none_when_entry_missing(root: MemoryRoot) -> None:
    writer = _DemoWriter(root)
    await writer.append("u_jason", "only", date=dt.date(2026, 4, 22))

    reader = _DemoReader(root)
    assert await reader.find_entry("u_jason", "demo_20260422_00000099") is None


async def test_find_entry_accepts_entryid_object(root: MemoryRoot) -> None:
    writer = _DemoWriter(root)
    await writer.append("u_jason", "alpha", date=dt.date(2026, 4, 22))

    reader = _DemoReader(root)
    eid = EntryId(prefix="demo", date=dt.date(2026, 4, 22), seq=1)
    e = await reader.find_entry("u_jason", eid)
    assert e is not None
    assert e.body == "alpha"


# ── find_structured ─────────────────────────────────────────────────────


async def test_find_structured_parses_audit_form(root: MemoryRoot) -> None:
    writer = _DemoWriter(root)
    body = render_structured_entry(
        header="demo_20260422_00000001",
        inline={"type": "demo", "user_id": "u_jason"},
        sections={"Body": "the body"},
    )
    await writer.append("u_jason", body, date=dt.date(2026, 4, 22))

    reader = _DemoReader(root)
    structured = await reader.find_structured("u_jason", "demo_20260422_00000001")
    assert structured is not None
    assert isinstance(structured, StructuredEntry)
    assert structured.id == "demo_20260422_00000001"
    assert structured.header == "demo_20260422_00000001"
    assert structured.inline == {"type": "demo", "user_id": "u_jason"}
    assert structured.sections == {"Body": "the body"}


async def test_find_structured_returns_none_when_missing(root: MemoryRoot) -> None:
    reader = _DemoReader(root)
    assert await reader.find_structured("u_jason", "demo_20260422_00000001") is None


# ── path_for ────────────────────────────────────────────────────────────


def test_path_for_matches_writer(tmp_path: Path, root: MemoryRoot) -> None:
    """Reader and writer resolve to the same path for the same schema."""
    reader = _DemoReader(root)
    writer = _DemoWriter(root)
    d = dt.date(2026, 4, 22)
    assert reader.path_for("u_jason", d) == writer.path_for("u_jason", d)


def test_path_for_does_not_create_files(tmp_path: Path, root: MemoryRoot) -> None:
    reader = _DemoReader(root)
    p = reader.path_for("u_jason", dt.date(2026, 4, 22))
    assert not p.exists()
    assert not (tmp_path / "users").exists()
