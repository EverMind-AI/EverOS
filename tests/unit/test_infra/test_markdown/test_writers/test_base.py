"""Tests for ``BaseDailyWriter`` skeleton.

Uses a dummy ``UserScopedFrontmatter`` subclass to exercise the path
resolution + entry-id construction + today-by-default logic without
pulling in any concrete business schema.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import ClassVar, Literal

import pytest

from everos.component.utils.datetime import today_with_timezone
from everos.core.persistence import (
    AgentScopedFrontmatter,
    MarkdownReader,
    MemoryRoot,
    UserScopedFrontmatter,
)
from everos.infra.persistence.markdown.writers import BaseDailyWriter


class _UserDemoFrontmatter(UserScopedFrontmatter):
    ENTRY_ID_PREFIX: ClassVar[str] = "demo"
    DIR_NAME: ClassVar[str] = "demos"
    FILE_PREFIX: ClassVar[str] = "demo"
    type: Literal["user_demo"] = "user_demo"


class _AgentDemoFrontmatter(AgentScopedFrontmatter):
    ENTRY_ID_PREFIX: ClassVar[str] = "ademo"
    DIR_NAME: ClassVar[str] = "demos"
    FILE_PREFIX: ClassVar[str] = "demo"
    type: Literal["agent_demo"] = "agent_demo"


class _UserDemoWriter(BaseDailyWriter):
    schema = _UserDemoFrontmatter


class _AgentDemoWriter(BaseDailyWriter):
    schema = _AgentDemoFrontmatter


@pytest.fixture
def root(tmp_path: Path) -> MemoryRoot:
    return MemoryRoot(tmp_path)


def test_constructor_rejects_missing_schema(root: MemoryRoot) -> None:
    class _NoSchema(BaseDailyWriter):
        pass

    with pytest.raises(TypeError, match="schema"):
        _NoSchema(root)


def test_constructor_rejects_schema_missing_classvars(root: MemoryRoot) -> None:
    class _IncompleteFrontmatter(UserScopedFrontmatter):
        # Missing ENTRY_ID_PREFIX / DIR_NAME / FILE_PREFIX.
        type: Literal["incomplete"] = "incomplete"

    class _IncompleteWriter(BaseDailyWriter):
        schema = _IncompleteFrontmatter

    with pytest.raises(TypeError, match="ENTRY_ID_PREFIX"):
        _IncompleteWriter(root)


async def test_append_writes_to_user_track(root: MemoryRoot) -> None:
    writer = _UserDemoWriter(root)
    eid = await writer.append("u_jason", "first", date=dt.date(2026, 4, 22))
    assert eid.prefix == "demo"
    assert eid.date == dt.date(2026, 4, 22)
    assert eid.seq == 1
    expected = root.users_dir() / "u_jason" / "demos" / "demo-2026-04-22.md"
    assert expected.exists()
    parsed = await MarkdownReader.read(expected)
    assert parsed.entries[0].id == "demo_20260422_00000001"
    assert parsed.entries[0].body == "first"


async def test_append_writes_to_agent_track(root: MemoryRoot) -> None:
    writer = _AgentDemoWriter(root)
    eid = await writer.append("agent_zhangsan", "trace", date=dt.date(2026, 4, 22))
    assert eid.prefix == "ademo"
    expected = root.agents_dir() / "agent_zhangsan" / "demos" / "demo-2026-04-22.md"
    assert expected.exists()


async def test_append_increments_seq_across_calls(root: MemoryRoot) -> None:
    writer = _UserDemoWriter(root)
    eids = [
        await writer.append("u_jason", f"body {i}", date=dt.date(2026, 4, 22))
        for i in range(3)
    ]
    assert [e.seq for e in eids] == [1, 2, 3]


async def test_append_date_defaults_to_today(root: MemoryRoot) -> None:
    """Omitting ``date`` falls back to today_with_timezone()."""
    writer = _UserDemoWriter(root)
    eid = await writer.append("u_jason", "body")
    today = today_with_timezone()
    assert eid.date == today
    expected = root.users_dir() / "u_jason" / "demos" / f"demo-{today.isoformat()}.md"
    assert expected.exists()


async def test_append_passes_frontmatter_updates(root: MemoryRoot) -> None:
    writer = _UserDemoWriter(root)
    await writer.append(
        "u_jason",
        "body",
        date=dt.date(2026, 4, 22),
        frontmatter_updates={"file_type": "user_demo_daily", "entry_count": 1},
    )
    path = root.users_dir() / "u_jason" / "demos" / "demo-2026-04-22.md"
    parsed = await MarkdownReader.read(path)
    assert parsed.frontmatter["file_type"] == "user_demo_daily"
    assert parsed.frontmatter["entry_count"] == 1


async def test_current_count_hook_can_be_overridden(root: MemoryRoot) -> None:
    """Subclass override of ``_current_count`` controls seq."""

    class _ConstantCount(BaseDailyWriter):
        schema = _UserDemoFrontmatter

        async def _current_count(self, path):  # noqa: ANN001
            return 41  # always claim 41 existing entries

    writer = _ConstantCount(root)
    eid = await writer.append("u_jason", "body", date=dt.date(2026, 4, 22))
    assert eid.seq == 42  # 41 + 1


async def test_frontmatter_updates_hook_supplies_defaults(root: MemoryRoot) -> None:
    """Subclass override of ``_frontmatter_updates`` populates frontmatter."""

    class _WithDefaults(BaseDailyWriter):
        schema = _UserDemoFrontmatter

        def _frontmatter_updates(self, scope_id, date, *, next_count):  # noqa: ANN001
            return {
                "user_id": scope_id,
                "entry_count": next_count,
                "marker": "from-hook",
            }

    writer = _WithDefaults(root)
    await writer.append("u_jason", "body", date=dt.date(2026, 4, 22))

    path = root.users_dir() / "u_jason" / "demos" / "demo-2026-04-22.md"
    parsed = await MarkdownReader.read(path)
    assert parsed.frontmatter["marker"] == "from-hook"
    assert parsed.frontmatter["entry_count"] == 1
    assert parsed.frontmatter["user_id"] == "u_jason"


async def test_explicit_frontmatter_updates_skip_hook(root: MemoryRoot) -> None:
    """Caller-supplied ``frontmatter_updates`` overrides the hook entirely."""

    class _WithDefaults(BaseDailyWriter):
        schema = _UserDemoFrontmatter

        def _frontmatter_updates(self, scope_id, date, *, next_count):  # noqa: ANN001
            return {"marker": "from-hook"}

    writer = _WithDefaults(root)
    await writer.append(
        "u_jason",
        "body",
        date=dt.date(2026, 4, 22),
        frontmatter_updates={"marker": "explicit"},
    )
    path = root.users_dir() / "u_jason" / "demos" / "demo-2026-04-22.md"
    parsed = await MarkdownReader.read(path)
    assert parsed.frontmatter["marker"] == "explicit"
