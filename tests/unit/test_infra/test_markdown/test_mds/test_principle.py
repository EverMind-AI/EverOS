"""Tests for :class:`PrincipleFrontmatter` and ProfileWriter reuse."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from everos.core.persistence import MemoryRoot
from everos.infra.persistence.markdown import (
    PrincipleFrontmatter,
    PrincipleItem,
    ProfileReader,
    ProfileWriter,
    mint_principle_id,
    render_principles_body,
)


def test_mint_principle_id_shape() -> None:
    pid = mint_principle_id()
    assert pid.startswith("pr_")
    assert len(pid) == 15
    assert pid != mint_principle_id()


def test_schema_pins_profile_chassis() -> None:
    assert PrincipleFrontmatter.PROFILE_FILENAME == "principles.md"
    assert PrincipleFrontmatter.SCOPE_DIR == "users"
    assert PrincipleFrontmatter.path_glob() == "*/*/users/*/principles.md"


def test_type_defaults_to_principle() -> None:
    fm = PrincipleFrontmatter(id="principle_u_alice", user_id="u_alice")
    assert fm.type == "principle"
    assert fm.track == "user"
    assert fm.principles == []


def test_nested_item_defaults() -> None:
    item = PrincipleItem(
        id="pr_aaaaaaaaaaaa",
        title="Use Rust on device",
        statement="Device Runtime uses Rust.",
    )
    assert item.source_entry_ids == []
    assert item.timestamp_ms == 0


def test_item_requires_id_title_statement() -> None:
    with pytest.raises(ValidationError):
        PrincipleItem(id="pr_1", title="T")  # type: ignore[call-arg]


def test_render_principles_body() -> None:
    items = [
        PrincipleItem(
            id="pr_aaaaaaaaaaaa",
            title="Use Rust on device",
            statement="Device Runtime uses Rust.",
        )
    ]
    assert render_principles_body(items) == (
        "- **Use Rust on device.** Device Runtime uses Rust.\n"
    )
    assert render_principles_body([]) == ""


async def test_profile_writer_round_trip(tmp_path: Path) -> None:
    """Principles reuse ProfileWriter — no fourth storage strategy."""
    root = MemoryRoot(tmp_path)
    writer = ProfileWriter(root)
    reader = ProfileReader(root)
    items = [
        PrincipleItem(
            id="pr_aaaaaaaaaaaa",
            title="Use Rust on device",
            statement="Device Runtime uses Rust.",
            source_entry_ids=["dc_20260517_0001"],
            timestamp_ms=1_700_000_000_000,
        )
    ]
    fm = PrincipleFrontmatter(
        id="principle_u_alice",
        user_id="u_alice",
        principles=items,
    )
    path = await writer.write(
        "u_alice",
        frontmatter=fm,
        body=render_principles_body(items),
    )
    expected = root.users_dir() / "u_alice" / "principles.md"
    assert path == expected
    assert expected.is_file()

    out = await reader.read("u_alice", schema=PrincipleFrontmatter)
    assert out is not None
    fm_out, body = out
    assert fm_out.type == "principle"
    assert len(fm_out.principles) == 1
    assert fm_out.principles[0].id == "pr_aaaaaaaaaaaa"
    assert fm_out.principles[0].source_entry_ids == ["dc_20260517_0001"]
    assert "Use Rust on device" in body
