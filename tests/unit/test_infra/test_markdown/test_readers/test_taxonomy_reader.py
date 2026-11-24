"""Tests for .taxonomy.md parsing and auto-generation."""

from __future__ import annotations

from pathlib import Path

import anyio
import pytest

from everos.infra.persistence.markdown import ensure_taxonomy, parse_taxonomy


@pytest.fixture()
def taxonomy_md(tmp_path: Path) -> Path:
    p = tmp_path / ".taxonomy.md"
    p.write_text(
        "---\n"
        "kind: knowledge_taxonomy\n"
        "categories:\n"
        "  - id: Sports\n"
        '    description: "Content about sports events."\n'
        "  - id: Others\n"
        '    description: "Catch-all."\n'
        "---\n"
    )
    return p


class TestParseTaxonomy:
    async def test_parses_categories(self, taxonomy_md: Path) -> None:
        entries = await parse_taxonomy(taxonomy_md)
        assert len(entries) == 2
        assert entries[0].id == "Sports"
        assert entries[0].description == "Content about sports events."

    async def test_returns_empty_when_file_vanishes_after_exists(
        self, taxonomy_md: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original_exists = anyio.Path.exists

        async def exists_then_delete(self: anyio.Path) -> bool:
            result = await original_exists(self)
            await self.unlink()
            return result

        monkeypatch.setattr(anyio.Path, "exists", exists_then_delete)
        assert await parse_taxonomy(taxonomy_md) == []

    async def test_empty_categories_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / ".taxonomy.md"
        p.write_text("---\nkind: knowledge_taxonomy\ncategories: []\n---\n")
        assert await parse_taxonomy(p) == []


class TestEnsureTaxonomy:
    async def test_creates_default_when_missing(self, tmp_path: Path) -> None:
        await ensure_taxonomy(tmp_path)
        p = tmp_path / ".taxonomy.md"
        assert p.exists()
        entries = await parse_taxonomy(p)
        assert len(entries) >= 20
        ids = [e.id for e in entries]
        assert "Others" in ids
        assert "Technology" in ids
        assert "Medical" in ids

    async def test_does_not_overwrite_existing(self, taxonomy_md: Path) -> None:
        await ensure_taxonomy(taxonomy_md.parent)
        entries = await parse_taxonomy(taxonomy_md)
        assert len(entries) == 2  # still the original 2, not overwritten
