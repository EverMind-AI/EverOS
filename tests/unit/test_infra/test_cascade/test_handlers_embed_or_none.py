"""Contract: every memory handler writes ``vector=None`` when embedding is
unavailable.

Handlers receive the embedder through :class:`HandlerDeps` and embed every
changed entry of a file in one batch. When it is ``None`` (embedding not
configured), each kind must still build
its row — BM25 tokenization and scalar
columns are populated as usual — but with ``vector`` (and, for Episode,
``subject_vector`` too) left as ``None`` rather than raising.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.component.tokenizer import Tokenizer
from everos.infra.cascade._projections import HandlerDeps, ImageMemoryHandler
from everos.infra.cascade._projections._daily_log_base import ParsedEntry
from everos.infra.cascade._projections.episode import EpisodeHandler
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown import StructuredEntry
from tests.fakes import handler_deps


class _StubTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return [tok for tok in text.split() if tok]

    def tokenize_batch(self, texts):  # type: ignore[no-untyped-def]
        return [self.tokenize(t) for t in texts]


@pytest.fixture
def memory_root(tmp_path: Path) -> MemoryRoot:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    return mr


def _deps(memory_root: MemoryRoot) -> HandlerDeps:
    """No embedder: the exact condition each handler must degrade against."""
    return handler_deps(
        memory_root=memory_root, tokenizer=_StubTokenizer(), embedding=None
    )


def _entry(
    entry_id: str,
    inline: dict[str, str],
    sections: dict[str, str],
    *,
    sha: str = "f" * 64,
) -> ParsedEntry:
    return ParsedEntry(
        entry_id=entry_id,
        structured=StructuredEntry(
            id=entry_id,
            body="",
            start=0,
            end=0,
            header=None,
            inline=inline,
            sections=sections,
        ),
        content_sha256=sha,
    )


async def _embed_one(handler, **kw):  # type: ignore[no-untyped-def]
    """Run one entry through the batch embed path (the production route)."""
    [row] = await handler._embed_entries(
        [kw["entry"]],
        kw["owner_id"],
        kw["owner_type"],
        kw.get("app_id", "default"),
        kw.get("project_id", "default"),
        kw["md_path"],
    )
    return row


async def test_image_memory_handler_writes_null_vector_when_embed_unavailable(
    memory_root: MemoryRoot,
) -> None:
    handler = ImageMemoryHandler(_deps(memory_root))
    row = await _embed_one(
        handler,
        owner_id="u1",
        owner_type="user",
        md_path="x.md",
        entry=_entry(
            "im_20260514_0001",
            inline={
                "owner_id": "u1",
                "session_id": "s1",
                "timestamp": "2026-05-14T10:00:00+00:00",
                "parent_id": "mc_1",
                "image_id": "c" * 64,
            },
            sections={"Description": "a red bicycle", "OCR": "SALE"},
        ),
    )
    assert row.vector is None
    assert row.description_tokens == "a red bicycle"
    assert row.ocr_tokens == "SALE"


async def test_episode_handler_writes_null_vectors_when_embed_unavailable(
    memory_root: MemoryRoot,
) -> None:
    """Episode has two vector columns — both must degrade to ``None``."""
    handler = EpisodeHandler(_deps(memory_root))
    row = await _embed_one(
        handler,
        owner_id="u1",
        owner_type="user",
        md_path="x.md",
        entry=_entry(
            "ep_20260514_0001",
            inline={
                "owner_id": "u1",
                "session_id": "s1",
                "timestamp": "2026-05-14T10:00:00+00:00",
                "parent_id": "mc_1",
            },
            sections={"Subject": "Test", "Summary": "Stub", "Content": "hello world"},
        ),
    )
    assert row.vector is None
    assert row.subject_vector is None
    assert row.episode_tokens == "hello world Test"
