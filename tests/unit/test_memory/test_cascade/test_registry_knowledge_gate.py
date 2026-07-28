"""Verify knowledge cascade handlers register unconditionally.

PR #361 review B3: the previous atomic-pair gate on embed + rerank
capability broke the delete path — when either capability went
missing, both knowledge handlers were unregistered, and the worker
marked every queued row ``failed(retryable=False)`` including the
delete events emitted by ``service.knowledge.delete_document``. That
stranded SQLite / LanceDB rows behind after ``shutil.rmtree`` had
already cleared the md, producing phantom documents on Tier 3 → Tier
2 downgrade.

The fix is at the handler level, not the registry:

- ``KnowledgeDocumentHandler`` is SQLite-only; no capability needed.
- ``KnowledgeTopicHandler.handle_added_or_modified`` calls
  ``embed_or_none``, which writes ``vector=None`` when the embedding
  capability is unavailable (the column has been nullable since the
  embedding-soft-dependency migration).
- ``handle_deleted`` on both handlers is a pure repo delete.

Search-side capability gating lives at the route level (Group C), so
cascade need not gate writes at all. This file pins the new contract:
``build_handlers`` always includes ``knowledge_topic`` and
``knowledge_document``, regardless of capability state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.component.embedding import EmbeddingCapability
from everos.component.rerank import RerankCapability
from everos.component.tokenizer import Tokenizer
from everos.core.persistence import MemoryRoot
from everos.memory.cascade.handlers import HandlerDeps
from everos.memory.cascade.registry import build_handlers


class _StubTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return [tok for tok in text.split() if tok]

    def tokenize_batch(self, texts):  # type: ignore[no-untyped-def]
        return [self.tokenize(t) for t in texts]


class _StubEmbeddingProvider:
    async def embed(self, text: str) -> list[float]:
        return [0.0]


class _StubRerankProvider:
    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [0.0 for _ in documents]


@pytest.fixture
def deps(tmp_path: Path) -> HandlerDeps:
    memory_root = MemoryRoot(tmp_path)
    memory_root.ensure()
    return HandlerDeps(memory_root=memory_root, tokenizer=_StubTokenizer())


def _set_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    *,
    embed_available: bool,
    rerank_available: bool,
) -> None:
    """Stub both process-wide capability singletons for one test."""
    import everos.component.embedding.accessor as embed_accessor
    import everos.component.rerank.accessor as rerank_accessor

    monkeypatch.setattr(
        embed_accessor,
        "_capability",
        EmbeddingCapability(
            provider=_StubEmbeddingProvider() if embed_available else None
        ),
    )
    monkeypatch.setattr(
        rerank_accessor,
        "_capability",
        RerankCapability(provider=_StubRerankProvider() if rerank_available else None),
    )


def test_knowledge_handlers_registered_when_all_available(
    monkeypatch: pytest.MonkeyPatch, deps: HandlerDeps
) -> None:
    """Both capabilities present → both knowledge handlers register."""
    _set_capabilities(monkeypatch, embed_available=True, rerank_available=True)
    handlers = build_handlers(deps)
    assert "knowledge_topic" in handlers
    assert "knowledge_document" in handlers


def test_knowledge_handlers_registered_when_embed_missing(
    monkeypatch: pytest.MonkeyPatch, deps: HandlerDeps
) -> None:
    """Embed missing → knowledge handlers still register.

    ``embed_or_none`` is the body-guard for the vector step; the
    column is nullable. Delete/modify events must keep flowing.
    """
    _set_capabilities(monkeypatch, embed_available=False, rerank_available=True)
    handlers = build_handlers(deps)
    assert "knowledge_topic" in handlers
    assert "knowledge_document" in handlers


def test_knowledge_handlers_registered_when_rerank_missing(
    monkeypatch: pytest.MonkeyPatch, deps: HandlerDeps
) -> None:
    """Rerank missing → knowledge handlers still register.

    Neither handler touches rerank; rerank gating belongs at the
    search endpoint, not at cascade write time.
    """
    _set_capabilities(monkeypatch, embed_available=True, rerank_available=False)
    handlers = build_handlers(deps)
    assert "knowledge_topic" in handlers
    assert "knowledge_document" in handlers


def test_knowledge_handlers_registered_when_both_missing(
    monkeypatch: pytest.MonkeyPatch, deps: HandlerDeps
) -> None:
    """Tier-1 scenario (no embed, no rerank): both handlers register.

    Regression guard for PR #361 review B3 — the previous atomic-pair
    gate stranded SQLite / LanceDB rows after ``delete_document``.
    """
    _set_capabilities(monkeypatch, embed_available=False, rerank_available=False)
    handlers = build_handlers(deps)
    assert "knowledge_topic" in handlers
    assert "knowledge_document" in handlers


def test_non_knowledge_handlers_always_registered(
    monkeypatch: pytest.MonkeyPatch, deps: HandlerDeps
) -> None:
    """Every non-knowledge kind is unaffected by capability state."""
    _set_capabilities(monkeypatch, embed_available=False, rerank_available=False)
    handlers = build_handlers(deps)
    for kind in (
        "episode",
        "atomic_fact",
        "foresight",
        "agent_case",
        "agent_skill",
        "user_profile",
    ):
        assert kind in handlers
