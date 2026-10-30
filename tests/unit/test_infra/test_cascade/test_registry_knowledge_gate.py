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
- ``KnowledgeTopicHandler.handle_added_or_modified`` writes
  ``vector=None`` when no embedder is injected (the column has been
  nullable since the embedding-soft-dependency migration).
- ``handle_deleted`` on both handlers is a pure repo delete.

Search-side capability gating lives at the route level (Group C), so
cascade need not gate writes at all. This file pins the new contract:
``build_handlers`` always includes ``knowledge_topic`` and
``knowledge_document``, whether or not an embedder is injected. Rerank
is not a cascade input at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.component.tokenizer import Tokenizer
from everos.infra.cascade._projections import HandlerDeps
from everos.infra.cascade.registry import build_handlers
from everos.infra.persistence import MemoryRoot
from tests.fakes import handler_deps


class _StubTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return [tok for tok in text.split() if tok]

    def tokenize_batch(self, texts):  # type: ignore[no-untyped-def]
        return [self.tokenize(t) for t in texts]


class _StubEmbeddingProvider:
    async def embed(self, text: str) -> list[float]:
        return [0.0]


def _deps(tmp_path: Path, *, embed_available: bool) -> HandlerDeps:
    memory_root = MemoryRoot(tmp_path)
    memory_root.ensure()
    return handler_deps(
        memory_root=memory_root,
        tokenizer=_StubTokenizer(),
        embedding=_StubEmbeddingProvider() if embed_available else None,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("embed_available", [True, False])
def test_knowledge_handlers_registered_whatever_the_embedder(
    tmp_path: Path, embed_available: bool
) -> None:
    """Embed missing → knowledge handlers still register.

    The vector step writes ``None`` into a nullable column, so delete /
    modify events keep flowing -- the regression guard for PR #361 review
    B3, where the previous atomic-pair gate stranded SQLite / LanceDB rows
    after ``delete_document``.
    """
    handlers = build_handlers(_deps(tmp_path, embed_available=embed_available))
    assert "knowledge_topic" in handlers
    assert "knowledge_document" in handlers


def test_non_knowledge_handlers_always_registered(tmp_path: Path) -> None:
    """Every non-knowledge kind registers on a keyword-only deployment."""
    handlers = build_handlers(_deps(tmp_path, embed_available=False))
    for kind in (
        "episode",
        "atomic_fact",
        "foresight",
        "agent_case",
        "agent_skill",
        "user_profile",
    ):
        assert kind in handlers
