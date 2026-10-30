"""Verify search requires embedding and reranker — no silent degradation.

An unconfigured provider is a client-actionable configuration fault, so
it surfaces as ``ProviderNotConfiguredError`` (HTTP 422) rather than a
retryable ``*ServiceError`` or an opaque HTTP 500.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from everos.core.errors import ProviderNotConfiguredError
from tests.fakes import make_knowledge_service


async def test_search_without_embedding_raises() -> None:
    """Search without an embedding provider raises ProviderNotConfiguredError."""
    service = make_knowledge_service(root=Path("/unused"))
    with pytest.raises(ProviderNotConfiguredError, match=r"[Ee]mbedding"):
        await service.search_knowledge(query="test", method="vector")


async def test_search_without_reranker_raises() -> None:
    """Search without a reranker raises ProviderNotConfiguredError."""
    mock_embedder = AsyncMock()
    mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)
    service = make_knowledge_service(root=Path("/unused"), embedding=mock_embedder)
    with pytest.raises(ProviderNotConfiguredError, match=r"[Rr]erank"):
        await service.search_knowledge(query="test", method="keyword")
