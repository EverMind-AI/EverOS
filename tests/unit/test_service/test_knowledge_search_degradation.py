"""Verify search requires embedding and reranker — no silent degradation.

An unconfigured provider is a client-actionable configuration fault, so
it surfaces as ``ProviderNotConfiguredError`` (HTTP 422) rather than a
retryable ``*ServiceError`` or an opaque HTTP 500.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from everos.core.errors import ProviderNotConfiguredError

_MOD = "everos.service.knowledge"


async def test_search_without_embedding_raises() -> None:
    """Search without an embedding provider raises ProviderNotConfiguredError."""
    from everos.service.knowledge import search_knowledge

    with (
        patch(f"{_MOD}._get_embedding", return_value=None),
        pytest.raises(ProviderNotConfiguredError, match=r"[Ee]mbedding"),
    ):
        await search_knowledge(query="test", method="vector")


async def test_search_without_reranker_raises() -> None:
    """Search without a reranker raises ProviderNotConfiguredError."""
    from everos.service.knowledge import search_knowledge

    mock_embedder = AsyncMock()
    mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)
    with (
        patch(f"{_MOD}._get_embedding", return_value=mock_embedder),
        patch(f"{_MOD}._get_reranker", return_value=None),
        pytest.raises(ProviderNotConfiguredError, match=r"[Rr]erank"),
    ):
        await search_knowledge(query="test", method="keyword")
