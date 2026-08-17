"""Tests for :class:`OpenAIEmbeddingProvider` edge cases."""

from __future__ import annotations

import unittest.mock as mock

import pytest

from everos.component.embedding.openai_provider import OpenAIEmbeddingProvider
from everos.component.embedding.protocol import EmbeddingServiceError


def _make_provider(**overrides) -> OpenAIEmbeddingProvider:
    defaults = dict(
        model="test-model",
        api_key="sk-test",
        base_url="https://example.test",
        dim=4,
        timeout=1.0,
        max_retries=0,
        batch_size=10,
        max_concurrent=1,
    )
    defaults.update(overrides)
    return OpenAIEmbeddingProvider(**defaults)


async def test_empty_response_data_raises_embedding_error() -> None:
    """API returning 200 with empty data must raise, not return []."""
    provider = _make_provider()
    empty_response = mock.MagicMock()
    empty_response.data = []
    provider._client = mock.AsyncMock()
    provider._client.embeddings.create = mock.AsyncMock(return_value=empty_response)

    with pytest.raises(EmbeddingServiceError, match="empty data"):
        await provider.embed("hello")
