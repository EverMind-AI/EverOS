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

    with pytest.raises(EmbeddingServiceError, match="0 vectors for 1 inputs"):
        await provider.embed("hello")


async def test_short_response_data_raises_embedding_error() -> None:
    """One vector back for two inputs must raise: a silent short list would
    shift every later vector onto the wrong text (cascade slices the batch
    back per entry)."""
    provider = _make_provider()
    short_response = mock.MagicMock()
    item = mock.MagicMock()
    item.embedding = [0.1, 0.2, 0.3, 0.4]
    short_response.data = [item]
    provider._client = mock.AsyncMock()
    provider._client.embeddings.create = mock.AsyncMock(return_value=short_response)

    with pytest.raises(EmbeddingServiceError, match="1 vectors for 2 inputs"):
        await provider.embed_batch(["hello", "world"])
