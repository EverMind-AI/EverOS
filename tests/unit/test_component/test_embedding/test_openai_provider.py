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


async def test_query_embed_does_not_queue_behind_write_embeds() -> None:
    """An interactive query must not wait on the cascade's budget.

    This is issue #473: with one shared semaphore, a burst of background
    embeds puts the query a user is waiting on at the back of the queue.
    Measured on a real server under three writers, vector search p50 was
    48 s while the keyword leg — which needs no embedding — answered in
    0.49 s.

    The write budget is saturated and held open here; the query embed has
    to come back anyway. On a shared semaphore it cannot, and the wait_for
    below times out.
    """
    import asyncio

    provider = _make_provider(max_concurrent=1, query_max_concurrent=1)
    release = asyncio.Event()
    started = asyncio.Event()

    async def blocking_create(**kwargs):  # type: ignore[no-untyped-def]
        # Only the write request blocks. If the query had to wait on it, that
        # would be the shared-semaphore bug rather than this stub.
        if "background" in kwargs["input"][0]:
            started.set()
            await release.wait()
        return _response_with(len(kwargs["input"]))

    provider._client = mock.AsyncMock()
    provider._client.embeddings.create = blocking_create

    writer = asyncio.create_task(provider.embed("a background write"))
    await asyncio.wait_for(started.wait(), timeout=2)  # write holds its slot

    try:
        vec = await asyncio.wait_for(provider.embed_query("a user's query"), timeout=2)
        assert len(vec) == 4
    finally:
        release.set()
        await writer


def _response_with(n: int):  # type: ignore[no-untyped-def]
    resp = mock.MagicMock()
    items = []
    for _ in range(n):
        item = mock.MagicMock()
        item.embedding = [0.1, 0.2, 0.3, 0.4]
        items.append(item)
    resp.data = items
    resp.usage = None
    return resp
