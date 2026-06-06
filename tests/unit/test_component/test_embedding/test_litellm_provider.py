"""LiteLLMEmbeddingProvider -- embed dispatch, batching, truncation, errors."""

from __future__ import annotations

import types
from typing import Any
from unittest import mock

import pytest

from everos.component.embedding.protocol import EmbeddingError


def _make_fake_litellm() -> types.ModuleType:
    mod = types.ModuleType("litellm")
    mod.aembedding = mock.AsyncMock(name="litellm.aembedding")  # type: ignore[attr-defined]
    return mod


def _patch_litellm(
    monkeypatch: pytest.MonkeyPatch,
    fake: types.ModuleType,
) -> None:
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake)


def _embedding_response(vectors: list[list[float]]) -> Any:
    """Return an object shaped like ``litellm.EmbeddingResponse``."""
    items = []
    for vec in vectors:
        item = mock.MagicMock()
        item.embedding = vec
        items.append(item)
    resp = mock.MagicMock()
    resp.data = items
    return resp


async def test_embed_single(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _make_fake_litellm()
    _patch_litellm(monkeypatch, fake)

    from everos.component.embedding.litellm_provider import LiteLLMEmbeddingProvider

    fake.aembedding.return_value = _embedding_response([[0.1, 0.2, 0.3]])  # type: ignore[attr-defined]

    provider = LiteLLMEmbeddingProvider(model="openai/text-embedding-3-small", dim=3)
    result = await provider.embed("hello")

    assert result == [0.1, 0.2, 0.3]
    fake.aembedding.assert_awaited_once()  # type: ignore[attr-defined]
    call_kwargs = fake.aembedding.call_args.kwargs  # type: ignore[attr-defined]
    assert call_kwargs["drop_params"] is True


async def test_embed_batch_preserves_order(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _make_fake_litellm()
    _patch_litellm(monkeypatch, fake)

    from everos.component.embedding.litellm_provider import LiteLLMEmbeddingProvider

    fake.aembedding.return_value = _embedding_response(  # type: ignore[attr-defined]
        [[1.0, 2.0], [3.0, 4.0]]
    )

    provider = LiteLLMEmbeddingProvider(
        model="m", dim=2, batch_size=10, max_concurrent=2
    )
    result = await provider.embed_batch(["a", "b"])

    assert len(result) == 2
    assert result[0] == [1.0, 2.0]
    assert result[1] == [3.0, 4.0]


async def test_embed_batch_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _make_fake_litellm()
    _patch_litellm(monkeypatch, fake)

    from everos.component.embedding.litellm_provider import LiteLLMEmbeddingProvider

    provider = LiteLLMEmbeddingProvider(model="m", dim=2)
    result = await provider.embed_batch([])

    assert result == []
    fake.aembedding.assert_not_awaited()  # type: ignore[attr-defined]


async def test_dimension_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _make_fake_litellm()
    _patch_litellm(monkeypatch, fake)

    from everos.component.embedding.litellm_provider import LiteLLMEmbeddingProvider

    fake.aembedding.return_value = _embedding_response(  # type: ignore[attr-defined]
        [[0.1, 0.2, 0.3, 0.4, 0.5]]
    )

    provider = LiteLLMEmbeddingProvider(model="m", dim=3)
    result = await provider.embed("test")

    assert len(result) == 3
    assert result == [0.1, 0.2, 0.3]


async def test_api_key_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _make_fake_litellm()
    _patch_litellm(monkeypatch, fake)

    from everos.component.embedding.litellm_provider import LiteLLMEmbeddingProvider

    fake.aembedding.return_value = _embedding_response([[0.1]])  # type: ignore[attr-defined]

    provider = LiteLLMEmbeddingProvider(model="m", api_key="sk-test", dim=1)
    await provider.embed("x")

    call_kwargs = fake.aembedding.call_args.kwargs  # type: ignore[attr-defined]
    assert call_kwargs["api_key"] == "sk-test"


async def test_api_key_omitted_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _make_fake_litellm()
    _patch_litellm(monkeypatch, fake)

    from everos.component.embedding.litellm_provider import LiteLLMEmbeddingProvider

    fake.aembedding.return_value = _embedding_response([[0.1]])  # type: ignore[attr-defined]

    provider = LiteLLMEmbeddingProvider(model="m", dim=1)
    await provider.embed("x")

    call_kwargs = fake.aembedding.call_args.kwargs  # type: ignore[attr-defined]
    assert "api_key" not in call_kwargs


async def test_litellm_error_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _make_fake_litellm()
    _patch_litellm(monkeypatch, fake)

    class FakeAPIError(Exception):
        pass

    FakeAPIError.__module__ = "litellm.exceptions"
    fake.aembedding.side_effect = FakeAPIError("rate limit")  # type: ignore[attr-defined]

    from everos.component.embedding.litellm_provider import LiteLLMEmbeddingProvider

    provider = LiteLLMEmbeddingProvider(model="m", dim=1)
    with pytest.raises(EmbeddingError, match="rate limit"):
        await provider.embed("x")


async def test_import_error_raises_embedding_error() -> None:
    from everos.component.embedding.litellm_provider import LiteLLMEmbeddingProvider

    provider = LiteLLMEmbeddingProvider(model="m", dim=1)
    with pytest.raises(EmbeddingError, match="litellm is not installed"):
        await provider.embed("x")


async def test_batching_chunks_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _make_fake_litellm()
    _patch_litellm(monkeypatch, fake)

    from everos.component.embedding.litellm_provider import LiteLLMEmbeddingProvider

    call_count = 0

    async def _fake_embed(**kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        n = len(kwargs["input"])
        return _embedding_response([[float(i)] for i in range(n)])

    fake.aembedding = _fake_embed  # type: ignore[attr-defined]

    provider = LiteLLMEmbeddingProvider(model="m", dim=1, batch_size=2)
    result = await provider.embed_batch(["a", "b", "c", "d", "e"])

    assert call_count == 3
    assert len(result) == 5
