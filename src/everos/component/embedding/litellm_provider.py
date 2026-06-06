"""LiteLLM AI gateway embedding provider.

Wraps ``litellm.aembedding`` so any provider litellm supports (OpenAI,
Cohere, Bedrock, HuggingFace, …) can serve embeddings without
per-provider forks.  Mirrors :class:`OpenAIEmbeddingProvider`'s
batching + concurrency model: inputs are chunked by ``batch_size``
and an :class:`asyncio.Semaphore` bounds in-flight requests.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from .protocol import EmbeddingError


class LiteLLMEmbeddingProvider:
    """LiteLLM-backed embedding provider with batching + concurrency.

    Args:
        model: LiteLLM embedding model id
            (e.g. ``"openai/text-embedding-3-small"``).
        api_key: Bearer credential forwarded to litellm. When *not* set
            litellm falls back to provider-specific env vars.
        base_url: Optional proxy / gateway URL.
        dim: Target vector dimension. Vectors longer than this are
            truncated client-side.
        timeout: Per-request timeout, seconds.
        batch_size: How many inputs per ``/embeddings`` call.
        max_concurrent: Cap on in-flight chunked requests.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        dim: int = 1024,
        timeout: float = 30.0,
        batch_size: int = 10,
        max_concurrent: int = 5,
    ) -> None:
        self.dim = dim
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._batch_size = batch_size
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def embed(self, text: str) -> list[float]:
        """Embed a single string."""
        vectors = await self._embed_chunk([text])
        return vectors[0]

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed many strings, preserving input order."""
        if not texts:
            return []
        chunks = [
            list(texts[i : i + self._batch_size])
            for i in range(0, len(texts), self._batch_size)
        ]
        results = await asyncio.gather(*(self._embed_chunk(c) for c in chunks))
        return [vec for chunk in results for vec in chunk]

    async def _embed_chunk(self, chunk: list[str]) -> list[list[float]]:
        """One embedding call, semaphore-guarded."""
        try:
            import litellm
        except ImportError as exc:
            raise EmbeddingError(
                "litellm is not installed. Install with: pip install 'everos[litellm]'"
            ) from exc

        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": chunk,
            "drop_params": True,
            "timeout": self._timeout,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["api_base"] = self._base_url

        async with self._semaphore:
            try:
                response = await litellm.aembedding(**kwargs)
            except Exception as exc:
                qualname = f"{type(exc).__module__}.{type(exc).__name__}"
                if qualname.startswith("litellm.exceptions.") or qualname.startswith(
                    "openai."
                ):
                    raise EmbeddingError(str(exc)) from exc
                raise

        return [list(item["embedding"][: self.dim]) for item in response.data]
