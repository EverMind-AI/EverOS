"""OpenAI-compatible embedding provider.

Wraps :class:`openai.AsyncOpenAI` so any OpenAI-protocol endpoint
(DeepInfra, OpenAI, Together, Fireworks, …) works without per-provider
forks. Self-hosted vLLM also exposes the same shape; the only quirk it
imposes is that the ``dimensions`` request parameter is ignored — we
truncate client-side to ``dim`` so callers always see the declared
shape regardless of backend.

Concurrency model:

- ``embed_batch`` splits the inputs into chunks of ``batch_size``.
- An :class:`asyncio.Semaphore` capped at ``max_concurrent`` bounds
  in-flight requests; remaining chunks queue and start as slots free.
- Retries / timeouts come from the openai SDK (``max_retries``,
  ``timeout`` constructor args).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import openai

from .protocol import EmbeddingServiceError


class OpenAIEmbeddingProvider:
    """OpenAI-compatible embedding provider with batching + concurrency.

    Args:
        model: Embedding model id (e.g. ``"Qwen/Qwen3-Embedding-4B"``).
        api_key: Bearer credential as a plain ``str``.
        base_url: OpenAI-protocol endpoint
            (e.g. ``"https://api.deepinfra.com/v1/openai"``).
        dim: Target vector dimension. Vectors longer than this are
            truncated client-side (matches the LanceDB column shape —
            see ``17_lancedb_tables_design.md``).
        timeout: Per-request timeout, seconds.
        max_retries: Retry budget exposed via the openai SDK.
        batch_size: How many inputs per ``/embeddings`` call.
        max_concurrent: Cap on in-flight chunked requests.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        dim: int = 1024,
        timeout: float = 30.0,
        max_retries: int = 3,
        batch_size: int = 10,
        max_concurrent: int = 5,
    ) -> None:
        self.dim = dim
        self._model = model
        self._batch_size = batch_size
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

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
        results = await asyncio.gather(*(self._embed_chunk(chunk) for chunk in chunks))
        # gather preserves order across awaitables, and each chunk preserves
        # its internal order — so flattening yields the input order back.
        return [vec for chunk in results for vec in chunk]

    async def _embed_chunk(self, chunk: list[str]) -> list[list[float]]:
        """One ``/embeddings`` call, semaphore-guarded."""
        async with self._semaphore:
            try:
                response = await self._client.embeddings.create(
                    model=self._model,
                    input=chunk,
                )
            except openai.OpenAIError as exc:
                raise EmbeddingServiceError(str(exc)) from exc
        # OpenAI returns ``data`` indexed by request order; truncate to ``dim``.
        return [list(item.embedding[: self.dim]) for item in response.data]
