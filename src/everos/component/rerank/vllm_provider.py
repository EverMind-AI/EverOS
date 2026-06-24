"""vLLM rerank provider.

Self-deployed vLLM (and other OpenAI-compatible rerank servers) expose
the OpenAI-style rerank endpoint::

    POST {base_url}/rerank
    Authorization: Bearer <api_key>  # optional for self-hosted ("EMPTY")
    Content-Type: application/json

Request body:

    {
        "model":     "<model>",
        "query":     "<query>",
        "documents": ["<doc 1>", "<doc 2>", ...]
    }

Response body:

    {
        "results": [
            {"index": 0, "relevance_score": 0.87},
            {"index": 1, "relevance_score": 0.43},
            ...
        ],
        "id": "...",
        ...
    }

We pass documents through as-is — caller is responsible for any
prompt-template formatting required by the underlying reranker. Output
ordering may already be score-descending; we sort defensively to honour
the :class:`RerankProvider` contract regardless of server behaviour.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import httpx

from ._errors import retries_exhausted_error, transport_error, upstream_http_error
from .protocol import RerankResult, RerankServiceError


class VllmRerankProvider:
    """Rerank provider for vLLM / OpenAI-compat ``/v1/rerank`` endpoints.

    Args:
        model: Reranker model id (e.g. ``"Qwen/Qwen3-Reranker-4B"``).
        api_key: Bearer credential. Pass ``""`` (empty string) for
            self-hosted endpoints that don't require auth — the
            ``Authorization`` header is omitted in that case.
        base_url: API root that *contains* the ``/v1`` prefix
            (e.g. ``"http://localhost:8000/v1"``). The ``/rerank``
            suffix is appended at request time.
        timeout: Per-request timeout, seconds.
        max_retries: Soft retry count on transport errors / 5xx.
        batch_size: Cap on documents per request.
        max_concurrent: Cap on in-flight requests across all batches.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        batch_size: int = 10,
        max_concurrent: int = 5,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._url = f"{base_url.rstrip('/')}/rerank"
        self._timeout = timeout
        self._max_retries = max_retries
        self._batch_size = batch_size
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: str | None = None,
    ) -> list[RerankResult]:
        """Score every document against ``query``; return sorted desc.

        ``instruction`` is accepted for protocol parity but not transmitted:
        the OpenAI-compatible ``/rerank`` endpoint applies the reranker's chat
        template (including any task instruction) server-side, so unlike the
        DeepInfra completion-style API there is no client-side template to fill.
        """
        if not documents:
            return []

        chunks: list[tuple[int, list[str]]] = [
            (offset, list(documents[offset : offset + self._batch_size]))
            for offset in range(0, len(documents), self._batch_size)
        ]
        chunk_results = await asyncio.gather(
            *(self._score_chunk(query, docs) for _, docs in chunks)
        )
        scored: list[RerankResult] = []
        for (offset, _), partial in zip(chunks, chunk_results, strict=True):
            scored.extend(
                RerankResult(index=offset + r.index, score=r.score) for r in partial
            )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored

    async def _score_chunk(
        self, query: str, documents: list[str]
    ) -> list[RerankResult]:
        payload: dict[str, Any] = {
            "model": self._model,
            "query": query,
            "documents": documents,
        }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async with self._semaphore:
            for attempt in range(self._max_retries + 1):
                try:
                    async with httpx.AsyncClient(timeout=self._timeout) as client:
                        response = await client.post(
                            self._url, json=payload, headers=headers
                        )
                except httpx.HTTPError as exc:
                    if attempt == self._max_retries:
                        raise transport_error("vLLM", exc) from exc
                    continue

                if response.status_code == 200:
                    return _parse_rerank_results(response.json())

                if response.status_code >= 500 or response.status_code == 429:
                    if attempt == self._max_retries:
                        raise upstream_http_error("vLLM", response)
                    continue
                raise upstream_http_error("vLLM", response)

            raise retries_exhausted_error("vLLM", self._max_retries)


def _parse_rerank_results(body: dict[str, Any]) -> list[RerankResult]:
    items = body.get("results")
    if not isinstance(items, list):
        raise RerankServiceError(f"vLLM rerank response missing results: {body!r}")
    parsed: list[RerankResult] = []
    for item in items:
        try:
            parsed.append(
                RerankResult(
                    index=int(item["index"]),
                    score=float(item["relevance_score"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RerankServiceError(
                f"malformed rerank result entry: {item!r}"
            ) from exc
    return parsed
