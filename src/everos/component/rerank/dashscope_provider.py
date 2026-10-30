"""DashScope (Aliyun Bailian) ``gte-rerank-v2`` provider.

DashScope's ``gte-rerank-v2`` uses a native ``text-rerank`` endpoint; the
``compatible-mode`` base_url used for LLM / embedding does not apply here::

    POST {base_url}/api/v1/services/rerank/text-rerank/text-rerank
    Authorization: Bearer <api_key>
    Content-Type: application/json

Request body (note the nested ``input`` / ``parameters``)::

    {
        "model": "<model>",
        "input": {"query": "<query>", "documents": ["<doc 1>", ...]},
        "parameters": {"return_documents": false, "top_n": <n>}
    }

Response body (results nested under ``output``)::

    {
        "output": {
            "results": [
                {"index": 0, "relevance_score": 0.87},
                {"index": 1, "relevance_score": 0.43},
                ...
            ]
        },
        "usage": {...},
        "request_id": "..."
    }

We request ``top_n = len(documents)`` so DashScope scores every input and
the :class:`RerankProvider` contract (one result per document) holds; the
returned list is sorted score-descending defensively regardless of server
ordering. ``api_key`` is required — DashScope has no anonymous tier.

``gte-rerank-v2`` and ``qwen3-rerank`` are supported. Both use the same native
endpoint and body; ``qwen3-rerank`` additionally takes the task instruction as
``parameters.instruct`` (local patch 2026-10-02: the intl region has no
``gte-rerank-v2``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import httpx

from ._errors import backoff_sleep
from .protocol import RerankError, RerankResult


class DashScopeRerankProvider:
    """Rerank provider for Aliyun Bailian / DashScope ``gte-rerank-v2``.

    Args:
        model: Reranker model id. Currently only ``"gte-rerank-v2"`` is
            supported.
        api_key: DashScope bearer credential (``sk-...``). Required.
        base_url: DashScope API root *without* the service path
            (e.g. ``"https://dashscope.aliyuncs.com"``). The
            ``/api/v1/services/rerank/text-rerank/text-rerank`` suffix is
            appended at request time.
        timeout: Per-request timeout, seconds.
        max_retries: Soft retry count on transport errors / 5xx.
        batch_size: Cap on documents per request.
        max_concurrent: Cap on in-flight requests across all batches.
    """

    _SERVICE_PATH = "/api/v1/services/rerank/text-rerank/text-rerank"
    _SUPPORTED_MODELS = frozenset({"gte-rerank-v2", "qwen3-rerank"})
    _INSTRUCT_MODELS = frozenset({"qwen3-rerank"})

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
        if model not in self._SUPPORTED_MODELS:
            raise ValueError(
                f"DashScope rerank supports {sorted(self._SUPPORTED_MODELS)}; "
                f"got {model!r}"
            )
        self._model = model
        self._api_key = api_key
        self._url = f"{base_url.rstrip('/')}{self._SERVICE_PATH}"
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

        ``instruction`` is sent as ``parameters.instruct`` for ``qwen3-rerank``;
        ``gte-rerank-v2`` takes no instruction, so it is dropped there.
        """
        if not documents:
            return []

        chunks: list[tuple[int, list[str]]] = [
            (offset, list(documents[offset : offset + self._batch_size]))
            for offset in range(0, len(documents), self._batch_size)
        ]
        chunk_results = await asyncio.gather(
            *(self._score_chunk(query, docs, instruction) for _, docs in chunks)
        )
        scored: list[RerankResult] = []
        for (offset, _), partial in zip(chunks, chunk_results, strict=True):
            scored.extend(
                RerankResult(index=offset + r.index, score=r.score) for r in partial
            )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored

    async def _score_chunk(
        self, query: str, documents: list[str], instruction: str | None = None
    ) -> list[RerankResult]:
        parameters: dict[str, Any] = {
            "return_documents": False,
            "top_n": len(documents),
        }
        if instruction and self._model in self._INSTRUCT_MODELS:
            parameters["instruct"] = instruction
        payload: dict[str, Any] = {
            "model": self._model,
            "input": {"query": query, "documents": documents},
            "parameters": parameters,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async with self._semaphore:
            for attempt in range(self._max_retries + 1):
                try:
                    async with httpx.AsyncClient(timeout=self._timeout) as client:
                        response = await client.post(
                            self._url, json=payload, headers=headers
                        )
                except httpx.HTTPError as exc:
                    if attempt == self._max_retries:
                        raise RerankError(
                            f"DashScope rerank transport failure: {exc}"
                        ) from exc
                    continue

                if response.status_code == 200:
                    return _parse_dashscope_results(response.json())

                if response.status_code >= 500 or response.status_code == 429:
                    if attempt == self._max_retries:
                        raise RerankError(
                            f"DashScope rerank HTTP {response.status_code}: "
                            f"{response.text[:200]}"
                        )
                    await backoff_sleep(attempt)
                    continue
                raise RerankError(
                    f"DashScope rerank HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )

            raise RerankError(
                f"DashScope rerank exhausted retries ({self._max_retries})"
            )


def _parse_dashscope_results(body: dict[str, Any]) -> list[RerankResult]:
    output = body.get("output")
    items = output.get("results") if isinstance(output, dict) else None
    if not isinstance(items, list):
        raise RerankError(f"DashScope rerank response missing results: {body!r}")
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
            raise RerankError(f"malformed rerank result entry: {item!r}") from exc
    return parsed
