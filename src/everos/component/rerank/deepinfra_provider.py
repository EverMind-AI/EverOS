"""DeepInfra inference-API rerank provider.

DeepInfra exposes reranker models (e.g. ``Qwen/Qwen3-Reranker-4B``) at::

    POST {base_url}/{model}
    Authorization: Bearer <api_key>
    Content-Type: application/json

The request shape is the inference-API convention used across DeepInfra
reranker / classifier models:

    {
        "queries":   ["<query>"],
        "documents": ["<doc 1>", "<doc 2>", ...]
    }

The response carries one ``scores`` array per query:

    {
        "scores":          [[0.12, 0.87, 0.43, ...]],
        "request_id":      "...",
        "inference_status": {...}
    }

We submit one query at a time (matches the :class:`RerankProvider`
contract) and unwrap the inner score list. Documents longer than the
model's input window are silently truncated server-side.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import httpx

from .protocol import RerankError, RerankResult

# Qwen3-Reranker chat template. The DeepInfra inference API treats the reranker
# as a yes/no generator, so the prompt scaffolding must be supplied client-side
# (verbatim mirror of the benchmark reranker client). Without it the
# model scores raw text off-template and returns uncalibrated relevance.
_QWEN3_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
_QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_DEFAULT_RERANK_INSTRUCTION = (
    "Given a question and a passage, determine if the passage contains "
    "information relevant to answering the question."
)


def _format_qwen3_inputs(
    query: str, documents: list[str], instruction: str | None
) -> tuple[str, list[str]]:
    """Wrap query + documents in the Qwen3-Reranker chat template."""
    instr = instruction or _DEFAULT_RERANK_INSTRUCTION
    formatted_query = f"{_QWEN3_PREFIX}<Instruct>: {instr}\n<Query>: {query}\n"
    formatted_docs = [f"<Document>: {doc}{_QWEN3_SUFFIX}" for doc in documents]
    return formatted_query, formatted_docs


class DeepInfraRerankProvider:
    """Rerank provider for the DeepInfra inference API.

    Args:
        model: Reranker model id (e.g. ``"Qwen/Qwen3-Reranker-4B"``).
        api_key: Bearer credential as plain ``str``.
        base_url: Inference endpoint root
            (e.g. ``"https://api.deepinfra.com/v1/inference"``). The
            ``/{model}`` suffix is appended at request time.
        timeout: Per-request timeout, seconds.
        max_retries: Soft retry count on transport errors / 5xx.
        batch_size: Cap on documents per request (large doc lists are
            split, scores merged in input order).
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
        self._url = f"{base_url.rstrip('/')}/{model}"
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
        """Score every document against ``query``; return sorted desc."""
        if not documents:
            return []

        chunks: list[tuple[int, list[str]]] = [
            (offset, list(documents[offset : offset + self._batch_size]))
            for offset in range(0, len(documents), self._batch_size)
        ]
        chunk_scores = await asyncio.gather(
            *(self._score_chunk(query, docs, instruction) for _, docs in chunks)
        )
        scored: list[RerankResult] = []
        for (offset, _), scores in zip(chunks, chunk_scores, strict=True):
            scored.extend(
                RerankResult(index=offset + i, score=score)
                for i, score in enumerate(scores)
            )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored

    async def _score_chunk(
        self, query: str, documents: list[str], instruction: str | None
    ) -> list[float]:
        formatted_query, formatted_docs = _format_qwen3_inputs(
            query, documents, instruction
        )
        payload: dict[str, Any] = {
            "queries": [formatted_query],
            "documents": formatted_docs,
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
                            f"DeepInfra rerank transport failure: {exc}"
                        ) from exc
                    continue

                if response.status_code == 200:
                    return _extract_scores(response.json(), len(documents))

                # Retry on 5xx / 429 only; surface 4xx immediately.
                if response.status_code >= 500 or response.status_code == 429:
                    if attempt == self._max_retries:
                        raise RerankError(
                            f"DeepInfra rerank HTTP {response.status_code}: "
                            f"{response.text[:200]}"
                        )
                    continue
                raise RerankError(
                    f"DeepInfra rerank HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )

            raise RerankError(
                f"DeepInfra rerank exhausted retries ({self._max_retries})"
            )


def _extract_scores(body: dict[str, Any], expected_len: int) -> list[float]:
    """Unwrap ``scores`` from the DeepInfra response body.

    Inference API returns ``scores`` as either:

    - ``[[s1, s2, ...]]`` — one score row per query (current single-query
      shape); take row 0.
    - ``[s1, s2, ...]`` — flat list (fallback for providers that drop
      the outer list when only one query is sent).
    """
    raw = body.get("scores")
    if not isinstance(raw, list):
        raise RerankError(f"DeepInfra rerank response missing scores: {body!r}")
    row = raw[0] if raw and isinstance(raw[0], list) else raw
    if len(row) != expected_len:
        raise RerankError(
            f"DeepInfra rerank returned {len(row)} scores, expected {expected_len}"
        )
    return [float(s) for s in row]
