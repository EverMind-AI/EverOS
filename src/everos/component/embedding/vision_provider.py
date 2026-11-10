"""OpenAI-compatible vision-embedding provider (chat-style ``/embeddings``).

Turns a picture into a vector by sending its bytes to an OpenAI-protocol
``/embeddings`` endpoint that accepts **chat-style** input — the only shape in
which a picture can travel:

.. code-block:: text

    POST /v1/embeddings
    {"model": "...", "input": [[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]}]]}

``input`` is a *list of conversations* (the batch form). One conversation per
picture keeps the batch in one request, so N pictures cost one round trip
instead of N — and the endpoint returns one vector per conversation, in order.

The provider sends the user turn only. The endpoint owns the model's input
recipe — for ``Qwen3-VL-Embedding-*`` the system instruction "Represent the
user's input." and the generation prompt, with the vector pooled at the
prompt's last token. That is a property of the server, and it fails silently:
stock vLLM 0.32's chat-embedding path loses the prompt's last token, so it
pools the wrong position (measured: cosine 0.35–0.56 with the reference
vectors; the model card's similarities off by up to 0.49). A server must be
checked against the model card's reference similarities before it is used.

The query side embeds text through the same model, because a picture's bytes
are only comparable to a question if the question went through the same
encoder. ``embed_query`` draws on its own concurrency budget for the same
reason the text provider does: a user waiting on a search must not queue
behind the cascade's background picture embeds (issue #473).

Width: ``dim`` is the ``visual_vector`` column's width, separate from
``[embedding] dim`` because the two columns come from different models.
Returned vectors are cut to it client-side, as the text provider does; the cut
is only lossless for Matryoshka-trained models (``Qwen3-VL-Embedding`` is one,
64–4096).
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Sequence

import openai

from everos.core.observability.tracing import memory_span, set_generation_usage

from .openai_provider import _classify
from .protocol import EmbeddingServiceError


class OpenAIVisionEmbeddingProvider:
    """OpenAI-compatible vision embedder over ``openai.AsyncOpenAI``.

    Args:
        model: Pooling model id (e.g. ``"Qwen/Qwen3-VL-Embedding-8B"``).
        api_key: Bearer credential as a plain ``str``.
        base_url: OpenAI-protocol endpoint (``http://host:port/v1``).
        dim: Width of the returned vector — the ``visual_vector`` column.
        timeout: Per-request timeout, seconds.
        max_retries: Retry budget exposed via the openai SDK.
        batch_size: How many pictures per ``/embeddings`` call.
        max_concurrent: Cap on in-flight chunked requests from write paths.
        query_max_concurrent: Separate cap for interactive query embeds.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        dim: int,
        timeout: float = 60.0,
        max_retries: int = 3,
        batch_size: int = 8,
        max_concurrent: int = 4,
        query_max_concurrent: int = 16,
    ) -> None:
        self.dim = dim
        self._model = model
        self._batch_size = batch_size
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._query_semaphore = asyncio.Semaphore(query_max_concurrent)
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def embed_image(self, image: bytes, media_type: str) -> list[float]:
        """Embed one picture on the write budget."""
        vectors = await self._embed_conversations(
            [self._picture_conversation(image, media_type)]
        )
        return vectors[0]

    async def embed_images(
        self, images: Sequence[tuple[bytes, str]]
    ) -> list[list[float]]:
        """Embed many pictures, preserving input order."""
        if not images:
            return []
        conversations = [
            self._picture_conversation(image, media_type)
            for image, media_type in images
        ]
        chunks = [
            conversations[i : i + self._batch_size]
            for i in range(0, len(conversations), self._batch_size)
        ]
        results = await asyncio.gather(
            *(self._embed_conversations(chunk) for chunk in chunks)
        )
        return [vec for chunk in results for vec in chunk]

    async def embed_query(self, text: str) -> list[float]:
        """Embed one text query in the picture space, on the query budget."""
        vectors = await self._embed_conversations(
            [self._text_conversation(text)], semaphore=self._query_semaphore
        )
        return vectors[0]

    @staticmethod
    def _picture_conversation(image: bytes, media_type: str) -> list[dict[str, object]]:
        encoded = base64.b64encode(image).decode("ascii")
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                    }
                ],
            }
        ]

    @staticmethod
    def _text_conversation(text: str) -> list[dict[str, object]]:
        return [{"role": "user", "content": [{"type": "text", "text": text}]}]

    async def _embed_conversations(
        self,
        conversations: list[list[dict[str, object]]],
        *,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[list[float]]:
        """One ``/embeddings`` call taking N conversations, in order."""
        with memory_span(
            "everos.embedding.vision", observation_type="embedding", nested_only=True
        ):
            async with semaphore or self._semaphore:
                try:
                    # ``cast_to=object`` is the SDK's raw-JSON path: it
                    # deserialises to plain dict/list without constructing a
                    # typed model. ``dict`` looks equivalent but the SDK feeds
                    # it through pydantic's generic-type construction and
                    # raises on the bare type.
                    response = await self._client.post(
                        "/embeddings",
                        cast_to=object,
                        body={"model": self._model, "input": conversations},
                    )
                except openai.OpenAIError as exc:
                    raise _classify(exc) from exc
            data = response.get("data") or []
            if len(data) != len(conversations):
                raise EmbeddingServiceError(
                    f"Vision embedding API returned {len(data)} vectors "
                    f"for {len(conversations)} inputs"
                )
            # Sorted by ``index``: the batch form is documented to preserve
            # order, but the field exists and a reordered answer would put a
            # picture's vector on the wrong row.
            ordered = sorted(data, key=lambda item: item.get("index", 0))
            usage = response.get("usage") or {}
            set_generation_usage(
                model=self._model, input_tokens=usage.get("prompt_tokens")
            )
            return [list(item["embedding"][: self.dim]) for item in ordered]


__all__ = ["OpenAIVisionEmbeddingProvider"]
