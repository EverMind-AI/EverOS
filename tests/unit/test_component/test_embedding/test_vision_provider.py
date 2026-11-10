"""Tests for :class:`OpenAIVisionEmbeddingProvider`.

What these pin:

- A picture travels as a chat conversation with one ``image_url`` part (a data
  URL of its bytes); a query travels as a conversation with one text part. The
  provider sends the user turn only — the endpoint owns the model's recipe.
- Pictures are batched by ``batch_size`` into one request each, and the vectors
  come back in input order even when the endpoint reorders ``data`` (``index``
  is what places them).
- Returned vectors are cut to ``dim``, as the text provider does.
- A short answer (fewer vectors than inputs) raises instead of shifting every
  later vector onto the wrong picture.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest

from everos.component.embedding import OpenAIVisionEmbeddingProvider
from everos.component.embedding.protocol import EmbeddingServiceError


class _Endpoint:
    """Records request bodies; answers one vector per conversation, reversed."""

    def __init__(self, *, width: int = 6, drop_last: bool = False) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.width = width
        self.drop_last = drop_last

    async def post(
        self, path: str, *, cast_to: Any, body: dict[str, Any]
    ) -> dict[str, Any]:
        assert path == "/embeddings"
        assert cast_to is object
        self.bodies.append(body)
        n = len(body["input"]) - (1 if self.drop_last else 0)
        data = [{"index": i, "embedding": [float(i)] * self.width} for i in range(n)]
        return {"data": list(reversed(data)), "usage": {"prompt_tokens": 7}}


def _provider(endpoint: _Endpoint, **overrides: Any) -> OpenAIVisionEmbeddingProvider:
    kwargs: dict[str, Any] = dict(
        model="vl-test",
        api_key="k",
        base_url="https://example.test",
        dim=4,
        batch_size=2,
        max_concurrent=1,
    )
    kwargs.update(overrides)
    provider = OpenAIVisionEmbeddingProvider(**kwargs)
    provider._client = endpoint  # type: ignore[assignment]
    return provider


async def test_a_picture_is_sent_as_one_image_part_of_its_bytes() -> None:
    endpoint = _Endpoint()

    await _provider(endpoint).embed_image(b"\x89PNG-bytes", "image/png")

    (conversation,) = endpoint.bodies[0]["input"]
    (turn,) = conversation
    assert turn["role"] == "user"
    (part,) = turn["content"]
    assert part["type"] == "image_url"
    url = part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"\x89PNG-bytes"


async def test_a_query_is_sent_as_one_text_part() -> None:
    endpoint = _Endpoint()

    await _provider(endpoint).embed_query("a red bicycle")

    assert endpoint.bodies[0]["input"] == [
        [{"role": "user", "content": [{"type": "text", "text": "a red bicycle"}]}]
    ]


async def test_pictures_are_batched_and_returned_in_input_order() -> None:
    endpoint = _Endpoint()

    vectors = await _provider(endpoint).embed_images(
        [(b"a", "image/png"), (b"b", "image/png"), (b"c", "image/jpeg")]
    )

    assert [len(b["input"]) for b in endpoint.bodies] == [2, 1]
    # The endpoint answers in reverse; ``index`` puts each vector back in place.
    assert vectors == [[0.0] * 4, [1.0] * 4, [0.0] * 4]


async def test_vectors_are_cut_to_dim() -> None:
    endpoint = _Endpoint(width=10)

    vector = await _provider(endpoint, dim=3).embed_query("q")

    assert len(vector) == 3


async def test_a_short_answer_raises() -> None:
    endpoint = _Endpoint(drop_last=True)

    with pytest.raises(EmbeddingServiceError, match="1 vectors for 2 inputs"):
        await _provider(endpoint).embed_images(
            [(b"a", "image/png"), (b"b", "image/png")]
        )
