"""DeepInfra rerank provider — happy path, batching, retries, errors.

httpx is faked via :class:`httpx.MockTransport`; the provider's
``httpx.AsyncClient(timeout=...)`` ctx manager is monkeypatched to
return a client wired to the transport.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from everos.component.rerank import (
    DeepInfraRerankProvider,
    RerankServiceError,
)


def _patch_httpx(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    """Make ``httpx.AsyncClient(timeout=...)`` use a MockTransport."""
    transport = httpx.MockTransport(handler)
    import everos.component.rerank.deepinfra_provider as mod

    real_cls = httpx.AsyncClient

    def factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_cls(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(mod.httpx, "AsyncClient", factory)


def _ok_response(scores: list[float]) -> httpx.Response:
    return httpx.Response(200, json={"scores": [scores]})


async def test_empty_documents_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _ok_response([])

    _patch_httpx(monkeypatch, handler)
    p = DeepInfraRerankProvider(model="m", api_key="k", base_url="https://api/v1")
    assert await p.rerank("q", []) == []
    assert calls == 0


async def test_scores_sorted_descending(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return _ok_response([0.1, 0.9, 0.5])

    _patch_httpx(monkeypatch, handler)
    p = DeepInfraRerankProvider(
        model="m", api_key="k", base_url="https://api/v1", batch_size=10
    )
    results = await p.rerank("q", ["a", "b", "c"])
    assert [r.index for r in results] == [1, 2, 0]
    assert results[0].score == pytest.approx(0.9)


async def test_batching_merges_chunk_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    """batch_size=2 with 3 documents → 2 chunks; merged indices respect offset."""
    seen_bodies: list[list[str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen_bodies.append(body["documents"])
        # Score by length so we can verify ordering.
        return _ok_response([float(len(d)) for d in body["documents"]])

    _patch_httpx(monkeypatch, handler)
    p = DeepInfraRerankProvider(
        model="m", api_key="k", base_url="https://api/v1", batch_size=2
    )
    docs = ["x", "yy", "zzz"]
    results = await p.rerank("q", docs)
    assert {len(b) for b in seen_bodies} == {1, 2}
    # Sorted desc by score = len: "zzz"=3 → idx 2, "yy"=2 → idx 1, "x"=1 → idx 0
    assert [r.index for r in results] == [2, 1, 0]


async def test_url_appends_model(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_urls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen_urls.append(str(req.url))
        return _ok_response([0.5])

    _patch_httpx(monkeypatch, handler)
    p = DeepInfraRerankProvider(
        model="Qwen/Q",
        api_key="k",
        # Trailing slash should be stripped before appending model path.
        base_url="https://api.deepinfra.com/v1/inference/",
    )
    await p.rerank("q", ["a"])
    assert seen_urls == ["https://api.deepinfra.com/v1/inference/Qwen/Q"]


async def test_4xx_raises_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="bad input")

    _patch_httpx(monkeypatch, handler)
    p = DeepInfraRerankProvider(
        model="m", api_key="k", base_url="https://api/v1", max_retries=3
    )
    with pytest.raises(RerankServiceError, match="HTTP 400"):
        await p.rerank("q", ["a"])
    assert calls == 1  # no retry on 4xx


async def test_5xx_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"calls": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] < 3:
            return httpx.Response(503, text="busy")
        return _ok_response([0.7])

    _patch_httpx(monkeypatch, handler)
    p = DeepInfraRerankProvider(
        model="m", api_key="k", base_url="https://api/v1", max_retries=3
    )
    results = await p.rerank("q", ["a"])
    assert state["calls"] == 3
    assert results[0].score == pytest.approx(0.7)


async def test_5xx_exhausts_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    _patch_httpx(monkeypatch, handler)
    p = DeepInfraRerankProvider(
        model="m", api_key="k", base_url="https://api/v1", max_retries=1
    )
    with pytest.raises(RerankServiceError, match="HTTP 500"):
        await p.rerank("q", ["a"])


async def test_429_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"calls": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] == 1:
            return httpx.Response(429, text="slow down")
        return _ok_response([0.4])

    _patch_httpx(monkeypatch, handler)
    p = DeepInfraRerankProvider(
        model="m", api_key="k", base_url="https://api/v1", max_retries=3
    )
    results = await p.rerank("q", ["a"])
    assert state["calls"] == 2
    assert results[0].score == pytest.approx(0.4)


async def test_transport_error_retries_then_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network down")

    _patch_httpx(monkeypatch, handler)
    p = DeepInfraRerankProvider(
        model="m", api_key="k", base_url="https://api/v1", max_retries=1
    )
    with pytest.raises(RerankServiceError, match="transport failure"):
        await p.rerank("q", ["a"])


async def test_malformed_scores_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"something_else": []})

    _patch_httpx(monkeypatch, handler)
    p = DeepInfraRerankProvider(model="m", api_key="k", base_url="https://api/v1")
    with pytest.raises(RerankServiceError, match="missing scores"):
        await p.rerank("q", ["a"])


async def test_score_length_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"scores": [[0.1, 0.2]]})

    _patch_httpx(monkeypatch, handler)
    p = DeepInfraRerankProvider(
        model="m", api_key="k", base_url="https://api/v1", batch_size=10
    )
    with pytest.raises(RerankServiceError, match="returned 2 scores, expected 3"):
        await p.rerank("q", ["a", "b", "c"])


async def test_payload_wraps_qwen3_template(monkeypatch: pytest.MonkeyPatch) -> None:
    """Query + documents are wrapped in the Qwen3-Reranker chat template.

    DeepInfra's inference API scores raw text, so the prompt scaffolding
    (system frame + ``<Instruct>``/``<Query>``/``<Document>`` markers) must be
    supplied client-side or the reranker returns uncalibrated scores.
    """
    captured: dict[str, list[str]] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return _ok_response([0.5])

    _patch_httpx(monkeypatch, handler)
    p = DeepInfraRerankProvider(model="m", api_key="k", base_url="https://api/v1")
    await p.rerank("what did Alice eat?", ["pasta"], instruction="find facts")

    query_sent = captured["queries"][0]
    assert query_sent.startswith("<|im_start|>system")
    assert "<Instruct>: find facts" in query_sent
    assert "<Query>: what did Alice eat?" in query_sent
    assert captured["documents"][0].startswith("<Document>: pasta")


async def test_default_instruction_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``None`` instruction falls back to the provider's default, not blank."""
    captured: dict[str, list[str]] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return _ok_response([0.5])

    _patch_httpx(monkeypatch, handler)
    p = DeepInfraRerankProvider(model="m", api_key="k", base_url="https://api/v1")
    await p.rerank("q", ["d"])
    assert "<Instruct>: Given a question and a passage" in captured["queries"][0]


async def test_flat_scores_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """If response is ``{"scores": [s1, s2]}`` (flat), the unwrap still works."""

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"scores": [0.3, 0.6]})

    _patch_httpx(monkeypatch, handler)
    p = DeepInfraRerankProvider(model="m", api_key="k", base_url="https://api/v1")
    results = await p.rerank("q", ["a", "b"])
    assert [r.score for r in results] == [0.6, 0.3]
