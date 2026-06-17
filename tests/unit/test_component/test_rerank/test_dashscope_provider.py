"""DashScope rerank provider — URL/body shape, results parsing, retries."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from everos.component.rerank import DashScopeRerankProvider, RerankError


def _patch_httpx(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    transport = httpx.MockTransport(handler)
    import everos.component.rerank.dashscope_provider as mod

    real_cls = httpx.AsyncClient

    def factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_cls(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(mod.httpx, "AsyncClient", factory)


def _ok_response(items: list[dict[str, float | int]]) -> httpx.Response:
    return httpx.Response(200, json={"output": {"results": items}})


def test_only_gte_rerank_v2_is_supported() -> None:
    with pytest.raises(ValueError, match="gte-rerank-v2 only"):
        DashScopeRerankProvider(
            model="qwen3-rerank",
            api_key="k",
            base_url="https://dashscope.aliyuncs.com",
        )


async def test_empty_documents_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _ok_response([])

    _patch_httpx(monkeypatch, handler)
    p = DashScopeRerankProvider(
        model="gte-rerank-v2", api_key="k", base_url="https://dashscope.aliyuncs.com"
    )
    assert await p.rerank("q", []) == []
    assert calls == 0


async def test_url_body_and_sort_desc(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_urls: list[str] = []
    seen_bodies: list[dict[str, object]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen_urls.append(str(req.url))
        seen_bodies.append(json.loads(req.content))
        return _ok_response(
            [
                {"index": 0, "relevance_score": 0.1},
                {"index": 1, "relevance_score": 0.9},
                {"index": 2, "relevance_score": 0.5},
            ]
        )

    _patch_httpx(monkeypatch, handler)
    p = DashScopeRerankProvider(
        model="gte-rerank-v2",
        api_key="k",
        base_url="https://dashscope.aliyuncs.com/",
    )
    results = await p.rerank("q", ["a", "b", "c"])
    # Trailing slash stripped, native service path appended.
    assert seen_urls == [
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    ]
    # Nested input/parameters body shape.
    body = seen_bodies[0]
    assert body["model"] == "gte-rerank-v2"
    assert body["input"] == {"query": "q", "documents": ["a", "b", "c"]}
    assert body["parameters"]["top_n"] == 3
    assert [r.index for r in results] == [1, 2, 0]


async def test_auth_header_always_present(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_headers: list[dict[str, str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen_headers.append(dict(req.headers))
        return _ok_response([{"index": 0, "relevance_score": 0.5}])

    _patch_httpx(monkeypatch, handler)
    p = DashScopeRerankProvider(
        model="gte-rerank-v2",
        api_key="sk-abc",
        base_url="https://dashscope.aliyuncs.com",
    )
    await p.rerank("q", ["a"])
    assert seen_headers[0].get("authorization") == "Bearer sk-abc"


async def test_batching_offsets_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    """With batch_size=2 and 3 docs, the second batch's result index 0 becomes 2."""

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        docs = body["input"]["documents"]
        return _ok_response(
            [{"index": i, "relevance_score": float(i)} for i in range(len(docs))]
        )

    _patch_httpx(monkeypatch, handler)
    p = DashScopeRerankProvider(
        model="gte-rerank-v2",
        api_key="k",
        base_url="https://dashscope.aliyuncs.com",
        batch_size=2,
    )
    results = await p.rerank("q", ["a", "b", "c"])
    assert sorted(r.index for r in results) == [0, 1, 2]


async def test_4xx_raises_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"calls": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        return httpx.Response(401, text="unauthorized")

    _patch_httpx(monkeypatch, handler)
    p = DashScopeRerankProvider(
        model="gte-rerank-v2",
        api_key="bad",
        base_url="https://dashscope.aliyuncs.com",
        max_retries=3,
    )
    with pytest.raises(RerankError, match="HTTP 401"):
        await p.rerank("q", ["a"])
    assert state["calls"] == 1


async def test_5xx_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"calls": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] < 2:
            return httpx.Response(503, text="unavailable")
        return _ok_response([{"index": 0, "relevance_score": 0.42}])

    _patch_httpx(monkeypatch, handler)
    p = DashScopeRerankProvider(
        model="gte-rerank-v2",
        api_key="k",
        base_url="https://dashscope.aliyuncs.com",
        max_retries=3,
    )
    results = await p.rerank("q", ["a"])
    assert state["calls"] == 2
    assert results[0].score == pytest.approx(0.42)


async def test_transport_error_exhausts(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout")

    _patch_httpx(monkeypatch, handler)
    p = DashScopeRerankProvider(
        model="gte-rerank-v2",
        api_key="k",
        base_url="https://dashscope.aliyuncs.com",
        max_retries=1,
    )
    with pytest.raises(RerankError, match="transport failure"):
        await p.rerank("q", ["a"])


async def test_malformed_results_missing_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"usage": {}})

    _patch_httpx(monkeypatch, handler)
    p = DashScopeRerankProvider(
        model="gte-rerank-v2", api_key="k", base_url="https://dashscope.aliyuncs.com"
    )
    with pytest.raises(RerankError, match="missing results"):
        await p.rerank("q", ["a"])


async def test_malformed_result_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"output": {"results": [{"index": 0}]}})

    _patch_httpx(monkeypatch, handler)
    p = DashScopeRerankProvider(
        model="gte-rerank-v2", api_key="k", base_url="https://dashscope.aliyuncs.com"
    )
    with pytest.raises(RerankError, match="malformed rerank result"):
        await p.rerank("q", ["a"])
