"""vLLM rerank provider — auth header conditional, results parsing, retries."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from everos.component.rerank import RerankError, VllmRerankProvider


def _patch_httpx(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    transport = httpx.MockTransport(handler)
    import everos.component.rerank.vllm_provider as mod

    real_cls = httpx.AsyncClient

    def factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_cls(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(mod.httpx, "AsyncClient", factory)


def _ok_response(items: list[dict[str, float | int]]) -> httpx.Response:
    return httpx.Response(200, json={"results": items})


async def test_empty_documents_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _ok_response([])

    _patch_httpx(monkeypatch, handler)
    p = VllmRerankProvider(model="m", api_key="", base_url="http://x/v1")
    assert await p.rerank("q", []) == []
    assert calls == 0


async def test_url_and_sort_desc(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_urls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen_urls.append(str(req.url))
        return _ok_response(
            [
                {"index": 0, "relevance_score": 0.1},
                {"index": 1, "relevance_score": 0.9},
                {"index": 2, "relevance_score": 0.5},
            ]
        )

    _patch_httpx(monkeypatch, handler)
    p = VllmRerankProvider(model="m", api_key="k", base_url="http://localhost:8000/v1/")
    results = await p.rerank("q", ["a", "b", "c"])
    # Trailing slash stripped, ``/rerank`` appended.
    assert seen_urls == ["http://localhost:8000/v1/rerank"]
    assert [r.index for r in results] == [1, 2, 0]


async def test_auth_header_added_when_api_key_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_headers: list[dict[str, str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen_headers.append(dict(req.headers))
        return _ok_response([{"index": 0, "relevance_score": 0.5}])

    _patch_httpx(monkeypatch, handler)
    p = VllmRerankProvider(model="m", api_key="sk-abc", base_url="http://x/v1")
    await p.rerank("q", ["a"])
    assert seen_headers[0].get("authorization") == "Bearer sk-abc"


async def test_auth_header_omitted_when_api_key_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_headers: list[dict[str, str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen_headers.append(dict(req.headers))
        return _ok_response([{"index": 0, "relevance_score": 0.5}])

    _patch_httpx(monkeypatch, handler)
    p = VllmRerankProvider(model="m", api_key="", base_url="http://x/v1")
    await p.rerank("q", ["a"])
    assert "authorization" not in seen_headers[0]


async def test_batching_offsets_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    """With batch_size=2 and 3 docs, the second batch's result index 0 becomes 2."""

    def handler(req: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(req.content)
        docs = body["documents"]
        # Each chunk: return per-chunk indices 0..len-1
        return _ok_response(
            [{"index": i, "relevance_score": float(i)} for i in range(len(docs))]
        )

    _patch_httpx(monkeypatch, handler)
    p = VllmRerankProvider(model="m", api_key="", base_url="http://x/v1", batch_size=2)
    results = await p.rerank("q", ["a", "b", "c"])
    # Returned indices should be 0, 1 from chunk 1; 2 from chunk 2.
    assert sorted(r.index for r in results) == [0, 1, 2]


async def test_4xx_raises_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"calls": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        return httpx.Response(401, text="unauthorized")

    _patch_httpx(monkeypatch, handler)
    p = VllmRerankProvider(
        model="m", api_key="bad", base_url="http://x/v1", max_retries=3
    )
    with pytest.raises(RerankError, match="HTTP 401"):
        await p.rerank("q", ["a"])
    assert state["calls"] == 1


async def test_5xx_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"calls": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] < 2:
            return httpx.Response(502, text="bad gw")
        return _ok_response([{"index": 0, "relevance_score": 0.42}])

    _patch_httpx(monkeypatch, handler)
    p = VllmRerankProvider(model="m", api_key="", base_url="http://x/v1", max_retries=3)
    results = await p.rerank("q", ["a"])
    assert state["calls"] == 2
    assert results[0].score == pytest.approx(0.42)


async def test_5xx_exhausts_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    _patch_httpx(monkeypatch, handler)
    p = VllmRerankProvider(model="m", api_key="", base_url="http://x/v1", max_retries=1)
    with pytest.raises(RerankError, match="HTTP 500"):
        await p.rerank("q", ["a"])


async def test_transport_error_exhausts(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout")

    _patch_httpx(monkeypatch, handler)
    p = VllmRerankProvider(model="m", api_key="", base_url="http://x/v1", max_retries=1)
    with pytest.raises(RerankError, match="transport failure"):
        await p.rerank("q", ["a"])


async def test_malformed_results_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    _patch_httpx(monkeypatch, handler)
    p = VllmRerankProvider(model="m", api_key="", base_url="http://x/v1")
    with pytest.raises(RerankError, match="missing results"):
        await p.rerank("q", ["a"])


async def test_malformed_result_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": 0}]})

    _patch_httpx(monkeypatch, handler)
    p = VllmRerankProvider(model="m", api_key="", base_url="http://x/v1")
    with pytest.raises(RerankError, match="malformed rerank result"):
        await p.rerank("q", ["a"])
