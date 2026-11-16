"""``MemoryService.search`` — MemTrace stage-trace wiring."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from everos.memory.search import SearchData, SearchMethod, SearchRequest, SearchResponse
from everos.memory.search.trace import record_stage

search_mod = importlib.import_module("everos.service.memory")


@dataclass(frozen=True)
class _Candidate:
    id: str
    score: float
    source: str


class _StubTracker:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.calls: list[tuple[Any, Any, Any]] = []
        self.error_calls: list[tuple[Any, Any, Any]] = []

    async def on_search(
        self, req: Any, resp: Any, *, stages: Any = None, request_id: Any = None
    ) -> None:
        self.calls.append((req, resp, stages))

    async def on_search_error(
        self, req: Any, *, stages: Any = None, error: Any = None, request_id: Any = None
    ) -> None:
        self.error_calls.append((req, stages, error))


def _stub_manager(
    resp: SearchResponse, record: Callable[[], None] | None = None
) -> Any:
    class _StubManager:
        async def search(self, req: Any) -> SearchResponse:
            if record is not None:
                record()
            return resp

    return _StubManager()


def _service(manager: Any) -> Any:
    service = search_mod.MemoryService.__new__(search_mod.MemoryService)
    service._search_manager = manager
    return service


def _req() -> SearchRequest:
    return SearchRequest(user_id="alice", query="hi", method=SearchMethod.KEYWORD)


def _resp() -> SearchResponse:
    return SearchResponse(request_id="r" * 32, data=SearchData())


async def test_search_passes_collected_stages_to_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resp = _resp()
    tracker = _StubTracker()
    service = _service(
        _stub_manager(
            resp,
            record=lambda: record_stage(
                "single_route",
                "episode",
                [_Candidate(id="ep_1", score=0.8, source="keyword")],
            ),
        )
    )
    monkeypatch.setattr(
        search_mod.MemoryService, "_memtrace_tracker", tracker, raising=False
    )
    monkeypatch.setattr(search_mod, "resolve_request_id", lambda: "request-1")

    out = await service.search(_req())

    assert out is resp
    assert len(tracker.calls) == 1
    _, _, stages = tracker.calls[0]
    assert [(s.phase, s.kind) for s in stages] == [("single_route", "episode")]
    assert [(c.id, c.score, c.source) for c in stages[0].candidates] == [
        ("ep_1", 0.8, "keyword")
    ]


async def test_search_starts_from_a_fresh_collector_per_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _StubTracker()
    service = _service(_stub_manager(_resp()))
    monkeypatch.setattr(
        search_mod.MemoryService, "_memtrace_tracker", tracker, raising=False
    )

    record_stage("sparse", "episode", [_Candidate(id="x", score=0.1, source="keyword")])
    await service.search(_req())

    assert len(tracker.calls) == 1
    assert tracker.calls[0][2] == []


async def test_search_skips_collection_and_hook_when_tracker_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _StubTracker(enabled=False)

    def _unexpected_collector() -> Any:
        raise AssertionError("disabled tracing must not create a stage collector")

    service = _service(_stub_manager(_resp()))
    monkeypatch.setattr(
        search_mod.MemoryService, "_memtrace_tracker", tracker, raising=False
    )
    monkeypatch.setattr(search_mod, "collect_search_trace", _unexpected_collector)

    out = await service.search(_req())

    assert out is not None
    assert tracker.calls == []


async def test_search_error_invokes_error_hook_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A manager failure reaches ``on_search_error`` with the partial stages."""
    tracker = _StubTracker()

    class _FailingManager:
        async def search(self, req: Any) -> SearchResponse:
            record_stage(
                "sparse",
                "episode",
                [_Candidate(id="ep_1", score=0.8, source="keyword")],
            )
            raise RuntimeError("manager blew up")

    service = _service(_FailingManager())
    monkeypatch.setattr(
        search_mod.MemoryService, "_memtrace_tracker", tracker, raising=False
    )

    with pytest.raises(RuntimeError, match="manager blew up"):
        await service.search(_req())

    assert tracker.calls == []
    assert len(tracker.error_calls) == 1
    req, stages, error = tracker.error_calls[0]
    assert req.query == "hi"
    assert isinstance(error, RuntimeError)
    assert [(s.phase, s.kind) for s in stages] == [("sparse", "episode")]
