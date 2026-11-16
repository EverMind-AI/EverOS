"""``MemoryService._memorize`` — MemTrace hook wiring on success and error."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from everos.memory.extract.trace import record_construction

mm = importlib.import_module("everos.service.memory")


class _StubTracker:
    def __init__(self) -> None:
        self.enabled = False
        self.calls: list[tuple[Any, ...]] = []
        self.error_calls: list[tuple[Any, ...]] = []
        self.events: list[dict[str, Any]] = []

    async def on_memorize(
        self,
        payload: Any,
        *,
        is_final: bool,
        defer_extraction: bool,
        result: Any,
        request_id: Any = None,
        construction_events: Any = None,
    ) -> None:
        self.calls.append((payload, is_final, defer_extraction, result))
        self.events.extend(construction_events or [])

    async def on_memorize_error(
        self,
        payload: Any,
        *,
        is_final: bool,
        defer_extraction: bool,
        error: Any,
        request_id: Any = None,
        construction_events: Any = None,
    ) -> None:
        self.error_calls.append((payload, is_final, defer_extraction, error))
        self.events.extend(construction_events or [])


@pytest.fixture
def tracker(monkeypatch: pytest.MonkeyPatch) -> Iterator[_StubTracker]:
    stub = _StubTracker()
    monkeypatch.setattr(mm.MemoryService, "_memtrace_tracker", stub, raising=False)
    monkeypatch.setattr(mm, "resolve_request_id", lambda: "request-1")
    yield stub


def _service(run: Any) -> Any:
    service = mm.MemoryService.__new__(mm.MemoryService)
    service._memorizer = SimpleNamespace(run=run)
    return service


async def test_success_calls_on_memorize_with_result(
    tracker: _StubTracker,
) -> None:
    result = mm.MemorizeResult(message_count=1, status="extracted")
    service = _service(AsyncMock(return_value=result))

    out = await service._memorize({"session_id": "s1", "messages": []})

    assert out is result
    assert tracker.error_calls == []
    assert len(tracker.calls) == 1
    _, is_final, defer_extraction, recorded = tracker.calls[0]
    assert (is_final, defer_extraction) == (False, False)
    assert recorded is result


async def test_failure_calls_on_memorize_error_and_reraises(
    tracker: _StubTracker,
) -> None:
    boom = RuntimeError("pipeline blew up")
    service = _service(AsyncMock(side_effect=boom))

    with pytest.raises(RuntimeError, match="pipeline blew up") as exc_info:
        await service._memorize({"session_id": "s1", "messages": []}, is_final=True)

    assert exc_info.value is boom
    assert tracker.calls == []
    assert len(tracker.error_calls) == 1
    _, is_final, defer_extraction, error = tracker.error_calls[0]
    assert (is_final, defer_extraction) == (True, False)
    assert error is boom


@pytest.mark.parametrize("fail", [False, True])
async def test_actual_events_reach_both_hooks(
    tracker: _StubTracker, fail: bool
) -> None:
    tracker.enabled = True

    async def run(*args: Any, **kwargs: Any) -> Any:
        record_construction("extraction_result", content="", owners=["alice"])
        if fail:
            raise ValueError("write failed")
        return mm.MemorizeResult(message_count=1, status="extracted")

    service = _service(run)
    if fail:
        with pytest.raises(ValueError, match="write failed"):
            await service._memorize({"session_id": "s1", "messages": []})
    else:
        await service._memorize({"session_id": "s1", "messages": []})
    assert len(tracker.events) == 1
    assert tracker.events[0]["content"] == ""
    assert tracker.events[0]["phase"] == "extraction_result"
