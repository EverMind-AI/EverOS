"""The extraction that outlives a cancelled memorize() call stays owned.

Once the boundary stage has committed a memcell, ``memorize()`` runs the
extraction shielded from the request's cancellation (a client that went
away). The task is handed to :class:`ExtractionSupervisor`, which keeps it,
retrieves its outcome, logs a failure with the memcell ids, and lets the
service's shutdown wait for what is still running. Review on #521 pointed
out that the first version left the task unowned: a later failure was
"never retrieved" and nobody could know a memcell lost its memory.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import structlog.testing

from everos.service import MemorizeResult
from everos.service.memory import ExtractionSupervisor
from tests.fakes import make_memory_service


async def _run_with_cancelled_request(outcome: str) -> list[dict[str, Any]]:
    """Cancel a call mid-extraction; let the extraction end the given way."""
    supervisor = ExtractionSupervisor()
    gate = asyncio.Event()

    async def extraction() -> Any:
        await gate.wait()
        if outcome == "fail":
            raise RuntimeError("extraction failed after disconnect")
        return MemorizeResult(message_count=1, status="extracted")

    with structlog.testing.capture_logs() as logs:
        task = supervisor.supervise(extraction(), session_id="s", memcell_ids=["mc_1"])

        async def request() -> Any:
            return await asyncio.shield(task)

        req = asyncio.create_task(request())
        await asyncio.sleep(0)
        req.cancel()
        with pytest.raises(asyncio.CancelledError):
            await req
        assert supervisor.in_flight == 1, "the request is gone, the extraction is not"
        gate.set()
        left = await supervisor.drain(wait_seconds=2.0)
        assert left == 0
        assert supervisor.in_flight == 0
        assert task.done() and not task.cancelled()
        if outcome == "fail":
            assert isinstance(task.exception(), RuntimeError)
        else:
            assert task.result().status == "extracted"
    return logs


async def test_extraction_succeeds_after_the_request_was_cancelled() -> None:
    logs = await _run_with_cancelled_request("ok")
    assert not [log for log in logs if log["event"] == "memorize_extraction_failed"]


async def test_extraction_failure_after_cancel_is_retrieved_and_logged() -> None:
    logs = await _run_with_cancelled_request("fail")
    failed = [log for log in logs if log["event"] == "memorize_extraction_failed"]
    assert failed and failed[0]["memcell_ids"] == ["mc_1"]
    assert "extraction failed after disconnect" in failed[0]["error"]


async def test_lifespan_drains_running_extractions() -> None:
    service = make_memory_service()
    supervisor = service._extractions
    gate = asyncio.Event()

    async def extraction() -> Any:
        await gate.wait()
        return MemorizeResult(message_count=1, status="extracted")

    supervisor.supervise(extraction(), session_id="s", memcell_ids=["mc_2"])
    assert supervisor.in_flight == 1
    shutdown = asyncio.create_task(service.aclose())
    await asyncio.sleep(0.05)
    assert not shutdown.done(), "shutdown waits for the running extraction"
    gate.set()
    await asyncio.wait_for(shutdown, timeout=2.0)
    assert supervisor.in_flight == 0


async def test_extraction_is_tracked_with_the_purge_past_a_cancel(
    tmp_path: Any,
) -> None:
    """A purge of the scope must wait for an extraction whose request is gone:
    the extraction is registered (origin = the request's, scope = its
    payload's) from hand-off to end, cancel or not."""
    from datetime import timedelta
    from types import SimpleNamespace

    from everos.core.purge_marks import MARKS, ORIGIN, PurgeScope
    from everos.core.utils.datetime import get_utc_now

    marks = MARKS.for_root(tmp_path)
    service = make_memory_service(purge_marks=marks)
    memorizer = service._memorizer
    gate = asyncio.Event()

    async def cut(*_: Any, **__: Any) -> Any:
        return object(), SimpleNamespace(memcell_ids=["mc_1"])

    async def extract(*_: Any, **__: Any) -> Any:
        await gate.wait()
        return MemorizeResult(message_count=1, status="extracted")

    memorizer._cut = cut  # type: ignore[method-assign]
    memorizer._extract = extract  # type: ignore[method-assign]
    arrived = get_utc_now() - timedelta(seconds=5)
    s = PurgeScope("project", "a1", "p1", "a1", "p1")
    other = PurgeScope("project", "a2", "p1", "a2", "p1")
    later = get_utc_now() + timedelta(hours=1)

    async def request() -> Any:
        ORIGIN.set(arrived)  # what the write-tracking middleware sets
        payload = {"session_id": "s", "app_id": "a1", "project_id": "p1"}
        return await memorizer.run(payload)

    req = asyncio.create_task(request())
    for _ in range(50):
        if marks.tracked_before(later, s):
            break
        await asyncio.sleep(0.01)
    assert marks.tracked_before(later, s) == 1
    assert marks.tracked_before(arrived + timedelta(seconds=1), s) == 1
    assert marks.tracked_before(arrived, s) == 0  # its origin is the arrival
    assert marks.tracked_before(later, other) == 0  # its scope is the payload's
    req.cancel()
    with pytest.raises(asyncio.CancelledError):
        await req
    assert marks.tracked_before(later, s) == 1, "the request is gone, not the work"
    gate.set()
    await service._extractions.drain(wait_seconds=2.0)
    assert marks.tracked_before(later, s) == 0
    MARKS.reset()
