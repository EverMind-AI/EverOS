"""The extraction that outlives a cancelled memorize() call stays owned.

Once the boundary stage has committed a memcell, ``memorize()`` runs the
extraction shielded from the request's cancellation (a client that went
away). The task is handed to :class:`ExtractionSupervisor`, which keeps it,
retrieves its outcome, logs a failure with the memcell ids, and lets the
lifespan wait for what is still running at shutdown. Review on #521 pointed
out that the first version left the task unowned: a later failure was
"never retrieved" and nobody could know a memcell lost its memory.
"""

from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace
from typing import Any

import pytest
import structlog.testing

memorize_mod = importlib.import_module("everos.service.memorize")


async def _run_with_cancelled_request(outcome: str) -> list[dict[str, Any]]:
    """Cancel a call mid-extraction; let the extraction end the given way."""
    supervisor = memorize_mod.ExtractionSupervisor()
    gate = asyncio.Event()

    async def extraction() -> Any:
        await gate.wait()
        if outcome == "fail":
            raise RuntimeError("extraction failed after disconnect")
        return memorize_mod.MemorizeResult(message_count=1, status="extracted")

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
    from fastapi import FastAPI

    from everos.entrypoints.api.lifespans.memorize import MemorizeLifespanProvider

    supervisor = memorize_mod.extractions
    gate = asyncio.Event()

    async def extraction() -> Any:
        await gate.wait()
        return memorize_mod.MemorizeResult(message_count=1, status="extracted")

    supervisor.supervise(extraction(), session_id="s", memcell_ids=["mc_2"])
    assert supervisor.in_flight == 1
    provider = MemorizeLifespanProvider()
    shutdown = asyncio.create_task(provider.shutdown(SimpleNamespace()))  # type: ignore[arg-type]
    await asyncio.sleep(0.05)
    assert not shutdown.done(), "shutdown waits for the running extraction"
    gate.set()
    await asyncio.wait_for(shutdown, timeout=2.0)
    assert supervisor.in_flight == 0
    _ = FastAPI
