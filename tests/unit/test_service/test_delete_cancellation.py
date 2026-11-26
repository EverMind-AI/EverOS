"""A cancelled /memory/delete still finishes its input-log cleanup."""

from __future__ import annotations

import asyncio

import pytest

from everos.service.memory._delete import DeleteResult
from tests.fakes import make_memory_service


class _SlowDeleter:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def delete(self, **_: object) -> DeleteResult:
        self.started.set()
        await self.release.wait()
        return DeleteResult(filters=["session_id"], count=0)


class _RecordingInputLog:
    def __init__(self) -> None:
        self.cleaned = asyncio.Event()
        self.owner_cleaned = asyncio.Event()
        self.session_cleaned = asyncio.Event()

    async def delete_session(self, **_: object) -> None:
        self.cleaned.set()
        self.session_cleaned.set()

    async def delete_owner(self, **_: object) -> None:
        self.owner_cleaned.set()


async def test_cancelled_delete_still_cleans_the_session_log() -> None:
    service = make_memory_service()
    deleter = _SlowDeleter()
    input_log = _RecordingInputLog()
    service._deleter = deleter  # type: ignore[attr-defined]
    service._input_log = input_log  # type: ignore[attr-defined]

    task = asyncio.create_task(
        service.delete(
            app_id="a",
            project_id="p",
            user_id=None,
            agent_id=None,
            session_id="s1",
        )
    )
    await deleter.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    deleter.release.set()
    await asyncio.wait_for(input_log.cleaned.wait(), timeout=1.0)


class _FastDeleter:
    async def delete(self, **_: object) -> DeleteResult:
        return DeleteResult(filters=["user_id"], count=0)


async def test_owner_delete_cleans_the_owner_log() -> None:
    service = make_memory_service()
    service._deleter = _FastDeleter()  # type: ignore[attr-defined]
    input_log = _RecordingInputLog()
    service._input_log = input_log  # type: ignore[attr-defined]

    await service.delete(
        app_id="a",
        project_id="p",
        user_id="u1",
        agent_id=None,
        session_id=None,
    )

    assert input_log.owner_cleaned.is_set()
    assert not input_log.session_cleaned.is_set()
