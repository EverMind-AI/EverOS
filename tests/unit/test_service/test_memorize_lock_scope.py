"""The memorize() lock is keyed by (app_id, project_id, session_id).

Buffers and memcells are scoped by app / project, so two sessions that
merely share a ``session_id`` across projects must not wait on each other;
two calls on the very same scope still serialise.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Iterator
from typing import Any

import pytest

from everos.config import Settings
from everos.service import _session_lock
from everos.service.memorize import MemorizeResult

mm = importlib.import_module("everos.service.memorize")


class _Gate:
    """``_memorize_locked`` stand-in: enters, waits for release, exits."""

    def __init__(self) -> None:
        self.inside: list[str] = []
        self.release = asyncio.Event()
        self.entered = asyncio.Event()

    async def __call__(self, payload: dict[str, Any], **_: Any) -> MemorizeResult:
        key = f"{payload.get('project_id', 'default')}/{payload['session_id']}"
        self.inside.append(key)
        self.entered.set()
        await self.release.wait()
        self.inside.remove(key)
        return MemorizeResult(message_count=1, status="extracted")


@pytest.fixture(autouse=True)
def _patch(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Gate]:
    monkeypatch.setattr(mm, "load_settings", lambda: Settings())
    gate = _Gate()
    monkeypatch.setattr(mm, "_memorize_locked", gate)
    _session_lock._reset_for_tests()
    yield gate
    _session_lock._reset_for_tests()


async def test_same_session_in_different_projects_runs_side_by_side(
    _patch: _Gate,
) -> None:
    a = asyncio.create_task(
        mm.memorize({"session_id": "s", "project_id": "p1", "messages": []})
    )
    b = asyncio.create_task(
        mm.memorize({"session_id": "s", "project_id": "p2", "messages": []})
    )
    await asyncio.sleep(0.05)
    assert sorted(_patch.inside) == ["p1/s", "p2/s"]  # both inside at once
    _patch.release.set()
    await asyncio.gather(a, b)


async def test_same_scope_still_serialises(_patch: _Gate) -> None:
    a = asyncio.create_task(
        mm.memorize({"session_id": "s", "project_id": "p1", "messages": []})
    )
    b = asyncio.create_task(
        mm.memorize({"session_id": "s", "project_id": "p1", "messages": []})
    )
    await asyncio.sleep(0.05)
    assert _patch.inside == ["p1/s"]  # the second waits on the lock
    _patch.release.set()
    await asyncio.gather(a, b)
