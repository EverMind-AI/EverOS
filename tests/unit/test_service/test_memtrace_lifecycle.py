"""Application shutdown performs the tracker's final markdown diff and export."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from everos.service import bootstrap


class _StubTracker:
    def __init__(self) -> None:
        self.sync_calls = 0

    async def sync_now(self) -> None:
        self.sync_calls += 1


async def test_application_close_syncs_memtrace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _StubTracker()
    monkeypatch.setattr(bootstrap, "shutdown_tracing", lambda: None)
    app = bootstrap.Application.__new__(bootstrap.Application)
    object.__setattr__(app, "_memtrace_tracker", tracker)
    object.__setattr__(app, "_stack", SimpleNamespace(aclose=AsyncMock()))
    object.__setattr__(
        app,
        "_admin",
        SimpleNamespace(close_index=AsyncMock(), dispose=AsyncMock()),
    )
    object.__setattr__(app, "_root_lock", SimpleNamespace(release=lambda: None))

    await app._close_components()

    assert tracker.sync_calls == 1
