"""SQLite manager singletons.

Verifies ``get_engine`` / ``get_session_factory`` / ``dispose_engine``
are idempotent and rebuild after dispose.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.infra.persistence.sqlite import sqlite_manager


@pytest.fixture(autouse=True)
async def _reset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the singleton at an isolated memory-root and reset module state."""
    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(tmp_path))
    sqlite_manager._engine = None
    sqlite_manager._session_factory = None
    yield
    await sqlite_manager.dispose_engine()


async def test_get_engine_is_singleton(tmp_path: Path) -> None:
    e1 = sqlite_manager.get_engine()
    e2 = sqlite_manager.get_engine()
    assert e1 is e2
    # Engine points at the redirected memory root.
    assert str(tmp_path) in str(e1.url)


async def test_get_session_factory_is_singleton() -> None:
    f1 = sqlite_manager.get_session_factory()
    f2 = sqlite_manager.get_session_factory()
    assert f1 is f2


async def test_dispose_resets_state() -> None:
    e1 = sqlite_manager.get_engine()
    await sqlite_manager.dispose_engine()
    assert sqlite_manager._engine is None
    assert sqlite_manager._session_factory is None
    e2 = sqlite_manager.get_engine()
    assert e2 is not e1


async def test_dispose_is_idempotent() -> None:
    await sqlite_manager.dispose_engine()  # nothing built yet
    sqlite_manager.get_engine()
    await sqlite_manager.dispose_engine()
    await sqlite_manager.dispose_engine()  # second call must not raise
