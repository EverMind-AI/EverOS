"""A refused server startup must leave an active owner's cascade claims alone.

Real API lifespan, SQLite, LanceDB and file locks; model providers fail if called.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from everos.config import Settings
from everos.entrypoints.api.app import create_app
from everos.infra.ome import OmeLock
from everos.infra.ome.exceptions import EngineLockHeldError
from everos.infra.persistence import MemoryRoot
from everos.service import build_application
from tests.fakes import make_providers, make_storage


async def test_refused_startup_preserves_an_active_cascade_claim(
    tmp_path: Path,
) -> None:
    settings = Settings()
    root = MemoryRoot(tmp_path)
    root.ensure()
    storage = make_storage(root, settings)
    path = "default_app/default_project/users/u1/episodes/episode-2026-11-06.md"
    try:
        await storage.create_schema()
        ledger = storage.sqlite.md_change_state
        await ledger.upsert(path, kind="episode", change_type="modified", mtime=1.0)
        claimed = await ledger.claim_one(path)
        assert claimed is not None and claimed.status == "processing"
        application = build_application(settings, providers=make_providers())
        api = create_app(settings, application=application)
        with OmeLock(root.ome_db).hold():
            with pytest.raises(EngineLockHeldError):
                async with api.router.lifespan_context(api):
                    pytest.fail("a second server must not start on the same root")
            current = await ledger.read_one(path)
            assert current is not None
            assert current.model_dump() == claimed.model_dump()
    finally:
        await storage.aclose()


async def test_server_holds_the_root_until_cascade_and_storage_close(
    tmp_path: Path,
) -> None:
    """OME stops first; it must not release the application's borrowed lock."""
    import dataclasses

    (tmp_path / "ome.toml").write_text("# Initialized test root\n", encoding="utf-8")

    settings = Settings()
    application = build_application(settings, providers=make_providers())
    original_ops = application.ops
    original_admin = application._admin
    probe = OmeLock(MemoryRoot(tmp_path).ome_db)
    checked: list[tuple[str, bool]] = []

    class ClosingOps:
        async def start(self) -> None:
            await original_ops.start()

        async def aclose(self) -> None:
            checked.append(("cascade", probe.is_free()))
            await original_ops.aclose()

    class ClosingAdmin:
        def __getattr__(self, name: str) -> Any:
            return getattr(original_admin, name)

        async def dispose(self) -> None:
            checked.append(("storage", probe.is_free()))
            await original_admin.dispose()

    application = dataclasses.replace(
        application,
        ops=ClosingOps(),
        _admin=ClosingAdmin(),
    )
    api = create_app(settings, application=application)
    async with api.router.lifespan_context(api):
        assert not probe.is_free()
    assert checked == [("cascade", False), ("storage", False)]
    assert probe.is_free()
