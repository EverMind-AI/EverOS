"""OmeLifespanProvider — startup wires engine, shutdown stops it."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi import FastAPI

from everos.entrypoints.api.lifespans import OmeLifespanProvider


async def test_lifespan_starts_and_stops_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from everos.core.persistence import MemoryRoot

    svc = importlib.import_module("everos.service.memorize")

    monkeypatch.setattr(
        MemoryRoot, "default", classmethod(lambda cls: MemoryRoot(root=tmp_path))
    )
    (tmp_path / "ome.toml").write_text("# test\n")
    monkeypatch.setattr(svc, "_ome_engine", None, raising=False)

    provider = OmeLifespanProvider()
    app = FastAPI()

    engine = await provider.startup(app)
    assert engine is not None
    assert engine._started is True

    await provider.shutdown(app)
    assert engine._started is False
