from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from everos.infra.ome.engine import OfflineEngine

# everos.service.__init__ re-exports `memorize` as a function attribute on
# the package, which shadows the submodule when using `import ... as svc`.
# importlib.import_module bypasses that shadowing and returns the real module.
_svc = importlib.import_module("everos.service.memorize")


async def test_get_engine_returns_singleton(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from everos.core.persistence import MemoryRoot

    monkeypatch.setattr(
        MemoryRoot, "default", classmethod(lambda cls: MemoryRoot(root=tmp_path))
    )
    monkeypatch.setattr(_svc, "_ome_engine", None, raising=False)

    engine1 = _svc._get_engine()
    engine2 = _svc._get_engine()
    assert isinstance(engine1, OfflineEngine)
    assert engine1 is engine2


async def test_get_user_pipeline_injects_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from everos.core.persistence import MemoryRoot

    monkeypatch.setattr(
        MemoryRoot, "default", classmethod(lambda cls: MemoryRoot(root=tmp_path))
    )
    monkeypatch.setattr(_svc, "_ome_engine", None, raising=False)
    monkeypatch.setattr(_svc, "_user_pipeline", None, raising=False)
    monkeypatch.setattr(_svc, "get_llm_client", lambda: object())

    pipeline = _svc._get_user_pipeline()
    assert pipeline._engine is _svc._get_engine()
