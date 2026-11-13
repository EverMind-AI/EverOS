"""``[purge]`` settings: shipped defaults, env binding, positive bounds."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from everos.config import PurgeSettings, load_settings


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in list(os.environ):
        if key.startswith("EVEROS_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    load_settings.cache_clear()


def test_purge_defaults_from_shipped_toml() -> None:
    purge = load_settings().purge
    assert (
        purge.run_budget_seconds,
        purge.nonce_ttl_seconds,
        purge.count_timeout_seconds,
    ) == (7200.0, 300.0, 30.0)


def test_purge_env_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVEROS_PURGE__NONCE_TTL_SECONDS", "2")
    assert load_settings().purge.nonce_ttl_seconds == 2.0


@pytest.mark.parametrize(
    "field",
    [
        "run_budget_seconds",
        "nonce_ttl_seconds",
        "count_timeout_seconds",
    ],
)
def test_purge_bounds_reject_zero(field: str) -> None:
    with pytest.raises(ValidationError):
        PurgeSettings(**{field: 0})
