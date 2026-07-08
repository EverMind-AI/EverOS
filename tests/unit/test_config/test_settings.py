"""Unit tests for Settings loading (everos.toml-based)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from everos.config import Settings, load_settings
from everos.config.settings import resolve_root


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Strip EVEROS_* env vars and pin root to tmp_path so no external
    everos.toml is ever discovered."""
    for key in list(os.environ):
        if key.startswith("EVEROS_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    load_settings.cache_clear()


def test_load_settings_defaults_from_shipped_toml() -> None:
    s = load_settings()
    assert s.memory.timezone == "UTC"
    assert s.sqlite.journal_mode == "WAL"
    assert s.sqlite.synchronous == "NORMAL"
    assert s.sqlite.busy_timeout_ms == 5000
    assert s.api.host == "127.0.0.1"
    assert s.api.port == 8000


def test_everos_toml_overrides_defaults(tmp_path: Path) -> None:
    """<root>/everos.toml overrides shipped default.toml values."""
    root = tmp_path / "myroot"
    root.mkdir()
    (root / "everos.toml").write_text(
        '[sqlite]\nbusy_timeout_ms = 7777\n[memory]\ntimezone = "Asia/Tokyo"\n',
        encoding="utf-8",
    )
    s = Settings(_everos_root=root)
    assert s.sqlite.busy_timeout_ms == 7777
    assert s.memory.timezone == "Asia/Tokyo"
    assert s.sqlite.journal_mode == "WAL"  # untouched → default


def test_env_var_overrides_everos_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """EVEROS_* env vars beat everos.toml."""
    root = tmp_path / "myroot"
    root.mkdir()
    (root / "everos.toml").write_text(
        "[sqlite]\nbusy_timeout_ms = 7777\n", encoding="utf-8"
    )
    monkeypatch.setenv("EVEROS_SQLITE__BUSY_TIMEOUT_MS", "9999")
    s = Settings(_everos_root=root)
    assert s.sqlite.busy_timeout_ms == 9999


def test_no_everos_toml_uses_defaults_only(tmp_path: Path) -> None:
    """Missing everos.toml is not an error — falls back to defaults."""
    s = Settings(_everos_root=tmp_path)
    assert s.sqlite.busy_timeout_ms == 5000


def test_env_overrides_toml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVEROS_SQLITE__BUSY_TIMEOUT_MS", "10000")
    monkeypatch.setenv("EVEROS_SQLITE__JOURNAL_MODE", "DELETE")
    s = Settings()
    assert s.sqlite.busy_timeout_ms == 10000
    assert s.sqlite.journal_mode == "DELETE"
    assert s.sqlite.synchronous == "NORMAL"


def test_init_args_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVEROS_SQLITE__BUSY_TIMEOUT_MS", "10000")
    from everos.config.settings import SqliteSettings

    s = Settings(sqlite=SqliteSettings(busy_timeout_ms=99999))
    assert s.sqlite.busy_timeout_ms == 99999


def test_invalid_journal_mode_rejected() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings.model_validate({"sqlite": {"journal_mode": "BOGUS"}})


def test_negative_busy_timeout_rejected() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings.model_validate({"sqlite": {"busy_timeout_ms": -1}})


def test_load_settings_is_cached() -> None:
    a = load_settings()
    b = load_settings()
    assert a is b
    load_settings.cache_clear()
    c = load_settings()
    assert c is not a


def test_embedding_rerank_defaults() -> None:
    s = Settings()
    assert s.embedding.model == "Qwen/Qwen3-Embedding-4B"
    assert s.embedding.base_url == "https://api.deepinfra.com/v1/openai"
    assert s.embedding.api_key.get_secret_value() == ""
    assert s.embedding.timeout_seconds == 30.0
    assert s.rerank.model == "Qwen/Qwen3-Reranker-4B"
    assert s.rerank.base_url == "https://api.deepinfra.com/v1/inference"
    assert s.rerank.api_key.get_secret_value() == ""
    assert s.rerank.timeout_seconds == 30.0
    assert s.llm.api_key.get_secret_value() == ""


def test_resolve_root_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """No --root, no EVEROS_ROOT → ~/.everos."""
    monkeypatch.delenv("EVEROS_ROOT", raising=False)
    assert resolve_root() == Path("~/.everos").expanduser().resolve()


def test_resolve_root_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVEROS_ROOT", "/data/everos")
    assert resolve_root() == Path("/data/everos").resolve()


def test_resolve_root_explicit() -> None:
    assert resolve_root("/custom/root") == Path("/custom/root").resolve()
