"""Unit tests for Settings loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.config import Settings, load_settings


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip any EVEROS_* env vars from the host so tests are deterministic."""
    for key in list(__import__("os").environ):
        if key.startswith("EVEROS_"):
            monkeypatch.delenv(key, raising=False)
    load_settings.cache_clear()


def test_load_settings_defaults_from_toml() -> None:
    s = load_settings()
    # Values straight out of config/default.toml
    assert s.memory.root == Path("~/.everos")
    assert s.memory.timezone == "UTC"
    assert s.sqlite.journal_mode == "WAL"
    assert s.sqlite.synchronous == "NORMAL"
    assert s.sqlite.foreign_keys is True
    assert s.sqlite.temp_store == "MEMORY"
    assert s.sqlite.busy_timeout_ms == 5000
    assert s.sqlite.journal_size_limit_bytes == 64 * 1024 * 1024
    assert s.sqlite.cache_size_kb == 2048
    assert s.lancedb.read_consistency_seconds is None


def test_env_overrides_toml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVEROS_SQLITE__BUSY_TIMEOUT_MS", "10000")
    monkeypatch.setenv("EVEROS_SQLITE__JOURNAL_MODE", "DELETE")
    s = Settings()
    assert s.sqlite.busy_timeout_ms == 10000
    assert s.sqlite.journal_mode == "DELETE"
    # Untouched values stay at TOML defaults.
    assert s.sqlite.synchronous == "NORMAL"


def test_init_args_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVEROS_SQLITE__BUSY_TIMEOUT_MS", "10000")
    from everos.config.settings import SqliteSettings

    s = Settings(sqlite=SqliteSettings(busy_timeout_ms=99999))
    assert s.sqlite.busy_timeout_ms == 99999  # init beats env


def test_invalid_journal_mode_rejected() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings.model_validate({"sqlite": {"journal_mode": "BOGUS"}})


def test_negative_busy_timeout_rejected() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings.model_validate({"sqlite": {"busy_timeout_ms": -1}})


def test_lancedb_read_consistency_optional_float() -> None:
    s = Settings.model_validate({"lancedb": {"read_consistency_seconds": 5.0}})
    assert s.lancedb.read_consistency_seconds == 5.0
    s2 = Settings.model_validate({"lancedb": {"read_consistency_seconds": None}})
    assert s2.lancedb.read_consistency_seconds is None


def test_memory_timezone_overridable_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVEROS_MEMORY__TIMEZONE", "Asia/Shanghai")
    s = Settings()
    assert s.memory.timezone == "Asia/Shanghai"


def test_memory_timezone_invalid_rejected() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="invalid timezone"):
        Settings.model_validate({"memory": {"timezone": "Not/A/Real_Zone"}})


def test_load_settings_is_cached() -> None:
    """Repeated calls return the same Settings object until cache_clear."""
    a = load_settings()
    b = load_settings()
    assert a is b
    load_settings.cache_clear()
    c = load_settings()
    assert c is not a


def test_embedding_rerank_defaults() -> None:
    """Embedding / rerank ship with runtime knobs but no model credentials."""
    # ``_isolate_env`` already strips shell env; ``_env_file=None`` further
    # prevents a developer's ``.env`` (which typically sets MODEL / API_KEY /
    # BASE_URL for live runs) from leaking into this default-state check.
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    # Credentials must be set explicitly (no default).
    assert s.embedding.model is None
    assert s.embedding.api_key is None
    assert s.embedding.base_url is None
    # Runtime knobs come from default.toml.
    assert s.embedding.timeout_seconds == 30.0
    assert s.embedding.max_retries == 3
    assert s.embedding.batch_size == 10
    assert s.embedding.max_concurrent == 5
    # Rerank mirrors the shape.
    assert s.rerank.model is None
    assert s.rerank.timeout_seconds == 30.0
    assert s.rerank.batch_size == 10


def test_embedding_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVEROS_EMBEDDING__MODEL", "intfloat/e5-large-v2")
    monkeypatch.setenv("EVEROS_EMBEDDING__BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("EVEROS_EMBEDDING__BATCH_SIZE", "32")
    s = Settings()
    assert s.embedding.model == "intfloat/e5-large-v2"
    assert s.embedding.base_url == "http://localhost:8000/v1"
    assert s.embedding.batch_size == 32


def test_rerank_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVEROS_RERANK__MODEL", "BAAI/bge-reranker-v2-m3")
    monkeypatch.setenv("EVEROS_RERANK__MAX_CONCURRENT", "8")
    s = Settings()
    assert s.rerank.model == "BAAI/bge-reranker-v2-m3"
    assert s.rerank.max_concurrent == 8


def test_dashscope_one_key_can_configure_llm_embedding_and_rerank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One DashScope key value can be reused across all three clients."""
    key = "sk-dashscope"
    compatible_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    monkeypatch.setenv("EVEROS_LLM__MODEL", "qwen-plus")
    monkeypatch.setenv("EVEROS_LLM__API_KEY", key)
    monkeypatch.setenv("EVEROS_LLM__BASE_URL", compatible_base_url)
    monkeypatch.setenv("EVEROS_EMBEDDING__MODEL", "text-embedding-v4")
    monkeypatch.setenv("EVEROS_EMBEDDING__API_KEY", key)
    monkeypatch.setenv("EVEROS_EMBEDDING__BASE_URL", compatible_base_url)
    monkeypatch.setenv("EVEROS_RERANK__PROVIDER", "dashscope")
    monkeypatch.setenv("EVEROS_RERANK__MODEL", "gte-rerank-v2")
    monkeypatch.setenv("EVEROS_RERANK__API_KEY", key)
    monkeypatch.setenv("EVEROS_RERANK__BASE_URL", "https://dashscope.aliyuncs.com")

    s = Settings()

    assert s.llm.api_key is not None
    assert s.embedding.api_key is not None
    assert s.rerank.api_key is not None
    assert s.llm.api_key.get_secret_value() == key
    assert s.embedding.api_key.get_secret_value() == key
    assert s.rerank.api_key.get_secret_value() == key
    assert s.llm.base_url == compatible_base_url
    assert s.embedding.base_url == compatible_base_url
    assert s.rerank.provider == "dashscope"

    from everos.component.embedding import (
        OpenAIEmbeddingProvider,
        build_embedding_provider,
    )
    from everos.component.llm import build_llm_provider
    from everos.component.llm.openai_provider import OpenAIProvider
    from everos.component.rerank import (
        DashScopeRerankProvider,
        build_rerank_provider,
    )

    assert isinstance(build_llm_provider(s.llm), OpenAIProvider)
    assert isinstance(build_embedding_provider(s.embedding), OpenAIEmbeddingProvider)
    assert isinstance(build_rerank_provider(s.rerank), DashScopeRerankProvider)


def test_user_toml_override_via_env_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``EVEROS_CONFIG_FILE`` points pydantic-settings at a user toml."""
    user_toml = tmp_path / "config.toml"
    user_toml.write_text(
        '[sqlite]\nbusy_timeout_ms = 7777\n[memory]\ntimezone = "Asia/Tokyo"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("EVEROS_CONFIG_FILE", str(user_toml))
    s = Settings()
    assert s.sqlite.busy_timeout_ms == 7777
    assert s.memory.timezone == "Asia/Tokyo"
    # Values not touched by the user toml still come from the shipped default.
    assert s.sqlite.journal_mode == "WAL"


def test_user_toml_loses_to_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """env vars beat the user-level toml."""
    user_toml = tmp_path / "config.toml"
    user_toml.write_text("[sqlite]\nbusy_timeout_ms = 7777\n", encoding="utf-8")
    monkeypatch.setenv("EVEROS_CONFIG_FILE", str(user_toml))
    monkeypatch.setenv("EVEROS_SQLITE__BUSY_TIMEOUT_MS", "9999")
    s = Settings()
    assert s.sqlite.busy_timeout_ms == 9999


def test_user_toml_missing_file_is_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-existent user toml path is silently skipped, not an error."""
    monkeypatch.setenv("EVEROS_CONFIG_FILE", str(tmp_path / "nope.toml"))
    s = Settings()
    # Falls back to shipped defaults.
    assert s.sqlite.busy_timeout_ms == 5000
