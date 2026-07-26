"""get_llm_client — raises on missing credentials, caches on success."""

from __future__ import annotations

import importlib

import pytest
from pydantic import SecretStr

from everos.component.llm import LLMNotConfiguredError
from everos.component.llm._usage_client import UsageRecordingClient
from everos.config import Settings
from everos.config.settings import LLMSettings, ObservabilitySettings

_client_mod = importlib.import_module("everos.component.llm.client")


def _reset_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_client_mod, "_llm_client", None, raising=False)


def _patch_settings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    api_key: str | None,
    base_url: str | None,
) -> None:
    """Stub the ``load_settings`` reference bound inside the client module."""
    cfg = Settings(
        llm=LLMSettings(
            model="gpt-4.1-mini",
            api_key=SecretStr(api_key) if api_key is not None else None,
            base_url=base_url,
        )
    )
    monkeypatch.setattr(_client_mod, "load_settings", lambda: cfg)


def test_raises_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_singleton(monkeypatch)
    _patch_settings(monkeypatch, api_key=None, base_url="https://example.test")

    with pytest.raises(LLMNotConfiguredError, match="EVEROS_LLM__API_KEY"):
        _client_mod.get_llm_client()


def test_raises_when_base_url_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_singleton(monkeypatch)
    _patch_settings(monkeypatch, api_key="sk-test", base_url=None)

    with pytest.raises(LLMNotConfiguredError, match="EVEROS_LLM__BASE_URL"):
        _client_mod.get_llm_client()


def test_returns_singleton_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_singleton(monkeypatch)
    _patch_settings(monkeypatch, api_key="sk-test", base_url="https://example.test")
    sentinel = object()
    monkeypatch.setattr(_client_mod, "build_client", lambda cfg: sentinel)

    first = _client_mod.get_llm_client()
    second = _client_mod.get_llm_client()

    assert first is sentinel
    assert first is second


def _patch_settings_with_observability(
    monkeypatch: pytest.MonkeyPatch, *, enabled: bool
) -> None:
    cfg = Settings(
        llm=LLMSettings(
            model="gpt-4.1-mini",
            api_key=SecretStr("sk-test"),
            base_url="https://example.test",
        ),
        observability=ObservabilitySettings(enabled=enabled),
    )
    monkeypatch.setattr(_client_mod, "load_settings", lambda: cfg)


def test_wraps_client_when_observability_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_singleton(monkeypatch)
    _patch_settings_with_observability(monkeypatch, enabled=True)
    sentinel = object()
    monkeypatch.setattr(_client_mod, "build_client", lambda cfg: sentinel)

    client = _client_mod.get_llm_client()

    assert isinstance(client, UsageRecordingClient)
    assert client._inner is sentinel


def test_does_not_wrap_client_when_observability_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_singleton(monkeypatch)
    _patch_settings_with_observability(monkeypatch, enabled=False)
    sentinel = object()
    monkeypatch.setattr(_client_mod, "build_client", lambda cfg: sentinel)

    assert _client_mod.get_llm_client() is sentinel
