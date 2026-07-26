"""service.search._get_llm_client — token-usage wrapping parity with memorize.

The hybrid/agentic search path is the most LLM-heavy flow (query
decomposition + refine + per-query fan-out + rerank judge). Its client
must be wrapped with ``UsageRecordingClient`` when observability is on,
exactly like ``memorize`` / the reflection strategies — otherwise the
biggest token spend is invisible in Langfuse.
"""

from __future__ import annotations

import importlib

import pytest
from pydantic import SecretStr

import everos.config as config_mod
from everos.component.llm._usage_client import UsageRecordingClient
from everos.config import Settings
from everos.config.settings import LLMSettings, ObservabilitySettings

# `everos.service.search` the submodule is shadowed by the re-exported
# `search` function on the package, so resolve the module explicitly.
search_mod = importlib.import_module("everos.service.search")


def _reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search_mod, "_llm_client", None, raising=False)
    monkeypatch.setattr(search_mod, "_llm_resolved", False, raising=False)


def _patch_settings(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> None:
    cfg = Settings(
        llm=LLMSettings(
            model="gpt-4.1-mini",
            api_key=SecretStr("sk-test"),
            base_url="https://example.test",
        ),
        observability=ObservabilitySettings(enabled=enabled),
    )
    monkeypatch.setattr(config_mod, "load_settings", lambda: cfg)


def test_search_llm_wrapped_when_observability_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset(monkeypatch)
    _patch_settings(monkeypatch, enabled=True)

    client = search_mod._get_llm_client()

    assert isinstance(client, UsageRecordingClient)


def test_search_llm_not_wrapped_when_observability_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset(monkeypatch)
    _patch_settings(monkeypatch, enabled=False)

    client = search_mod._get_llm_client()

    assert client is not None
    assert not isinstance(client, UsageRecordingClient)


def test_search_llm_none_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    # No credentials → graceful None (keyword-only degradation), regardless
    # of observability. The wrapper must not change this contract.
    _reset(monkeypatch)
    cfg = Settings(
        llm=LLMSettings(model="m", api_key=None, base_url=None),
        observability=ObservabilitySettings(enabled=True),
    )
    monkeypatch.setattr(config_mod, "load_settings", lambda: cfg)

    assert search_mod._get_llm_client() is None
