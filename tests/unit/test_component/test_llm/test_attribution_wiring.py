"""Attribution headers reach the wire, and only for aimlapi.com.

The header helper is unit-tested separately; what breaks silently is the
*wiring* — a client built without the headers still works, just
unattributed, so nothing fails loudly. These tests pin that each client
construction site actually forwards them, and that a non-aimlapi
``base_url`` adds no request key at all (a stray ``extra_headers`` or a
``None`` valued key is a 400 on some upstreams).
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest
from pydantic import SecretStr

from everos.component.llm.openai_provider import OpenAIProvider
from everos.config import Settings
from everos.config.settings import LLMSettings, MultimodalSettings

_client_mod = importlib.import_module("everos.component.llm.client")

_AIMLAPI = "https://api.aimlapi.com/v1"
_OPENROUTER = "https://openrouter.ai/api/v1"
_PARTNER_ID = "part_VxTyAUvoIVbl30dPrB7kbRZk"


def _recorder(captured: dict[str, Any]) -> Any:
    """Return a ``build_client`` stub that records the config it is given."""

    def _build(cfg: Any) -> object:
        captured["cfg"] = cfg
        return object()

    return _build


def _capture_llm_config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    base_url: str,
    config_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the LLM singleton against ``base_url`` and return its config."""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(_client_mod, "_llm_client", None, raising=False)
    monkeypatch.setattr(
        _client_mod,
        "load_settings",
        lambda: Settings(
            llm=LLMSettings(
                model="openai/gpt-4.1-mini",
                api_key=SecretStr("sk-test"),
                base_url=base_url,
                extra=config_extra or {},
            )
        ),
    )
    monkeypatch.setattr(_client_mod, "build_client", _recorder(captured))
    _client_mod.get_llm_client()
    return captured["cfg"].extra


def _capture_multimodal_config(
    monkeypatch: pytest.MonkeyPatch, *, base_url: str
) -> dict[str, Any]:
    """Build the multimodal singleton and return its config ``extra``."""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(_client_mod, "_multimodal_client", None, raising=False)
    monkeypatch.setattr(
        _client_mod,
        "load_settings",
        lambda: Settings(
            multimodal=MultimodalSettings(
                model="google/gemini-3.8-flash",
                api_key=SecretStr("sk-test"),
                base_url=base_url,
            )
        ),
    )
    monkeypatch.setattr(_client_mod, "build_client", _recorder(captured))
    _client_mod.get_multimodal_llm_client()
    return captured["cfg"].extra


def test_llm_client_sends_attribution_to_aimlapi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extra = _capture_llm_config(monkeypatch, base_url=_AIMLAPI)
    assert extra["extra_headers"]["X-AIMLAPI-Partner-ID"] == _PARTNER_ID
    assert extra["extra_headers"]["X-AIMLAPI-Source"] == "agent/everos"


def test_llm_client_adds_no_request_key_for_other_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _capture_llm_config(monkeypatch, base_url=_OPENROUTER) == {}


def test_llm_client_config_extra_wins_over_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key set in ``[llm].extra`` overrides the attribution value."""
    extra = _capture_llm_config(
        monkeypatch,
        base_url=_AIMLAPI,
        config_extra={"extra_headers": {"X-Custom": "user"}},
    )
    assert extra["extra_headers"] == {"X-Custom": "user"}


def test_multimodal_client_sends_attribution_to_aimlapi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extra = _capture_multimodal_config(monkeypatch, base_url=_AIMLAPI)
    assert extra["extra_headers"]["X-AIMLAPI-Partner-ID"] == _PARTNER_ID


def test_multimodal_client_adds_no_request_key_for_other_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _capture_multimodal_config(monkeypatch, base_url=_OPENROUTER) == {}


def test_openai_provider_sets_default_headers_for_aimlapi() -> None:
    provider = OpenAIProvider(model="m", api_key="sk-test", base_url=_AIMLAPI)
    sent = provider._client.default_headers
    assert sent["X-AIMLAPI-Partner-ID"] == _PARTNER_ID
    # The SDK's own defaults survive — the partner headers merge in.
    assert "Content-Type" in sent


def test_openai_provider_sends_no_partner_headers_elsewhere() -> None:
    provider = OpenAIProvider(model="m", api_key="sk-test", base_url=_OPENROUTER)
    assert "X-AIMLAPI-Partner-ID" not in provider._client.default_headers
