"""``MultimodalLLMCapability`` + ``get_multimodal_llm_capability`` accessor.

Contract: MultimodalLLMCapability mirrors RerankCapability exactly — 2 methods
(available + require), frozen dataclass, no soft-degrade variant.
"""

from __future__ import annotations

import importlib

import pytest
from pydantic import SecretStr

from everos.component.multimodal import (
    MultimodalLLMCapability,
    get_multimodal_llm_capability,
)
from everos.config import Settings
from everos.config.settings import MultimodalSettings
from everos.core.errors import ProviderNotConfiguredError

_accessor_mod = importlib.import_module("everos.component.multimodal.accessor")
_llm_client_mod = importlib.import_module("everos.component.llm.client")


class _MockMultimodalLLM:
    async def chat(self, messages):
        return {}


def _reset_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_accessor_mod, "_capability", None, raising=False)
    monkeypatch.setattr(_llm_client_mod, "_multimodal_client", None, raising=False)


def _patch_settings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> None:
    kwargs = {"api_key": SecretStr(api_key) if api_key is not None else None}
    if base_url is not None:
        kwargs["base_url"] = base_url
    else:
        kwargs["base_url"] = None
    if model is not None:
        kwargs["model"] = model
    cfg = Settings(multimodal=MultimodalSettings(**kwargs))
    monkeypatch.setattr(_llm_client_mod, "load_settings", lambda: cfg)


def test_capability_available_false_when_provider_none() -> None:
    cap = MultimodalLLMCapability(provider=None)
    assert cap.available is False


def test_capability_available_true_when_provider_present() -> None:
    cap = MultimodalLLMCapability(provider=_MockMultimodalLLM())
    assert cap.available is True


def test_require_raises_when_unavailable() -> None:
    cap = MultimodalLLMCapability(provider=None)
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        cap.require()
    assert excinfo.value.provider == "multimodal_llm"


def test_require_returns_provider_when_available() -> None:
    provider = _MockMultimodalLLM()
    cap = MultimodalLLMCapability(provider=provider)
    assert cap.require() is provider


def test_capability_is_frozen() -> None:
    cap = MultimodalLLMCapability(provider=None)
    with pytest.raises((AttributeError, TypeError)):
        cap.provider = _MockMultimodalLLM()  # type: ignore[misc]


def test_capability_has_no_soft_degrade_method() -> None:
    """Multimodal LLM has no soft-degrade write path, unlike
    EmbeddingCapability's ``embed_or_none`` — callers either require it or
    skip multimodal parsing entirely."""
    cap = MultimodalLLMCapability(provider=None)
    assert not hasattr(cap, "embed_or_none")
    assert not hasattr(cap, "parse_or_none")


def test_get_multimodal_llm_capability_empty_when_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_capability(monkeypatch)
    _patch_settings(monkeypatch, model=None, api_key=None, base_url=None)

    cap = get_multimodal_llm_capability()

    assert isinstance(cap, MultimodalLLMCapability)
    assert cap.available is False


def test_get_multimodal_llm_capability_populated_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_capability(monkeypatch)
    _patch_settings(
        monkeypatch, model="m", api_key="sk-test", base_url="https://example.test"
    )

    cap = get_multimodal_llm_capability()

    assert cap.available is True


def test_get_multimodal_llm_capability_caches_singleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_capability(monkeypatch)
    _patch_settings(monkeypatch, model=None, api_key=None, base_url=None)

    first = get_multimodal_llm_capability()
    second = get_multimodal_llm_capability()

    assert first is second
