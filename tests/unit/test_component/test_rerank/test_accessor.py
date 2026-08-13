"""``RerankCapability`` + ``get_rerank_capability`` accessor."""

from __future__ import annotations

import importlib

import pytest
import structlog.testing
from pydantic import SecretStr

from everos.component.rerank import RerankCapability, get_rerank_capability
from everos.config import Settings
from everos.config.settings import RerankSettings
from everos.core.errors import ProviderNotConfiguredError

_accessor_mod = importlib.import_module("everos.component.rerank.accessor")


class _MockReranker:
    async def rerank(self, query, documents, *, instruction=None):
        return []


def _reset_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_accessor_mod, "_capability", None, raising=False)


def _patch_settings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
) -> None:
    cfg = Settings(
        rerank=RerankSettings(
            model=model,
            api_key=SecretStr(api_key) if api_key is not None else None,
            base_url=base_url,
        )
    )
    monkeypatch.setattr(_accessor_mod, "load_settings", lambda: cfg)


def test_capability_available_false_when_provider_none() -> None:
    cap = RerankCapability(provider=None)
    assert cap.available is False


def test_capability_available_true_when_provider_present() -> None:
    cap = RerankCapability(provider=_MockReranker())
    assert cap.available is True


def test_require_raises_when_unavailable() -> None:
    cap = RerankCapability(provider=None)
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        cap.require()
    assert excinfo.value.provider == "rerank"


def test_require_returns_provider_when_available() -> None:
    provider = _MockReranker()
    cap = RerankCapability(provider=provider)
    assert cap.require() is provider


def test_capability_is_frozen() -> None:
    cap = RerankCapability(provider=None)
    with pytest.raises((AttributeError, TypeError)):
        cap.provider = _MockReranker()  # type: ignore[misc]


def test_capability_has_no_soft_degrade_method() -> None:
    """Rerank has no soft-degrade write path, unlike EmbeddingCapability's
    ``embed_or_none`` — callers either require rerank or skip it entirely."""
    cap = RerankCapability(provider=None)
    assert not hasattr(cap, "embed_or_none")
    assert not hasattr(cap, "rerank_or_none")


def test_get_rerank_capability_empty_when_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_capability(monkeypatch)
    _patch_settings(monkeypatch, model=None, api_key=None, base_url=None)

    cap = get_rerank_capability()

    assert isinstance(cap, RerankCapability)
    assert cap.available is False


def test_get_rerank_capability_populated_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_capability(monkeypatch)
    _patch_settings(
        monkeypatch, model="m", api_key="sk-test", base_url="https://example.test"
    )

    cap = get_rerank_capability()

    assert cap.available is True


def test_get_rerank_capability_caches_singleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_capability(monkeypatch)
    _patch_settings(monkeypatch, model=None, api_key=None, base_url=None)

    first = get_rerank_capability()
    second = get_rerank_capability()

    assert first is second


def test_get_rerank_capability_logs_when_build_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ValueError from the factory surfaces as a warning log line.

    Both "user hasn't configured it" (missing fields) and "user
    configured it wrong" (unsupported provider name, malformed URL)
    reach the accessor as a :class:`ValueError`, and the downstream
    :class:`ProviderNotConfiguredError` message maps both onto the same
    HTTP 422 for the operator. The log line is the only place the two
    states can be distinguished, so this test pins that it fires with a
    non-empty reason kwarg.
    """
    _reset_capability(monkeypatch)

    def _raise(_settings: object) -> object:
        raise ValueError("cohere provider not supported")

    monkeypatch.setattr(_accessor_mod, "build_rerank_provider", _raise)

    with structlog.testing.capture_logs() as cap_logs:
        cap = get_rerank_capability()

    assert cap.available is False
    events = [
        log for log in cap_logs if log["event"] == "rerank_capability_build_failed"
    ]
    assert len(events) == 1
    assert events[0]["reason"] == "cohere provider not supported"
