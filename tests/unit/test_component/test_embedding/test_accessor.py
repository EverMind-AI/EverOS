"""``EmbeddingCapability`` + ``get_embedding_capability`` accessor."""

from __future__ import annotations

import importlib

import pytest
import structlog.testing
from pydantic import SecretStr

from everos.component.embedding import EmbeddingCapability, get_embedding_capability
from everos.config import Settings
from everos.config.settings import EmbeddingSettings
from everos.core.errors import ProviderNotConfiguredError

_accessor_mod = importlib.import_module("everos.component.embedding.accessor")


class _MockEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


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
        embedding=EmbeddingSettings(
            model=model,
            api_key=SecretStr(api_key) if api_key is not None else None,
            base_url=base_url,
        )
    )
    monkeypatch.setattr(_accessor_mod, "load_settings", lambda: cfg)


def test_capability_available_false_when_provider_none() -> None:
    cap = EmbeddingCapability(provider=None)
    assert cap.available is False


def test_capability_available_true_when_provider_present() -> None:
    cap = EmbeddingCapability(provider=_MockEmbedder())
    assert cap.available is True


async def test_embed_or_none_returns_none_when_unavailable() -> None:
    cap = EmbeddingCapability(provider=None)
    result = await cap.embed_or_none("hello")
    assert result is None


async def test_embed_or_none_returns_vector_when_available() -> None:
    cap = EmbeddingCapability(provider=_MockEmbedder())
    result = await cap.embed_or_none("hello")
    assert result == [0.1, 0.2, 0.3]


def test_require_raises_when_unavailable() -> None:
    cap = EmbeddingCapability(provider=None)
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        cap.require()
    assert excinfo.value.provider == "embedding"


def test_require_returns_provider_when_available() -> None:
    provider = _MockEmbedder()
    cap = EmbeddingCapability(provider=provider)
    assert cap.require() is provider


def test_capability_is_frozen() -> None:
    cap = EmbeddingCapability(provider=None)
    with pytest.raises((AttributeError, TypeError)):
        cap.provider = _MockEmbedder()  # type: ignore[misc]


def test_get_embedding_capability_empty_when_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_capability(monkeypatch)
    _patch_settings(monkeypatch, model=None, api_key=None, base_url=None)

    cap = get_embedding_capability()

    assert isinstance(cap, EmbeddingCapability)
    assert cap.available is False


def test_get_embedding_capability_populated_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_capability(monkeypatch)
    _patch_settings(
        monkeypatch, model="m", api_key="sk-test", base_url="https://example.test"
    )

    cap = get_embedding_capability()

    assert cap.available is True


def test_get_embedding_capability_caches_singleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_capability(monkeypatch)
    _patch_settings(monkeypatch, model=None, api_key=None, base_url=None)

    first = get_embedding_capability()
    second = get_embedding_capability()

    assert first is second


def test_get_embedding_capability_logs_when_build_fails(
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

    monkeypatch.setattr(_accessor_mod, "build_embedding_provider", _raise)

    with structlog.testing.capture_logs() as cap_logs:
        cap = get_embedding_capability()

    assert cap.available is False
    events = [
        log for log in cap_logs if log["event"] == "embedding_capability_build_failed"
    ]
    assert len(events) == 1
    assert events[0]["reason"] == "cohere provider not supported"
