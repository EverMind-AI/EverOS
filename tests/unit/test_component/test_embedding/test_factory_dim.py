"""``build_embedding_provider`` — where the target vector width comes from.

Two callers with opposite needs share this factory. The write path wants the
width the index columns were built at, which is ``settings.dim``. The startup
probe wants the model's *real* output length, which it can only see by asking
for far more than the model returns, because
:class:`~everos.component.embedding.openai_provider.OpenAIEmbeddingProvider`
truncates to ``dim`` before returning and so can never report a longer vector.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from everos.component.embedding import build_embedding_provider
from everos.config import EmbeddingSettings


def _settings(**overrides: object) -> EmbeddingSettings:
    base: dict[str, object] = {
        "model": "fake-model",
        "api_key": SecretStr("key"),
        "base_url": "http://127.0.0.1:1/v1",
    }
    base.update(overrides)
    return EmbeddingSettings(**base)  # type: ignore[arg-type]


def test_default_follows_settings_dim() -> None:
    """No explicit ``dim``: the provider targets the configured column width.

    A hardcoded default here would silently truncate every vector from a
    wider model down to 1024 while the tables were built at the configured
    width — invisible to any case configured at or below 1024.
    """
    assert build_embedding_provider(_settings(dim=1536)).dim == 1536
    assert build_embedding_provider(_settings(dim=768)).dim == 768


def test_explicit_dim_overrides_settings() -> None:
    """The probe asks for far more than any model returns, so the slice is a no-op."""
    provider = build_embedding_provider(_settings(dim=768), dim=1 << 20)
    assert provider.dim == 1 << 20


@pytest.mark.parametrize("missing", ["model", "api_key", "base_url"])
def test_missing_credentials_still_refused(missing: str) -> None:
    """The three-field guard is what keyword-only deployments rely on."""
    with pytest.raises(ValueError):
        build_embedding_provider(_settings(**{missing: None}))
