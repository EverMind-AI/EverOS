"""``build_rerank_provider`` — settings validation + provider routing."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from everos.component.rerank import (
    DeepInfraRerankProvider,
    VllmRerankProvider,
    build_rerank_provider,
)
from everos.config.settings import RerankSettings


def test_raises_when_model_missing() -> None:
    s = RerankSettings(model=None, api_key=SecretStr("k"), base_url="https://x")
    with pytest.raises(ValueError, match="EVEROS_RERANK__MODEL"):
        build_rerank_provider(s)


def test_raises_when_base_url_missing() -> None:
    s = RerankSettings(model="m", api_key=SecretStr("k"), base_url=None)
    with pytest.raises(ValueError, match="EVEROS_RERANK__BASE_URL"):
        build_rerank_provider(s)


def test_deepinfra_requires_api_key() -> None:
    s = RerankSettings(
        provider="deepinfra", model="m", api_key=None, base_url="https://x"
    )
    with pytest.raises(ValueError, match="EVEROS_RERANK__API_KEY"):
        build_rerank_provider(s)


def test_deepinfra_builds_provider() -> None:
    s = RerankSettings(
        provider="deepinfra",
        model="m",
        api_key=SecretStr("k"),
        base_url="https://api/v1/inference",
    )
    p = build_rerank_provider(s)
    assert isinstance(p, DeepInfraRerankProvider)


def test_vllm_accepts_empty_api_key() -> None:
    """vLLM self-hosted: empty api_key is allowed (no auth header)."""
    s = RerankSettings(
        provider="vllm",
        model="m",
        api_key=None,
        base_url="http://localhost:8000/v1",
    )
    p = build_rerank_provider(s)
    assert isinstance(p, VllmRerankProvider)


def test_vllm_with_api_key() -> None:
    s = RerankSettings(
        provider="vllm",
        model="m",
        api_key=SecretStr("k"),
        base_url="http://localhost:8000/v1",
    )
    p = build_rerank_provider(s)
    assert isinstance(p, VllmRerankProvider)
