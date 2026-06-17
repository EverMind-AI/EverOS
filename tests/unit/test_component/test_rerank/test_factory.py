"""``build_rerank_provider`` — settings validation + provider routing."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from everos.component.rerank import (
    DashScopeRerankProvider,
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


def test_deepinfra_rejects_empty_api_key() -> None:
    s = RerankSettings(
        provider="deepinfra",
        model="m",
        api_key=SecretStr(""),
        base_url="https://api.deepinfra.com/v1/inference",
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


def test_dashscope_requires_api_key() -> None:
    s = RerankSettings(
        provider="dashscope",
        model="gte-rerank-v2",
        api_key=None,
        base_url="https://dashscope.aliyuncs.com",
    )
    with pytest.raises(ValueError, match="EVEROS_RERANK__API_KEY"):
        build_rerank_provider(s)


def test_dashscope_builds_provider() -> None:
    s = RerankSettings(
        provider="dashscope",
        model="gte-rerank-v2",
        api_key=SecretStr("k"),
        base_url="https://dashscope.aliyuncs.com",
    )
    p = build_rerank_provider(s)
    assert isinstance(p, DashScopeRerankProvider)


def test_dashscope_supports_gte_rerank_v2_only() -> None:
    s = RerankSettings(
        provider="dashscope",
        model="qwen3-rerank",
        api_key=SecretStr("k"),
        base_url="https://dashscope.aliyuncs.com",
    )
    with pytest.raises(ValueError, match="gte-rerank-v2 only"):
        build_rerank_provider(s)


def test_infers_dashscope_from_base_url() -> None:
    """provider unset + DashScope host -> DashScopeRerankProvider."""
    s = RerankSettings(
        provider=None,
        model="gte-rerank-v2",
        api_key=SecretStr("k"),
        base_url="https://dashscope.aliyuncs.com",
    )
    assert isinstance(build_rerank_provider(s), DashScopeRerankProvider)


def test_infers_deepinfra_from_base_url() -> None:
    """provider unset + DeepInfra host -> DeepInfraRerankProvider."""
    s = RerankSettings(
        provider=None,
        model="m",
        api_key=SecretStr("k"),
        base_url="https://api.deepinfra.com/v1/inference",
    )
    assert isinstance(build_rerank_provider(s), DeepInfraRerankProvider)


def test_unknown_host_falls_back_to_deepinfra() -> None:
    """provider unset + unrecognized host -> historical deepinfra default."""
    s = RerankSettings(
        provider=None,
        model="m",
        api_key=SecretStr("k"),
        base_url="https://rerank.internal.example/v1",
    )
    assert isinstance(build_rerank_provider(s), DeepInfraRerankProvider)


def test_explicit_provider_overrides_inference() -> None:
    """Explicit provider wins even when the host hints another."""
    s = RerankSettings(
        provider="vllm",
        model="m",
        api_key=SecretStr("k"),
        base_url="https://dashscope.aliyuncs.com",
    )
    assert isinstance(build_rerank_provider(s), VllmRerankProvider)
