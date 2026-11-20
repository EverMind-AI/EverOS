"""NVIDIA NIM -- an ``[llm]`` section pointed at NIM reaches the SDK client.

NIM serves the OpenAI Chat Completions protocol, so it needs no provider of its
own: ``resolve_llm_config`` + ``build_llm_provider`` turn the section into an
``OpenAIProvider`` whose ``openai.AsyncOpenAI`` client carries NIM's base URL,
key, and model. No request is sent.
"""

from __future__ import annotations

from pydantic import SecretStr

from everos.component.llm import (
    OpenAIProvider,
    build_llm_provider,
    observer_for,
    resolve_llm_config,
)
from everos.config import Settings
from everos.config.settings import LLMSettings

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"


def _settings(base_url: str = NVIDIA_BASE_URL) -> Settings:
    return Settings(
        llm=LLMSettings(
            model=NVIDIA_MODEL,
            api_key=SecretStr("nvidia-test-key"),
            base_url=base_url,
        )
    )


def test_nvidia_section_resolves_to_its_endpoint() -> None:
    config = resolve_llm_config(_settings())

    assert config.model == NVIDIA_MODEL
    assert config.api_key == "nvidia-test-key"
    assert config.base_url == NVIDIA_BASE_URL


def test_nvidia_endpoint_reaches_the_sdk_client() -> None:
    settings = _settings()

    provider = build_llm_provider(
        resolve_llm_config(settings), observer=observer_for(settings)
    )

    assert isinstance(provider, OpenAIProvider)
    assert str(provider._client.base_url) == f"{NVIDIA_BASE_URL}/"
    assert provider._client.api_key == "nvidia-test-key"
    assert provider._model == NVIDIA_MODEL


def test_self_hosted_nim_base_url_is_kept() -> None:
    provider = build_llm_provider(
        resolve_llm_config(_settings(base_url="http://localhost:8000/v1"))
    )

    assert str(provider._client.base_url) == "http://localhost:8000/v1/"
