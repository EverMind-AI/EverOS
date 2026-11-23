"""NVIDIA NIM -- an ``[llm]`` section pointed at NIM reaches every task's client.

NIM serves the OpenAI Chat Completions protocol, so it needs no provider of its
own, and ``[llm]`` is the base every ``[llm.<task>]`` is laid over: with no task
section set, every task resolves to NIM's base URL, key, and model. The
bootstrap's path -- ``resolve_llm_tasks`` + ``llm_client_config`` +
``build_llm_task_providers`` -- then hands each task an ``OpenAIProvider`` whose
``openai.AsyncOpenAI`` client carries them. No request is sent.
"""

from __future__ import annotations

from pydantic import SecretStr

from everos.component.llm import (
    LLMClientConfig,
    OpenAIProvider,
    build_llm_task_providers,
    llm_client_config,
    observer_for,
    resolve_llm_tasks,
)
from everos.config import LLMTask, Settings
from everos.config.settings import LLMSettings

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
NVIDIA_KEY = "nvidia-test-key"


def _settings(base_url: str = NVIDIA_BASE_URL) -> Settings:
    return Settings(
        llm=LLMSettings(
            model=NVIDIA_MODEL,
            api_key=SecretStr(NVIDIA_KEY),
            base_url=base_url,
        )
    )


def _configs(settings: Settings) -> dict[LLMTask, LLMClientConfig]:
    return {
        task: llm_client_config(resolution)
        for task, resolution in resolve_llm_tasks(settings).items()
    }


def _providers(settings: Settings) -> dict[LLMTask, OpenAIProvider]:
    """Every task's provider, assembled as the bootstrap assembles them."""
    return build_llm_task_providers(_configs(settings), observer=observer_for(settings))


def _sdk_endpoint(provider: OpenAIProvider) -> tuple[str, str, str]:
    """The base URL, key, and model the provider's SDK client sends."""
    client = provider._client
    return str(client.base_url), client.api_key, provider._model


def test_nvidia_section_resolves_to_its_endpoint_for_every_task() -> None:
    configs = _configs(_settings())

    assert {
        task: (config.base_url, config.api_key, config.model)
        for task, config in configs.items()
    } == dict.fromkeys(LLMTask, (NVIDIA_BASE_URL, NVIDIA_KEY, NVIDIA_MODEL))


def test_nvidia_endpoint_reaches_every_tasks_sdk_client() -> None:
    providers = _providers(_settings())

    assert all(isinstance(p, OpenAIProvider) for p in providers.values())
    assert {
        task: _sdk_endpoint(provider) for task, provider in providers.items()
    } == dict.fromkeys(LLMTask, (f"{NVIDIA_BASE_URL}/", NVIDIA_KEY, NVIDIA_MODEL))


def test_self_hosted_nim_base_url_is_kept() -> None:
    providers = _providers(_settings(base_url="http://localhost:8000/v1"))

    assert {
        task: str(provider._client.base_url) for task, provider in providers.items()
    } == dict.fromkeys(LLMTask, "http://localhost:8000/v1/")
