"""``build_providers``: the LLM is required, every other provider optional.

Without ``[llm]`` the build raises -- the application exists to extract
memory. The bootstrap is the one place that decides what an unconfigured
optional provider means. Unconfigured and misconfigured both become ``None``
-- they surface as the same 422 later -- so the warning logged here is the
only place an operator can tell the two apart.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr

from everos.component.llm import LLMNotConfiguredError
from everos.config import Settings
from everos.config.settings import (
    EmbeddingSettings,
    LLMSettings,
    MultimodalSettings,
    RerankSettings,
)
from everos.service import bootstrap as bootstrap_mod
from everos.service import build_providers


class _RecordingLogger:
    """Stands in for the bootstrap's module logger and keeps its warnings.

    Replacing the logger rather than capturing structlog output keeps the
    assertion independent of whether an earlier test configured logging
    (a configured structlog logger is cached and bypasses capture).
    """

    def __init__(self) -> None:
        self.warnings: list[dict[str, Any]] = []

    def warning(self, event: str, **fields: Any) -> None:
        self.warnings.append({"event": event, **fields})


@pytest.fixture
def logs(monkeypatch: pytest.MonkeyPatch) -> _RecordingLogger:
    recorder = _RecordingLogger()
    monkeypatch.setattr(bootstrap_mod, "logger", recorder)
    return recorder


_LLM = LLMSettings(api_key=SecretStr("sk-test"), base_url="https://l.test")


def _unconfigured() -> Settings:
    """Only the required LLM is configured."""
    return Settings(
        llm=_LLM,
        embedding=EmbeddingSettings(model=None, api_key=None, base_url=None),
        rerank=RerankSettings(model=None, api_key=None, base_url=None),
        multimodal=MultimodalSettings(api_key=None, base_url=None),
    )


def test_unconfigured_optional_providers_are_none() -> None:
    providers = build_providers(_unconfigured())

    assert providers.embedding is None
    assert providers.rerank is None
    assert providers.multimodal_llm is None


@pytest.mark.parametrize(
    "llm",
    [
        LLMSettings(api_key=None, base_url="https://l.test"),
        LLMSettings(api_key=SecretStr(""), base_url="https://l.test"),
        LLMSettings(api_key=SecretStr("sk-test"), base_url=None),
    ],
    ids=["no-key", "blank-key", "no-base-url"],
)
def test_an_unconfigured_llm_fails_the_build(llm: LLMSettings) -> None:
    settings = _unconfigured().model_copy(update={"llm": llm})

    with pytest.raises(LLMNotConfiguredError, match=r"everos\.toml.*\[llm\]"):
        build_providers(settings)


def test_the_llm_and_decider_are_built_from_llm_alone() -> None:
    providers = build_providers(_unconfigured())

    assert providers.llm is not None
    assert providers.decider is not None
    assert providers.decider is not providers.llm


def test_configured_optional_providers_are_built() -> None:
    settings = Settings(
        llm=_LLM,
        embedding=EmbeddingSettings(
            model="m", api_key=SecretStr("sk-test"), base_url="https://e.test"
        ),
        rerank=RerankSettings(
            model="r", api_key=SecretStr("sk-test"), base_url="https://r.test"
        ),
        multimodal=MultimodalSettings(
            model="mm", api_key=SecretStr("sk-test"), base_url="https://m.test"
        ),
    )

    providers = build_providers(settings)

    assert providers.embedding is not None
    assert providers.rerank is not None
    assert providers.multimodal_llm is not None


@pytest.mark.parametrize(
    ("factory", "event"),
    [
        ("build_embedding_provider", "embedding_capability_build_failed"),
        ("build_rerank_provider", "rerank_capability_build_failed"),
    ],
)
def test_a_factory_failure_is_logged_with_its_reason(
    monkeypatch: pytest.MonkeyPatch,
    logs: _RecordingLogger,
    factory: str,
    event: str,
) -> None:
    def _raise(_settings: object) -> object:
        raise ValueError("cohere provider not supported")

    monkeypatch.setattr(bootstrap_mod, factory, _raise)

    build_providers(_unconfigured())

    failures = [log for log in logs.warnings if log["event"] == event]
    assert [log["reason"] for log in failures] == ["cohere provider not supported"]


def test_only_the_optional_multimodal_llm_is_logged_as_not_configured(
    logs: _RecordingLogger,
) -> None:
    build_providers(_unconfigured())

    roles = [
        log["role"] for log in logs.warnings if log["event"] == "llm_not_configured"
    ]
    assert roles == ["multimodal_llm"]
