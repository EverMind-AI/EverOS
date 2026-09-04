"""The decider's SDK retry count reaches the client the decider actually uses.

``DeciderSettings.sdk_max_retries`` was added to stop the OpenAI SDK's own retries from
multiplying with the decider's: four decider attempts times three SDK requests against a
60s deadline is a 728s round that still ends in the fixed top-3 fallback. It was
declared and documented and reached nothing. Two gaps, both invisible:

* ``get_decider_llm_client()`` built an everalgo ``LLMConfig``, whose fields are
  ``model, api_key, base_url, temperature, max_tokens, timeout, extra`` -- there is no
  ``max_retries``, so ``AsyncOpenAI`` kept its default of 2.
* With no ``[decider].model`` the function returned the shared extraction client, which
  cannot carry a decider-specific retry count even in principle: the SDK reads
  ``max_retries`` at construction.

So the assertions here go all the way down to ``AsyncOpenAI.max_retries`` on both paths.
Asserting that the settings field exists, or that some config object carries it, is what
let the original change look done.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

import everos.component.llm.client as client_mod
from everos.config import load_settings
from everos.config.settings import DeciderSettings

NO_THINK_PARSED = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
NO_THINK = '{"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}'


@pytest.fixture(autouse=True)
def _fresh_clients(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Both singletons are process-wide; a leaked one would answer for the next test."""
    monkeypatch.setattr(client_mod, "_decider_client", None)
    monkeypatch.setattr(client_mod, "_llm_client", None)
    monkeypatch.setenv("EVEROS_LLM__API_KEY", "k")
    monkeypatch.setenv("EVEROS_LLM__BASE_URL", "http://gw.invalid/v1")
    load_settings.cache_clear()
    yield
    load_settings.cache_clear()


def _sdk_of(client: Any) -> Any:
    """Unwrap the logging / usage decorators down to the ``AsyncOpenAI`` instance."""
    while hasattr(client, "_inner"):
        client = client._inner
    return client._client


def test_an_explicit_decider_model_turns_the_sdk_retries_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVEROS_DECIDER__MODEL", "qwen3.6-27b")
    load_settings.cache_clear()
    assert _sdk_of(client_mod.get_decider_llm_client()).max_retries == 0


def test_an_inherited_decider_config_turns_the_sdk_retries_off() -> None:
    """The path with no ``[decider].model`` at all, which used to share the client.

    This is the configuration every store built before ``[decider]`` existed runs
    under, so it is the one that mattered most and the one that could not work.
    """
    client = client_mod.get_decider_llm_client()
    assert _sdk_of(client).max_retries == 0
    assert client is not client_mod.get_llm_client()


def test_a_configured_count_is_what_the_sdk_gets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not hardcoded to zero somewhere along the way."""
    monkeypatch.setenv("EVEROS_DECIDER__SDK_MAX_RETRIES", "4")
    load_settings.cache_clear()
    assert _sdk_of(client_mod.get_decider_llm_client()).max_retries == 4


def test_none_leaves_the_sdk_default_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented opt-out: say nothing and the SDK behaves as it always did."""
    monkeypatch.setattr(
        client_mod,
        "resolve_decider_config",
        lambda *_a, **_k: client_mod.DeciderClientConfig(
            model="m",
            api_key="k",
            base_url="http://gw.invalid/v1",
            timeout=60.0,
            max_retries=None,
        ),
    )
    assert _sdk_of(client_mod.get_decider_llm_client()).max_retries == 2


def test_a_negative_retry_count_is_rejected() -> None:
    """The SDK reads it as a count, so a negative value is a typo, not "fewer"."""
    with pytest.raises(ValidationError):
        DeciderSettings(sdk_max_retries=-1)


def test_the_decider_deadline_reaches_the_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """It bounds one round inside a live search, not a background extraction."""
    monkeypatch.setenv("EVEROS_LLM__TIMEOUT_SECONDS", "300")
    monkeypatch.setenv("EVEROS_DECIDER__TIMEOUT_SECONDS", "20")
    load_settings.cache_clear()
    assert _sdk_of(client_mod.get_decider_llm_client()).timeout == 20.0


# ── `extra` travels with whichever section supplied the endpoint ──────────────


def test_inheriting_the_endpoint_keeps_its_provider_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Losing this would re-enable thinking on a deployment that turned it off once.

    A Qwen gateway configured once under ``[llm]`` serves both roles when
    ``[decider]`` names no model of its own; the decider must not quietly drop the
    field that endpoint needs.
    """
    monkeypatch.setenv("EVEROS_LLM__EXTRA", NO_THINK)
    load_settings.cache_clear()
    assert client_mod.resolve_decider_config().extra == NO_THINK_PARSED


def test_its_own_endpoint_does_not_inherit_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``extra`` is endpoint vocabulary; another gateway rejects unknown names."""
    monkeypatch.setenv("EVEROS_LLM__EXTRA", NO_THINK)
    monkeypatch.setenv("EVEROS_DECIDER__MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("EVEROS_DECIDER__BASE_URL", "http://other.invalid/v1")
    load_settings.cache_clear()
    assert client_mod.resolve_decider_config().extra == {}


def test_the_model_falls_back_to_the_main_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVEROS_LLM__MODEL", "backbone-model")
    load_settings.cache_clear()
    cfg = client_mod.resolve_decider_config()
    assert (cfg.model, cfg.inherits_model) == ("backbone-model", True)


def test_no_credentials_anywhere_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVEROS_LLM__API_KEY", raising=False)
    monkeypatch.delenv("EVEROS_LLM__BASE_URL", raising=False)
    load_settings.cache_clear()
    with pytest.raises(client_mod.LLMNotConfiguredError):
        client_mod.resolve_decider_config()


# ── and the fields the decider sends per call ────────────────────────────────


class _StopError(Exception):
    """Ends the call once the outgoing kwargs have been captured."""


async def test_client_level_extra_and_per_call_kwargs_both_reach_the_request() -> None:
    """Per-call wins on a collision, matching everalgo's own merge order."""
    from everos.component.llm.openai_provider import OpenAIProvider
    from everos.component.llm.protocol import ChatMessage

    provider = OpenAIProvider(
        model="m", api_key="k", base_url="http://gw.invalid/v1", extra=NO_THINK_PARSED
    )
    seen: dict[str, Any] = {}

    class _Completions:
        async def create(self, **kw: Any) -> Any:
            seen.update(kw)
            raise _StopError

    provider._client = type(  # type: ignore[assignment]
        "_C", (), {"chat": type("_Chat", (), {"completions": _Completions()})()}
    )()
    with pytest.raises(_StopError):
        await provider.chat(
            [ChatMessage(role="user", content="hi")], max_tokens=512, top_p=0.9
        )

    assert seen["max_tokens"] == 512
    assert seen["extra_body"] == NO_THINK_PARSED["extra_body"]
    assert seen["top_p"] == 0.9
