"""The extraction client's deadline and provider-specific fields are configurable.

Regression cover for a silent misconfiguration: ``LLMSettings`` had no timeout
and no passthrough, so ``LLMConfig`` was built with three arguments and took the
algo defaults for the rest. A benchmark harness exported
``EVEROS_LLM__TIMEOUT_SECONDS=300`` for months and it did nothing, and there was
no way at all to reach a gateway's own request fields.

That combination is what made a reasoning model unusable as an extraction
backbone. Served over an OpenAI-compatible endpoint it thinks by default, and
thinking is billed against ``max_tokens``, so an atomic-facts prompt spent the
whole budget reasoning and returned an empty ``content``. Measured through the
real client on one gateway: knob nested correctly = 12.7s / 2264 characters;
knob omitted = 42.4s / **zero** characters. Every attempt then passed the 60s
deadline, so all three retries timed out and the memory was dead-lettered --
while the run's own logs said "timeout", pointing at the gateway rather than at
the request. Across ten servers that aborted 69 conversations with 0 successes;
with the knob in place the same fleet ran 981 extractions at 0 failures and a
2.3s mean.

The knob's *shape* is the second trap, and the reason
:func:`test_only_extra_body_survives_the_sdk_signature` exists: everalgo merges
``extra`` into the kwargs it hands the OpenAI SDK, and the SDK rejects unknown
top-level names. Nesting under ``extra_body`` is not a style choice.
"""

from __future__ import annotations

from typing import Any

import pytest

from everos.config import load_settings
from everos.config.settings import DeciderSettings, LLMSettings

NO_THINK = '{"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}'
NO_THINK_PARSED = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Any:
    load_settings.cache_clear()
    yield
    load_settings.cache_clear()


def test_defaults_match_the_algo_defaults() -> None:
    """Absent config, behaviour is byte-identical to before these fields existed."""
    assert LLMSettings().timeout_seconds == 60.0
    assert LLMSettings().extra == {}
    assert DeciderSettings().timeout_seconds == 60.0
    assert DeciderSettings().extra == {}


def test_extra_defaults_are_not_shared_between_instances() -> None:
    """A mutable default must not leak across settings objects."""
    first = LLMSettings()
    first.extra["chat_template_kwargs"] = {"enable_thinking": False}
    assert LLMSettings().extra == {}


@pytest.mark.parametrize("section", ["LLM", "DECIDER"])
def test_env_supplies_timeout_and_extra(
    section: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The env spelling the harness already used has to be the one that works."""
    monkeypatch.setenv(f"EVEROS_{section}__TIMEOUT_SECONDS", "300")
    monkeypatch.setenv(f"EVEROS_{section}__EXTRA", NO_THINK)
    load_settings.cache_clear()

    cfg = getattr(load_settings(), section.lower())
    assert cfg.timeout_seconds == 300.0
    assert cfg.extra == NO_THINK_PARSED


def test_timeout_must_be_positive() -> None:
    """A zero deadline fails every call; reject it at load rather than at runtime."""
    with pytest.raises(ValueError):
        LLMSettings(timeout_seconds=0)


def test_client_passes_both_through_to_the_algo_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The settings must reach ``LLMConfig`` -- the step that was missing.

    Asserted against the object handed to ``build_client``, because that is the
    boundary where the values were being dropped: the settings were correct and
    the algo client honoured what it was given, but nothing carried one to the
    other.
    """
    import everos.component.llm.client as client_mod

    monkeypatch.setenv("EVEROS_LLM__API_KEY", "k")
    monkeypatch.setenv("EVEROS_LLM__BASE_URL", "http://gw.invalid/v1")
    monkeypatch.setenv("EVEROS_LLM__TIMEOUT_SECONDS", "300")
    monkeypatch.setenv("EVEROS_LLM__EXTRA", NO_THINK)
    load_settings.cache_clear()

    seen: list[Any] = []
    monkeypatch.setattr(client_mod, "_llm_client", None)
    monkeypatch.setattr(
        client_mod, "build_client", lambda cfg: seen.append(cfg) or object()
    )
    client_mod.get_llm_client()

    (cfg,) = seen
    assert cfg.timeout == 300.0
    assert cfg.extra == NO_THINK_PARSED


def test_decider_client_passes_both_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """The decider needs it too: it runs the same model inside every search."""
    import everos.component.llm.client as client_mod

    monkeypatch.setenv("EVEROS_LLM__API_KEY", "k")
    monkeypatch.setenv("EVEROS_LLM__BASE_URL", "http://gw.invalid/v1")
    monkeypatch.setenv("EVEROS_DECIDER__MODEL", "qwen3.8-27b")
    monkeypatch.setenv("EVEROS_DECIDER__TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("EVEROS_DECIDER__EXTRA", NO_THINK)
    load_settings.cache_clear()

    seen: list[Any] = []
    monkeypatch.setattr(client_mod, "_decider_client", None)
    monkeypatch.setattr(
        client_mod, "build_client", lambda cfg: seen.append(cfg) or object()
    )
    client_mod.get_decider_llm_client()

    (cfg,) = seen
    assert cfg.timeout == 120.0
    assert cfg.extra == NO_THINK_PARSED


def test_config_extra_reaches_the_request_kwargs() -> None:
    """``extra`` is only worth configuring if it lands in the outgoing request.

    Calls the algo provider's own kwargs assembly rather than re-implementing
    the merge here, so the assertion still means something if that merge
    changes: an upstream that stopped forwarding ``config_extra`` would make
    every setting above inert while all the tests above still passed.
    """
    from everalgo.llm.providers.openai_compat import _build_request_kwargs
    from everalgo.llm.types import ChatMessage

    kwargs = _build_request_kwargs(
        messages=[ChatMessage(role="user", content="hi")],
        model="qwen3.8-27b",
        temperature=0.0,
        max_tokens=4096,
        config_extra=NO_THINK_PARSED,
        extra={},
    )
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_per_call_extra_overrides_the_config() -> None:
    """A caller can opt back into what the config disables."""
    from everalgo.llm.providers.openai_compat import _build_request_kwargs
    from everalgo.llm.types import ChatMessage

    kwargs = _build_request_kwargs(
        messages=[ChatMessage(role="user", content="hi")],
        model="qwen3.8-27b",
        temperature=0.0,
        max_tokens=None,
        config_extra=NO_THINK_PARSED,
        extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
    )
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}


def test_only_extra_body_survives_the_sdk_signature() -> None:
    """A gateway-specific name must be nested, not passed at the top level.

    ``_build_request_kwargs`` is a ``dict.update``, so it accepts any key and
    cannot catch this -- the rejection happens one layer down, where the kwargs
    are splatted into the SDK call. Asserting against the SDK's own signature is
    what makes the nesting requirement a checked fact rather than a comment: a
    top-level ``chat_template_kwargs`` raised ``TypeError`` on the first call of
    a ten-server run.
    """
    import inspect

    from openai.resources.chat.completions import AsyncCompletions

    params = inspect.signature(AsyncCompletions.create).parameters
    assert not any(p.kind is p.VAR_KEYWORD for p in params.values()), (
        "SDK grew **kwargs; unknown names would now pass silently and this "
        "config's nesting requirement needs rechecking"
    )
    assert "extra_body" in params
    assert "chat_template_kwargs" not in params
