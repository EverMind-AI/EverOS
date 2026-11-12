"""Per-task LLM resolution: ``[llm]`` with each ``[llm.<task>]`` laid over it.

One rule: every field is taken from the first section that sets it -- the task's
own section, then (decider only) the deprecated ``[decider]`` fields, then the
task's own defaults (decider only), then ``[llm]``. ``extra`` is one value,
replaced whole. The cases below pin that rule, the source each field reports,
which tasks share a provider, and that ``temperature`` / ``max_tokens`` are
defaults a call can still override -- read off the wire, because a provider that
dropped them would leave every assertion on the settings green.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import openai
import pytest
from everalgo.llm import LLMMessage, StrictOutputModel
from pydantic import SecretStr

from everos.component.llm import (
    LLMNotConfiguredError,
    build_llm_task_providers,
    llm_client_config,
    resolve_llm_task,
    resolve_llm_tasks,
)
from everos.config import LLMTask, LLMTaskSettings, Settings
from everos.config.settings import DeciderSettings, LLMSettings

_FIELDS = (
    "model",
    "api_key",
    "base_url",
    "timeout_seconds",
    "extra",
    "response_format_mode",
    "invalid_reply_retries",
    "temperature",
    "max_tokens",
    "sdk_max_retries",
)

_BASE: dict[str, Any] = {
    "model": "base-model",
    "api_key": SecretStr("sk-base"),
    "base_url": "https://base.test/v1",
}


def _settings(
    llm: dict[str, Any] | None = None, decider: dict[str, Any] | None = None
) -> Settings:
    """Settings with ``[llm]`` = ``_BASE`` | ``llm`` and ``[decider]`` = ``decider``."""
    return Settings(
        llm=LLMSettings(**{**_BASE, **(llm or {})}),
        decider=DeciderSettings(**(decider or {})),
    )


def _effective(settings: Settings, task: LLMTask) -> dict[str, Any]:
    resolved = resolve_llm_task(settings, task).settings
    return {name: getattr(resolved, name) for name in _FIELDS}


# ── the inheritance rule, table-driven ───────────────────────────────────────

_FULL_OVERRIDE: dict[str, Any] = {
    "model": "own-model",
    "api_key": SecretStr("sk-own"),
    "base_url": "http://localhost:8000/v1",
    "timeout_seconds": 300.0,
    "extra": {"extra_body": {"own": True}},
    "response_format_mode": "prompt_only",
    "invalid_reply_retries": 3,
    "temperature": 0.7,
    "max_tokens": 2048,
    "sdk_max_retries": 4,
}

_INHERIT_CASES: list[tuple[str, dict[str, Any], LLMTask, dict[str, Any]]] = [
    (
        "everything inherited",
        {"extra": {"extra_body": {"a": 1}}, "temperature": 0.3},
        LLMTask.PROFILE,
        {"model": "base-model", "extra": {"extra_body": {"a": 1}}, "temperature": 0.3},
    ),
    (
        "only the model changes",
        {
            "extra": {"extra_body": {"a": 1}},
            "boundary": LLMTaskSettings(model="nano"),
        },
        LLMTask.BOUNDARY,
        {
            "model": "nano",
            "base_url": "https://base.test/v1",
            "api_key": SecretStr("sk-base"),
            "extra": {"extra_body": {"a": 1}},
        },
    ),
    (
        "the whole set overridden",
        {"profile": LLMTaskSettings(**_FULL_OVERRIDE)},
        LLMTask.PROFILE,
        _FULL_OVERRIDE,
    ),
    (
        "extra replaced whole, never merged",
        {
            "extra": {"extra_body": {"a": 1}, "top_p": 0.5},
            "search": LLMTaskSettings(extra={"extra_body": {"b": 2}}),
        },
        LLMTask.SEARCH,
        {"extra": {"extra_body": {"b": 2}}},
    ),
    (
        "decider keeps its own four defaults",
        {
            "timeout_seconds": 300.0,
            "max_tokens": 8000,
            "sdk_max_retries": 5,
            "invalid_reply_retries": 4,
            "temperature": 0.3,
            "extra": {"extra_body": {"a": 1}},
            "response_format_mode": "json_object",
        },
        LLMTask.DECIDER,
        {
            "timeout_seconds": 60.0,
            "max_tokens": 512,
            "sdk_max_retries": 0,
            "invalid_reply_retries": 1,
            "temperature": 0.3,
            "extra": {"extra_body": {"a": 1}},
            "response_format_mode": "json_object",
            "model": "base-model",
        },
    ),
    (
        "decider section beats its defaults",
        {"decider": LLMTaskSettings(timeout_seconds=20.0, max_tokens=64)},
        LLMTask.DECIDER,
        {"timeout_seconds": 20.0, "max_tokens": 64, "sdk_max_retries": 0},
    ),
]


@pytest.mark.parametrize(
    ("llm", "task", "expected"),
    [case[1:] for case in _INHERIT_CASES],
    ids=[case[0] for case in _INHERIT_CASES],
)
def test_each_field_comes_from_the_first_section_that_sets_it(
    llm: dict[str, Any], task: LLMTask, expected: dict[str, Any]
) -> None:
    effective = _effective(_settings(llm), task)
    assert {name: effective[name] for name in expected} == expected


def test_a_task_section_changes_no_other_task() -> None:
    settings = _settings({"boundary": LLMTaskSettings(model="nano")})
    for task in set(LLMTask) - {LLMTask.BOUNDARY}:
        assert resolve_llm_task(settings, task).settings.model == "base-model"


def test_blank_strings_in_a_task_section_inherit() -> None:
    """A template's ``api_key = ""`` must not mask the key it inherits."""
    settings = _settings({"episode": {"model": "", "api_key": "", "base_url": ""}})
    effective = _effective(settings, LLMTask.EPISODE)
    assert (effective["model"], effective["base_url"]) == (
        "base-model",
        "https://base.test/v1",
    )
    assert effective["api_key"] == SecretStr("sk-base")


# ── where each field came from ───────────────────────────────────────────────


def test_sources_name_the_section_or_the_default() -> None:
    settings = _settings(
        {"timeout_seconds": 90.0, "boundary": LLMTaskSettings(model="nano")}
    )
    sources = resolve_llm_task(settings, LLMTask.BOUNDARY).sources

    assert sources["model"] == "llm.boundary"
    assert (sources["base_url"], sources["timeout_seconds"]) == ("llm", "llm")
    assert (sources["extra"], sources["max_tokens"]) == ("default", "default")
    assert set(sources) == set(_FIELDS)


def test_the_deciders_own_defaults_report_default() -> None:
    sources = resolve_llm_task(_settings(), LLMTask.DECIDER).sources
    assert {sources[n] for n in ("timeout_seconds", "max_tokens")} == {"default"}


def test_host_is_the_endpoints_host_name() -> None:
    resolved = resolve_llm_task(
        _settings({"profile": LLMTaskSettings(base_url="http://localhost:8000/v1")}),
        LLMTask.PROFILE,
    )
    assert resolved.host == "localhost"


def test_every_task_resolves_in_task_order() -> None:
    assert list(resolve_llm_tasks(_settings())) == list(LLMTask)


# ── the deprecated [decider] spelling ────────────────────────────────────────


def test_the_old_decider_section_still_applies() -> None:
    resolved = resolve_llm_task(
        _settings(decider={"model": "qwen3.6-27b", "timeout_seconds": 600}),
        LLMTask.DECIDER,
    )
    assert (resolved.settings.model, resolved.settings.timeout_seconds) == (
        "qwen3.6-27b",
        600.0,
    )
    assert resolved.deprecated_fields == ("model", "timeout_seconds")


def test_the_new_section_wins_over_the_old_one() -> None:
    settings = _settings(
        {"decider": LLMTaskSettings(model="new-model")},
        decider={"model": "old-model", "max_tokens": 128},
    )
    resolved = resolve_llm_task(settings, LLMTask.DECIDER)
    assert (resolved.settings.model, resolved.sources["model"]) == (
        "new-model",
        "llm.decider",
    )
    assert (resolved.settings.max_tokens, resolved.sources["max_tokens"]) == (
        128,
        "decider",
    )


def test_blank_old_fields_play_no_part() -> None:
    """Blank, as ``default.toml`` shipped ``model`` / ``api_key`` / ``base_url``."""
    resolved = resolve_llm_task(
        _settings(decider={"model": "", "api_key": "", "base_url": ""}),
        LLMTask.DECIDER,
    )
    assert resolved.deprecated_fields == ()
    assert resolved.settings.model == "base-model"


def test_the_old_env_names_still_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVEROS_LLM__API_KEY", "sk-env")
    monkeypatch.setenv("EVEROS_DECIDER__MODEL", "env-decider")
    resolved = resolve_llm_task(Settings(), LLMTask.DECIDER)
    assert (resolved.settings.model, resolved.sources["model"]) == (
        "env-decider",
        "decider",
    )


def test_no_other_task_reads_the_old_section() -> None:
    settings = _settings(decider={"model": "old-model"})
    assert resolve_llm_task(settings, LLMTask.SEARCH).settings.model == "base-model"


# ── credentials ──────────────────────────────────────────────────────────────


def test_a_task_without_a_key_is_not_configured() -> None:
    settings = _settings({"api_key": None})
    with pytest.raises(LLMNotConfiguredError, match=r"task boundary.*\[llm\]"):
        llm_client_config(resolve_llm_task(settings, LLMTask.BOUNDARY))


def test_a_task_section_can_supply_its_own_key() -> None:
    settings = _settings(
        {"api_key": None, "profile": LLMTaskSettings(api_key=SecretStr("local"))}
    )
    assert llm_client_config(resolve_llm_task(settings, LLMTask.PROFILE)).api_key == (
        "local"
    )


# ── provider sharing ─────────────────────────────────────────────────────────


def _providers(settings: Settings) -> dict[LLMTask, Any]:
    configs = {
        task: llm_client_config(resolution)
        for task, resolution in resolve_llm_tasks(settings).items()
    }
    return build_llm_task_providers(configs)


def test_by_default_eleven_tasks_share_one_provider_and_the_decider_has_its_own() -> (
    None
):
    providers = _providers(_settings())
    shared = {id(p) for task, p in providers.items() if task is not LLMTask.DECIDER}
    assert len(shared) == 1
    assert id(providers[LLMTask.DECIDER]) not in shared


def test_tasks_configured_alike_share_and_the_rest_split() -> None:
    nano = LLMTaskSettings(model="nano")
    providers = _providers(_settings({"boundary": nano, "summary": nano}))
    assert providers[LLMTask.BOUNDARY] is providers[LLMTask.SUMMARY]
    assert providers[LLMTask.BOUNDARY] is not providers[LLMTask.EPISODE]
    assert len({id(p) for p in providers.values()}) == 3


def test_the_decider_is_never_shared_even_when_configured_alike() -> None:
    """Its SDK retry count is fixed at construction and must stay its own."""
    same = LLMTaskSettings(
        timeout_seconds=60.0, max_tokens=512, sdk_max_retries=0, invalid_reply_retries=1
    )
    providers = _providers(_settings({"search": same}))
    assert providers[LLMTask.DECIDER] is not providers[LLMTask.SEARCH]


# ── what reaches the wire ────────────────────────────────────────────────────


class _Reply(StrictOutputModel):
    text: str


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every request body sent by a provider built after this fixture runs."""
    bodies: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        message = {"role": "assistant", "content": '{"text": "ok"}'}
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            },
        )

    real: Callable[..., openai.AsyncOpenAI] = openai.AsyncOpenAI

    def with_transport(**kwargs: Any) -> openai.AsyncOpenAI:
        transport = httpx.MockTransport(handle)
        return real(**kwargs, http_client=httpx.AsyncClient(transport=transport))

    monkeypatch.setattr(openai, "AsyncOpenAI", with_transport)
    return bodies


async def _send(provider: Any, **call: Any) -> None:
    await provider.achat(
        [LLMMessage(role="user", content="hi")], response_format=_Reply, **call
    )


async def test_configured_defaults_are_sent_when_a_call_names_none(
    wire: list[dict[str, Any]],
) -> None:
    settings = _settings({"temperature": 0.3, "max_tokens": 16384})
    await _send(_providers(settings)[LLMTask.EPISODE])
    (body,) = wire
    assert (body["temperature"], body["max_tokens"]) == (0.3, 16384)


async def test_a_calls_own_values_win_even_zero(wire: list[dict[str, Any]]) -> None:
    """everalgo's agentic retrieval asks for temperature 0.0 and must get it."""
    settings = _settings({"temperature": 0.3, "max_tokens": 16384})
    await _send(_providers(settings)[LLMTask.SEARCH], temperature=0.0, max_tokens=64)
    (body,) = wire
    assert (body["temperature"], body["max_tokens"]) == (0.0, 64)


async def test_unset_defaults_leave_both_out_of_the_request(
    wire: list[dict[str, Any]],
) -> None:
    """Only ``[llm]`` configured: each extraction request is what it was before."""
    providers = _providers(_settings())
    for task in set(LLMTask) - {LLMTask.DECIDER}:
        await _send(providers[task])
    assert {tuple(sorted(body)) for body in wire} == {
        ("messages", "model", "response_format")
    }


async def test_the_decider_sends_its_reply_cap_and_extra(
    wire: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap and ``extra`` that bound one round are the decider client's own.

    Moved here from per-call arguments; a Qwen endpoint gets
    ``enable_thinking=false`` through ``extra`` -- left on, an un-finetuned
    decider spends the whole deadline reasoning (39.3% of rounds on
    Qwen3.5-0.8B). Set through the deprecated names the launch scripts export.
    """
    monkeypatch.setenv("EVEROS_LLM__API_KEY", "sk-env")
    monkeypatch.setenv("EVEROS_DECIDER__MAX_TOKENS", "256")
    monkeypatch.setenv(
        "EVEROS_DECIDER__EXTRA",
        '{"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}',
    )
    await _send(_providers(Settings())[LLMTask.DECIDER])
    (body,) = wire
    assert body["max_tokens"] == 256
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


async def test_the_deciders_default_request_is_unchanged(
    wire: list[dict[str, Any]],
) -> None:
    """Before tasks the decider passed 512 per call; its client now sends it."""
    await _send(_providers(_settings())[LLMTask.DECIDER])
    (body,) = wire
    assert sorted(body) == ["max_tokens", "messages", "model", "response_format"]
    assert body["max_tokens"] == 512
