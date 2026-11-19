"""``[llm.<task>]`` sections: how they load, and what a mistyped one does.

A task section is optional and takes any of ``[llm]``'s fields; TOML and the
nested ``EVEROS_LLM__<TASK>__<FIELD>`` variables both reach it. ``[llm]`` and
its task sections are the one place that rejects unknown keys -- a mistyped
task or field would otherwise be ignored and the task silently keep ``[llm]``'s
model -- and the message lists what is valid. Every other section keeps
ignoring unknown keys.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from everos.config import (
    LLMClientSettings,
    LLMSettings,
    LLMTask,
    LLMTaskSettings,
    Settings,
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No developer ``EVEROS_*`` variable may reach these loads."""
    for key in list(os.environ):
        if key.startswith("EVEROS_") and key != "EVEROS_ROOT":
            monkeypatch.delenv(key, raising=False)


def _load(tmp_path: Path, toml: str) -> Settings:
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    (root / "everos.toml").write_text(toml, encoding="utf-8")
    return Settings(_everos_root=root)


# ── loading ──────────────────────────────────────────────────────────────────


def test_a_toml_task_section_loads(tmp_path: Path) -> None:
    settings = _load(
        tmp_path,
        '[llm.boundary]\nmodel = "openai/gpt-4.1-nano"\n'
        '[llm.profile]\nbase_url = "http://localhost:8000/v1"\napi_key = "local"\n',
    )
    assert settings.llm.boundary.model == "openai/gpt-4.1-nano"
    assert settings.llm.profile.base_url == "http://localhost:8000/v1"
    assert settings.llm.profile.api_key is not None
    assert settings.llm.model == "openai/gpt-4.1-mini", "[llm] is untouched"


def test_nested_env_variables_reach_a_task_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVEROS_LLM__BOUNDARY__MODEL", "nano")
    monkeypatch.setenv("EVEROS_LLM__BOUNDARY__TEMPERATURE", "0.2")
    monkeypatch.setenv("EVEROS_LLM__SEARCH__EXTRA", '{"extra_body": {"a": 1}}')

    llm = Settings().llm
    assert (llm.boundary.model, llm.boundary.temperature) == ("nano", 0.2)
    assert llm.search.extra == {"extra_body": {"a": 1}}


def test_an_absent_section_sets_nothing() -> None:
    section = Settings().llm.episode
    assert all(getattr(section, name) is None for name in LLMTaskSettings.model_fields)


def test_the_new_llm_fields_default_to_unset() -> None:
    """Unset, the request carries no temperature or cap, as before they existed."""
    llm = Settings().llm
    assert (llm.temperature, llm.max_tokens, llm.sdk_max_retries) == (None, None, None)


# ── what a mistake does ──────────────────────────────────────────────────────


def test_an_unknown_task_fails_and_lists_the_tasks(tmp_path: Path) -> None:
    with pytest.raises(ValidationError) as caught:
        _load(tmp_path, '[llm.boundry]\nmodel = "nano"\n')
    message = str(caught.value)
    assert "'boundry'" in message
    assert all(task.value in message for task in LLMTask)


def test_an_unknown_field_in_a_task_fails_and_lists_the_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVEROS_LLM__PROFILE__MODLE", "qwen3-32b")
    with pytest.raises(ValidationError) as caught:
        Settings()
    message = str(caught.value)
    assert "llm.profile" in message and "'modle'" in message
    assert all(name in message for name in LLMClientSettings.model_fields)


def test_an_unknown_field_in_llm_fails(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="'temprature'"):
        _load(tmp_path, "[llm]\ntemprature = 0.3\n")


def test_the_error_does_not_echo_a_key(tmp_path: Path) -> None:
    """A mistyped ``api_key`` holds the key itself; the message must not print it."""
    with pytest.raises(ValidationError) as caught:
        _load(tmp_path, '[llm]\napi_kye = "sk-or-v1-secret-tail-0123456789"\n')
    assert "secret-tail" not in str(caught.value)


def test_other_sections_still_ignore_unknown_keys(tmp_path: Path) -> None:
    settings = _load(tmp_path, "[memory]\nnot_a_field = 1\n[decider]\nbogus = 2\n")
    assert settings.memory.timezone == "UTC"


def test_values_are_validated_in_a_task_section() -> None:
    with pytest.raises(ValidationError):
        LLMTaskSettings(max_tokens=0)
    with pytest.raises(ValidationError):
        LLMTaskSettings(response_format_mode="xml")  # type: ignore[arg-type]  # invalid on purpose


def test_blank_strings_in_a_task_section_mean_unset() -> None:
    section = LLMTaskSettings(model="", api_key="", base_url="")  # type: ignore[arg-type]  # str coerces to SecretStr
    assert (section.model, section.api_key, section.base_url) == (None, None, None)


# ── one field set, one task list ─────────────────────────────────────────────


def test_a_task_section_takes_exactly_llms_fields() -> None:
    """The same field set, so anything ``[llm]`` can say a task can override."""
    assert list(LLMTaskSettings.model_fields) == list(LLMClientSettings.model_fields)


def test_llm_has_one_section_per_task() -> None:
    task_fields = set(LLMSettings.model_fields) - set(LLMClientSettings.model_fields)
    assert task_fields == {task.value for task in LLMTask}
    for name in task_fields:
        assert LLMSettings.model_fields[name].annotation is LLMTaskSettings


def test_the_task_names_are_stable() -> None:
    """Section names in users' configs: renaming one breaks every config using it."""
    assert [task.value for task in LLMTask] == [
        "boundary",
        "episode",
        "summary",
        "atomic_fact",
        "foresight",
        "profile",
        "reflection",
        "agent_case",
        "agent_skill",
        "knowledge",
        "search",
        "decider",
    ]
