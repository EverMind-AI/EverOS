"""Every LLM consumer the bootstrap wires gets its own task's client.

Each task is given a model of its own, then each consumer is asked which model
its client names: boundary detection, the episode and summary calls, agentic
search, the decider, the case gate, knowledge extraction, and the strategy
dependencies (each strategy picking its own attribute is pinned beside it, in
``tests/unit/test_memory/test_strategies``). Startup logs where every task's
settings came from, never the key, and warns when the deprecated ``[decider]``
spelling is in use.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from everos.config import LLMTask, Settings, load_settings
from everos.memory.deps import LLMClients, MemoryDeps
from everos.service import bootstrap as bootstrap_mod
from everos.service import build_application, build_providers, describe_llm_tasks

_KEY = "sk-test-not-for-logs"


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """Every task on a model named after it; the case gate on; nothing optional."""
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("EVEROS_LLM__API_KEY", _KEY)
    for task in LLMTask:
        monkeypatch.setenv(f"EVEROS_LLM__{task.upper()}__MODEL", f"m-{task}")
    for section in ("DECIDER", "MULTIMODAL", "EMBEDDING", "RERANK"):
        monkeypatch.setenv(f"EVEROS_{section}__API_KEY", "")
    monkeypatch.setenv("EVEROS_AGENT_MEMORY__CASE_GATE_ENABLED", "true")
    load_settings.cache_clear()
    yield load_settings()
    load_settings.cache_clear()


def _spy(monkeypatch: pytest.MonkeyPatch, name: str) -> list[tuple[Any, ...]]:
    """Record the arguments of every call to the bootstrap's ``name``."""
    real: Callable[..., Any] = getattr(bootstrap_mod, name)
    calls: list[tuple[Any, ...]] = []

    def record(*args: Any, **kwargs: Any) -> Any:
        calls.append((*args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(bootstrap_mod, name, record)
    return calls


def _model(client: Any) -> str:
    return client._model


def test_each_consumer_gets_its_own_tasks_client(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    boundary = _spy(monkeypatch, "BoundaryDetector")
    pipeline = _spy(monkeypatch, "UserMemoryPipeline")
    search = _spy(monkeypatch, "SearchManager")
    judge = _spy(monkeypatch, "LLMCaseJudge")
    knowledge = _spy(monkeypatch, "_knowledge_extractor")

    build_application(settings)

    assert _model(boundary[0][-1]["llm"]) == "m-boundary"
    assert _model(pipeline[0][-1]["episode_llm"]) == "m-episode"
    assert _model(pipeline[0][-1]["summary_llm"]) == "m-summary"
    assert _model(search[0][-1]["llm_client"]) == "m-search"
    assert _model(search[0][-1]["decider_client"]) == "m-decider"
    assert _model(judge[0][0]) == "m-agent_case"
    assert _model(knowledge[0][0]) == "m-knowledge"


def test_strategies_are_bound_to_every_tasks_client(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    bound = _spy(monkeypatch, "to_ome_strategy")

    build_application(settings)

    deps: MemoryDeps = bound[0][1]
    assert {f.name: _model(getattr(deps.llms, f.name)) for f in _tasks()} == {
        task.value: f"m-{task}" for task in LLMTask
    }


def test_the_client_bundle_has_one_client_per_task() -> None:
    """The domain's bundle and the configuration's task list cannot drift apart."""
    assert [f.name for f in _tasks()] == [task.value for task in LLMTask]


def _tasks() -> tuple[dataclasses.Field[Any], ...]:
    return dataclasses.fields(LLMClients)


def test_resolutions_are_described_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``everos config show`` must work before a key is configured."""
    monkeypatch.setenv("EVEROS_LLM__API_KEY", "")
    load_settings.cache_clear()
    described = describe_llm_tasks(load_settings())
    assert [r.task for r in described] == list(LLMTask)


# ── startup log ──────────────────────────────────────────────────────────────


class _RecordingLogger:
    """Stands in for the bootstrap's module logger and keeps what it logs."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def info(self, event: str, **fields: Any) -> None:
        self.records.append({"level": "info", "event": event, **fields})

    def warning(self, event: str, **fields: Any) -> None:
        self.records.append({"level": "warning", "event": event, **fields})

    def events(self, event: str) -> list[dict[str, Any]]:
        return [r for r in self.records if r["event"] == event]


@pytest.fixture
def logs(monkeypatch: pytest.MonkeyPatch) -> _RecordingLogger:
    recorder = _RecordingLogger()
    monkeypatch.setattr(bootstrap_mod, "logger", recorder)
    return recorder


def test_every_task_logs_its_model_host_and_sources(
    settings: Settings, logs: _RecordingLogger
) -> None:
    build_providers(settings)

    resolved = logs.events("llm.task.resolved")
    assert [r["task"] for r in resolved] == [task.value for task in LLMTask]
    boundary = resolved[0]
    assert (boundary["model"], boundary["host"]) == ("m-boundary", "openrouter.ai")
    assert boundary["sources"]["model"] == "llm.boundary"
    assert boundary["sources"]["api_key"] == "llm"


def test_the_key_is_never_logged(settings: Settings, logs: _RecordingLogger) -> None:
    build_providers(settings)
    assert logs.records
    assert _KEY not in repr(logs.records)


def test_the_deprecated_decider_section_is_warned_about(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, logs: _RecordingLogger
) -> None:
    monkeypatch.setenv("EVEROS_DECIDER__TIMEOUT_SECONDS", "600")
    monkeypatch.setenv("EVEROS_DECIDER__MAX_TOKENS", "256")
    load_settings.cache_clear()

    build_providers(load_settings())

    (warning,) = logs.events("llm.decider.deprecated_section")
    assert warning["level"] == "warning"
    assert warning["fields"] == ["timeout_seconds", "max_tokens"]
    assert "[llm.decider]" in warning["hint"]


def test_no_warning_without_the_deprecated_section(
    settings: Settings, logs: _RecordingLogger
) -> None:
    build_providers(settings)
    assert logs.events("llm.decider.deprecated_section") == []
