"""``build_application`` assembles the app; ``Application`` starts and stops it.

Assembly: one call turns settings into three facades over one set of
providers, stores and indexes, with every OME strategy registered whatever
the tier -- the tier lives in the providers the strategies are bound to, so
a tier change needs a restart. Building opens nothing and writes nothing.

Lifecycle: the server's ``start`` brings storage, index, cascade, OME and the
memorize queue up in that order (the queue publishes into OME, so it comes
last) and ``aclose`` unwinds exactly what was started, in reverse, without
one failing stop skipping the rest. A server without an extraction LLM
refuses to start; building the same application for a CLI command does not.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pytest

from everos.component.llm import LLMCaseJudge, LLMNotConfiguredError
from everos.config import Settings, load_settings
from everos.infra.ome import OmeEventBus
from everos.memory.strategies import STRATEGY_SPECS
from everos.service import (
    Application,
    KnowledgeService,
    MemoryService,
    OpsService,
    build_application,
    build_providers,
    configure_process,
)
from everos.service.bootstrap import _case_judge
from tests.fakes import FakeEmbeddingProvider, make_application, unscripted_llm
from tests.fakes.application import _ALL_CAPABILITIES

_STARTED = ["sqlite.create", "index.start", "ops.start", "ome.start", "memory.start"]


_PROVIDER_SECTIONS = ("LLM", "DECIDER", "MULTIMODAL", "EMBEDDING", "RERANK")


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Unconfigured settings over a fresh root, whatever the host carries.

    A developer's ``.env`` can reach the test process with real keys, so
    every provider key is blanked explicitly rather than assumed absent.
    """
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path / "root"))
    for section in _PROVIDER_SECTIONS:
        monkeypatch.setenv(f"EVEROS_{section}__API_KEY", "")
    load_settings.cache_clear()
    yield load_settings()
    load_settings.cache_clear()


def _build(settings: Settings, **providers: object) -> Application:
    return build_application(
        settings,
        providers=dataclasses.replace(build_providers(settings), **providers),
    )


# ── assembly ───────────────────────────────────────────────────────────────


def test_unconfigured_settings_build_three_facades_without_touching_disk(
    settings: Settings, tmp_path: Path
) -> None:
    application = build_application(settings)

    assert isinstance(application.memory, MemoryService)
    assert isinstance(application.knowledge, KnowledgeService)
    assert isinstance(application.ops, OpsService)
    snapshot = application.capabilities
    assert (snapshot.llm, snapshot.decider, snapshot.embedding, snapshot.rerank) == (
        False,
        False,
        False,
        False,
    )
    assert snapshot.index_backend == settings.index.backend
    assert not (tmp_path / "root").exists(), "building must not write anything"


def _with_level(settings: Settings, level: str) -> Settings:
    return settings.model_copy(
        update={"logging": settings.logging.model_copy(update={"level": level})}
    )


def test_building_configures_logging_from_the_logging_section(
    settings: Settings,
) -> None:
    build_application(_with_level(settings, "ERROR"))

    assert logging.getLogger().level == logging.ERROR


def test_configure_process_can_run_again_and_the_last_call_wins(
    settings: Settings,
) -> None:
    configure_process(_with_level(settings, "DEBUG"))
    configure_process(_with_level(settings, "WARNING"))

    assert logging.getLogger().level == logging.WARNING


@pytest.mark.parametrize("embedding", [True, False], ids=["tier2", "tier1"])
def test_every_strategy_registers_whatever_the_tier(
    settings: Settings, embedding: bool
) -> None:
    application = _build(
        settings, embedding=FakeEmbeddingProvider() if embedding else None
    )

    registered = {meta.name for meta in application._engine._registry.all()}
    assert registered == {spec.name for spec in STRATEGY_SPECS}
    assert application.capabilities.embedding is embedding


def test_memory_is_wired_over_one_engine_and_one_set_of_stores(
    settings: Settings,
) -> None:
    llm = unscripted_llm()
    application = _build(settings, llm=llm)
    memory = application.memory
    memorizer = memory._memorizer

    assert isinstance(memorizer._bus, OmeEventBus)
    assert memorizer._bus._engine is application._engine
    assert memory._search_manager._llm is llm
    assert memorizer._boundary._llm is llm
    assert memorizer._boundary._session_buffer is memory._search_manager._session_buffer
    assert memorizer._extractions is memory._extractions


@pytest.mark.parametrize(
    ("enabled", "has_llm", "judged"),
    [(True, True, True), (False, True, False), (True, False, False)],
)
def test_the_case_judge_exists_only_with_the_gate_on_and_an_llm(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, has_llm: bool, judged: bool
) -> None:
    monkeypatch.setenv("EVEROS_AGENT_MEMORY__CASE_GATE_ENABLED", str(enabled))
    load_settings.cache_clear()
    judge = _case_judge(load_settings(), unscripted_llm() if has_llm else None)
    assert isinstance(judge, LLMCaseJudge) is judged
    load_settings.cache_clear()


@pytest.mark.parametrize(
    ("value", "runs"),
    [("1", False), ("true", False), ("Yes", False), ("", True), ("0", True)],
)
def test_the_disable_switch_keeps_the_cascade_from_running(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, value: str, runs: bool
) -> None:
    """A retrieval-only server opts out; anything not clearly "yes" ingests."""
    monkeypatch.setenv("EVEROS_DISABLE_CASCADE", value)
    assert build_application(settings).ops._run_cascade is runs


# ── lifecycle ──────────────────────────────────────────────────────────────


async def test_start_brings_the_queue_up_last_and_aclose_unwinds_in_reverse(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    application = make_application(root=tmp_path, lifecycle=events)

    await application.start()
    assert events == _STARTED

    events.clear()
    await application.aclose()
    assert events == [
        "memory.aclose",
        "ome.stop",
        "ops.aclose",
        "index.close",
        "sqlite.dispose",
    ]


async def test_aclose_without_start_closes_only_the_storage(tmp_path: Path) -> None:
    """A CLI command opens storage without starting; closing must still close it."""
    events: list[str] = []
    application = make_application(root=tmp_path, lifecycle=events)

    await application.aclose()

    assert events == ["index.close", "sqlite.dispose"]


async def test_aclose_is_idempotent(tmp_path: Path) -> None:
    events: list[str] = []
    application = make_application(root=tmp_path, lifecycle=events)
    await application.start()
    await application.aclose()
    events.clear()

    await application.aclose()

    assert events == []


async def test_async_with_closes_on_exit_without_starting(tmp_path: Path) -> None:
    events: list[str] = []

    async with make_application(root=tmp_path, lifecycle=events) as application:
        assert events == []
        assert application.memory is not None

    assert events == ["index.close", "sqlite.dispose"]


async def test_a_failing_stop_does_not_skip_the_rest(tmp_path: Path) -> None:
    events: list[str] = []
    application = make_application(root=tmp_path, lifecycle=events)

    async def broken() -> None:
        raise RuntimeError("engine stop failed")

    application._engine.stop = broken  # type: ignore[method-assign]
    await application.start()
    events.clear()
    await application.aclose()

    assert events == ["memory.aclose", "ops.aclose", "index.close", "sqlite.dispose"]


async def test_a_server_without_an_extraction_llm_refuses_to_start(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    application = make_application(
        root=tmp_path,
        lifecycle=events,
        capabilities=dataclasses.replace(_ALL_CAPABILITIES, llm=False),
    )

    with pytest.raises(LLMNotConfiguredError):
        await application.start()
    assert events == [], "nothing is brought up before the check"
    await application.aclose()
