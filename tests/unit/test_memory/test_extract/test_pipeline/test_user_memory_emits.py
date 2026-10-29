"""``UserMemoryPipeline.run_cell``: one episode per cell, one copy per user sender.

The episode operator is real (everalgo ``EpisodeExtractor``); the model
answers from the user-track canned scripts.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from everalgo.llm import LLMOutputError
from everalgo.types import ChatMessage, MemCell

from everos.memory import IngestResult
from everos.memory.events import EpisodeExtracted, UserPipelineStarted
from everos.memory.extract.llm_retry import OUTPUT_ATTEMPTS
from everos.memory.extract.pipeline import PreparedCell, UserMemoryPipeline
from everos.memory.ports import MemoryRef
from tests.fakes import bundled_instructions, user_track_llm, user_track_script

_EPISODE = json.loads(user_track_script("episode"))

_REF = MemoryRef(
    id="u1_ep_20260517_00000001",
    entry_id="ep_20260517_00000001",
    md_path="default_app/default_project/users/u1/episodes/episode-2026-05-17.md",
)


def _message(role: str, sender: str) -> ChatMessage:
    return ChatMessage(
        id=f"m_{sender}",
        role=role,
        content="hello",
        timestamp=1_700_000_000_000,
        sender_id=sender,
    )


def _prepared(*senders: str) -> PreparedCell:
    cell = MemCell(
        items=[_message("user", s) for s in senders or ("u1",)],
        timestamp=1_700_000_000_000,
    )
    return PreparedCell(cell=cell, memcell_id="mc_a", sender_ids=list(senders))


def _pipeline(llm: object, episode_store: object | None = None) -> UserMemoryPipeline:
    if episode_store is None:
        episode_store = MagicMock()
        episode_store.append = AsyncMock(return_value=_REF)
    return UserMemoryPipeline(
        episode_store=episode_store,  # type: ignore[arg-type]  # recording double
        instructions=bundled_instructions(),
        llm_client=llm,  # type: ignore[arg-type]  # FakeLLMClient or None
    )


_INGESTED = IngestResult(session_id="s1", messages=[])


async def test_pipeline_started_leads_the_cell_events() -> None:
    """``UserPipelineStarted`` is the cell's first event, even when the cell
    has no user sender to extract an Episode for."""
    llm = user_track_llm()
    cell = MemCell(items=[_message("assistant", "agent")], timestamp=1_700_000_000_000)
    ingested = IngestResult(
        session_id="s1", app_id="claude_code", project_id="oss", messages=[]
    )

    outcome = await _pipeline(llm).run_cell(
        ingested, PreparedCell(cell=cell, memcell_id="mc_a")
    )

    assert outcome.status == "extracted"
    assert outcome.message_count == 1
    (started,) = outcome.events
    assert isinstance(started, UserPipelineStarted)
    assert (started.memcell_id, started.session_id) == ("mc_a", "s1")
    assert (started.app_id, started.project_id) == ("claude_code", "oss")
    assert started.memcell is cell
    assert llm.call_count == 0


async def test_no_llm_skips_the_cell_without_events() -> None:
    outcome = await _pipeline(None).run_cell(_INGESTED, _prepared("u1"))

    assert outcome.status == "skipped"
    assert outcome.events == ()


async def test_emit_episode_extracted_after_md_write() -> None:
    """Each per-sender Episode write emits EpisodeExtracted with the md entry id."""
    episode_store = MagicMock()
    episode_store.append = AsyncMock(return_value=_REF)

    outcome = await _pipeline(user_track_llm(), episode_store).run_cell(
        _INGESTED, _prepared("u1")
    )

    assert outcome.status == "extracted"
    assert outcome.md_paths == [_REF.md_path]
    assert isinstance(outcome.events[0], UserPipelineStarted)
    (extracted,) = [e for e in outcome.events if isinstance(e, EpisodeExtracted)]
    assert extracted.memcell_id == "mc_a"
    assert extracted.episode_entry_id == "ep_20260517_00000001"
    assert extracted.episode_text == _EPISODE["content"]
    assert extracted.episode_timestamp_ms == 1_700_000_000_000
    assert (extracted.owner_id, extracted.session_id) == ("u1", "s1")
    assert extracted.source == "pipeline"
    (written,) = [call.args[1] for call in episode_store.append.await_args_list]
    assert written.summary == _EPISODE["summary"]
    assert written.subject == _EPISODE["title"]
    assert written.parent_id == "mc_a"


async def test_one_call_fans_out_to_every_user_sender() -> None:
    llm = user_track_llm()
    episode_store = MagicMock()
    episode_store.append = AsyncMock(return_value=_REF)

    outcome = await _pipeline(llm, episode_store).run_cell(
        _INGESTED, _prepared("u1", "u2")
    )

    owners = [e.owner_id for e in outcome.events if isinstance(e, EpisodeExtracted)]
    assert owners == ["u1", "u2"]
    assert llm.call_count == 1


async def test_a_blank_episode_is_tried_again_then_fails_the_cell() -> None:
    """A blank reply is unusable: retried, then the cell fails, nothing written."""
    llm = user_track_llm(episode="episode_blank")
    episode_store = MagicMock()
    episode_store.append = AsyncMock(return_value=_REF)

    with pytest.raises(LLMOutputError):
        await _pipeline(llm, episode_store).run_cell(_INGESTED, _prepared("u1"))

    assert llm.call_count == OUTPUT_ATTEMPTS
    episode_store.append.assert_not_awaited()


async def test_run_emits_extract_and_persist_spans() -> None:
    """run_cell opens an everos.extract generation span (where LLM token
    usage lands) and an everos.persist.markdown span around the md write."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from everos.config.settings import ObservabilitySettings
    from everos.core.observability.tracing import (
        force_flush,
        init_tracing,
        shutdown_tracing,
    )

    exporter = InMemorySpanExporter()
    init_tracing(
        ObservabilitySettings(enabled=True, endpoint="http://collector.invalid"),
        span_processor=SimpleSpanProcessor(exporter),
    )
    try:
        await _pipeline(user_track_llm()).run_cell(_INGESTED, _prepared("u1"))
        force_flush()
        spans = {s.name: s for s in exporter.get_finished_spans()}
        assert "everos.extract" in spans
        assert "everos.persist.markdown" in spans
        assert spans["everos.extract"].attributes["langfuse.observation.type"] == (
            "generation"
        )
    finally:
        shutdown_tracing()


async def test_extract_persist_capture_content_when_on() -> None:
    """With capture_content on, everos.extract carries the episode text and
    everos.persist.markdown carries the written .md path."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from everos.config.settings import ObservabilitySettings
    from everos.core.observability.tracing import (
        force_flush,
        init_tracing,
        set_capture_content,
        shutdown_tracing,
    )

    exporter = InMemorySpanExporter()
    init_tracing(
        ObservabilitySettings(enabled=True, endpoint="http://collector.invalid"),
        span_processor=SimpleSpanProcessor(exporter),
    )
    set_capture_content(True)
    try:
        await _pipeline(user_track_llm()).run_cell(_INGESTED, _prepared("u1"))
        force_flush()
    finally:
        set_capture_content(False)
        shutdown_tracing()

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert (
        spans["everos.extract"].attributes["langfuse.observation.output"]
        == (_EPISODE["content"])
    )
    assert (
        spans["everos.persist.markdown"].attributes["langfuse.observation.output"]
        == _REF.md_path
    )
