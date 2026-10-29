from __future__ import annotations

import datetime as _dt
from unittest.mock import AsyncMock, MagicMock, patch

from everalgo.types import ChatMessage, MemCell
from everalgo.types import Episode as AlgoEpisode

from everos.memory import IngestResult
from everos.memory.events import EpisodeExtracted, UserPipelineStarted
from everos.memory.extract.pipeline import PreparedCell, UserMemoryPipeline
from everos.memory.models import CanonicalMessage
from everos.memory.ports import MemoryRef


def _sample_memcell() -> MemCell:
    return MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="hello",
                timestamp=1_700_000_000_000,
                sender_id="u1",
            ),
        ],
        timestamp=1_700_000_000_000,
    )


_REF = MemoryRef(
    id="u1_ep_20260517_00000001",
    entry_id="ep_20260517_00000001",
    md_path="default_app/default_project/users/u1/episodes/episode-2026-05-17.md",
)


def _prepared() -> PreparedCell:
    return PreparedCell(cell=_sample_memcell(), memcell_id="mc_a", sender_ids=["u1"])


async def test_pipeline_started_leads_the_cell_events() -> None:
    """``UserPipelineStarted`` is the cell's first event, even when the cell
    has no user sender to extract an Episode for."""
    pipeline = UserMemoryPipeline(
        episode_store=MagicMock(),
        prompt_loader=MagicMock(),
        llm_client=MagicMock(),
    )
    cell = MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="assistant",
                content="hello",
                timestamp=1_700_000_000_000,
                sender_id="agent",
            ),
        ],
        timestamp=1_700_000_000_000,
    )
    ingested = IngestResult(
        session_id="s1", app_id="claude_code", project_id="oss", messages=[]
    )

    outcome = await pipeline.run_cell(
        ingested, PreparedCell(cell=cell, memcell_id="mc_a")
    )

    assert outcome.status == "extracted"
    assert outcome.message_count == 1
    (started,) = outcome.events
    assert isinstance(started, UserPipelineStarted)
    assert started.memcell_id == "mc_a"
    assert started.session_id == "s1"
    assert started.app_id == "claude_code"
    assert started.project_id == "oss"
    assert started.memcell is cell


async def test_no_llm_skips_the_cell_without_events() -> None:
    pipeline = UserMemoryPipeline(
        episode_store=MagicMock(), prompt_loader=MagicMock(), llm_client=None
    )

    outcome = await pipeline.run_cell(
        IngestResult(session_id="s1", messages=[]), _prepared()
    )

    assert outcome.status == "skipped"
    assert outcome.events == ()


async def test_emit_episode_extracted_after_md_write() -> None:
    """Each per-sender Episode write emits EpisodeExtracted with the md entry id."""
    episode_store = MagicMock()
    episode_store.append = AsyncMock(return_value=_REF)
    prompt_loader = MagicMock()
    prompt_loader.load = MagicMock(return_value="<prompt>")
    llm_client = MagicMock()

    pipeline = UserMemoryPipeline(
        episode_store=episode_store,
        prompt_loader=prompt_loader,
        llm_client=llm_client,
    )

    ingested = IngestResult(
        session_id="s1",
        messages=[
            CanonicalMessage(
                message_id="m1",
                session_id="s1",
                sender_id="u1",
                role="user",
                timestamp=_dt.datetime.fromtimestamp(1_700_000_000, tz=_dt.UTC),
                text="hello",
            )
        ],
    )
    algo_ep = AlgoEpisode(
        owner_id="u1", episode="they said hello", timestamp=1_700_000_000_000
    )
    with patch.object(
        pipeline._ep_ext, "aextract", new=AsyncMock(return_value=algo_ep)
    ):
        outcome = await pipeline.run_cell(ingested, _prepared())

    assert outcome.status == "extracted"
    assert outcome.md_paths == [_REF.md_path]
    assert isinstance(outcome.events[0], UserPipelineStarted)
    extracted = [e for e in outcome.events if isinstance(e, EpisodeExtracted)]
    assert len(extracted) == 1
    assert extracted[0].memcell_id == "mc_a"
    assert extracted[0].episode_entry_id == "ep_20260517_00000001"
    assert extracted[0].episode_text == "they said hello"
    assert extracted[0].episode_timestamp_ms == 1_700_000_000_000
    assert extracted[0].owner_id == "u1"
    assert extracted[0].session_id == "s1"
    assert extracted[0].source == "pipeline"


async def test_run_emits_extract_and_persist_spans() -> None:
    """pipeline.run opens an everos.extract generation span (where LLM token
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

    episode_store = MagicMock()
    episode_store.append = AsyncMock(return_value=_REF)
    prompt_loader = MagicMock()
    prompt_loader.load = MagicMock(return_value="<prompt>")
    pipeline = UserMemoryPipeline(
        episode_store=episode_store,
        prompt_loader=prompt_loader,
        llm_client=MagicMock(),
    )
    ingested = IngestResult(
        session_id="s1",
        messages=[
            CanonicalMessage(
                message_id="m1",
                session_id="s1",
                sender_id="u1",
                role="user",
                timestamp=_dt.datetime.fromtimestamp(1_700_000_000, tz=_dt.UTC),
                text="hello",
            )
        ],
    )
    algo_ep = AlgoEpisode(
        owner_id="u1", episode="they said hello", timestamp=1_700_000_000_000
    )

    exporter = InMemorySpanExporter()
    init_tracing(
        ObservabilitySettings(enabled=True, endpoint="http://collector.invalid"),
        span_processor=SimpleSpanProcessor(exporter),
    )
    try:
        with patch.object(
            pipeline._ep_ext, "aextract", new=AsyncMock(return_value=algo_ep)
        ):
            await pipeline.run_cell(ingested, _prepared())
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
    everos.persist.markdown carries the written .md path; off → neither."""
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

    episode_store = MagicMock()
    episode_store.append = AsyncMock(return_value=_REF)
    prompt_loader = MagicMock()
    prompt_loader.load = MagicMock(return_value="<prompt>")
    pipeline = UserMemoryPipeline(
        episode_store=episode_store,
        prompt_loader=prompt_loader,
        llm_client=MagicMock(),
    )
    ingested = IngestResult(
        session_id="s1",
        messages=[
            CanonicalMessage(
                message_id="m1",
                session_id="s1",
                sender_id="u1",
                role="user",
                timestamp=_dt.datetime.fromtimestamp(1_700_000_000, tz=_dt.UTC),
                text="hello",
            )
        ],
    )
    algo_ep = AlgoEpisode(
        owner_id="u1", episode="they said hello", timestamp=1_700_000_000_000
    )

    exporter = InMemorySpanExporter()
    init_tracing(
        ObservabilitySettings(enabled=True, endpoint="http://collector.invalid"),
        span_processor=SimpleSpanProcessor(exporter),
    )
    set_capture_content(True)
    try:
        with patch.object(
            pipeline._ep_ext, "aextract", new=AsyncMock(return_value=algo_ep)
        ):
            await pipeline.run_cell(ingested, _prepared())
        force_flush()
    finally:
        set_capture_content(False)
        shutdown_tracing()

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans["everos.extract"].attributes["langfuse.observation.output"] == (
        "they said hello"
    )
    assert (
        spans["everos.persist.markdown"].attributes["langfuse.observation.output"]
        == _REF.md_path
    )
