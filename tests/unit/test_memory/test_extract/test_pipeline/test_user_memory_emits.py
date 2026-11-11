"""``UserMemoryPipeline.run_cell``: one episode per cell, one copy per user sender.

The episode and summary operators are real (everalgo ``EpisodeExtractor`` and
``EpisodeSummarizer``); the model answers from the user-track canned scripts.
The summary call runs beside the episode call; when it fails, the episode keeps
extraction's own summary, the content's first 200 characters.
"""

from __future__ import annotations

import json
from datetime import timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from everalgo.llm import LLMOutputError
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, MemCell

from everos.memory import IngestResult
from everos.memory.events import EpisodeExtracted, UserPipelineStarted
from everos.memory.extract.llm_retry import OUTPUT_ATTEMPTS
from everos.memory.extract.pipeline import PreparedCell, UserMemoryPipeline
from everos.memory.ports import MemoryRef
from tests.fakes import (
    bundled_instructions,
    memorize_llm,
    user_track_llm,
    user_track_script,
)
from tests.fakes.stores import InMemoryEpisodeStore, InMemoryMemcellStore

_EPISODE = json.loads(user_track_script("episode"))
_SUMMARY = json.loads(user_track_script("episode_summary"))["summary"]

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


def _pipeline(
    llm: object,
    episode_store: object | None = None,
    *,
    memcell_exists: bool = True,
) -> UserMemoryPipeline:
    if episode_store is None:
        episode_store = MagicMock()
        episode_store.append = AsyncMock(return_value=_REF)
    memcells = InMemoryMemcellStore()
    if memcell_exists:
        memcells.seed("mc_a", _prepared().cell)
    return UserMemoryPipeline(
        episode_store=episode_store,  # type: ignore[arg-type]  # recording double
        memcell_store=memcells,
        instructions=bundled_instructions(),
        llm_client=llm,  # type: ignore[arg-type]  # FakeLLMClient
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

    assert outcome.message_count == 1
    (started,) = outcome.events
    assert isinstance(started, UserPipelineStarted)
    assert (started.memcell_id, started.session_id) == ("mc_a", "s1")
    assert (started.app_id, started.project_id) == ("claude_code", "oss")
    assert started.memcell is cell
    assert llm.call_count == 0


async def test_emit_episode_extracted_after_md_write() -> None:
    """Each per-sender Episode write emits EpisodeExtracted with the md entry id."""
    episode_store = MagicMock()
    episode_store.append = AsyncMock(return_value=_REF)

    outcome = await _pipeline(user_track_llm(), episode_store).run_cell(
        _INGESTED, _prepared("u1")
    )

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
    assert written.summary == _SUMMARY
    assert written.subject == _EPISODE["title"]
    assert written.parent_id == "mc_a"


@pytest.mark.parametrize(
    ("content", "language"),
    [
        ("The user booked a trip to Lisbon for the spring.", "English"),
        ("用户预订了春天去里斯本的行程。", "Chinese"),
        ("2026-05-17 12:00", None),
    ],
)
async def test_episode_and_event_carry_the_detected_language(
    content: str, language: str | None
) -> None:
    """The episode's language is detected once and pinned on every copy."""
    episode_store = MagicMock()
    episode_store.append = AsyncMock(return_value=_REF)

    outcome = await _pipeline(
        memorize_llm(episode_content=content), episode_store
    ).run_cell(_INGESTED, _prepared("u1", "u2"))

    events = [e for e in outcome.events if isinstance(e, EpisodeExtracted)]
    written = [call.args[1] for call in episode_store.append.await_args_list]
    assert [e.language for e in events] == [language, language]
    assert [ep.language for ep in written] == [language, language]


async def test_one_call_fans_out_to_every_user_sender() -> None:
    """One episode call and one summary call per cell, whatever the senders."""
    llm = user_track_llm()
    episode_store = MagicMock()
    episode_store.append = AsyncMock(return_value=_REF)

    outcome = await _pipeline(llm, episode_store).run_cell(
        _INGESTED, _prepared("u1", "u2")
    )

    owners = [e.owner_id for e in outcome.events if isinstance(e, EpisodeExtracted)]
    assert owners == ["u1", "u2"]
    assert sorted(_requested(llm)) == ["_EpisodeReply", "_SummaryReply"]


async def test_a_blank_episode_is_tried_again_then_fails_the_cell() -> None:
    """A blank reply is unusable: retried, then the cell fails, nothing written."""
    llm = user_track_llm(episode="episode_blank")
    episode_store = MagicMock()
    episode_store.append = AsyncMock(return_value=_REF)

    with pytest.raises(LLMOutputError):
        await _pipeline(llm, episode_store).run_cell(_INGESTED, _prepared("u1"))

    assert _requested(llm).count("_EpisodeReply") == OUTPUT_ATTEMPTS
    episode_store.append.assert_not_awaited()


async def test_an_unusable_summary_falls_back_to_extractions_own() -> None:
    """Retried, then the episode keeps the content's first 200 characters."""
    llm = user_track_llm(episode_summary="episode_summary_blank")
    episode_store = MagicMock()
    episode_store.append = AsyncMock(return_value=_REF)

    outcome = await _pipeline(llm, episode_store).run_cell(_INGESTED, _prepared("u1"))

    assert outcome.md_paths == [_REF.md_path]
    assert _requested(llm).count("_SummaryReply") == OUTPUT_ATTEMPTS
    (written,) = [call.args[1] for call in episode_store.append.await_args_list]
    assert written.summary == _EPISODE["content"][:200]
    assert written.episode == _EPISODE["content"]


async def test_a_failing_summary_call_never_fails_the_cell() -> None:
    """A client error on the summary call is not retried; the cell goes on."""
    episode = user_track_script("episode")

    def _answer(_messages: object, *, response_format: type, **_kwargs: object) -> str:
        if response_format.__name__ == "_SummaryReply":
            raise TimeoutError("summary endpoint timed out")
        return episode

    episode_store = MagicMock()
    episode_store.append = AsyncMock(return_value=_REF)

    outcome = await _pipeline(FakeLLMClient(handler=_answer), episode_store).run_cell(
        _INGESTED, _prepared("u1")
    )

    assert outcome.md_paths == [_REF.md_path]
    (written,) = [call.args[1] for call in episode_store.append.await_args_list]
    assert written.summary == _EPISODE["content"][:200]


def _requested(llm: FakeLLMClient) -> list[str]:
    """The reply model each call asked for, in call order."""
    return [call.response_format.__name__ for call in llm.calls]


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


async def test_add_returns_ok_and_emits_nothing_when_memcell_gone() -> None:
    """A23: a session deletion removed the memcell while the episode was being
    extracted. The cell still succeeds, writes no episode and announces none,
    so no downstream extraction starts from a deleted conversation."""
    episodes = InMemoryEpisodeStore()

    outcome = await _pipeline(
        user_track_llm(), episodes, memcell_exists=False
    ).run_cell(_INGESTED, _prepared("u1"))

    assert episodes.records == []
    assert outcome.md_paths == []
    assert [type(e) for e in outcome.events] == [UserPipelineStarted]


async def test_episode_written_when_memcell_exists() -> None:
    """Positive control for the test above: same cell, memcell present."""
    episodes = InMemoryEpisodeStore()

    outcome = await _pipeline(user_track_llm(), episodes).run_cell(
        _INGESTED, _prepared("u1")
    )

    assert len(episodes.records) == 1
    assert [type(e) for e in outcome.events] == [UserPipelineStarted, EpisodeExtracted]


async def test_episode_and_summary_requests_show_times_in_the_configured_zone() -> None:
    """Both calls over the cell render its message times in the pipeline's zone."""
    llm = user_track_llm()
    pipeline = UserMemoryPipeline(
        episode_store=_recording_store(),
        memcell_store=_alive_memcells(),
        instructions=bundled_instructions(),
        llm_client=llm,
        tz=timezone(timedelta(hours=8)),
    )

    await pipeline.run_cell(_INGESTED, _prepared("u1"))

    users = {c.response_format.__name__: str(c.messages[-1].content) for c in llm.calls}
    assert "UTC+08:00" in users["_EpisodeReply"]
    assert "UTC+08:00" in users["_SummaryReply"]


def _recording_store() -> MagicMock:
    store = MagicMock()
    store.append = AsyncMock(return_value=_REF)
    return store


def _alive_memcells() -> InMemoryMemcellStore:
    memcells = InMemoryMemcellStore()
    memcells.seed("mc_a", _prepared().cell)
    return memcells
