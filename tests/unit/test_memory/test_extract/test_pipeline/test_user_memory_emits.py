"""``UserMemoryPipeline.run_cell``: one episode per cell, one copy per user sender.

The episode operator is real (everalgo ``EpisodeExtractor``); the model answers
from the user-track canned scripts. A cell costs two model calls: the language
decision, then the episode written in that language. Every copy is written with
the placeholder summary and the decided language (else the one read off the
episode text), and one ``EpisodeExtracted`` lists the copies written.
"""

from __future__ import annotations

import json
from datetime import timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog
from everalgo.llm import LLMOutputError
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, MemCell

from everos.memory import IngestResult
from everos.memory.episode_summary import placeholder_summary
from everos.memory.events import EpisodeCopy, EpisodeExtracted
from everos.memory.extract.llm_retry import OUTPUT_ATTEMPTS
from everos.memory.extract.pipeline import PreparedCell, UserMemoryPipeline
from everos.memory.extract.trace import collect_construction_trace
from everos.memory.ports import MemoryRef
from tests.fakes import (
    bundled_instructions,
    memorize_llm,
    user_track_llm,
    user_track_script,
)
from tests.fakes.stores import InMemoryEpisodeStore, InMemoryMemcellStore

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
        language_llm=llm,  # type: ignore[arg-type]  # FakeLLMClient
        episode_llm=llm,  # type: ignore[arg-type]  # FakeLLMClient
    )


_INGESTED = IngestResult(session_id="s1", messages=[])


async def test_a_cell_without_a_user_sender_emits_no_event() -> None:
    """A cell with no user sender to extract an Episode for costs no model
    call and announces nothing."""
    llm = user_track_llm()
    cell = MemCell(items=[_message("assistant", "agent")], timestamp=1_700_000_000_000)
    ingested = IngestResult(
        session_id="s1", app_id="claude_code", project_id="oss", messages=[]
    )

    outcome = await _pipeline(llm).run_cell(
        ingested, PreparedCell(cell=cell, memcell_id="mc_a")
    )

    assert outcome.message_count == 1
    assert outcome.md_paths == []
    assert outcome.events == []
    assert llm.call_count == 0


async def test_emit_episode_extracted_after_md_write() -> None:
    """The cell's episode event names the written copy by its md entry id."""
    episode_store = MagicMock()
    episode_store.append = AsyncMock(return_value=_REF)

    with collect_construction_trace() as events:
        outcome = await _pipeline(user_track_llm(), episode_store).run_cell(
            _INGESTED, _prepared("u1")
        )

    assert [event["phase"] for event in events] == [
        "extraction_attempt",
        "extraction_result",
        "persisted_episode",
    ]
    assert events[1]["content"] == _EPISODE["content"]
    assert events[2]["entry_id"] == _REF.entry_id
    assert outcome.md_paths == [_REF.md_path]
    (extracted,) = outcome.events
    assert isinstance(extracted, EpisodeExtracted)
    assert extracted.memcell_id == "mc_a"
    assert extracted.copies == [
        EpisodeCopy(owner_id="u1", entry_id=_REF.entry_id, owner_name="u1")
    ]
    assert extracted.episode_text == _EPISODE["content"]
    assert extracted.episode_timestamp_ms == 1_700_000_000_000
    assert extracted.session_id == "s1"
    assert extracted.source == "pipeline"
    (written,) = [call.args[1] for call in episode_store.append.await_args_list]
    assert written.subject == _EPISODE["title"]
    assert written.parent_id == "mc_a"


async def test_every_copy_is_written_with_the_placeholder_summary() -> None:
    """S1 / S5: the only summary written online is the placeholder."""
    episodes = InMemoryEpisodeStore()

    await _pipeline(user_track_llm(), episodes).run_cell(
        _INGESTED, _prepared("u1", "u2")
    )

    placeholder = placeholder_summary(_EPISODE["content"])
    assert placeholder.endswith("\u2026")  # the scripted episode is over the cap
    assert [s.item.summary for s in episodes.records] == [placeholder, placeholder]
    assert [s.item.episode for s in episodes.records] == [_EPISODE["content"]] * 2


async def test_the_decided_language_writes_the_episode_and_is_pinned() -> None:
    """The episode is asked for in the decided language, and every copy and the
    event carry it -- even over the language its text reads as."""
    llm = memorize_llm(
        episode_content="The user booked a trip to Lisbon.", language="language_chinese"
    )
    episode_store = _recording_store()

    with structlog.testing.capture_logs() as logs:
        outcome = await _pipeline(llm, episode_store).run_cell(
            _INGESTED, _prepared("u1", "u2")
        )

    (episode_call,) = _calls_for(llm, "_EpisodeReply")
    assert f"Write every output field in Chinese{_NAMED}" in str(episode_call.messages)
    (event,) = outcome.events
    written = [call.args[1] for call in episode_store.append.await_args_list]
    assert event.language == "Chinese"
    assert [ep.language for ep in written] == ["Chinese", "Chinese"]
    assert _written_log(logs) == {
        "memcell_id": "mc_a",
        "owners": 2,
        "language": "Chinese",
        "language_from": "decided",
    }


@pytest.mark.parametrize(
    ("content", "language"),
    [
        ("The user booked a trip to Lisbon for the spring.", "English"),
        ("用户预订了春天去里斯本的行程。", "Chinese"),
        ("2026-05-17 12:00", None),
    ],
)
async def test_an_undecided_language_is_read_off_the_episode_text(
    content: str, language: str | None
) -> None:
    """A failed decision leaves the language to the model; what it wrote in is
    detected once and pinned on every copy."""
    llm = memorize_llm(episode_content=content, language="language_unusable")
    episode_store = _recording_store()

    with structlog.testing.capture_logs() as logs:
        outcome = await _pipeline(llm, episode_store).run_cell(
            _INGESTED, _prepared("u1", "u2")
        )

    (episode_call,) = _calls_for(llm, "_EpisodeReply")
    assert _NAMED not in str(episode_call.messages)
    (event,) = outcome.events
    written = [call.args[1] for call in episode_store.append.await_args_list]
    assert event.language == language
    assert [ep.language for ep in written] == [language, language]
    assert _written_log(logs)["language_from"] == "text"


_NAMED = ", whatever language the input is in"
"""How everalgo's rule for a caller-named language continues past the name."""


def _calls_for(llm: FakeLLMClient, reply: str) -> list[object]:
    return [call for call in llm.calls if call.response_format.__name__ == reply]


def _written_log(logs: list[dict[str, object]]) -> dict[str, object]:
    (line,) = [entry for entry in logs if entry["event"] == "episode_written"]
    return {k: line[k] for k in ("memcell_id", "owners", "language", "language_from")}


async def test_two_calls_one_event_lists_every_user_sender() -> None:
    """The language decision, then one episode call per cell, and one event
    naming each copy's own entry id."""
    llm = user_track_llm()
    episodes = InMemoryEpisodeStore()

    outcome = await _pipeline(llm, episodes).run_cell(_INGESTED, _prepared("u1", "u2"))

    (event,) = [e for e in outcome.events if isinstance(e, EpisodeExtracted)]
    assert [(c.owner_id, c.entry_id) for c in event.copies] == [
        (s.scope.owner_id, s.ref.entry_id) for s in episodes.records
    ]
    assert [c.owner_id for c in event.copies] == ["u1", "u2"]
    assert outcome.md_paths == [s.ref.md_path for s in episodes.records]
    assert _requested(llm) == ["_LanguageReply", "_EpisodeReply"]


async def test_a_copy_whose_write_was_skipped_is_not_listed() -> None:
    episode_store = MagicMock()
    episode_store.append = AsyncMock(side_effect=[_REF, None])

    outcome = await _pipeline(user_track_llm(), episode_store).run_cell(
        _INGESTED, _prepared("u1", "u2")
    )

    (event,) = [e for e in outcome.events if isinstance(e, EpisodeExtracted)]
    assert [c.owner_id for c in event.copies] == ["u1"]
    assert outcome.md_paths == [_REF.md_path]


async def test_a_blank_episode_is_tried_again_then_fails_the_cell() -> None:
    """A blank reply is unusable: retried, then the cell fails, nothing written.

    The language is decided once, before the attempts, not once per attempt.
    """
    llm = user_track_llm(episode="episode_blank")
    episode_store = MagicMock()
    episode_store.append = AsyncMock(return_value=_REF)

    with pytest.raises(LLMOutputError):
        await _pipeline(llm, episode_store).run_cell(_INGESTED, _prepared("u1"))

    assert _requested(llm).count("_LanguageReply") == 1
    assert _requested(llm).count("_EpisodeReply") == OUTPUT_ATTEMPTS
    episode_store.append.assert_not_awaited()


def _requested(llm: FakeLLMClient) -> list[str]:
    """The reply model each call asked for, in call order."""
    return [call.response_format.__name__ for call in llm.calls]


async def test_run_emits_extract_and_persist_spans() -> None:
    """run_cell opens generation spans for the language decision and the
    episode (where LLM token usage lands) and an everos.persist.markdown span
    around the md write."""
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
        assert "everos.persist.markdown" in spans
        for name in ("everos.extract.language", "everos.extract"):
            assert spans[name].attributes["langfuse.observation.type"] == ("generation")
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
    assert outcome.events == []


async def test_episode_written_when_memcell_exists() -> None:
    """Positive control for the test above: same cell, memcell present."""
    episodes = InMemoryEpisodeStore()

    outcome = await _pipeline(user_track_llm(), episodes).run_cell(
        _INGESTED, _prepared("u1")
    )

    assert len(episodes.records) == 1
    assert [type(e) for e in outcome.events] == [EpisodeExtracted]


async def test_the_episode_request_shows_times_in_the_configured_zone() -> None:
    """The call over the cell renders its message times in the pipeline's zone."""
    llm = user_track_llm()
    pipeline = UserMemoryPipeline(
        episode_store=_recording_store(),
        memcell_store=_alive_memcells(),
        instructions=bundled_instructions(),
        language_llm=user_track_llm(),
        episode_llm=llm,
        tz=timezone(timedelta(hours=8)),
    )

    await pipeline.run_cell(_INGESTED, _prepared("u1"))

    (call,) = llm.calls
    assert "UTC+08:00" in str(call.messages[-1].content)


def _recording_store() -> MagicMock:
    store = MagicMock()
    store.append = AsyncMock(return_value=_REF)
    return store


def _alive_memcells() -> InMemoryMemcellStore:
    memcells = InMemoryMemcellStore()
    memcells.seed("mc_a", _prepared().cell)
    return memcells


async def test_each_owner_event_carries_the_label_of_that_owners_lines() -> None:
    """``sender_name`` (stripped) when a line carries one, else the sender id."""
    cell = MemCell(
        items=[
            _message("user", "u1").model_copy(update={"sender_name": " Alice "}),
            _message("user", "u2"),
            _message("user", "u3").model_copy(update={"sender_name": "  "}),
            _message("assistant", "bot").model_copy(update={"sender_name": "Kai"}),
        ],
        timestamp=1_700_000_000_000,
    )
    prepared = PreparedCell(cell=cell, memcell_id="mc_a", sender_ids=["u1", "u2"])

    outcome = await _pipeline(user_track_llm()).run_cell(_INGESTED, prepared)

    (event,) = [e for e in outcome.events if isinstance(e, EpisodeExtracted)]
    labels = {c.owner_id: c.owner_name for c in event.copies}
    assert labels == {"u1": "Alice", "u2": "u2", "u3": "u3"}
