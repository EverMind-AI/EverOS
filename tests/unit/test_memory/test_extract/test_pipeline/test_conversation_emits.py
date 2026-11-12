"""``ConversationPipeline.run_cell``: one episode per cell, in the cell's space.

A one-to-one cell is the one person's own memory; a group's cell is the
group's, announced with its human speakers. Nothing is copied per sender.
The episode call is real (``pipeline.episode``, everalgo's rules); the model
answers from the user-track canned scripts.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from everalgo.llm import LLMOutputError
from everalgo.types import ChatMessage, MemCell

from everos.memory import IngestResult
from everos.memory.events import EpisodeExtracted
from everos.memory.extract.llm_retry import OUTPUT_ATTEMPTS
from everos.memory.extract.pipeline import ConversationPipeline, PreparedCell
from everos.memory.models import Group, RosterEvent
from everos.memory.ports import MemoryRef
from tests.fakes import (
    bundled_instructions,
    memorize_llm,
    user_track_llm,
    user_track_script,
)
from tests.fakes.stores import (
    InMemoryEpisodeStore,
    InMemoryGroupStore,
    InMemoryMemcellStore,
)

_EPISODE = json.loads(user_track_script("episode"))

_REF = MemoryRef(
    id="u1_ep_20260517_00000001",
    entry_id="ep_20260517_00000001",
    md_path="default_app/default_project/users/u1/episodes/episode-2026-05-17.md",
)


def _message(role: str, sender: str, content: str = "hello") -> ChatMessage:
    return ChatMessage(
        id=f"m_{sender}",
        role=role,
        content=content,
        timestamp=1_700_000_000_000,
        sender_id=sender,
    )


def _prepared(*senders: str, said: str = "hello") -> PreparedCell:
    cell = MemCell(
        items=[_message("user", s, said) for s in senders or ("u1",)],
        timestamp=1_700_000_000_000,
    )
    return PreparedCell(cell=cell, memcell_id="mc_a", sender_ids=list(senders))


def _pipeline(
    llm: object,
    episode_store: object | None = None,
    *,
    memcell_exists: bool = True,
) -> ConversationPipeline:
    if episode_store is None:
        episode_store = MagicMock()
        episode_store.append = AsyncMock(return_value=_REF)
    memcells = InMemoryMemcellStore()
    if memcell_exists:
        memcells.seed("mc_a", _prepared().cell)
    groups = InMemoryGroupStore()
    key = ("default", "default", "trip")
    groups.groups[key] = Group(group_id="trip", created_at=0)
    groups.events[key].extend(
        [
            RosterEvent(op="join", member_id="u1", at=0, display_name="Uma"),
            RosterEvent(op="join", member_id="u2", at=0, display_name="Ugo"),
            RosterEvent(
                op="join", member_id="bot", at=0, kind="assistant", role="assistant"
            ),
        ]
    )
    return ConversationPipeline(
        episode_store=episode_store,  # type: ignore[arg-type]  # recording double
        memcell_store=memcells,
        groups=groups,
        instructions=bundled_instructions(),
        llm_client=llm,  # type: ignore[arg-type]  # FakeLLMClient
    )


_INGESTED = IngestResult(session_id="s1", messages=[])


async def test_a_cell_without_a_user_sender_extracts_and_announces_nothing() -> None:
    """A cell with no user sender has no Episode to extract: no model call,
    no md write and no event, but its messages still count."""
    llm = user_track_llm()
    episode_store = MagicMock()
    episode_store.append = AsyncMock(return_value=_REF)
    cell = MemCell(items=[_message("assistant", "agent")], timestamp=1_700_000_000_000)

    outcome = await _pipeline(llm, episode_store).run_cell(
        _INGESTED, PreparedCell(cell=cell, memcell_id="mc_a")
    )

    assert outcome.message_count == 1
    assert not outcome.events
    assert not outcome.md_paths
    assert llm.call_count == 0
    episode_store.append.assert_not_awaited()


async def test_emit_episode_extracted_after_md_write() -> None:
    """The episode write emits EpisodeExtracted with the md entry id."""
    episode_store = MagicMock()
    episode_store.append = AsyncMock(return_value=_REF)

    outcome = await _pipeline(user_track_llm(), episode_store).run_cell(
        _INGESTED, _prepared("u1")
    )

    assert outcome.md_paths == [_REF.md_path]
    (extracted,) = outcome.events
    assert isinstance(extracted, EpisodeExtracted)
    assert extracted.memcell_id == "mc_a"
    assert extracted.episode_entry_id == "ep_20260517_00000001"
    assert extracted.episode_text == _EPISODE["content"]
    assert extracted.episode_timestamp_ms == 1_700_000_000_000
    assert (extracted.owner_id, extracted.session_id) == ("u1", "s1")
    assert (extracted.owner_type, extracted.speaker_ids) == ("user", [])
    assert extracted.group_key is None
    (written,) = [call.args[1] for call in episode_store.append.await_args_list]
    assert written.summary == _EPISODE["summary"]
    assert written.subject == _EPISODE["title"]
    assert written.parent_id == "mc_a"


@pytest.mark.parametrize(
    ("said", "content", "language"),
    [
        # The conversation decides, whatever language the reply came back in.
        ("hello", "The user booked a trip to Lisbon for the spring.", "English"),
        ("hello", "用户预订了春天去里斯本的行程。", "English"),
        ("你好", "The user booked a trip to Lisbon for the spring.", "Chinese"),
        # A conversation without a letter leaves it to the episode's text.
        ("2026-05-17 12:00", "用户预订了春天去里斯本的行程。", "Chinese"),
        ("2026-05-17 12:00", "2026-05-17 12:00", None),
    ],
)
async def test_episode_and_event_carry_the_conversation_s_language(
    said: str, content: str, language: str | None
) -> None:
    """The language is resolved once per cell and pinned on the episode."""
    episode_store = MagicMock()
    episode_store.append = AsyncMock(return_value=_REF)

    outcome = await _pipeline(
        memorize_llm(episode_content=content), episode_store
    ).run_cell(_INGESTED, _prepared("u1", said=said))

    events = [e for e in outcome.events if isinstance(e, EpisodeExtracted)]
    written = [call.args[1] for call in episode_store.append.await_args_list]
    assert [e.language for e in events] == [language]
    assert [ep.language for ep in written] == [language]


async def test_a_group_cell_is_one_episode_of_the_group() -> None:
    """Two people in a group's cell: one call, one episode under the group,
    its event naming the group and the people who spoke -- no copy per
    sender."""
    llm = user_track_llm()
    episode_store = MagicMock()
    episode_store.append = AsyncMock(return_value=_REF)
    cell = MemCell(
        items=[
            _message("user", "u1"),
            _message("user", "u2"),
            _message("assistant", "bot"),
        ],
        timestamp=1_700_000_000_000,
    )

    outcome = await _pipeline(llm, episode_store).run_cell(
        IngestResult(session_id="s1", messages=[], group_id="trip"),
        PreparedCell(cell=cell, memcell_id="mc_a", sender_ids=["u1", "u2", "bot"]),
    )

    assert llm.call_count == 1
    (scope, written) = episode_store.append.await_args.args
    assert (scope.owner_type, scope.owner_id) == ("group", "trip")
    assert written.sender_ids == ["u1", "u2", "bot"]
    assert written.origin == "mixed"
    (event,) = outcome.events
    assert isinstance(event, EpisodeExtracted)
    assert (event.owner_type, event.owner_id) == ("group", "trip")
    assert event.speaker_ids == ["u1", "u2"]
    assert event.group_key == "default/default/trip"


async def test_several_people_outside_a_group_are_never_copied_per_sender() -> None:
    """A session of two or more people is an implicit group's, decided before
    its cells are cut; a cell of several people without a group is refused
    rather than copied into each sender's memory."""
    llm = user_track_llm()
    episode_store = MagicMock()
    episode_store.append = AsyncMock(return_value=_REF)

    with pytest.raises(ValueError, match="several human senders"):
        await _pipeline(llm, episode_store).run_cell(_INGESTED, _prepared("u1", "u2"))

    assert llm.call_count == 0
    episode_store.append.assert_not_awaited()


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
