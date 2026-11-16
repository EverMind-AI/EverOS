"""The conversation pipeline and the profile cues: who is asked, and when they are kept.

What these pin:

- a cell of a space whose profiles read cues is sent the cue block, and its
  cues -- one record per human speaker, an empty one when nothing was marked
  -- are kept before any memory of the segment is written (its event starts
  the profile update that reads them); a one-to-one cell's two humans each
  get their record from the one call;
- a group cell is asked for its human speakers' cues only, never the
  assistant's;
- a space whose profiles read none is sent the request without the block and
  keeps no cue, and a re-extraction after a forget asks for none and changes
  none.
"""

from __future__ import annotations

import json
from typing import Any

from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, MemCell

from everos.memory import IngestResult
from everos.memory.extract.pipeline import (
    ConversationPipeline,
    CueConsumers,
    PreparedCell,
)
from everos.memory.models import Group, RosterEvent
from everos.memory.ports import PersonCues, Scope
from tests.fakes import bundled_instructions
from tests.fakes.stores import (
    InMemoryEpisodeStore,
    InMemoryGroupStore,
    InMemoryMemcellStore,
    InMemoryProfileCueStore,
)

_T0 = 1_700_000_000_000


def _message(index: int, sender: str, text: str, *, role: str = "user") -> ChatMessage:
    return ChatMessage(
        id=f"m_{index}",
        role=role,  # type: ignore[arg-type]
        content=text,
        timestamp=_T0 + index,
        sender_id=sender,
        sender_name=sender.removeprefix("u_").title(),
    )


_CELL = MemCell(
    items=[
        _message(0, "u_ann", "I teach maths at a high school."),
        _message(1, "u_bob", "Nice. I just got back from a run."),
        _message(2, "bot", "Sounds like a good day.", role="assistant"),
    ],
    timestamp=_T0 + 3,
)


class _Model:
    """Answers the episode call, with cues when it asks for them."""

    def __init__(self, cues: list[dict[str, Any]]) -> None:
        self.cues = cues
        self.asked: list[str] = []
        self.prompts: list[str] = []
        self.llm = FakeLLMClient(handler=self._answer)

    def _answer(self, messages: Any, *, response_format: type, **_: Any) -> str:
        self.asked.append(response_format.__name__)
        content = messages[-1].content
        self.prompts.append(content if isinstance(content, str) else str(content))
        reply: dict[str, Any] = {"title": "t", "content": "c"}
        if response_format.__name__.startswith("Cued"):
            reply["profile_cues"] = self.cues
        return json.dumps(reply)


class _OrderedEpisodes(InMemoryEpisodeStore):
    """Episodes that note whether the segment's cues were kept before them."""

    def __init__(self, cues: InMemoryProfileCueStore) -> None:
        super().__init__()
        self.cues = cues
        self.cues_before: list[bool] = []

    async def append(self, scope: Scope, episode: Any, **kwargs: Any) -> Any:
        self.cues_before.append(bool(self.cues.records))
        return await super().append(scope, episode, **kwargs)


def _pipeline(
    model: _Model, consumers: CueConsumers, cues: InMemoryProfileCueStore
) -> tuple[ConversationPipeline, _OrderedEpisodes]:
    memcells = InMemoryMemcellStore()
    memcells.seed("mc_a", _CELL)
    groups = InMemoryGroupStore()
    key = ("default", "default", "trip")
    groups.groups[key] = Group(group_id="trip", created_at=0)
    groups.events[key].extend(
        [
            RosterEvent(op="join", member_id="u_ann", at=0, display_name="Ann"),
            RosterEvent(op="join", member_id="u_bob", at=0, display_name="Bob"),
            RosterEvent(
                op="join", member_id="bot", at=0, kind="assistant", role="assistant"
            ),
        ]
    )
    episodes = _OrderedEpisodes(cues)
    pipeline = ConversationPipeline(
        episode_store=episodes,
        memcell_store=memcells,
        groups=groups,
        instructions=bundled_instructions(),
        llm_client=model.llm,  # type: ignore[arg-type]  # FakeLLMClient
        cue_consumers=consumers,
        profile_cues=cues,
    )
    return pipeline, episodes


_PREPARED = PreparedCell(cell=_CELL, memcell_id="mc_a", sender_ids=["u_ann", "u_bob"])


async def test_a_one_to_one_cell_keeps_each_human_s_cues_before_its_memories() -> None:
    model = _Model(
        [
            {"speaker": "Ann", "quote": "I teach maths"},
            {"speaker": "Sounds", "quote": "nothing"},
        ]
    )
    cues = InMemoryProfileCueStore()
    pipeline, episodes = _pipeline(model, CueConsumers(users=True), cues)

    outcome = await pipeline.run_cell(
        IngestResult(session_id="s1", messages=[]), _PREPARED
    )

    assert model.asked == ["CuedEpisodeReply"]
    assert "<profile_cues>" in model.prompts[0]
    assert cues.records == {
        ("mc_a", "u_ann"): PersonCues(message_ids=("m_0",)),
        ("mc_a", "u_bob"): PersonCues(),
    }
    assert episodes.cues_before == [True, True]  # one copy per human, cues first
    assert len(outcome.events) == 2


async def test_a_group_cell_marks_its_human_speakers_only() -> None:
    model = _Model(
        [
            {"speaker": "Bob", "quote": "got back from a run"},
            {"speaker": "Bot (assistant)", "quote": "Sounds like"},
        ]
    )
    cues = InMemoryProfileCueStore()
    pipeline, _ = _pipeline(model, CueConsumers(groups=True), cues)

    await pipeline.run_cell(
        IngestResult(session_id="s1", messages=[], group_id="trip"), _PREPARED
    )

    assert model.asked == ["CuedEpisodeReply"]
    prompt = model.prompts[0]
    assert prompt.index("<group_chat>") < prompt.index("<profile_cues>")
    assert cues.records == {
        ("mc_a", "u_ann"): PersonCues(),
        ("mc_a", "u_bob"): PersonCues(message_ids=("m_1",)),
    }


async def test_a_space_whose_profiles_read_no_cue_is_not_asked() -> None:
    model = _Model([{"speaker": "Ann", "quote": "I teach maths"}])
    cues = InMemoryProfileCueStore()
    pipeline, _ = _pipeline(model, CueConsumers(users=False, groups=True), cues)

    await pipeline.run_cell(IngestResult(session_id="s1", messages=[]), _PREPARED)

    assert model.asked == ["EpisodeReply"]
    assert "profile_cues" not in model.prompts[0]
    assert cues.records == {}


async def test_a_re_extraction_asks_for_no_cue_and_changes_none() -> None:
    model = _Model([{"speaker": "Ann", "quote": "I teach maths"}])
    cues = InMemoryProfileCueStore()
    pipeline, _ = _pipeline(model, CueConsumers(users=True, groups=True), cues)
    await pipeline.run_cell(IngestResult(session_id="s1", messages=[]), _PREPARED)
    kept = dict(cues.records)

    episode = await pipeline.reextract(
        scope=Scope(
            app_id="default", project_id="default", owner_id="u_ann", owner_type="user"
        ),
        session_id="s1",
        memcell_id="mc_a",
        cell=_CELL.model_copy(update={"items": _CELL.items[1:]}),
        senders=["u_ann", "u_bob"],
    )

    assert episode is not None
    assert model.asked == ["CuedEpisodeReply", "EpisodeReply"]
    assert "profile_cues" not in model.prompts[1]
    assert cues.records == kept
