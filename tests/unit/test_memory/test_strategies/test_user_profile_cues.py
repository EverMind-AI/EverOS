"""A user's profile update in ``cues`` mode, its idle trigger and its catch-up.

What these pin (``[profile] input = "cues"``, the default):

- an update shows the model only the owner's marked messages, each after the
  message before it by someone else, and a marked picture as its picture
  memory's text, whose item still names the picture (so forgetting it reaches
  the item); ``picture_pixels`` attaches the marked picture itself, and only
  it;
- segments that marked nothing for the owner make no call: they leave the
  queue and are marked read, so a replay of their event is not queued again;
- a marked message forgotten since is not shown, and when it was the only
  cue no call is made;
- ``full`` sends the update of 2026-11-13 whatever cues are recorded;
- the idle tick of a user's space, :func:`update_pending_profile` and a
  catch-up's ``UserProfileDue`` read what the profile has queued.
"""

from __future__ import annotations

import importlib
import json
from typing import Any

import pytest
from everalgo.llm import ImageUrlPart, LLMMessage, TextPart
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, MemCell

from everos.component.images import ImageLimits
from everos.config import Settings
from everos.memory._partition_locks import _reset_for_tests
from everos.memory.deps import MemoryDeps
from everos.memory.events import EpisodeExtracted, UserProfileDue
from everos.memory.models import Episode, ExplicitInfoItem, ImageMemory, Profile
from everos.memory.ports import (
    NO_SOURCE_CHECK,
    IdleTick,
    PersonCues,
    Scope,
    SessionKey,
)
from everos.memory.strategies import extract_user_profile, update_pending_profile
from tests.fakes import make_deps

_TS = 1_700_000_000_000
_SESSION = SessionKey("default", "default", "sess_1")
_LIMITS = ImageLimits(
    max_bytes=1 << 20,
    max_per_message=4,
    max_request_bytes=1 << 22,
    max_per_request=8,
    formats=("jpeg",),
)
_CAT = "c" * 64
_DOG = "d" * 64


@pytest.fixture(autouse=True)
def _isolate_partition_locks() -> None:
    _reset_for_tests()


def _every(monkeypatch: pytest.MonkeyPatch, n: int) -> None:
    base = Settings()
    settings = base.model_copy(
        update={"profile": base.profile.model_copy(update={"update_every": n})}
    )
    module = importlib.import_module("everos.memory.strategies.extract_user_profile")
    monkeypatch.setattr(module, "load_settings", lambda: settings)


@pytest.fixture(autouse=True)
def _every_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    _every(monkeypatch, 1)


def _scope(owner_id: str = "u_alice") -> Scope:
    return Scope(
        app_id="default", project_id="default", owner_id=owner_id, owner_type="user"
    )


def _event(memcell_id: str = "mc_1") -> EpisodeExtracted:
    return EpisodeExtracted(
        memcell_id=memcell_id,
        episode_entry_id=f"ep_{memcell_id}",
        episode_text="...",
        episode_timestamp_ms=_TS,
        owner_id="u_alice",
        session_id="sess_1",
        language="English",
    )


def _memcell(*lines: tuple[str, str], ts: int = _TS) -> MemCell:
    return MemCell(
        items=[
            ChatMessage(
                id=f"m{i}",
                role="user",
                content=text,
                timestamp=ts + i,
                sender_id=sender,
                sender_name=sender.removeprefix("u_").title(),
            )
            for i, (sender, text) in enumerate(lines)
        ],
        timestamp=ts + len(lines),
    )


def _add(description: str, *, pictures: list[str] | None = None) -> dict[str, Any]:
    return {
        "action": "add",
        "type": "explicit_info",
        "data": {"evidence": "e", "description": description, "category": "c"},
        "sources": ["s1"],
        "pictures": pictures or [],
    }


def _none() -> str:
    return json.dumps({"operations": [{"action": "none"}]})


class _Model:
    """A scripted model that records what each call was sent."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.sent: list[list[LLMMessage]] = []
        self.llm = FakeLLMClient(handler=self._answer)

    def _answer(self, messages: list[LLMMessage], **_kwargs: Any) -> str:
        self.sent.append(messages)
        return self.replies.pop(0)

    def parts(self, call: int = 0) -> list[Any]:
        content = self.sent[call][1].content
        return [TextPart(text=content)] if isinstance(content, str) else list(content)

    def user_text(self, call: int = 0) -> str:
        return "\n".join(p.text for p in self.parts(call) if isinstance(p, TextPart))


def _deps(model: _Model, **kwargs: Any) -> MemoryDeps:
    deps = make_deps(llm=model.llm, **kwargs)
    seed = deps.stores.memcell.seed

    def seed_with_episode(memcell_id: str, cell: MemCell, *args: Any) -> None:
        seed(memcell_id, cell, *args)
        deps.stores.message_forget.segments[memcell_id] = _SESSION
        deps.stores.episode.seed(
            _scope(),
            f"ep_{memcell_id}",
            Episode(
                owner_id="u_alice",
                episode="...",
                summary="...",
                timestamp=cell.timestamp,
                session_id="sess_1",
                parent_id=memcell_id,
            ),
        )

    deps.stores.memcell.seed = seed_with_episode  # type: ignore[method-assign]
    return deps


async def _cue(deps: MemoryDeps, memcell_id: str, **people: PersonCues) -> None:
    assert deps.stores.profile_cues is not None
    await deps.stores.profile_cues.record(_SESSION, memcell_id, people)


_TALK = _memcell(
    ("u_bob", "How is the new job going?"),
    ("u_alice", "Great, I am a pediatric nurse now."),
    ("u_bob", "Did you watch the game yesterday?"),
    ("u_alice", "No, I was busy all evening."),
)


async def test_an_update_shows_only_the_marked_messages_after_their_context() -> None:
    model = _Model(json.dumps({"operations": [_add("Works as a pediatric nurse.")]}))
    deps = _deps(model, profile_input="cues")
    deps.stores.memcell.seed("mc_1", _TALK)
    await _cue(deps, "mc_1", u_alice=PersonCues(message_ids=("m1",)))

    await extract_user_profile(_event(), deps)

    text = model.user_text()
    assert "How is the new job going?" in text
    assert "I am a pediatric nurse now." in text
    assert "game yesterday" not in text and "busy all evening" not in text
    stored = await deps.stores.profile.read(_scope())
    assert stored is not None
    (item,) = stored.explicit_info
    # The item rests on the whole segment: its span is the segment's.
    extra = item.model_extra or {}
    assert extra["source_memcell_ids"] == ["mc_1"]
    assert (extra["source_start"], extra["source_end"]) == (_TS, _TS + 4)


async def test_full_sends_the_whole_segment_whatever_cues_are_recorded() -> None:
    cued, plain = _Model(_none()), _Model(_none())
    with_cues = _deps(cued, profile_input="full")
    without = _deps(plain, profile_input="full")
    for deps in (with_cues, without):
        deps.stores.memcell.seed("mc_1", _TALK)
    await _cue(with_cues, "mc_1", u_alice=PersonCues(message_ids=("m1",)))

    await extract_user_profile(_event(), with_cues)
    await extract_user_profile(_event(), without)

    assert [m.model_dump() for m in cued.sent[0]] == [
        m.model_dump() for m in plain.sent[0]
    ]
    assert "busy all evening" in cued.user_text()


async def test_no_cue_no_call_and_the_segment_is_read() -> None:
    model = _Model()
    deps = _deps(model, profile_input="cues")
    deps.stores.memcell.seed("mc_1", _TALK)
    await deps.stores.profile.write(
        _scope(),
        Profile(
            owner_id="u_alice",
            summary="s",
            timestamp=_TS,
            explicit_info=[ExplicitInfoItem(description="Lives in Lyon.")],
            pending_memcell_ids=["mc_0"],
        ),
    )
    await _cue(deps, "mc_1", u_alice=PersonCues(), u_bob=PersonCues())

    await extract_user_profile(_event(), deps)

    assert model.sent == []
    stored = await deps.stores.profile.read(_scope())
    assert stored is not None and stored.pending_memcell_ids == []
    assert [i.description for i in stored.explicit_info] == ["Lives in Lyon."]
    assert deps.stores.profile_cues is not None
    record = await deps.stores.profile_cues.of_person("u_alice", ["mc_1"])
    assert record["mc_1"].read
    unread = await deps.stores.profile_cues.of_person("u_bob", ["mc_1"])
    assert not unread["mc_1"].read

    await extract_user_profile(_event(), deps)  # a replay is not queued again
    stored = await deps.stores.profile.read(_scope())
    assert stored is not None and stored.pending_memcell_ids == []
    assert model.sent == []


async def test_a_segment_read_with_a_call_is_not_queued_again_on_a_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model(_none())
    deps = _deps(model, profile_input="cues")
    deps.stores.memcell.seed("mc_1", _TALK)
    await _cue(deps, "mc_1", u_alice=PersonCues(message_ids=("m1",)))
    await extract_user_profile(_event(), deps)  # read (the reply changed nothing)
    _every(monkeypatch, 5)
    await deps.stores.profile.write(
        _scope(),
        Profile(
            owner_id="u_alice",
            summary="s",
            timestamp=_TS,
            explicit_info=[ExplicitInfoItem(description="Lives in Lyon.")],
        ),
    )

    await extract_user_profile(_event(), deps)

    stored = await deps.stores.profile.read(_scope())
    assert stored is not None and stored.pending_memcell_ids == []
    assert len(model.sent) == 1


async def test_a_forgotten_cued_message_is_not_read() -> None:
    model = _Model(_none())
    deps = _deps(model, profile_input="cues")
    deps.stores.memcell.seed("mc_1", _TALK)
    deps.stores.memcell.seed("mc_2", _memcell(("u_alice", "I play chess."), ts=_TS + 9))
    await _cue(deps, "mc_1", u_alice=PersonCues(message_ids=("m1", "m3")))
    await _cue(deps, "mc_2", u_alice=PersonCues(message_ids=("m0",)))
    await deps.stores.message_forget.record(_SESSION, ["m1"], forget_id="fg_1")

    await extract_user_profile(_event("mc_1"), deps)

    text = model.user_text()
    assert "pediatric nurse" not in text
    assert "busy all evening" in text

    # The forget took out the only cue of a segment: nothing to read.
    deps2 = _deps(_Model(), profile_input="cues")
    deps2.stores.memcell.seed("mc_1", _TALK)
    await _cue(deps2, "mc_1", u_alice=PersonCues(message_ids=("m1",)))
    await deps2.stores.message_forget.record(_SESSION, ["m1"], forget_id="fg_1")
    await extract_user_profile(_event("mc_1"), deps2)
    assert await deps2.stores.profile.read(_scope()) is None


async def _picture_deps(model: _Model, **kwargs: Any) -> MemoryDeps:
    deps = _deps(model, profile_input="cues", picture_limits=_LIMITS, **kwargs)
    deps.stores.images.images[_CAT] = (b"cat", "jpeg")
    deps.stores.images.images[_DOG] = (b"dog", "jpeg")
    deps.stores.memcell.seed(
        "mc_1",
        _memcell(
            ("u_bob", "Any pets?"),
            ("u_alice", f"This is Luna [IMAGE: {_CAT}]"),
            ("u_alice", f"And my lunch [IMAGE: {_DOG}]"),
        ),
    )
    for image_id, description in ((_CAT, "A black cat on a sofa."), (_DOG, "Pasta.")):
        await deps.stores.image_memory.append(
            _scope(),
            ImageMemory(
                owner_id="u_alice",
                description=description,
                timestamp=_TS,
                image_id=image_id,
                parent_id="mc_1",
            ),
            only_if=NO_SOURCE_CHECK,
        )
    await _cue(deps, "mc_1", u_alice=PersonCues(image_ids=(_CAT,)))
    return deps


async def test_a_cued_picture_is_read_as_its_memory_text_and_keeps_its_lineage() -> (
    None
):
    model = _Model(
        json.dumps({"operations": [_add("Has a cat, Luna.", pictures=["img-1"])]})
    )
    deps = await _picture_deps(model)

    await extract_user_profile(_event(), deps)

    parts = model.parts()
    assert not any(isinstance(p, ImageUrlPart) for p in parts)
    text = model.user_text()
    assert "This is Luna [IMAGE: img-1]" in text
    assert "<picture>\nlabel: img-1\n</picture>" in text
    assert "A black cat on a sofa." in text
    assert "Pasta." not in text and "my lunch" not in text
    stored = await deps.stores.profile.read(_scope())
    assert stored is not None
    assert (stored.explicit_info[0].model_extra or {})["source_image_ids"] == [_CAT]


async def test_picture_pixels_attach_the_cued_picture_itself() -> None:
    model = _Model(_none())
    deps = await _picture_deps(model, profile_picture_pixels=True)

    await extract_user_profile(_event(), deps)

    assert sum(isinstance(p, ImageUrlPart) for p in model.parts()) == 1
    assert "A black cat on a sofa." not in model.user_text()


async def test_a_forgotten_cued_picture_is_not_read() -> None:
    """The only cue of the segment is a picture the owner forgot: the picture
    is taken out before the cues are chosen, so nothing is left and no call
    is made -- neither the picture nor its memory's text reaches a model."""
    model = _Model()
    deps = await _picture_deps(model)
    await deps.stores.forget_log.record(
        target="image",
        target_id=_CAT,
        app_id="default",
        project_id="default",
        owner_id="u_alice",
    )

    await extract_user_profile(_event(), deps)

    assert model.sent == []
    assert await deps.stores.profile.read(_scope()) is None


async def test_an_idle_tick_and_the_catch_up_read_the_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _every(monkeypatch, 5)
    model = _Model(_none(), _none())
    deps = _deps(model, profile_input="cues")
    for n in (1, 2):
        deps.stores.memcell.seed(
            f"mc_{n}", _TALK.model_copy(update={"timestamp": _TS + n})
        )
        await _cue(deps, f"mc_{n}", u_alice=PersonCues(message_ids=("m1",)))
    profile = Profile(
        owner_id="u_alice",
        summary="s",
        timestamp=_TS,
        explicit_info=[ExplicitInfoItem(description="Lives in Lyon.")],
    )
    await deps.stores.profile.write(_scope(), profile)
    await extract_user_profile(_event("mc_1"), deps)
    assert model.sent == []  # queued

    tick = IdleTick(
        strategy_name="extract_user_profile_idle",
        bucket_key="default/default/u_alice",
        idle_seconds=600,
    )
    await extract_user_profile(tick, deps)
    assert len(model.sent) == 1
    stored = await deps.stores.profile.read(_scope())
    assert stored is not None and stored.pending_memcell_ids == []

    await extract_user_profile(_event("mc_2"), deps)
    assert await update_pending_profile(deps, _scope()) == 1
    assert len(model.sent) == 2
    assert await update_pending_profile(deps, _scope()) == 0
    await extract_user_profile(tick, deps)
    assert len(model.sent) == 2  # nothing left to read


async def test_a_catch_up_reads_the_queue_and_then_finds_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _every(monkeypatch, 5)
    model = _Model(_none())
    deps = _deps(model, profile_input="cues")
    deps.stores.memcell.seed("mc_1", _TALK)
    await _cue(deps, "mc_1", u_alice=PersonCues(message_ids=("m1",)))
    await deps.stores.profile.write(
        _scope(),
        Profile(
            owner_id="u_alice",
            summary="s",
            timestamp=_TS,
            explicit_info=[ExplicitInfoItem(description="Lives in Lyon.")],
        ),
    )
    await extract_user_profile(_event("mc_1"), deps)
    assert model.sent == []  # queued: one segment of five

    due = UserProfileDue(user_id="u_alice", app_id="default", project_id="default")
    await extract_user_profile(due, deps)
    assert len(model.sent) == 1
    assert "pediatric nurse" in model.user_text()
    stored = await deps.stores.profile.read(_scope())
    assert stored is not None and stored.pending_memcell_ids == []

    await extract_user_profile(due, deps)
    assert len(model.sent) == 1  # nothing queued: no call
