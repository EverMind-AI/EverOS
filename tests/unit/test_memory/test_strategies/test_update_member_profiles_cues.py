"""A group's batched member update reading the episode calls' cues.

``[groups] member_profile_mode = "batched"`` with ``[profile] input = "cues"``
(both defaults). In-memory stores and a scripted model. What these pin:

- a member with no cue in the batch is not a target: no label, no
  operation, no token, and their document is not written;
- the batch's conversation is rendered once from the union of the targets'
  marked messages, each after the nearest earlier message by someone other
  than the member it was marked for, cut to 200 characters; a message marked
  for one member and another's context is shown whole; a segment nobody
  marked anything in is not shown; a segment never marked is shown whole;
- when no member has a cue there is no call: the batch is cleared, its
  segments are marked read, and a replay of one is not queued again;
- a marked picture goes as its picture memory's text, and only it
  (``picture_pixels`` attaches the picture itself);
- ``full`` sends the batched request of before whatever cues are recorded.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any

import pytest
from everalgo.llm import ImageUrlPart, TextPart
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, MemCell

from everos.component.images import ImageLimits
from everos.config import load_settings
from everos.memory.deps import MemoryDeps
from everos.memory.events import EpisodeExtracted
from everos.memory.models import Group, ImageMemory, RosterEvent
from everos.memory.ports import NO_SOURCE_CHECK, PersonCues, Scope, SessionKey
from everos.memory.strategies import update_member_profiles
from tests.fakes import make_deps

_SCOPE = Scope(app_id="default", project_id="p", owner_id="trip", owner_type="group")
_SESSION = SessionKey("default", "p", "s")
_TARGET = re.compile(r"^- (p\d+): .+?, lines tagged user_id:(\S+),", re.M)
_LIMITS = ImageLimits(
    max_bytes=1 << 20,
    max_per_message=4,
    max_request_bytes=1 << 22,
    max_per_request=8,
    formats=("jpeg",),
)
_CAT = "c" * 64
_DOG = "d" * 64


class _Model:
    """Answers a batched update with one item per target; records each call."""

    def __init__(self) -> None:
        self.sent: list[Any] = []

    async def __call__(self, messages: Any, *, response_format: type, **_: Any) -> str:
        assert response_format.__name__ == "_ProfileUpdateManyReply"
        self.sent.append(messages)
        people = [
            {"user_id": uid, "operations": [_add(f"Fact about {uid}")]}
            for _, uid in _TARGET.findall(self.text(len(self.sent) - 1))
        ]
        return json.dumps({"people": people})

    def parts(self, call: int = 0) -> list[Any]:
        return [
            part
            for message in self.sent[call]
            for part in (
                [TextPart(text=message.content)]
                if isinstance(message.content, str)
                else message.content
            )
        ]

    def text(self, call: int = 0) -> str:
        return "\n".join(p.text for p in self.parts(call) if isinstance(p, TextPart))

    def conversation(self, call: int = 0) -> str:
        found = re.search(
            r"<new_conversation>\n(.*?)\n</new_conversation>", self.text(call), re.S
        )
        assert found is not None
        return found.group(1)

    def targets(self, call: int = 0) -> list[str]:
        return [uid for _, uid in _TARGET.findall(self.text(call))]


def _add(description: str) -> dict[str, Any]:
    return {
        "action": "add",
        "type": "explicit_info",
        "data": {"evidence": "x", "description": description, "category": "c"},
        "sources": [],
        "pictures": [],
    }


@pytest.fixture(autouse=True)
def _one_segment_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVEROS_GROUPS__GROUP_PROFILE_BATCH", "1")
    load_settings.cache_clear()
    yield
    load_settings.cache_clear()


def _deps(model: _Model, **kwargs: Any) -> MemoryDeps:
    deps = make_deps(llm=FakeLLMClient(handler=model), **kwargs)
    deps.stores.groups.groups[("default", "p", "trip")] = Group(
        group_id="trip", created_at=0
    )
    deps.stores.groups.events[("default", "p", "trip")].extend(
        [
            RosterEvent(op="join", member_id=m, at=0, display_name=name)
            for m, name in (("u_chen", "Chen"), ("u_betty", "Betty"), ("u_ann", "Ann"))
        ]
    )
    return deps


def _segment(deps: MemoryDeps, memcell_id: str, *lines: tuple[str, str]) -> None:
    items = [
        ChatMessage(
            id=f"m{i}",
            role="assistant" if sender == "bot" else "user",
            sender_id=sender,
            sender_name=sender,
            content=text,
            timestamp=1_000 + i,
        )
        for i, (sender, text) in enumerate(lines)
    ]
    deps.stores.memcell.seed(  # type: ignore[attr-defined]
        memcell_id, MemCell(items=items, timestamp=1_000 + len(items))
    )


def _event(memcell_id: str, *speakers: str) -> EpisodeExtracted:
    return EpisodeExtracted(
        memcell_id=memcell_id,
        episode_entry_id=f"gep_{memcell_id}",
        episode_text="text",
        episode_timestamp_ms=1,
        owner_id="trip",
        owner_type="group",
        app_id="default",
        project_id="p",
        speaker_ids=list(speakers),
    )


async def _cue(deps: MemoryDeps, memcell_id: str, **people: PersonCues) -> None:
    assert deps.stores.profile_cues is not None
    await deps.stores.profile_cues.record(_SESSION, memcell_id, people)


async def _read(deps: MemoryDeps, member: str, memcell_id: str) -> bool:
    assert deps.stores.profile_cues is not None
    found = await deps.stores.profile_cues.of_person(member, [memcell_id])
    return found[memcell_id].read


_LONG = ("We should split the costs evenly, " * 10).strip()  # 339 characters


async def test_a_member_with_no_cue_is_not_a_target() -> None:
    model = _Model()
    deps = _deps(model, profile_input="cues")
    _segment(
        deps,
        "mc_1",
        ("u_chen", "I always want refundable tickets."),
        ("u_betty", "Sounds good to me."),
        ("u_ann", "I am allergic to peanuts."),
    )
    await _cue(
        deps,
        "mc_1",
        u_chen=PersonCues(message_ids=("m0",)),
        u_betty=PersonCues(),
        u_ann=PersonCues(message_ids=("m2",)),
    )

    await update_member_profiles(_event("mc_1", "u_chen", "u_betty", "u_ann"), deps)

    (_,) = model.sent
    assert model.targets() == ["u_ann", "u_chen"]
    assert await deps.stores.profile.read(_SCOPE, "u_betty") is None
    assert await deps.stores.profile.read(_SCOPE, "u_chen") is not None
    for member in ("u_ann", "u_betty", "u_chen"):
        assert await _read(deps, member, "mc_1")


async def test_the_conversation_is_the_union_of_the_marked_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVEROS_GROUPS__GROUP_PROFILE_BATCH", "2")
    load_settings.cache_clear()
    model = _Model()
    deps = _deps(model, profile_input="cues")
    _segment(
        deps,
        "mc_1",
        ("u_betty", "Where do we stay in Lisbon?"),  # m0: Chen's context
        ("u_chen", "I always want refundable tickets."),  # m1: Chen's cue
        ("u_ann", "Chen, " + _LONG),  # m2: Ann's cue, Betty's context
        ("u_betty", "I am vegetarian, by the way."),  # m3: Betty's cue
        ("bot", "Noted, I will look for options."),  # m4: nobody's
        ("u_chen", "Lunch was great today."),  # m5: nobody's
    )
    _segment(deps, "mc_2", ("u_ann", "Small talk only."))
    await _cue(
        deps,
        "mc_1",
        u_chen=PersonCues(message_ids=("m1",)),
        u_ann=PersonCues(message_ids=("m2",)),
        u_betty=PersonCues(message_ids=("m3",)),
    )
    await _cue(deps, "mc_2", u_ann=PersonCues())

    for memcell_id, who in (
        ("mc_1", ("u_betty", "u_chen", "u_ann")),
        ("mc_2", ("u_ann",)),
    ):
        await update_member_profiles(_event(memcell_id, *who), deps)

    (_,) = model.sent  # one call for the batch of two
    conversation = model.conversation()
    assert conversation.count("<segment") == 1  # mc_2 marked nothing
    lines = [line.split("): ", 1)[1] for line in conversation.splitlines()[1:-1]]
    assert lines == [
        "Where do we stay in Lisbon?",  # the context of Chen's cue
        "I always want refundable tickets.",  # Chen's cue, Ann's context
        "Chen, " + _LONG,  # Ann's cue and Betty's context: whole, never cut
        "I am vegetarian, by the way.",  # Betty's cue
    ]
    assert model.targets() == ["u_ann", "u_betty", "u_chen"]
    assert await _read(deps, "u_ann", "mc_2")


async def test_the_nearest_context_and_a_segment_never_marked_shown_whole() -> None:
    model = _Model()
    deps = _deps(model, profile_input="cues")
    _segment(
        deps,
        "mc_1",
        ("u_betty", _LONG),  # m0: only Chen's context
        ("u_ann", "Agreed."),
        ("u_chen", "I always want refundable tickets."),  # m2
    )
    _segment(
        deps,
        "mc_2",
        ("u_ann", "We moved to Porto."),
        ("u_betty", "Nice place."),
    )
    await _cue(
        deps,
        "mc_1",
        u_chen=PersonCues(message_ids=("m2",)),
        u_betty=PersonCues(),
        u_ann=PersonCues(),
    )
    # mc_2 has no record: extracted before cues, it is read in full.
    await update_member_profiles(_event("mc_1", "u_betty", "u_ann", "u_chen"), deps)
    await update_member_profiles(_event("mc_2", "u_ann", "u_betty"), deps)

    first, second = model.conversation(0), model.conversation(1)
    assert "Agreed." in first  # the nearest earlier message by someone else
    assert "We should split" not in first
    assert model.targets(0) == ["u_chen"]
    assert "We moved to Porto." in second and "Nice place." in second
    assert model.targets(1) == ["u_ann", "u_betty"]


async def test_the_context_skips_the_member_s_own_unmarked_lines() -> None:
    model = _Model()
    deps = _deps(model, profile_input="cues")
    _segment(
        deps,
        "mc_1",
        ("u_betty", _LONG),  # m0: Chen's context, cut
        ("u_chen", "Hmm."),  # m1: Chen's own, unmarked
        ("u_chen", "I always want refundable tickets."),  # m2: Chen's cue
    )
    await _cue(
        deps,
        "mc_1",
        u_chen=PersonCues(message_ids=("m2",)),
        u_betty=PersonCues(),
    )

    await update_member_profiles(_event("mc_1", "u_betty", "u_chen"), deps)

    conversation = model.conversation()
    assert "Hmm." not in conversation
    cut = _LONG[:200].rstrip() + "..."
    assert cut in conversation and _LONG not in conversation
    assert "I always want refundable tickets." in conversation


async def test_no_cue_no_call_and_a_replay_is_not_queued_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    deps = _deps(model, profile_input="cues")
    _segment(deps, "mc_1", ("u_chen", "Hi all."), ("u_betty", "Hello!"))
    await _cue(deps, "mc_1", u_chen=PersonCues(), u_betty=PersonCues())
    event = _event("mc_1", "u_chen", "u_betty")

    await update_member_profiles(event, deps)

    assert model.sent == []
    batch = await deps.stores.group_profile.member_pending_batch(_SCOPE)
    assert batch.memcell_ids == ()
    assert await _read(deps, "u_chen", "mc_1") and await _read(deps, "u_betty", "mc_1")
    assert await deps.stores.profile.read(_SCOPE, "u_chen") is None

    monkeypatch.setenv("EVEROS_GROUPS__GROUP_PROFILE_BATCH", "2")
    load_settings.cache_clear()
    await update_member_profiles(event, deps)  # a replay: not queued

    batch = await deps.stores.group_profile.member_pending_batch(_SCOPE)
    assert batch.memcell_ids == () and model.sent == []


async def _with_pictures(model: _Model, **kwargs: Any) -> MemoryDeps:
    deps = dataclasses.replace(
        _deps(model, profile_input="cues", **kwargs), picture_limits=_LIMITS
    )
    deps.stores.images.images[_CAT] = (b"cat", "jpeg")  # type: ignore[attr-defined]
    deps.stores.images.images[_DOG] = (b"dog", "jpeg")  # type: ignore[attr-defined]
    _segment(
        deps,
        "mc_1",
        ("u_chen", f"This is our cat Luna [IMAGE: {_CAT}]"),
        ("u_betty", f"And my lunch [IMAGE: {_DOG}]"),
        ("u_betty", "I am vegetarian."),
    )
    for image_id, description in ((_CAT, "A black cat on a sofa."), (_DOG, "Pasta.")):
        await deps.stores.image_memory.append(
            _SCOPE,
            ImageMemory(
                owner_id="trip",
                description=description,
                timestamp=1_000,
                image_id=image_id,
                parent_id="mc_1",
            ),
            only_if=NO_SOURCE_CHECK,
        )
    await _cue(
        deps,
        "mc_1",
        u_chen=PersonCues(image_ids=(_CAT,)),
        u_betty=PersonCues(message_ids=("m2",)),
    )
    return deps


async def test_a_marked_picture_goes_as_its_memory_s_text() -> None:
    model = _Model()
    deps = await _with_pictures(model)

    await update_member_profiles(_event("mc_1", "u_chen", "u_betty"), deps)

    assert not any(isinstance(p, ImageUrlPart) for p in model.parts())
    text = model.text()
    assert "This is our cat Luna [IMAGE: img-1]" in text
    assert "A black cat on a sofa." in text
    assert "Pasta." not in text
    assert "user_id:u_chen, pictures posted: img-1" in text
    assert "user_id:u_betty, pictures posted: none" in text
    # Betty's lunch picture is her own unmarked message's: not shown at all,
    # nor is it attached.
    assert "my lunch" not in model.conversation()


async def test_picture_pixels_attach_the_marked_picture_itself() -> None:
    model = _Model()
    deps = await _with_pictures(model, profile_picture_pixels=True)

    await update_member_profiles(_event("mc_1", "u_chen", "u_betty"), deps)

    assert sum(isinstance(p, ImageUrlPart) for p in model.parts()) == 1
    assert "A black cat on a sofa." not in model.text()


async def test_full_sends_the_batched_request_whatever_cues_are_recorded() -> None:
    cued, plain = _Model(), _Model()
    with_cues = _deps(cued, profile_input="full")
    without = _deps(plain, profile_input="full")
    for deps in (with_cues, without):
        _segment(
            deps,
            "mc_1",
            ("u_chen", "I always want refundable tickets."),
            ("u_betty", "Sounds good to me."),
        )
    await _cue(
        with_cues, "mc_1", u_chen=PersonCues(message_ids=("m0",)), u_betty=PersonCues()
    )

    for deps in (with_cues, without):
        await update_member_profiles(_event("mc_1", "u_chen", "u_betty"), deps)

    assert [m.model_dump() for m in cued.sent[0]] == [
        m.model_dump() for m in plain.sent[0]
    ]
    assert cued.targets() == ["u_betty", "u_chen"]
    assert not await _read(with_cues, "u_betty", "mc_1")
