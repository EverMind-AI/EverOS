"""What a profile update reads (``[profile] input``): every message, or the cues.

What these pin:

- ``full`` hands the update its segments as they are, and every picture the
  person posted (nothing to choose);
- ``cues`` shows only the person's marked messages, each after the nearest
  earlier message by someone else -- cut to 200 characters, its pictures bare
  references -- and a marked picture's own message; the picture is attached
  as its current picture memory's text, or itself with ``picture_pixels``;
- a segment never marked is read whole, one that marked nothing for the
  person is read without being shown, and a cue whose message a forget took
  out (it is not in the readable cell) selects nothing.
"""

from __future__ import annotations

from everalgo.types import ChatMessage, MemCell

from everos.memory.extract.pipeline.profile_input import (
    CONTEXT_CHARS,
    select_profile_input,
)
from everos.memory.models import ImageMemory
from everos.memory.ports import NO_SOURCE_CHECK, PersonCues, Scope, SessionKey
from tests.fakes.stores import InMemoryImageMemoryStore, InMemoryProfileCueStore

_T0 = 1_700_000_000_000
_CAT = "c" * 64
_DOG = "d" * 64
_SESSION = SessionKey("default", "default", "s1")
_SCOPE = Scope(
    app_id="default", project_id="default", owner_id="u_mel", owner_type="user"
)


def _message(index: int, sender: str, text: str, *, role: str = "user") -> ChatMessage:
    return ChatMessage(
        id=f"m_{index}",
        role=role,  # type: ignore[arg-type]
        content=text,
        timestamp=_T0 + index * 60_000,
        sender_id=sender,
        sender_name=sender.removeprefix("u_").title(),
    )


def _cell(*messages: ChatMessage) -> MemCell:
    return MemCell(items=list(messages), timestamp=messages[-1].timestamp)


_LONG = "So tell me everything about your work, " * 10
_SEGMENT = _cell(
    _message(0, "u_car", "Hi! How is the new flat?"),
    _message(1, "u_mel", "Great, it is close to the hospital."),
    _message(2, "u_car", _LONG + f"[IMAGE: {_DOG}]"),
    _message(3, "u_mel", "Well,"),
    _message(4, "u_mel", "I work night shifts as a nurse."),
    _message(5, "u_mel", f"And this is Luna [IMAGE: {_CAT}]"),
    _message(6, "bot", "Lovely cat!", role="assistant"),
)


async def _cues(**records: PersonCues) -> InMemoryProfileCueStore:
    store = InMemoryProfileCueStore()
    for memcell_id, record in records.items():
        await store.record(_SESSION, memcell_id, {"u_mel": record})
    return store


def _ids(cell: MemCell) -> list[str]:
    return [item.id for item in cell.items if isinstance(item, ChatMessage)]


async def test_full_reads_every_segment_as_it_is() -> None:
    store = await _cues(mc_1=PersonCues(message_ids=("m_4",)))

    chosen = await select_profile_input(
        [("mc_1", _SEGMENT)], "u_mel", mode="full", cues=store
    )

    assert chosen.segments == [("mc_1", _SEGMENT)]
    assert chosen.shown == [_SEGMENT]
    assert (chosen.pictures, chosen.picture_text, chosen.read) == (None, None, [])


async def test_cues_show_the_marked_messages_each_after_its_context() -> None:
    store = await _cues(mc_1=PersonCues(message_ids=("m_1", "m_4")))

    chosen = await select_profile_input(
        [("mc_1", _SEGMENT)], "u_mel", mode="cues", cues=store
    )

    assert chosen.segments == [("mc_1", _SEGMENT)]
    (shown,) = chosen.shown
    assert _ids(shown) == ["m_0", "m_1", "m_2", "m_4"]
    context = shown.items[2]
    assert isinstance(context, ChatMessage)
    assert len(context.content) == CONTEXT_CHARS + len("...")
    assert context.content.endswith("...") and _DOG not in context.content
    assert shown.items[3] == _SEGMENT.items[4]
    assert (chosen.read, chosen.skipped) == (["mc_1"], [])
    assert chosen.pictures == frozenset()


async def test_a_short_context_keeps_its_words_and_bare_picture_references() -> None:
    cell = _cell(
        _message(0, "u_car", f"Look [IMAGE: {_DOG}] is mine. What about you?"),
        _message(1, "u_mel", "I have a cat."),
    )
    store = await _cues(mc_1=PersonCues(message_ids=("m_1",)))

    chosen = await select_profile_input(
        [("mc_1", cell)], "u_mel", mode="cues", cues=store
    )

    (shown,) = chosen.shown
    assert [m.content for m in shown.items] == [  # type: ignore[union-attr]
        "Look [IMAGE] is mine. What about you?",
        "I have a cat.",
    ]


async def test_a_marked_picture_is_attached_as_its_memory_text() -> None:
    store = await _cues(mc_1=PersonCues(image_ids=(_CAT,)))
    memories = InMemoryImageMemoryStore()
    refs = [
        await memories.append(
            _SCOPE,
            ImageMemory(
                owner_id="u_mel",
                description=description,
                ocr=ocr,
                timestamp=_T0,
                image_id=_CAT,
                parent_id="mc_1",
            ),
            only_if=NO_SOURCE_CHECK,
        )
        for description, ocr in (("An old description.", ""), ("A black cat.", "LUNA"))
    ]
    old, new = refs
    assert old is not None and new is not None
    await memories.deprecate(
        _SCOPE,
        entries=[(old.md_path, old.entry_id)],
        replaced_by=new.entry_id,
        only_if=NO_SOURCE_CHECK,
    )

    chosen = await select_profile_input(
        [("mc_1", _SEGMENT)],
        "u_mel",
        mode="cues",
        cues=store,
        image_memory=memories,
        scope=_SCOPE,
    )

    (shown,) = chosen.shown
    assert _ids(shown) == ["m_2", "m_5"]  # the picture's message, after its context
    assert chosen.pictures == frozenset({_CAT})
    assert chosen.picture_text == {
        _CAT: "Shown in words (the picture is not attached): A black cat.\n"
        "Text in the picture: LUNA"
    }


async def test_picture_pixels_attach_the_marked_pictures_themselves() -> None:
    store = await _cues(mc_1=PersonCues(image_ids=(_CAT,)))

    chosen = await select_profile_input(
        [("mc_1", _SEGMENT)],
        "u_mel",
        mode="cues",
        cues=store,
        picture_pixels=True,
    )

    assert chosen.pictures == frozenset({_CAT})
    assert chosen.picture_text is None


async def test_unmarked_never_marked_and_forgotten_segments() -> None:
    older = _cell(_message(0, "u_mel", "I live in Lyon."))
    empty = _cell(_message(0, "u_mel", "Nice weather today."))
    # m_4 was forgotten: the readable segment no longer holds it.
    forgotten = _cell(*[m for m in _SEGMENT.items if m.id != "m_4"])  # type: ignore[union-attr]
    store = await _cues(
        mc_empty=PersonCues(),
        mc_forgot=PersonCues(message_ids=("m_4",)),
    )

    chosen = await select_profile_input(
        [("mc_old", older), ("mc_empty", empty), ("mc_forgot", forgotten)],
        "u_mel",
        mode="cues",
        cues=store,
    )

    assert chosen.segments == [("mc_old", older)]
    assert chosen.shown == [older]
    assert chosen.read == ["mc_empty", "mc_forgot"]
    assert chosen.skipped == ["mc_empty", "mc_forgot"]


async def test_nothing_to_show_means_no_segment_to_call_with() -> None:
    store = await _cues(mc_1=PersonCues())

    chosen = await select_profile_input(
        [("mc_1", _SEGMENT)], "u_mel", mode="cues", cues=store
    )

    assert (chosen.segments, chosen.shown) == ([], [])
    assert chosen.read == chosen.skipped == ["mc_1"]
