"""The profile-cue block of the episode call, and how its cues are read.

What these pin:

- the block is registered (``chat.episode_profile_cues`` version 1, two
  variants) and recorded in the prompt manifest;
- a cell is sent the block only when it is asked for: without it the request
  is the one of 2026-11-14 byte for byte (the text, the picture and the group
  routes), with it the block follows what the cell adds and precedes the
  language block, in its text or pictures variant, and the call asks for the
  cued reply models and a wider output cap;
- the cues are read leniently in form (unknown keys, ``null``, one entry or
  one label given bare, an unreadable entry) and strictly in meaning: a quote
  maps to the first message of its speaker containing it, compared without
  case, extra whitespace or curly quotation marks; a label maps only to a
  picture that speaker posted; an assistant, an unknown speaker or a quote
  that is nowhere maps to nothing and is counted; every human speaker gets a
  record, an empty one when nothing was marked.
"""

from __future__ import annotations

import json

import pytest
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, MemCell
from everalgo.user_memory import OutputLanguage
from everalgo.utils import tagged

from everos.memory.extract.pipeline.episode import (
    CuedEpisodeReply,
    CuedImageEpisodeReply,
    EpisodeReply,
    ImageEpisodeReply,
    build_messages,
    build_original_prompt,
    extract,
    extract_cell,
    reply_token_cap,
)
from everos.memory.extract.pipeline.pictures import Picture
from everos.memory.extract.pipeline.profile_cues import (
    CHAT_PROFILE_CUES,
    ProfileCue,
    cue_block,
    map_cues,
)
from everos.memory.ports import PersonCues
from everos.memory.prompt_manifest import prompt_manifest

_T0 = 1_700_000_000_000
_PNG = b"\x89PNG\r\n\x1a\nnot really a picture"
_ROSTER = 'This conversation is a group chat in "Trip team".'
_CAT = "c" * 64
_DOG = "d" * 64


def _message(
    index: int, sender: str, name: str, text: str, *, role: str = "user"
) -> ChatMessage:
    return ChatMessage(
        id=f"m_{index}",
        role=role,  # type: ignore[arg-type]
        content=text,
        timestamp=_T0 + index * 60_000,
        sender_id=sender,
        sender_name=name,
    )


def _cell(*messages: ChatMessage) -> MemCell:
    return MemCell(items=list(messages), timestamp=messages[-1].timestamp)


_TALK = _cell(
    _message(0, "u_car", "Caroline", "Hi Mel! How was the week?"),
    _message(1, "u_mel", "Melanie", "Busy. I work as a nurse   at the  “City Clinic”."),
    _message(2, "u_car", "Caroline", "I adopted a dog last year, Max."),
    _message(3, "bot", "Ava", "I love dogs too.", role="assistant"),
)


def _picture(label: str, image_id: str, poster: str) -> Picture:
    return Picture(
        label=label,
        image_id=image_id,
        data=_PNG,
        mime="image/png",
        poster=poster,
        sender_id=poster,
        at=_T0,
    )


# ── the block ────────────────────────────────────────────────────────────


def test_the_block_is_registered_with_two_variants_and_in_the_manifest() -> None:
    assert (CHAT_PROFILE_CUES.id, CHAT_PROFILE_CUES.version) == (
        "chat.episode_profile_cues",
        1,
    )
    # Pinned: a change to its text is a new version.
    assert CHAT_PROFILE_CUES.fingerprint == "53fb220dee230906"
    assert CHAT_PROFILE_CUES.variant_fingerprint("text") == "82b7364890baeb5a"
    assert CHAT_PROFILE_CUES.variant_fingerprint("pictures") == "82f6d093c78a2a20"
    entry = prompt_manifest()["prompts"]["chat.episode_profile_cues"]
    assert (entry["owner"], entry["used"], entry["version"]) == ("everos", "write", 1)
    assert entry["variants"] == {
        "text": "82b7364890baeb5a",
        "pictures": "82f6d093c78a2a20",
    }


def test_the_text_variant_asks_for_quotes_and_the_pictures_one_for_labels() -> None:
    text, pictures = cue_block(pictures=False), cue_block(pictures=True)

    assert '"profile_cues"' in text and '"pictures"' not in text
    assert 'carries a third field, "profile_cues"' in text
    assert '"pictures": ["img-1"]' in pictures
    assert 'carries a fourth field, "profile_cues"' in pictures
    assert (
        "An empty list is the right answer" in text
        and pictures.count("An empty list") == 1
    )


@pytest.mark.parametrize("language", [None, OutputLanguage("English")])
@pytest.mark.parametrize("custom", [None, "Name every product version."])
@pytest.mark.parametrize("labels", [(), ("img-1", "img-2")])
@pytest.mark.parametrize("roster", [None, _ROSTER])
def test_without_cues_the_request_is_unchanged_and_with_them_one_block_is_added(
    language: OutputLanguage | None,
    custom: str | None,
    labels: tuple[str, ...],
    roster: str | None,
) -> None:
    kwargs = {
        "custom_instructions": custom,
        "output_language": language,
        "roster": roster,
        "labels": labels,
    }
    plain = build_original_prompt(_TALK, **kwargs)  # type: ignore[arg-type]
    asked = build_original_prompt(_TALK, cues=True, **kwargs)  # type: ignore[arg-type]

    assert "<profile_cues>" not in plain and "profile_cues" not in plain
    block = tagged("profile_cues", cue_block(pictures=bool(labels)))
    head, tail = plain.rsplit("\n\n<output_language>", 1)
    assert asked == f"{head}\n\n{block}\n\n<output_language>{tail}"


def test_the_block_follows_the_pictures_and_precedes_the_language() -> None:
    (message,) = build_messages(
        _TALK,
        pictures=[("img-1", _PNG, "image/png")],
        custom_instructions=None,
        output_language=OutputLanguage("English"),
        roster=_ROSTER,
        cues=True,
    )
    first = message.content[0]  # type: ignore[index]
    text = first.text if hasattr(first, "text") else first["text"]
    order = [
        text.index(tag)
        for tag in ("<group_chat>", "<pictures>", "<profile_cues>", "<output_language>")
    ]
    assert order == sorted(order)


@pytest.mark.parametrize(
    ("pictures", "cues", "model"),
    [
        (0, False, EpisodeReply),
        (1, False, ImageEpisodeReply),
        (0, True, CuedEpisodeReply),
        (1, True, CuedImageEpisodeReply),
    ],
)
async def test_the_call_asks_for_the_reply_model_of_its_blocks(
    pictures: int, cues: bool, model: type
) -> None:
    reply = {"title": "t", "content": "c"}
    if pictures:
        reply["images"] = [{"ref": "img-1", "description": "d", "ocr": ""}]  # type: ignore[assignment]
    llm = FakeLLMClient(responses=[json.dumps(reply)])

    await extract(
        llm,
        _TALK,
        pictures=[("img-1", _PNG, "image/png")] * pictures,
        custom_instructions=None,
        output_language=None,
        cues=cues,
    )

    (call,) = llm.calls
    assert call.response_format is model


def test_a_call_that_asks_for_cues_gets_the_conversation_allowance_twice() -> None:
    plain = reply_token_cap(pictures=0, conversation_chars=4_000)
    asked = reply_token_cap(pictures=0, conversation_chars=4_000, cues=True)

    assert (plain, asked) == (2_048 + 1_000, 2_048 + 2_000)
    assert reply_token_cap(pictures=0, conversation_chars=10**6, cues=True) == 32_768


# ── reading the cues ─────────────────────────────────────────────────────


def test_cues_are_read_leniently_in_form() -> None:
    reply = CuedEpisodeReply.model_validate_json(
        json.dumps(
            {
                "title": "t",
                "content": "c",
                "profile_cues": [
                    {"speaker": "Melanie", "quote": "I work as a nurse", "why": "job"},
                    {"speaker": "Caroline", "quote": None, "pictures": "img-1"},
                    {"quote": "no speaker"},
                    "not an object",
                    {"speaker": "  ", "quote": "blank speaker"},
                ],
                "other": 1,
            }
        )
    )

    assert reply.profile_cues == [
        ProfileCue(speaker="Melanie", quote="I work as a nurse"),
        ProfileCue(speaker="Caroline", quote="", pictures=["img-1"]),
    ]
    alone = CuedEpisodeReply.model_validate_json(
        json.dumps(
            {
                "title": "t",
                "content": "c",
                "profile_cues": {"speaker": "Melanie", "quote": "nurse"},
            }
        )
    )
    assert alone.profile_cues == [ProfileCue(speaker="Melanie", quote="nurse")]
    none = CuedEpisodeReply.model_validate_json(
        '{"title": "t", "content": "c", "profile_cues": null}'
    )
    assert none.profile_cues == []


def test_a_reply_without_the_field_has_no_cues_and_still_validates() -> None:
    reply = CuedEpisodeReply.model_validate_json('{"title": "t", "content": "c"}')
    assert reply.profile_cues == []


def test_a_quote_maps_to_its_speakers_message_without_case_space_or_curly_marks() -> (
    None
):
    found = map_cues(
        [
            ProfileCue(speaker="melanie", quote='"I WORK as a nurse at the "City'),
            ProfileCue(speaker="Caroline", quote="...adopted a dog last year…"),
        ],
        _TALK,
        people=["u_car", "u_mel"],
        pictures=[],
    )

    assert found.by_person == {
        "u_car": PersonCues(message_ids=("m_2",)),
        "u_mel": PersonCues(message_ids=("m_1",)),
    }
    assert (found.given, found.mapped, found.dropped) == (2, 2, 0)


def test_a_cue_maps_to_nothing_for_another_speaker_an_assistant_or_no_message() -> None:
    found = map_cues(
        [
            # Melanie's words, named as Caroline's.
            ProfileCue(speaker="Caroline", quote="I work as a nurse"),
            ProfileCue(speaker="Ava", quote="I love dogs"),
            ProfileCue(speaker="Nobody", quote="Hi"),
            ProfileCue(speaker="Melanie", quote="I am a pilot"),
            ProfileCue(speaker="Melanie", quote=""),
        ],
        _TALK,
        people=["u_car", "u_mel"],
        pictures=[],
    )

    assert found.by_person == {"u_car": PersonCues(), "u_mel": PersonCues()}
    assert all(record.empty for record in found.by_person.values())
    assert (found.given, found.mapped, found.dropped) == (5, 0, 5)


def test_a_label_maps_only_to_a_picture_its_speaker_posted() -> None:
    cell = _cell(
        _message(0, "u_car", "Caroline", f"My dog Max [IMAGE: {_DOG}]"),
        _message(1, "u_mel", "Melanie", f"My cat [IMAGE: {_CAT}]"),
    )
    pictures = [_picture("img-1", _DOG, "u_car"), _picture("img-2", _CAT, "u_mel")]

    found = map_cues(
        [
            ProfileCue(speaker="Caroline", quote="", pictures=["img-1", "img-2"]),
            ProfileCue(speaker="Melanie", quote="My cat", pictures=["img-9"]),
        ],
        cell,
        people=["u_car", "u_mel"],
        pictures=pictures,
    )

    assert found.by_person == {
        "u_car": PersonCues(image_ids=(_DOG,)),
        "u_mel": PersonCues(message_ids=("m_1",)),
    }
    assert (found.mapped, found.dropped) == (2, 0)


def test_a_repeated_quote_maps_once_to_the_first_message_holding_it() -> None:
    cell = _cell(
        _message(0, "u_mel", "Melanie", "I run every morning."),
        _message(1, "u_mel", "Melanie", "Yes, I run every morning, rain or shine."),
    )

    found = map_cues(
        [
            ProfileCue(speaker="Melanie", quote="I run every morning"),
            ProfileCue(speaker="Melanie", quote="rain or shine"),
            ProfileCue(speaker="Melanie", quote="I run every morning"),
        ],
        cell,
        people=["u_mel"],
        pictures=[],
    )

    assert found.by_person == {"u_mel": PersonCues(message_ids=("m_0", "m_1"))}


async def test_extract_cell_maps_the_cues_only_when_asked() -> None:
    reply = json.dumps(
        {
            "title": "t",
            "content": "c",
            "profile_cues": [{"speaker": "Melanie", "quote": "work as a nurse"}],
        }
    )

    asked = await extract_cell(
        FakeLLMClient(responses=[reply]),
        _TALK,
        pictures=[],
        custom_instructions=None,
        output_language=None,
        memcell_id="mc_1",
        cue_people=["u_car", "u_mel"],
    )
    plain_llm = FakeLLMClient(responses=[reply])
    plain = await extract_cell(
        plain_llm,
        _TALK,
        pictures=[],
        custom_instructions=None,
        output_language=None,
        memcell_id="mc_1",
    )

    assert asked[2] is not None
    assert asked[2].by_person == {
        "u_car": PersonCues(),
        "u_mel": PersonCues(message_ids=("m_1",)),
    }
    assert plain[2] is None
    (call,) = plain_llm.calls
    assert call.response_format is EpisodeReply
    assert asked[0].episode == plain[0].episode == "c"
