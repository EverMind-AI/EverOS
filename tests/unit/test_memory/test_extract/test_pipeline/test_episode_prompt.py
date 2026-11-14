"""The one episode prompt, assembled by rule (``chat.episode`` version 4).

What these pin:

- the prompt is registered: id, version, and a fingerprint for the whole
  template and for each of its two variants, ``text`` and ``pictures``;
- every cell is sent one user message whose text is the original request
  (everalgo-user-memory 0.3.1's) byte for byte, then one blank line and the
  blocks the cell adds -- a group's roster block, a picture cell's
  ``<pictures>`` block, then the language block -- whatever the language and
  the custom instructions; a picture cell's message then carries each
  picture after its label;
- a text cell asks for ``EpisodeReply`` (``title`` and ``content``), a
  picture cell for ``ImageEpisodeReply`` (and ``images``); each call asks for
  its output cap; a reply that leaves ``title`` or ``content`` blank is
  unusable, and the summary is the content's first 200 characters;
- a cell is never given a picture it did not send: a text cell's reply loses
  any picture entries, and a reference to a label the cell did not attach is
  taken out of the episode.
"""

from __future__ import annotations

import json

import pytest
from everalgo.llm import LLMOutputError
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, MemCell
from everalgo.user_memory import OutputLanguage
from everalgo.user_memory._language import (
    PARTICIPANT_LANGUAGE_RULE,
    build_language_rule,
)
from everalgo.user_memory.episode._render import (
    _original_conversation,
    _render_original_prompt,
)
from everalgo.user_memory.prompts.en.episode_original import (
    DEFAULT_CUSTOM_INSTRUCTIONS,
    EPISODE_GENERATION_PROMPT,
)
from everalgo.utils import tagged

from everos.memory.extract.pipeline.episode import (
    CHAT_EPISODE,
    EpisodeReply,
    ImageEpisodeReply,
    build_messages,
    extract,
    extract_cell,
    reply_token_cap,
    to_episode,
)
from everos.memory.extract.pipeline.pictures import Picture

_T0 = 1_700_000_000_000
_IMAGE_ID = "9f" + "0" * 62
_PNG = b"\x89PNG\r\n\x1a\nnot really a picture"
_ROSTER = 'This conversation is a group chat in "Trip team".'


def _cell(*texts: str) -> MemCell:
    items = [
        ChatMessage(
            id=f"m_{i}",
            role="user",
            content=text,
            timestamp=_T0 + i,
            sender_id="u1",
            sender_name="Caroline",
        )
        for i, text in enumerate(texts)
    ]
    return MemCell(items=items, timestamp=_T0 + len(texts))


_TEXT = _cell("I'm hiking at Mount Rainier this weekend.", "", "Can't wait!")
_PICTURES = _cell("here is my latte", f"[IMAGE: {_IMAGE_ID}]")


def test_the_prompt_and_its_two_variants_are_registered_and_pinned() -> None:
    assert CHAT_EPISODE.id == "chat.episode"
    assert CHAT_EPISODE.version == 4
    # Pinned: a change to its text -- everalgo's included -- is a new version.
    assert CHAT_EPISODE.fingerprint == "b7c8031efa943152"
    assert CHAT_EPISODE.variant_fingerprint("text") == "f9c003f109a466ea"
    assert CHAT_EPISODE.variant_fingerprint("pictures") == "b7c8031efa943152"


def test_the_core_parts_are_everalgo_s_0_3_1_prompt() -> None:
    assert CHAT_EPISODE.parts[:2] == (
        EPISODE_GENERATION_PROMPT,
        DEFAULT_CUSTOM_INSTRUCTIONS,
    )


@pytest.mark.parametrize(
    "language", [None, OutputLanguage("English"), OutputLanguage("Chinese")]
)
@pytest.mark.parametrize("custom", [None, "", "Name every product version."])
def test_a_one_to_one_text_cell_is_sent_the_original_then_the_language(
    language: OutputLanguage | None, custom: str | None
) -> None:
    rule = build_language_rule(language, fallback=PARTICIPANT_LANGUAGE_RULE)

    (user,) = build_messages(
        _TEXT, pictures=[], custom_instructions=custom, output_language=language
    )

    assert user.role == "user"
    assert user.content == (
        _render_original_prompt(_TEXT, custom_instructions=custom)
        + "\n"
        + tagged("output_language", rule)
    )
    assert user.content.startswith("\nYou are an episodic memory generation expert.")
    assert "<pictures>" not in user.content and '"images"' not in user.content


@pytest.mark.parametrize("language", [None, OutputLanguage("English")])
def test_a_group_text_cell_is_sent_the_original_then_the_roster_and_language(
    language: OutputLanguage | None,
) -> None:
    rule = build_language_rule(language, fallback=PARTICIPANT_LANGUAGE_RULE)

    (user,) = build_messages(
        _TEXT,
        pictures=[],
        custom_instructions=None,
        output_language=language,
        roster=_ROSTER,
    )

    assert user.content == (
        _render_original_prompt(_TEXT, custom_instructions=None)
        + "\n"
        + tagged("group_chat", _ROSTER)
        + "\n\n"
        + tagged("output_language", rule)
    )


def test_a_text_cell_drops_an_empty_message() -> None:
    (user,) = build_messages(
        _TEXT, pictures=[], custom_instructions=None, output_language=None
    )

    assert user.content.count('"speaker": Caroline') == 2


def _picture_block(*labels: str) -> str:
    rules, reply_field, labels_line = CHAT_EPISODE.parts[2:5]
    names = labels_line.format(labels=", ".join(labels))
    return tagged("pictures", f"{rules}\n\n{reply_field}\n\n{names}")


@pytest.mark.parametrize("roster", [None, _ROSTER])
def test_a_picture_cell_is_sent_the_original_then_its_blocks_then_each_picture(
    roster: str | None,
) -> None:
    (user,) = build_messages(
        _PICTURES,
        pictures=[("img-1", _PNG, "image/png"), ("img-2", _PNG, "image/png")],
        custom_instructions="Write tersely.",
        output_language=OutputLanguage("English"),
        roster=roster,
    )

    parts = [part.model_dump() for part in user.content]
    assert user.role == "user"
    assert [part["type"] for part in parts] == [
        "text",
        "text",
        "image_url",
        "text",
        "image_url",
    ]
    rule = build_language_rule(OutputLanguage("English"))
    blocks = [tagged("group_chat", roster)] if roster else []
    blocks += [_picture_block("img-1", "img-2"), tagged("output_language", rule)]
    assert parts[0]["text"] == (
        _render_original_prompt(_PICTURES, custom_instructions="Write tersely.")
        + "\n"
        + "\n\n".join(blocks)
    )
    assert parts[1]["text"] == "<picture>\nlabel: img-1\n</picture>"
    assert parts[2]["image_url"]["detail"] == "high"
    assert parts[3]["text"] == "<picture>\nlabel: img-2\n</picture>"


def test_the_picture_block_asks_for_no_summary() -> None:
    block = _picture_block("img-1")

    assert "summary" not in block
    assert '"images"' in block and "<conversation>" not in block


_REPLY = {"content": "Caroline plans a hike.", "summary": "A hike.", "title": "Hike"}


@pytest.mark.parametrize("roster", [None, _ROSTER])
async def test_a_text_cell_asks_for_episode_reply_and_its_cap(
    roster: str | None,
) -> None:
    llm = FakeLLMClient(responses=[json.dumps(_REPLY)])

    reply = await extract(
        llm,
        _TEXT,
        pictures=[],
        custom_instructions=None,
        output_language=None,
        roster=roster,
    )

    (call,) = llm.calls
    assert call.response_format is EpisodeReply
    assert call.max_tokens == reply_token_cap(
        pictures=0, conversation_chars=len(_original_conversation(_TEXT))
    )
    assert call.extra == {"retry_cut": True}
    assert [message.role for message in call.messages] == ["user"]
    assert reply.title == "Hike"


async def test_a_text_reply_s_summary_is_the_content_s_start() -> None:
    content = "On 2023-11-14, Caroline planned a sunrise hike at Mount Rainier. " * 6
    llm = FakeLLMClient(
        responses=[json.dumps({"title": "Hike", "content": content, "summary": "x"})]
    )

    episode, _ = await extract_cell(
        llm,
        _TEXT,
        pictures=[],
        custom_instructions=None,
        output_language=None,
        memcell_id="mc_0",
    )

    assert episode.episode == content
    assert episode.summary == content[:200]


async def test_a_picture_cell_asks_for_image_episode_reply() -> None:
    reply = _REPLY | {"images": [{"ref": "img-1", "description": "a latte"}]}
    llm = FakeLLMClient(responses=[json.dumps(reply)])

    got = await extract(
        llm,
        _PICTURES,
        pictures=[("img-1", _PNG, "image/png")],
        custom_instructions=None,
        output_language=None,
    )

    (call,) = llm.calls
    assert call.response_format is ImageEpisodeReply
    assert call.max_tokens == reply_token_cap(
        pictures=1, conversation_chars=len(_original_conversation(_PICTURES))
    )
    assert [message.role for message in call.messages] == ["user"]
    assert isinstance(got, ImageEpisodeReply)
    assert [fact.ref for fact in got.images] == ["img-1"]


@pytest.mark.parametrize("blank", ["content", "title"])
async def test_a_reply_with_a_blank_field_is_unusable(blank: str) -> None:
    llm = FakeLLMClient(responses=[json.dumps(_REPLY | {blank: "  "})])

    with pytest.raises(LLMOutputError, match=blank):
        await extract(
            llm, _TEXT, pictures=[], custom_instructions=None, output_language=None
        )


@pytest.mark.parametrize("blank", ["content", "title"])
async def test_a_picture_reply_with_a_blank_field_is_unusable(blank: str) -> None:
    reply = _REPLY | {blank: "  ", "images": [{"ref": "img-1", "description": "x"}]}
    llm = FakeLLMClient(responses=[json.dumps(reply)])

    with pytest.raises(LLMOutputError, match=blank):
        await extract(
            llm,
            _PICTURES,
            pictures=[("img-1", _PNG, "image/png")],
            custom_instructions=None,
            output_language=None,
        )


def test_a_picture_reply_s_summary_is_its_content_s_start_with_labels() -> None:
    content = "Caroline sent her latte (img-1). " * 10
    reply = ImageEpisodeReply(title="t", content=content, images=[])

    episode = to_episode(reply, _PICTURES, labels={"img-1": _IMAGE_ID})

    assert episode.summary == content[:200]
    assert f"(img-1: {_IMAGE_ID})" in episode.episode


async def test_a_text_cell_is_given_no_picture_it_did_not_send() -> None:
    """The 2 of 43 LoCoMo text cells of 2026-11-12 (chat.episode version 1)."""
    hallucinated = {
        "content": "Caroline shared a photo of the trail (img-1). She is excited.",
        "summary": "Caroline shared a trail photo (img-1).",
        "title": "Trail Photo (img-1)",
        "images": [{"ref": "img-1", "description": "a trail", "ocr": ""}],
    }
    llm = FakeLLMClient(responses=[json.dumps(hallucinated)])

    episode, described = await extract_cell(
        llm,
        _TEXT,
        pictures=[],
        custom_instructions=None,
        output_language=None,
        memcell_id="mc_1",
    )

    assert described == {}
    assert episode.episode == "Caroline shared a photo of the trail. She is excited."
    assert episode.summary == episode.episode
    assert episode.subject == "Trail Photo"


async def test_a_reference_to_a_label_not_attached_is_taken_out() -> None:
    reply = {
        "content": "Caroline sent her latte (img-1), then (img-2).",
        "summary": "A latte.",
        "title": "Latte",
        "images": [
            {"ref": "img-1", "description": "a latte", "ocr": ""},
            {"ref": "img-2", "description": "nothing sent", "ocr": ""},
        ],
    }
    llm = FakeLLMClient(responses=[json.dumps(reply)])
    picture = Picture(label="img-1", image_id=_IMAGE_ID, data=_PNG, mime="image/png")

    episode, described = await extract_cell(
        llm,
        _PICTURES,
        pictures=[picture],
        custom_instructions=None,
        output_language=None,
        memcell_id="mc_2",
    )

    assert described == {"img-1": ("a latte", "")}
    assert episode.episode == f"Caroline sent her latte (img-1: {_IMAGE_ID}), then."
    assert episode.summary == "Caroline sent her latte (img-1), then."
