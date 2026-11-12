"""The one episode prompt, assembled by rule (``chat.episode`` version 2).

What these pin:

- the prompt is registered: id, version, and a fingerprint for the whole
  template and for each of its two variants, ``text`` and ``pictures``;
- a cell without pictures is sent everalgo's episode prompt byte for byte --
  ``assemble_system_prompt``'s output, whatever the language and the custom
  instructions -- and the conversation alone, as everalgo renders it; it is
  shown no picture rule;
- a cell with pictures is sent the same rules first, then the ``<pictures>``
  block, one reply format (with ``images``) and one language block, the
  custom instructions last; its user message adds the labels, then each
  picture after its label;
- a text cell asks for ``EpisodeReply`` and a picture cell for
  ``ImageEpisodeReply``; each call asks for its output cap; a reply that
  leaves a field blank is unusable and a summary over 400 units is cut at a
  sentence, as everalgo's extractor holds them;
- a cell is never given a picture it did not send: a text cell's reply loses
  any picture entries, and a reference to a label the cell did not attach is
  taken out of the episode.
"""

from __future__ import annotations

import json
import re

import pytest
from everalgo.llm import LLMOutputError
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, MemCell
from everalgo.user_memory import OutputLanguage
from everalgo.user_memory._language import (
    PARTICIPANT_LANGUAGE_RULE,
    build_language_rule,
)
from everalgo.user_memory.episode._render import _render_conversation
from everalgo.user_memory.prompts.en.episode import (
    EPISODE_SYSTEM_PROMPT,
    assemble_system_prompt,
)

from everos.memory.extract.pipeline.episode import (
    CHAT_EPISODE,
    EPISODE_RULES,
    EpisodeReply,
    ImageEpisodeReply,
    build_messages,
    build_system_prompt,
    extract,
    extract_cell,
    render_conversation,
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


def _blocks(system: str) -> list[str]:
    """The top-level blocks of a system message, in order."""
    return re.findall(r"^<([a-z_]+)>$", system, flags=re.MULTILINE)


def test_the_prompt_and_its_two_variants_are_registered_and_pinned() -> None:
    assert CHAT_EPISODE.id == "chat.episode"
    assert CHAT_EPISODE.version == 2
    # Pinned: a change to its text -- everalgo's included -- is a new version.
    assert CHAT_EPISODE.fingerprint == "2962c35cf3bd2482"
    assert CHAT_EPISODE.variant_fingerprint("text") == "2117ea4fa75e421a"
    assert CHAT_EPISODE.variant_fingerprint("pictures") == "bae4c0c544622565"


def test_the_text_parts_are_everalgo_s_prompt_byte_for_byte() -> None:
    rules, text_format = CHAT_EPISODE.parts[:2]

    assert rules == EPISODE_RULES
    assert f"{rules}\n\n<reply_format>\n{text_format}\n</reply_format>" == (
        EPISODE_SYSTEM_PROMPT
    )


@pytest.mark.parametrize(
    "language", [None, OutputLanguage("English"), OutputLanguage("Chinese")]
)
@pytest.mark.parametrize("custom", [None, "", _ROSTER])
def test_a_text_cell_is_sent_everalgo_s_system_prompt(
    language: OutputLanguage | None, custom: str | None
) -> None:
    rule = build_language_rule(language, fallback=PARTICIPANT_LANGUAGE_RULE)

    system = build_system_prompt(
        pictures=False, custom_instructions=custom, output_language=language
    )

    assert system == assemble_system_prompt(
        custom_instructions=custom, language_rule=rule
    )
    assert "<pictures>" not in system and '"images"' not in system


@pytest.mark.parametrize("language", [None, OutputLanguage("English")])
def test_a_picture_cell_shares_the_rules_then_gets_one_of_each_block(
    language: OutputLanguage | None,
) -> None:
    system = build_system_prompt(
        pictures=True, custom_instructions=_ROSTER, output_language=language
    )

    assert system.startswith(EPISODE_RULES + "\n\n<pictures>\n")
    assert _blocks(system.removeprefix(EPISODE_RULES)) == [
        "pictures",
        "reply_format",
        "output_language",
        "custom_instructions",
    ]
    assert system.count("Reply with one JSON object") == 1
    assert system.endswith(f"<custom_instructions>\n{_ROSTER}\n</custom_instructions>")


def test_a_text_cell_s_user_message_is_everalgo_s() -> None:
    _, user = build_messages(
        _TEXT, pictures=[], custom_instructions=None, output_language=None
    )

    assert user.content == _render_conversation(_TEXT)
    assert user.content == render_conversation(_TEXT)
    assert '<conversation count="2">' in user.content  # the empty message dropped


def test_a_picture_cell_adds_the_labels_then_each_picture() -> None:
    _, user = build_messages(
        _PICTURES,
        pictures=[("img-1", _PNG, "image/png"), ("img-2", _PNG, "image/png")],
        custom_instructions=None,
        output_language=None,
    )

    parts = [part.model_dump() for part in user.content]
    assert [part["type"] for part in parts] == [
        "text",
        "text",
        "image_url",
        "text",
        "image_url",
    ]
    assert parts[0]["text"] == (
        render_conversation(_PICTURES)
        + "\n\n<pictures>\nLabels in order: img-1, img-2. "
        "Read every attached picture.\n</pictures>"
    )
    assert parts[1]["text"] == "<picture>\nlabel: img-1\n</picture>"
    assert parts[2]["image_url"]["detail"] == "high"


_REPLY = {"content": "Caroline plans a hike.", "summary": "A hike.", "title": "Hike"}


async def test_a_text_cell_asks_for_episode_reply_and_its_cap() -> None:
    llm = FakeLLMClient(responses=[json.dumps(_REPLY)])

    reply = await extract(
        llm, _TEXT, pictures=[], custom_instructions=None, output_language=None
    )

    (call,) = llm.calls
    assert call.response_format is EpisodeReply
    assert call.max_tokens == reply_token_cap(
        pictures=0, conversation_chars=len(render_conversation(_TEXT))
    )
    assert call.extra == {"retry_cut": True}
    assert isinstance(call.messages[1].content, str)
    assert reply.title == "Hike"


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
    assert isinstance(got, ImageEpisodeReply)
    assert [fact.ref for fact in got.images] == ["img-1"]


@pytest.mark.parametrize("blank", ["content", "summary", "title"])
async def test_a_reply_with_a_blank_field_is_unusable(blank: str) -> None:
    llm = FakeLLMClient(responses=[json.dumps(_REPLY | {blank: "  "})])

    with pytest.raises(LLMOutputError, match=blank):
        await extract(
            llm, _TEXT, pictures=[], custom_instructions=None, output_language=None
        )


def test_a_summary_over_its_width_is_cut_at_a_sentence() -> None:
    summary = "Caroline planned a sunrise hike at Mount Rainier. " * 12
    reply = EpisodeReply(content="c", summary=summary, title="t")

    episode = to_episode(reply, _TEXT)

    assert len(episode.summary) < len(summary)
    assert episode.summary.endswith("…")
    assert episode.summary.startswith("Caroline planned a sunrise hike")


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
    assert episode.summary == "Caroline shared a trail photo."
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
