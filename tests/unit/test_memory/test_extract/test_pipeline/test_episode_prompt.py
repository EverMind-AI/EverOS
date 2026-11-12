"""The one episode prompt every chat cell sends (``chat.episode``).

What these pin:

- the prompt is registered: id, version and the fingerprint of its text;
- it opens with everalgo's episode rules byte for byte (``EPISODE_SYSTEM_PROMPT``
  up to its reply format) and then holds one ``<pictures>`` block, one
  ``<reply_format>`` and one ``<output_language>``, in that order: every
  static block before the language; custom instructions (a group's roster
  block) come last;
- a text cell and a picture cell send the same system message; a text cell's
  user message is everalgo's rendering of the conversation, a plain string;
  a picture cell's adds the labels, then each picture after its label;
- every call asks for its output cap, a text cell's included; a reply that
  leaves a field blank is unusable and a summary over 400 units is cut at a
  sentence, as everalgo's extractor holds them.
"""

from __future__ import annotations

import json
import re

import pytest
from everalgo.llm import LLMOutputError
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, MemCell
from everalgo.user_memory import OutputLanguage
from everalgo.user_memory.episode._render import _render_conversation
from everalgo.user_memory.prompts.en.episode import EPISODE_SYSTEM_PROMPT

from everos.memory.extract.pipeline.episode import (
    CHAT_EPISODE,
    EPISODE_RULES,
    EpisodeReply,
    build_messages,
    build_system_prompt,
    extract,
    render_conversation,
    reply_token_cap,
    to_episode,
)

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


def test_the_prompt_is_registered_and_pinned() -> None:
    assert CHAT_EPISODE.id == "chat.episode"
    assert CHAT_EPISODE.version == 1
    # Pinned: a change to its text -- everalgo's rules included -- is a new
    # version, on purpose.
    assert CHAT_EPISODE.fingerprint == "888b4142faa46412"


def test_the_rules_are_everalgo_s_byte_for_byte() -> None:
    assert CHAT_EPISODE.parts[0] == EPISODE_RULES
    assert EPISODE_SYSTEM_PROMPT.startswith(EPISODE_RULES + "\n\n<reply_format>\n")
    rest = EPISODE_SYSTEM_PROMPT.removeprefix(EPISODE_RULES)
    assert rest.count("<reply_format>") == 1
    assert rest.endswith("</reply_format>")
    system = build_system_prompt(custom_instructions=None, output_language=None)
    assert system.startswith(EPISODE_RULES + "\n\n<pictures>\n")


@pytest.mark.parametrize("language", [OutputLanguage("English"), None])
def test_one_block_of_each_static_ones_first(language: OutputLanguage | None) -> None:
    system = build_system_prompt(custom_instructions=_ROSTER, output_language=language)
    tail = _blocks(system.removeprefix(EPISODE_RULES))

    assert tail == [
        "pictures",
        "reply_format",
        "output_language",
        "custom_instructions",
    ]
    assert system.endswith(f"<custom_instructions>\n{_ROSTER}\n</custom_instructions>")


def test_the_picture_rules_apply_only_with_pictures() -> None:
    pictures, reply_format = CHAT_EPISODE.parts[1], CHAT_EPISODE.parts[2]

    assert pictures.startswith(
        "Apply these rules only when the user message carries pictures"
    )
    assert reply_format.count("Reply with one JSON object") == 1
    assert "with no picture attached it is []" in reply_format


def test_a_text_and_a_picture_cell_send_the_same_system_message() -> None:
    language = OutputLanguage("English")
    text = build_messages(
        _TEXT, pictures=[], custom_instructions=None, output_language=language
    )
    picture = build_messages(
        _PICTURES,
        pictures=[("img-1", _PNG, "image/png")],
        custom_instructions=None,
        output_language=language,
    )

    assert text[0].content == picture[0].content


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


async def test_a_text_cell_asks_for_its_cap_and_sends_no_picture() -> None:
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
    assert (reply.title, reply.images) == ("Hike", [])


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
