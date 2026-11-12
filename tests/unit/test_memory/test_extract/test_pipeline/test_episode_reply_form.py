"""The episode call's reply: lenient about form, strict about meaning.

Every reply below is one of the shapes DeepSeek V4.1 Flash wrote in JSON mode
on 2026-11-11 (MemEye at temperature 1.0: 7 of 168 picture calls; CUE-Mem at
temperature 0: 3 of 125), cut down to what makes it fail. Each one cost a
second call that carried every picture again. The reply no longer carries
profile operations (the profile update reads the pictures itself), so a reply
that still writes an empty ``profile`` reads as one without it. What these
pin:

- an unknown key left empty reads as absent, in a picture's entry and at the
  top;
- an array element left open, and a reply split in two objects, are mended
  before validation -- in the provider's check and in the operator's parse
  alike, so the call is answered by its first reply;
- ``null`` reads as empty (``ocr``) or as no entries (``images``), and one
  entry given bare as a list of it; a text cell's reply (``EpisodeReply``)
  keeps no ``images`` at all, whatever it wrote, and is mended the same way;
- a reply that leaves ``content``, ``summary`` or ``title`` blank is unusable
  (``LLMOutputError``, which the pipeline retries), as it is for everalgo;
- what is not a slip of form still fails: an unknown key that carries
  something (profile operations included), a missing description or episode,
  a wrong type, a form slip the repair does not know;
- the JSON schema the call sends is pinned.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import httpx
import pytest
from everalgo.llm import to_openai_response_format
from everalgo.types import ChatMessage, MemCell
from pydantic import ValidationError

from everos.component.llm import OpenAIProvider
from everos.memory.extract.pipeline import episode
from everos.memory.extract.pipeline.episode import EpisodeReply, ImageEpisodeReply

_HEAD = (
    '"content": "Wang Lei sent her latte (img-1).", '
    '"summary": "A latte.", "title": "Latte"'
)
_IMAGE = '{"ref": "img-1", "description": "a latte with heart foam", "ocr": ""}'
_IMAGE_2 = '{"ref": "img-2", "description": "a cafe sign", "ocr": "EST. 2019"}'
_OPEN = _IMAGE[:-1]  # the entry's own closing brace left off
_OPEN_2 = _IMAGE_2[:-1]
_OP = (
    '{"action": "add", "type": "explicit_info", "source_image_id": "img-1", '
    '"data": {"category": "pets", "description": "Has a cat named Biscuit", '
    '"evidence": "2026-10-30: the tag reads Biscuit"}}'
)


def _reply(images: str = f"[{_IMAGE}]") -> str:
    return f'{{{_HEAD}, "images": {images}}}'


# ---------------------------------------------------------------------------
# Measured slips of form, each read as the reply it stands for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "image",
    [
        '{"ref": "img-1", "description": "a latte with heart foam", "ocr": "", '
        '"profile": []}',
        '{"ref": "img-1", "description": "a latte with heart foam", "ocr": "", '
        '"ocr_note": "", "unused": ""}',
    ],
)
def test_an_empty_unknown_key_in_a_pictures_entry_reads_as_absent(image: str) -> None:
    reply = ImageEpisodeReply.model_validate_json(_reply(images=f"[{image}]"))

    assert [(f.ref, f.description, f.ocr) for f in reply.images] == [
        ("img-1", "a latte with heart foam", "")
    ]


@pytest.mark.parametrize("extra", [', "notes": null', ', "profile": []'])
def test_an_empty_unknown_key_at_the_top_reads_as_absent(extra: str) -> None:
    reply = ImageEpisodeReply.model_validate_json(_reply()[:-1] + extra + "}")

    assert (reply.title, [f.ref for f in reply.images]) == ("Latte", ["img-1"])


@pytest.mark.parametrize(
    "images",
    [
        f"[{_OPEN}, {_OPEN_2}]",  # every element left open
        f"[{_OPEN}, {_IMAGE_2}]",  # the first one only
        f"[{_IMAGE}, {_OPEN_2}]",  # the last one, before the array closes
    ],
)
def test_a_pictures_entry_left_open_is_closed(images: str) -> None:
    text = _reply(images=images)
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)

    reply = ImageEpisodeReply.model_validate_json(text)

    assert [f.ref for f in reply.images] == ["img-1", "img-2"]


def test_a_reply_split_in_two_objects_is_read_as_one() -> None:
    text = f'{{{_HEAD}}}\n{{"images": [{_IMAGE}]}}'

    reply = ImageEpisodeReply.model_validate_json(text)

    assert reply.title == "Latte"
    assert [f.ref for f in reply.images] == ["img-1"]


# ---------------------------------------------------------------------------
# null, and one entry where a list is expected
# ---------------------------------------------------------------------------


def test_null_reads_as_empty_or_as_no_entries() -> None:
    image = '{"ref": "img-1", "description": "a latte", "ocr": null}'

    reply = ImageEpisodeReply.model_validate_json(_reply(images=f"[{image}]"))
    bare = ImageEpisodeReply.model_validate_json(_reply(images="null"))

    assert reply.images[0].ocr == ""
    assert bare.images == []


def test_one_entry_given_bare_reads_as_a_list_of_it() -> None:
    reply = ImageEpisodeReply.model_validate_json(_reply(images=_IMAGE))

    assert [f.ref for f in reply.images] == ["img-1"]


@pytest.mark.parametrize(
    "text", [f"{{{_HEAD}}}", _reply(images="[]"), _reply(), _reply(images=_IMAGE)]
)
def test_a_text_cell_s_reply_keeps_no_picture_entry(text: str) -> None:
    """A text cell attached no picture: whatever ``images`` it wrote is dropped."""
    reply = EpisodeReply.model_validate_json(text)

    assert reply.title == "Latte"
    assert "images" not in reply.model_dump()


def test_a_text_cell_s_reply_is_mended_like_a_picture_cell_s() -> None:
    reply = EpisodeReply.model_validate_json(f"{{{_HEAD}}}" + '\n{"extra": ""}')

    assert reply.content == "Wang Lei sent her latte (img-1)."


# ---------------------------------------------------------------------------
# What is not a slip of form still fails
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # an unknown key that carries something: content in the wrong place
        _reply(
            images='[{"ref": "img-1", "description": "a latte", "profile": ['
            + _OP
            + "]}]"
        ),
        _reply()[:-1] + f', "profile": [{_OP}]}}',  # the route writes no profile
        _reply(images='[{"ref": "img-1", "ocr": ""}]'),  # no description
        _reply(images='[{"ref": "img-1", "description": "a latte", "ocr": 7}]'),
        '{"summary": "s", "title": "t", "images": []}',  # no episode
        _reply(images=f"[{_IMAGE[:-2]}, {_IMAGE}]"),  # a string left unclosed
        _reply() + '\n{"title": "again"}',  # a key given twice across a split
    ],
)
def test_what_is_not_a_form_slip_still_fails(text: str) -> None:
    with pytest.raises(ValidationError):
        ImageEpisodeReply.model_validate_json(text)


def test_the_schema_sent_is_pinned() -> None:
    """sha256 of the json_schema request format: the reply without the profile.

    The reply models' names and docstrings are part of it; a change here is a
    change to what every json_schema-mode picture call sends. Unchanged by the
    rule-assembled prompt (2026-11-12); a text cell's ``EpisodeReply`` is
    pinned beside it.
    """
    sent = to_openai_response_format(ImageEpisodeReply, "json_schema")
    text = json.dumps(sent, sort_keys=True)
    text_cell = json.dumps(
        to_openai_response_format(EpisodeReply, "json_schema"), sort_keys=True
    )

    assert hashlib.sha256(text.encode()).hexdigest()[:16] == "2e848a4bc74cae48"
    assert hashlib.sha256(text_cell.encode()).hexdigest()[:16] == "069791f7e1c15613"
    assert "profile" not in json.dumps(sent)
    assert to_openai_response_format(ImageEpisodeReply, "json_object") == {
        "type": "json_object"
    }


# ---------------------------------------------------------------------------
# End to end: the provider's check and the operator's parse agree
# ---------------------------------------------------------------------------


def _provider(replies: list[str], seen: list[dict[str, Any]]) -> OpenAIProvider:
    async def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        content = replies.pop(0)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "served-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    return OpenAIProvider(
        model="m",
        api_key="k",
        base_url="http://test.local/v1",
        max_retries=0,
        response_format_mode="json_object",
        invalid_reply_retries=1,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )


def _cell() -> MemCell:
    message = ChatMessage(
        id="m_1",
        role="user",
        content="my cat [IMAGE: img-1]",
        timestamp=1_700_000_000_000,
        sender_id="u1",
    )
    return MemCell(items=[message], timestamp=1_700_000_000_000)


async def test_a_mendable_reply_answers_the_call_without_a_second_request() -> None:
    seen: list[dict[str, Any]] = []
    llm = _provider([_reply(images=f"[{_OPEN}, {_OPEN_2}]")], seen)

    reply = await episode.extract(
        llm,
        _cell(),
        pictures=[("img-1", b"picture bytes", "image/png")],
        custom_instructions=None,
        output_language=None,
    )

    assert len(seen) == 1
    assert [f.ref for f in reply.images] == ["img-1", "img-2"]


async def test_a_slip_the_repair_does_not_know_is_still_retried() -> None:
    seen: list[dict[str, Any]] = []
    llm = _provider([_reply(images=f"[{_IMAGE[:-2]}, {_IMAGE}]"), _reply()], seen)

    reply = await episode.extract(
        llm,
        _cell(),
        pictures=[("img-1", b"picture bytes", "image/png")],
        custom_instructions=None,
        output_language=None,
    )

    assert len(seen) == 2
    assert [f.ref for f in reply.images] == ["img-1"]
