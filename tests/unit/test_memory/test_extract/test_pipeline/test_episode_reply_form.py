"""The episode call's reply: lenient about form, strict about meaning.

Every reply below is one of the shapes DeepSeek V4.1 Flash wrote in JSON mode
on 2026-11-11 (MemEye at temperature 1.0: 7 of 168 picture calls; CUE-Mem at
temperature 0: 3 of 125), cut down to what makes it fail. Each one cost a
second call that carried every picture again. The reply no longer carries
profile operations (the profile update reads the pictures itself), so a reply
that still writes an empty ``profile`` reads as one without it. What these
pin:

- a top-level key the reply model does not define is ignored, empty or not
  -- the keys DeepSeek V4.1 Flash added to the original's reply on the LoCoMo
  probe of 2026-11-14 (``type``, ``_meta``, ``_comment``, ``emotional_tone``,
  ``_time_resolution``) included -- as everalgo-user-memory 0.3.1 read it;
  in a picture's entry an unknown key left empty reads as absent;
- an array element left open, and a reply split in two objects, are mended
  before validation -- in the provider's check and in the operator's parse
  alike, so the call is answered by its first reply;
- ``null`` reads as empty (``ocr``) or as no entries (``images``), and one
  entry given bare as a list of it; a text cell's reply (``EpisodeReply``)
  keeps no ``images`` at all, whatever it wrote, and is mended the same way;
- a ``summary`` the reply writes is dropped (the summary is cut in code), and
  a reply that leaves ``title`` or ``content`` blank is unusable
  (``LLMOutputError``, which the pipeline retries), as it is for everalgo;
- what is not a slip of form still fails: an unknown key that carries
  something inside a picture's entry (profile operations included), a missing
  description or episode, a wrong type, a form slip the repair does not know;
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
    change to what every json_schema-mode picture call sends. Changed by the
    original request (2026-11-14): ``title``, ``content`` and ``images``, no
    ``summary``; a text cell's ``EpisodeReply`` (``title`` and ``content``)
    is pinned beside it.
    """
    sent = to_openai_response_format(ImageEpisodeReply, "json_schema")
    text = json.dumps(sent, sort_keys=True)
    text_cell = json.dumps(
        to_openai_response_format(EpisodeReply, "json_schema"), sort_keys=True
    )

    assert hashlib.sha256(text.encode()).hexdigest()[:16] == "019f7f82befff2be"
    assert hashlib.sha256(text_cell.encode()).hexdigest()[:16] == "0b226fcf3ff40d51"
    assert "profile" not in json.dumps(sent)
    assert to_openai_response_format(ImageEpisodeReply, "json_object") == {
        "type": "json_object"
    }


# ---------------------------------------------------------------------------
# Keys a model adds at the top: ignored, as the original read its reply
# ---------------------------------------------------------------------------

# Five replies DeepSeek V4.1 Flash wrote to the original request on the LoCoMo
# probe of 2026-11-14 (2.0 c785d98, conversation 0), cut to their shape: 123 of
# its 127 replies carried keys of their own, and the strict reading rejected
# all but 4. The keys and their values are the model's, verbatim.
_TITLE = "Caroline and Melanie Discuss LGBTQ Support Group, Career Goals, and Painting"
_CONTENT = "On May 8, 2023 at 1:56 PM UTC, Caroline and Melanie greeted each other."
_ADDED_KEYS = [
    {"type": "json_object"},
    {
        "_meta": {
            "timestamps": {
                "start": "2023-05-08T13:56:00Z",
                "end": "2023-05-08T14:04:00Z",
            }
        }
    },
    {
        "_comment": (
            "The content field is a string, not an object. The JSON object "
            "contains only title and content as required."
        )
    },
    {"emotional_tone": "supportive, reflective, encouraging"},
    {"_time_resolution": {"yesterday": "May 7, 2023", "last year": "2022"}},
]


@pytest.mark.parametrize("added", _ADDED_KEYS, ids=lambda d: next(iter(d)))
def test_a_key_the_model_adds_at_the_top_is_ignored(added: dict[str, Any]) -> None:
    text = json.dumps({"title": _TITLE, "content": _CONTENT} | added)

    text_reply = EpisodeReply.model_validate_json(text)
    picture_reply = ImageEpisodeReply.model_validate_json(
        json.dumps({"title": _TITLE, "content": _CONTENT, "images": []} | added)
    )

    assert (text_reply.title, text_reply.content) == (_TITLE, _CONTENT)
    assert text_reply.model_dump() == {"title": _TITLE, "content": _CONTENT}
    assert (picture_reply.title, picture_reply.content) == (_TITLE, _CONTENT)


def test_every_added_key_at_once_and_a_summary_are_ignored() -> None:
    added = {k: v for d in _ADDED_KEYS for k, v in d.items()} | {"summary": "s"}
    text = json.dumps({"title": _TITLE, "content": _CONTENT} | added)

    assert EpisodeReply.model_validate_json(text).model_dump() == {
        "title": _TITLE,
        "content": _CONTENT,
    }


def test_profile_operations_at_the_top_are_ignored() -> None:
    reply = ImageEpisodeReply.model_validate_json(
        _reply()[:-1] + f', "profile": [{_OP}]}}'
    )

    assert [f.ref for f in reply.images] == ["img-1"]
    assert "profile" not in reply.model_dump()


@pytest.mark.parametrize("missing", ["title", "content"])
def test_a_reply_without_title_or_content_still_fails(missing: str) -> None:
    fields = {"title": _TITLE, "content": _CONTENT, "type": "json_object"}
    fields.pop(missing)

    with pytest.raises(ValidationError):
        EpisodeReply.model_validate_json(json.dumps(fields))


async def test_a_probe_reply_with_added_keys_answers_the_call() -> None:
    """The probe's failure: a reply with keys of its own failed validation."""
    seen: list[dict[str, Any]] = []
    reply = json.dumps({"title": _TITLE, "content": _CONTENT, "_meta": {"a": 1}})
    llm = _provider([reply], seen)
    message = ChatMessage(
        id="m_1",
        role="user",
        content="Hey Mel! Good to see you!",
        timestamp=1_683_554_160_000,
        sender_id="caroline",
        sender_name="Caroline",
    )
    cell = MemCell(items=[message], timestamp=1_683_554_160_000)

    got = await episode.extract(
        llm, cell, pictures=[], custom_instructions=None, output_language=None
    )

    assert len(seen) == 1
    assert (got.title, got.content) == (_TITLE, _CONTENT)


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
