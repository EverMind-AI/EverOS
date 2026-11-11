"""The picture route's reply: lenient about form, strict about meaning.

Every reply below is one of the shapes DeepSeek V4.1 Flash wrote in JSON mode
on 2026-11-11 (MemEye at temperature 1.0: 7 of 168 picture calls; CUE-Mem at
temperature 0: 3 of 125), cut down to what makes it fail. Each one cost a
second call that carried every picture again. What these pin:

- an unknown key left empty reads as absent, in a picture's entry, in a
  profile operation and at the top;
- an array element left open, and a reply split in two objects, are mended
  before validation -- in the provider's check and in the operator's parse
  alike, so the call is answered by its first reply;
- ``null`` reads as empty (``ocr``, ``source_image_id``) or as no entries
  (``images``, ``profile``), and one entry given bare as a list of it;
- what is not a slip of form still fails: an unknown key that carries
  something, a missing description or episode, a wrong type, an operation
  the route does not offer, a form slip the repair does not know;
- the JSON schema the call sends is unchanged, byte for byte.
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
from everos.memory.extract.pipeline import image_memory
from everos.memory.extract.pipeline.image_memory import ImageEpisodeReply

_HEAD = (
    '"content": "Wang Lei sent her latte (img-1).", '
    '"summary": "A latte.", "title": "Latte"'
)
_IMAGE = '{"ref": "img-1", "description": "a latte with heart foam", "ocr": ""}'
_OP = (
    '{"action": "add", "type": "explicit_info", "source_image_id": "img-1", '
    '"data": {"category": "pets", "description": "Has a cat named Biscuit", '
    '"evidence": "2026-10-30: the tag reads Biscuit"}}'
)
_OP_OPEN = _OP[:-1]  # the operation's own closing brace left off


def _reply(images: str = f"[{_IMAGE}]", profile: str = "[]") -> str:
    return f'{{{_HEAD}, "images": {images}, "profile": {profile}}}'


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


def test_an_empty_unknown_key_at_the_top_reads_as_absent() -> None:
    reply = ImageEpisodeReply.model_validate_json(_reply()[:-1] + ', "notes": null}')

    assert (reply.title, [f.ref for f in reply.images]) == ("Latte", ["img-1"])


def test_an_empty_unknown_key_in_a_profile_operation_reads_as_absent() -> None:
    op = _OP.replace('"action": "add",', '"action": "add", "action_note": "",')

    reply = ImageEpisodeReply.model_validate_json(_reply(profile=f"[{op}]"))

    assert [o.data.description for o in reply.profile] == ["Has a cat named Biscuit"]


@pytest.mark.parametrize(
    "profile",
    [
        f"[{_OP_OPEN}, {_OP_OPEN}]",  # every element left open
        f"[{_OP_OPEN}, {_OP}]",  # the first one only
        f"[{_OP_OPEN}]",  # the last one, before the array closes
    ],
)
def test_a_profile_operation_left_open_is_closed(profile: str) -> None:
    text = _reply(profile=profile)
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)

    reply = ImageEpisodeReply.model_validate_json(text)

    assert {o.source_image_id for o in reply.profile} == {"img-1"}
    assert len(reply.profile) == profile.count('"action"')
    assert [f.ref for f in reply.images] == ["img-1"]


def test_a_reply_split_in_two_objects_is_read_as_one() -> None:
    text = f'{{{_HEAD}, "images": [{_IMAGE}]}}\n{{"profile": [{_OP}]}}'

    reply = ImageEpisodeReply.model_validate_json(text)

    assert reply.title == "Latte"
    assert [f.ref for f in reply.images] == ["img-1"]
    assert [o.source_image_id for o in reply.profile] == ["img-1"]


# ---------------------------------------------------------------------------
# null, and one entry where a list is expected
# ---------------------------------------------------------------------------


def test_null_reads_as_empty_or_as_no_entries() -> None:
    image = '{"ref": "img-1", "description": "a latte", "ocr": null}'
    op = _OP.replace('"source_image_id": "img-1"', '"source_image_id": null')

    reply = ImageEpisodeReply.model_validate_json(
        _reply(images=f"[{image}]", profile=f"[{op}]")
    )
    bare = ImageEpisodeReply.model_validate_json(_reply(images="null", profile="null"))

    assert reply.images[0].ocr == "" and reply.profile[0].source_image_id == ""
    assert bare.images == [] and bare.profile == []


def test_one_entry_given_bare_reads_as_a_list_of_it() -> None:
    reply = ImageEpisodeReply.model_validate_json(_reply(images=_IMAGE, profile=_OP))

    assert [f.ref for f in reply.images] == ["img-1"]
    assert [o.source_image_id for o in reply.profile] == ["img-1"]


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
        _reply(profile=f'[{_OP[:-2]}, "basis": "a tag"}}}}]'),
        # a trait's grounding is its basis; evidence on a trait is not dropped
        _reply(
            profile='[{"action": "add", "type": "implicit_traits", '
            '"data": {"trait": "t", "description": "d", "basis": "b", '
            '"evidence": "2026-10-30: a tag"}}]'
        ),
        _reply(images='[{"ref": "img-1", "ocr": ""}]'),  # no description
        _reply(images='[{"ref": "img-1", "description": "a latte", "ocr": 7}]'),
        _reply(profile='[{"action": "update", "type": "explicit_info", "data": {}}]'),
        '{"summary": "s", "title": "t", "images": [], "profile": []}',  # no episode
        _reply(profile=f"[{_OP[:-2]}, {_OP}]"),  # two braces missing
        _reply() + '\n{"title": "again"}',  # a key given twice across a split
    ],
)
def test_what_is_not_a_form_slip_still_fails(text: str) -> None:
    with pytest.raises(ValidationError):
        ImageEpisodeReply.model_validate_json(text)


def test_the_schema_sent_is_unchanged() -> None:
    """sha256 of the json_schema request format at a3dd742, before this fix.

    The reply models' docstrings are part of it; a change here is a change to
    what every json_schema-mode picture call sends.
    """
    sent = to_openai_response_format(ImageEpisodeReply, "json_schema")
    text = json.dumps(sent, sort_keys=True)

    assert hashlib.sha256(text.encode()).hexdigest()[:16] == "c833a52284f4d7c5"
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
    llm = _provider([_reply(profile=f"[{_OP_OPEN}, {_OP_OPEN}]")], seen)

    reply = await image_memory.extract(
        llm,
        _cell(),
        pictures=[("img-1", b"picture bytes", "image/png")],
        custom_instructions=None,
        output_language=None,
    )

    assert len(seen) == 1
    assert [o.data.description for o in reply.profile] == [
        "Has a cat named Biscuit"
    ] * 2


async def test_a_slip_the_repair_does_not_know_is_still_retried() -> None:
    seen: list[dict[str, Any]] = []
    llm = _provider([_reply(profile=f"[{_OP[:-2]}, {_OP}]"), _reply()], seen)

    reply = await image_memory.extract(
        llm,
        _cell(),
        pictures=[("img-1", b"picture bytes", "image/png")],
        custom_instructions=None,
        output_language=None,
    )

    assert len(seen) == 2
    assert reply.profile == []
