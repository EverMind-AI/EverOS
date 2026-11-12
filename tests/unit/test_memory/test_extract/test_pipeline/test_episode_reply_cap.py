"""The episode call's output cap and its retry on a cut.

At temperature 0, DeepSeek V4.1 Flash sometimes transcribes a text-dense
picture into ``ocr`` and repeats it until the output limit: on 2026-11-11, two
or three of MemEye's 168 picture calls did so in each of three builds, ran to
the 32,768 tokens the builds allowed, and their cells kept no episode and no
picture memory. What these pin:

- the cap is ``2,048`` output tokens, plus ``1,024`` per picture, plus one per
  four characters of the rendered conversation, never above 32,768; every
  complete reply measured fits under its own -- the longest picture replies
  of 2026-11-11 and the longest text replies on record, which came from long
  transcript cells and would not fit under ``2,048`` alone;
- the call asks for that cap even when the deployment configures a larger
  ``max_tokens`` (the builds' 32,768 in ``[llm] extra``), and keeps the
  configured temperature;
- a reply cut at the cap is sent again once, at ``retry_temperature``, with
  the same cap; a second cut fails the call (``cut_short``), as one cut did
  before;
- a normal reply is answered by the first request, unchanged.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from everalgo.types import ChatMessage, MemCell

from everos.component.llm import LLMCallError, OpenAIProvider
from everos.memory.extract.pipeline import episode
from everos.memory.extract.pipeline.episode import reply_token_cap

_REPLY = json.dumps(
    {
        "content": "Alley Oop shared page 1 of Champ Comics (img-1).",
        "summary": "A comic page.",
        "title": "Champ Comics Page 1",
        "images": [{"ref": "img-1", "description": "a comic page", "ocr": "POW"}],
    }
)
_LOOP = '{"content": "c", "summary": "s", "title": "t", "images": [{"ref": "img-1", '
_LOOP += '"description": "a comic page", "ocr": "' + "I'D RATHER BE A BARRIER! " * 40


@pytest.mark.parametrize(
    ("pictures", "chars", "cap"),
    [
        (0, 0, 2048),
        (1, 0, 3072),
        (8, 0, 10240),
        (1, 400, 3172),
        (0, 24_120, 8078),
        (30, 0, 32768),
        (0, 200_000, 32768),
    ],
)
def test_the_cap_grows_by_picture_and_conversation_and_stops_at_the_ceiling(
    pictures: int, chars: int, cap: int
) -> None:
    assert reply_token_cap(pictures=pictures, conversation_chars=chars) == cap


@pytest.mark.parametrize(
    ("pictures", "chars", "longest"),
    [
        # The longest picture replies of 2026-11-11, their text not counted.
        (1, 0, 1486),
        (8, 0, 2423),
        # The longest text replies on record: transcript cells of 15-29K chars.
        (0, 24_120, 2613),
        (0, 28_895, 2191),
        (0, 15_232, 2105),
    ],
)
def test_the_longest_replies_measured_fit_under_their_cap(
    pictures: int, chars: int, longest: int
) -> None:
    assert longest < reply_token_cap(pictures=pictures, conversation_chars=chars)


def test_a_cap_of_2048_alone_would_cut_the_longest_text_replies() -> None:
    assert reply_token_cap(pictures=0, conversation_chars=0) < 2613


def _completion(content: str, finish_reason: str) -> httpx.Response:
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
                    "finish_reason": finish_reason,
                }
            ],
        },
    )


def _provider(
    replies: list[httpx.Response], seen: list[dict[str, Any]]
) -> OpenAIProvider:
    """The provider as the 2.0 builds configure it: T 0, retry at 0.3, 32,768 cap."""

    async def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return replies.pop(0)

    return OpenAIProvider(
        model="m",
        api_key="k",
        base_url="http://test.local/v1",
        max_retries=0,
        temperature=0.0,
        retry_temperature=0.3,
        extra={"max_tokens": 32768},
        response_format_mode="json_object",
        invalid_reply_retries=1,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )


def _cell() -> MemCell:
    message = ChatMessage(
        id="m_1",
        role="user",
        content="Page 1 of Champ Comics. [IMAGE: img-1]",
        timestamp=1_700_000_000_000,
        sender_id="u1",
    )
    return MemCell(items=[message], timestamp=1_700_000_000_000)


_CAP = reply_token_cap(
    pictures=1, conversation_chars=len(episode.render_conversation(_cell()))
)


async def _extract(llm: OpenAIProvider) -> episode.ImageEpisodeReply:
    return await episode.extract(
        llm,
        _cell(),
        pictures=[("img-1", b"picture bytes", "image/jpeg")],
        custom_instructions=None,
        output_language=None,
    )


async def test_a_normal_reply_is_answered_by_the_first_request() -> None:
    seen: list[dict[str, Any]] = []

    reply = await _extract(_provider([_completion(_REPLY, "stop")], seen))

    assert reply.title == "Champ Comics Page 1"
    assert [(b["temperature"], b["max_tokens"]) for b in seen] == [(0.0, _CAP)]


async def test_a_cut_reply_is_retried_once_at_the_retry_temperature() -> None:
    seen: list[dict[str, Any]] = []
    replies = [_completion(_LOOP, "length"), _completion(_REPLY, "stop")]

    reply = await _extract(_provider(replies, seen))

    assert reply.images[0].ocr == "POW"
    assert [(b["temperature"], b["max_tokens"]) for b in seen] == [
        (0.0, _CAP),
        (0.3, _CAP),
    ]
    first, second = seen
    assert {k: v for k, v in first.items() if k != "temperature"} == {
        k: v for k, v in second.items() if k != "temperature"
    }


async def test_a_second_cut_fails_the_call() -> None:
    seen: list[dict[str, Any]] = []
    replies = [_completion(_LOOP, "length"), _completion(_LOOP, "length")]

    with pytest.raises(LLMCallError) as info:
        await _extract(_provider(replies, seen))

    assert info.value.reason == "cut_short"
    assert len(seen) == 2
