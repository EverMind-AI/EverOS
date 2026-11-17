"""The episode call's last resort: a failed reply salvaged, or one fallback request.

Every reply here is synthetic, cut to the shape of a failure the multimodal
builds of 2026-11-17 recorded (DeepSeek V4.1 Flash, both attempts of a cell
failing the same way, and again on every re-send). What these pin:

- the salvage runs only once every attempt failed in form -- cut at the cap
  or not JSON -- and keeps the title, the episode, every picture entry whose
  ``ref`` and ``description`` were read whole (an ``ocr`` that was cut, ran
  away or ran long trimmed: each run cut to one copy, then cut to
  :data:`SALVAGED_OCR_CHARS`) and every profile cue read whole;
- an ``ocr`` closed by a typographic quote, so the string runs into the next
  line, and a quote copied into an ``ocr`` unescaped are read whole;
- a reply that is JSON but breaks the reply model in meaning is never
  salvaged: the call fails as it did, after the same two requests;
- when no reply keeps a title and an episode, a picture cell sends exactly
  one more request -- the first one's body with one text part appended -- and
  its reply is validated or salvaged; a text cell sends none;
- each salvage and fallback is counted (``everos_episode_reply_salvage_total``);
- the fallback instruction is a registered prompt.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from everalgo.types import ChatMessage, MemCell

from everos.component.llm import LLMCallError, OpenAIProvider
from everos.core.observability.metrics import get_metrics_registry
from everos.memory.extract.pipeline import episode
from everos.memory.extract.pipeline.episode_salvage import (
    CHAT_EPISODE_FALLBACK,
    SALVAGED_OCR_CHARS,
    collapse_runs,
    salvage_reply,
    trim_ocr,
)
from everos.memory.prompt_manifest import prompt_manifest

_TITLE = '"title": "Planning the garden"'
_CONTENT = '"content": "Ana sent the seed plan (img-1) and the poster (img-2)."'
_IMG_1 = '{"ref": "img-1", "description": "a seed plan table", "ocr": "ROW A"}'
_RUNAWAY = "1 2 3 4 5 6 7 8 9 0 " * 300


def _completion(content: str, finish_reason: str = "stop") -> httpx.Response:
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
    """The provider as the 2.0 builds configure it: T 0, retry at 0.3, one retry."""

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


def _cell(pictures: int) -> MemCell:
    refs = " ".join(f"[IMAGE: img-{i}]" for i in range(1, pictures + 1))
    message = ChatMessage(
        id="m_1",
        role="user",
        content=f"our garden plan {refs}".strip(),
        timestamp=1_700_000_000_000,
        sender_id="u1",
    )
    return MemCell(items=[message], timestamp=1_700_000_000_000)


async def _extract(llm: OpenAIProvider, *, pictures: int = 2) -> Any:
    return await episode.extract(
        llm,
        _cell(pictures),
        pictures=[
            (f"img-{i}", b"picture bytes", "image/png") for i in range(1, pictures + 1)
        ],
        custom_instructions=None,
        output_language=None,
        memcell_id="mc_test",
    )


def _salvages(step: str, outcome: str) -> float:
    value = get_metrics_registry().get_sample_value(
        "everos_episode_reply_salvage_total", {"step": step, "outcome": outcome}
    )
    return value or 0.0


# ---------------------------------------------------------------------------
# Salvaged: every attempt failed in form
# ---------------------------------------------------------------------------


async def test_an_ocr_closed_by_a_typographic_quote_is_read_whole() -> None:
    """Both replies close the second ``ocr`` with ”: not JSON, all there."""
    lost = (
        "{\n  "
        + _TITLE
        + ",\n  "
        + _CONTENT
        + ',\n  "images": [\n    '
        + _IMG_1
        + ',\n    {\n      "ref": "img-2",\n      "description": "a poster",\n'
        '      "ocr": "STEP 1 “plant” STEP 2 “water”\n    }\n  ]\n}'
    )
    recovered = _salvages("salvage", "recovered")
    seen: list[dict[str, Any]] = []

    reply = await _extract(_provider([_completion(lost), _completion(lost)], seen))

    assert len(seen) == 2
    assert reply.title == "Planning the garden"
    assert [(f.ref, f.ocr) for f in reply.images] == [
        ("img-1", "ROW A"),
        ("img-2", "STEP 1 “plant” STEP 2 “water”"),
    ]
    assert _salvages("salvage", "recovered") == recovered + 1


async def test_a_runaway_ocr_cut_at_the_cap_twice_keeps_its_description() -> None:
    cut = (
        "{" + _TITLE + ", " + _CONTENT + ', "images": [' + _IMG_1 + ', {"ref": '
        '"img-2", "description": "a poster", "ocr": "GARDEN DAY ' + _RUNAWAY
    )
    seen: list[dict[str, Any]] = []

    reply = await _extract(
        _provider([_completion(cut, "length"), _completion(cut, "length")], seen)
    )

    assert len(seen) == 2
    assert [(f.ref, f.description, f.ocr) for f in reply.images] == [
        ("img-1", "a seed plan table", "ROW A"),
        ("img-2", "a poster", "GARDEN DAY 1 2 3 4 5 6 7 8 9 0"),
    ]


async def test_an_entry_whose_description_was_not_read_whole_is_dropped() -> None:
    cut = (
        "{" + _TITLE + ", " + _CONTENT + ', "images": [' + _IMG_1 + ', {"ref": '
        '"img-2", "description": "a poster of ' + _RUNAWAY
    )
    seen: list[dict[str, Any]] = []

    reply = await _extract(
        _provider([_completion(cut, "length"), _completion(cut, "length")], seen)
    )

    assert [f.ref for f in reply.images] == ["img-1"]
    # The episode still refers to both pictures; only the record is missing.
    assert "(img-2)" in reply.content


async def test_the_attempt_that_keeps_most_is_used_the_first_on_a_tie() -> None:
    one = "{" + _TITLE + ", " + _CONTENT + ', "images": [' + _IMG_1 + ', {"ref'
    both = (
        "{" + _TITLE + ', "content": "second", "images": [' + _IMG_1 + ', {"ref": '
        '"img-2", "description": "a poster", "ocr": "' + _RUNAWAY
    )
    seen: list[dict[str, Any]] = []

    reply = await _extract(
        _provider([_completion(one, "length"), _completion(both, "length")], seen)
    )

    assert reply.content == "second"
    assert [f.ref for f in reply.images] == ["img-1", "img-2"]


async def test_whole_profile_cues_are_kept_and_a_cut_one_dropped() -> None:
    cut = (
        "{" + _TITLE + ", " + _CONTENT + ', "images": [' + _IMG_1 + "], "
        '"profile_cues": [{"speaker": "u1", "quote": "our garden plan"}, '
        '{"speaker": "u1", "quote": "our gar'
    )
    found = salvage_reply(cut, episode.CuedImageEpisodeReply)

    assert found is not None
    assert [(c.speaker, c.quote) for c in found.reply.profile_cues] == [
        ("u1", "our garden plan")
    ]
    assert found.cues_dropped == 1


async def test_a_text_cell_s_reply_is_salvaged_too() -> None:
    """A quote copied into the episode unescaped: not JSON, read whole."""
    broken = '{"title": "Garden", "content": "Ana said "plant it" to Bo."}'
    seen: list[dict[str, Any]] = []

    reply = await _extract(
        _provider([_completion(broken), _completion(broken)], seen), pictures=0
    )

    assert (reply.title, reply.content) == ("Garden", 'Ana said "plant it" to Bo.')
    assert len(seen) == 2


# ---------------------------------------------------------------------------
# Not salvaged: a failure in meaning, or a reply that validates
# ---------------------------------------------------------------------------


async def test_a_reply_that_breaks_the_model_in_meaning_is_never_salvaged() -> None:
    """JSON, but the entry has no description: the call fails as it did."""
    meaning = '{"title": "t", "content": "c", "images": [{"ref": "img-1"}]}'
    seen: list[dict[str, Any]] = []
    replies = [_completion(meaning), _completion(meaning)]

    with pytest.raises(LLMCallError) as info:
        await _extract(_provider(replies, seen), pictures=1)

    assert info.value.reason == "invalid_reply"
    assert len(seen) == 2


async def test_one_attempt_failing_in_meaning_stops_the_salvage() -> None:
    cut = "{" + _TITLE + ", " + _CONTENT + ', "images": [' + _IMG_1 + ', {"ref'
    meaning = '{"title": "t", "content": "c", "images": [{"ref": "img-1"}]}'
    seen: list[dict[str, Any]] = []
    replies = [_completion(cut, "length"), _completion(meaning)]

    with pytest.raises(LLMCallError):
        await _extract(_provider(replies, seen))

    assert len(seen) == 2


async def test_a_reply_that_validates_is_answered_as_before() -> None:
    valid = "{" + _TITLE + ", " + _CONTENT + ', "images": [' + _IMG_1 + "]}"
    seen: list[dict[str, Any]] = []

    reply = await _extract(_provider([_completion(valid)], seen))

    assert len(seen) == 1
    assert [f.ref for f in reply.images] == ["img-1"]


# ---------------------------------------------------------------------------
# The one fallback request
# ---------------------------------------------------------------------------


def _cut_in_content() -> httpx.Response:
    return _completion('{"title": "t", "content": "' + _RUNAWAY, "length")


async def test_no_episode_in_any_reply_sends_one_fallback_request() -> None:
    valid = "{" + _TITLE + ", " + _CONTENT + ', "images": [' + _IMG_1 + "]}"
    answered = _salvages("fallback", "recovered")
    seen: list[dict[str, Any]] = []
    replies = [_cut_in_content(), _cut_in_content(), _completion(valid)]

    reply = await _extract(_provider(replies, seen))

    assert reply.title == "Planning the garden"
    assert len(seen) == 3
    first, _, fallback = seen
    parts = fallback["messages"][0]["content"]
    assert parts[:-1] == first["messages"][0]["content"]
    assert parts[-1] == {"type": "text", "text": CHAT_EPISODE_FALLBACK.parts[0]}
    assert {k: v for k, v in fallback.items() if k != "messages"} == {
        k: v for k, v in first.items() if k != "messages"
    }
    assert _salvages("fallback", "recovered") == answered + 1


async def test_a_fallback_reply_cut_again_is_salvaged() -> None:
    cut = (
        "{" + _TITLE + ", " + _CONTENT + ', "images": [' + _IMG_1 + ', {"ref": '
        '"img-2", "description": "a poster", "ocr": "' + _RUNAWAY
    )
    seen: list[dict[str, Any]] = []
    replies = [_cut_in_content(), _cut_in_content(), _completion(cut, "length")]

    reply = await _extract(_provider(replies, seen))

    assert len(seen) == 3
    assert [f.ref for f in reply.images] == ["img-1", "img-2"]


async def test_a_fallback_that_yields_nothing_fails_the_call_as_before() -> None:
    failed = _salvages("fallback", "failed")
    seen: list[dict[str, Any]] = []
    replies = [_cut_in_content(), _cut_in_content(), _cut_in_content()]

    with pytest.raises(LLMCallError) as info:
        await _extract(_provider(replies, seen))

    assert info.value.reason == "cut_short"
    assert len(seen) == 3
    assert _salvages("fallback", "failed") == failed + 1


async def test_a_text_cell_sends_no_fallback_request() -> None:
    seen: list[dict[str, Any]] = []
    replies = [_cut_in_content(), _cut_in_content()]

    with pytest.raises(LLMCallError):
        await _extract(_provider(replies, seen), pictures=0)

    assert len(seen) == 2


def test_the_fallback_instruction_is_a_registered_prompt() -> None:
    entry = prompt_manifest()["prompts"]["chat.episode_fallback"]

    assert (entry["owner"], entry["used"], entry["version"]) == ("everos", "write", 1)
    assert entry["fingerprint"] == CHAT_EPISODE_FALLBACK.fingerprint


# ---------------------------------------------------------------------------
# Trimming an ocr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("LABEL " + "1 2 3 " * 40 + "END", "LABEL 1 2 3 END"),
        ("SALE " + "-" * 120, "SALE -"),
        # fewer than four copies, or under 100 characters: not a run
        ("ab ab ab", "ab ab ab"),
        ("x" * 99, "x" * 99),
        ("an ordinary line of text", "an ordinary line of text"),
    ],
)
def test_a_run_is_cut_to_one_copy(text: str, expected: str) -> None:
    assert collapse_runs(text) == expected


def test_a_trimmed_ocr_is_cut_to_its_length_after_its_runs() -> None:
    text = "".join(f"line {i}; " for i in range(2000))

    trimmed = trim_ocr(text)

    assert SALVAGED_OCR_CHARS - 10 < len(trimmed) <= SALVAGED_OCR_CHARS
    assert trimmed.startswith("line 0; line 1; ")
    assert trim_ocr("HEAD " + _RUNAWAY) == "HEAD 1 2 3 4 5 6 7 8 9 0"
