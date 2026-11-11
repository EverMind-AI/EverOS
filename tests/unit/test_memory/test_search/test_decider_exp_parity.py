"""The prompt-LLM decider asks and reads the way the exp decider does.

A differential check against the exp tree (EverOS-train@f06d67b, 2026-11-10) ran
both deciders on identical inputs and found two deviations on this side:

* R1 -- every request carried a strict ``response_format`` json_schema, so a vLLM
  endpoint grammar-constrained the decider. The exp decider sends none.
* R2 -- the provider held the reply to a strict model: fenced JSON, prose around
  the object, an extra key, a string index, a missing or null ``next_queries``, a
  float or bool index, a ``<think>`` prefix, or a ``length`` finish after a
  complete object each failed the round -- eight requests, then the fixed top-3
  fallback, which empties the question's core-only result. The exp decider takes
  each of them from one request.

Every case below is one of the differential harness's scenarios (S5a-S5r), run
through the ``OpenAIProvider`` the bootstrap builds for ``[decider]``, over
``httpx.MockTransport``. The decision and the number of requests are the exp
decider's, read off its run. One case differs since 2026-11-11, when
``[decider].salvage_truncated_core`` became the default: a reply cut after its core
list (S5l) keeps that core, as the exp decider does with the same switch on
(EverOS-train@c9711b5). Switched off, it falls back exactly as before.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from everalgo.testing import FakeLLMClient

from everos.component.llm import OpenAIProvider, resolve_decider_config
from everos.config import load_settings
from everos.config.search_runtime import SearchConfigSource
from everos.memory.search.llm_multiround import (
    LLMRoundDecider,
    Picture,
    _coerce_core_indices,
    _parse_decision,
)

_NO_THINK = '{"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}'
_N = 50  # candidates on the round, as on a seed round
_FALLBACK = [0, 1, 2]  # [decider].fallback_core = 3

type Reply = tuple[str | None, str]  # (content, finish_reason)


class _Endpoint:
    """A Chat Completions endpoint replaying ``replies``; the last one repeats."""

    def __init__(self, *replies: Reply) -> None:
        self._replies = list(replies)
        self.bodies: list[dict[str, Any]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        content, finish_reason = self._replies[
            min(len(self.bodies), len(self._replies)) - 1
        ]
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "policy",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                },
            },
        )


@pytest.fixture(autouse=True)
def _decider_env(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The exp chain's decider environment, with the retry backoff dropped."""
    monkeypatch.setenv("EVEROS_DECIDER__MODEL", "policy")
    monkeypatch.setenv("EVEROS_DECIDER__BASE_URL", "http://decider.test/v1")
    monkeypatch.setenv("EVEROS_DECIDER__API_KEY", "EMPTY")
    monkeypatch.setenv("EVEROS_DECIDER__EXTRA", _NO_THINK)
    monkeypatch.setenv("EVEROS_LLMMR_DECIDER_BACKOFF_S", "0")
    load_settings.cache_clear()
    yield
    load_settings.cache_clear()


def _decider(endpoint: _Endpoint) -> LLMRoundDecider:
    """The decider over the provider ``build_llm_provider`` makes for ``[decider]``."""
    cfg = resolve_decider_config(load_settings())
    provider = OpenAIProvider(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
        extra=cfg.extra,
        response_format_mode=cfg.response_format_mode,
        invalid_reply_retries=cfg.invalid_reply_retries,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(endpoint.handle)),
    )
    return LLMRoundDecider(provider, tuning_source=SearchConfigSource().tuning)


def _j(core: object, next_queries: object) -> str:
    return json.dumps({"core": core, "next_queries": next_queries})


# (scenario, reply, finish reason, core, next queries, failed, requests sent)
_EXP_DECISIONS = [
    ("S5a_fenced_json", '```json\n{"core": [0, 2], "next_queries": []}\n```', "stop",
     [0, 2], [], False, 1),
    ("S5b_prose_then_json",
     'Items 0 and 2 matter.\n\n{"core": [0, 2], "next_queries": []}', "stop",
     [0, 2], [], False, 1),
    ("S5c_extra_key", '{"core": [0, 2], "next_queries": [], "reason": "x"}', "stop",
     [0, 2], [], False, 1),
    ("S5d_string_indices", '{"core": ["0", "2"], "next_queries": []}', "stop",
     [0, 2], [], False, 1),
    ("S5e_missing_next_queries", '{"core": [0, 2]}', "stop", [0, 2], [], False, 1),
    ("S5f_null_next_queries", '{"core": [0, 2], "next_queries": null}', "stop",
     [0, 2], [], False, 1),
    ("S5g_float_bool_indices", '{"core": [0, 2.0, true, 5], "next_queries": []}',
     "stop", [0, 5], [], False, 1),
    ("S5h_json_then_trailing_text", '{"core": [0, 2], "next_queries": []}\nDone.',
     "stop", [0, 2], [], False, 1),
    ("S5i_range_neg_dup_and_query_cleanup",
     _j([2, 2, -1, 999, 0], ["  ", "dup q", "dup q", "a q", "b q", "c q"]), "stop",
     [2, 0], ["dup q", "dup q", "a q"], False, 1),
    ("S5j_empty_reply", "", "stop", _FALLBACK, [], True, 4),
    ("S5k_prose_only", "I cannot decide from these items.", "stop",
     _FALLBACK, [], True, 4),
    # Salvaged (on by default): the complete core indices, after every attempt.
    ("S5l_truncated_length", '{"core": [0, 2], "next_queries": ["abc', "length",
     [0, 2], [], False, 4),
    ("S5m_json_then_length",
     '{"core": [0, 2], "next_queries": []}\nNow let me explain my reasoning',
     "length", [0, 2], [], False, 1),
    ("S5n_think_prefix",
     '<think>\n\n</think>\n\n{"core": [0, 2], "next_queries": []}', "stop",
     [0, 2], [], False, 1),
    ("S5o_nonstring_queries", '{"core": [0], "next_queries": [123, "real q"]}',
     "stop", [0], ["123", "real q"], False, 1),
    ("S5p_core_not_list", '{"core": 3, "next_queries": []}', "stop",
     [], [], False, 1),
    ("S5r_null_content", None, "stop", _FALLBACK, [], True, 4),
]  # fmt: skip


@pytest.mark.parametrize(
    ("content", "finish", "core", "queries", "failed", "requests"),
    [pytest.param(*case[1:], id=case[0]) for case in _EXP_DECISIONS],
)
async def test_the_decision_and_the_request_count_are_the_exp_deciders(
    content: str | None,
    finish: str,
    core: list[int],
    queries: list[str],
    failed: bool,
    requests: int,
) -> None:
    endpoint = _Endpoint((content, finish))

    decision = await _decider(endpoint)("q?", "(none)", "x", _N, 0)

    assert (decision.core, decision.queries) == (core, queries)
    assert decision.stop is (not queries)
    assert decision.failed is failed
    assert len(endpoint.bodies) == requests


async def test_switched_off_a_truncated_reply_falls_back_as_the_exp_decider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S5l with ``[decider].salvage_truncated_core = false``: the f06d67b behaviour."""
    monkeypatch.setenv("EVEROS_DECIDER__SALVAGE_TRUNCATED_CORE", "false")
    endpoint = _Endpoint(('{"core": [0, 2], "next_queries": ["abc', "length"))

    decision = await _decider(endpoint)("q?", "(none)", "x", _N, 0)

    assert (decision.core, decision.queries) == (_FALLBACK, [])
    assert decision.failed is True and decision.salvaged is None
    assert len(endpoint.bodies) == 4


async def test_one_attempt_is_one_request() -> None:
    """S5q: an empty reply costs one request, and the next attempt is used."""
    endpoint = _Endpoint(("", "stop"), (_j([0, 2], []), "stop"))

    decision = await _decider(endpoint)("q?", "(none)", "x", _N, 0)

    assert (decision.core, decision.failed) == ([0, 2], False)
    assert len(endpoint.bodies) == 2
    assert endpoint.bodies[0] == endpoint.bodies[1]


async def test_no_request_carries_a_response_format() -> None:
    """R1: the format is asked for in the prompt only, as the exp decider asks."""
    endpoint = _Endpoint((_j([0], []), "stop"))

    await _decider(endpoint)("q?", "(none)", "x", _N, 0)

    (body,) = endpoint.bodies
    assert set(body) == {
        "messages",
        "model",
        "max_tokens",
        "temperature",
        "chat_template_kwargs",
    }
    assert (body["model"], body["max_tokens"], body["temperature"]) == (
        "policy",
        512,
        0.0,
    )
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


async def test_a_response_format_is_an_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``[decider].response_format_mode`` still works when set, and only then."""
    monkeypatch.setenv("EVEROS_DECIDER__RESPONSE_FORMAT_MODE", "json_object")
    load_settings.cache_clear()
    endpoint = _Endpoint((_j([0], []), "stop"))

    await _decider(endpoint)("q?", "(none)", "x", _N, 0)

    (body,) = endpoint.bodies
    assert body["response_format"] == {"type": "json_object"}


def test_the_llm_sections_format_mode_is_not_inherited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A decider running ``[llm]``'s model must not pick up its json_schema."""
    monkeypatch.delenv("EVEROS_DECIDER__MODEL")
    monkeypatch.setenv("EVEROS_LLM__RESPONSE_FORMAT_MODE", "json_schema")
    load_settings.cache_clear()

    cfg = resolve_decider_config(load_settings())

    assert cfg.inherits_model is True
    assert cfg.response_format_mode == "prompt_only"


async def test_a_failed_round_keeps_the_last_reply_and_its_tokens() -> None:
    """As in the exp trace: a degraded round can be read back afterwards."""
    endpoint = _Endpoint(("I cannot decide.", "stop"))

    decision = await _decider(endpoint)("q?", "(none)", "x", _N, 0)

    assert decision.failed is True
    assert decision.raw == "I cannot decide."
    assert decision.usage == {"prompt_tokens": 11, "completion_tokens": 7}


async def test_a_picture_round_goes_out_as_parts_with_no_format() -> None:
    """The picture path rides the same unchecked call: content parts, no format."""
    endpoint = _Endpoint((_j([3], []), "stop"))
    picture = Picture(label="3", data=b"\xff\xd8thumbnail", mime="image/jpeg")

    decision = await _decider(endpoint)("q?", "(none)", "x", _N, 0, pictures=(picture,))

    (body,) = endpoint.bodies
    assert "response_format" not in body
    parts = body["messages"][0]["content"]
    assert [part["type"] for part in parts] == ["text", "text", "image_url"]
    assert parts[1]["text"] == "Picture for evidence item 3:"
    assert decision.core == [3]


async def test_a_client_without_a_raw_call_is_parsed_the_same_way() -> None:
    """A test double or a policy's own client goes through ``achat``; its reply
    gets the same tolerant reading."""
    llm = FakeLLMClient(['```json\n{"core": ["1"], "next_queries": null}\n```'])

    decision = await LLMRoundDecider(llm, tuning_source=SearchConfigSource().tuning)(
        "q?", "(none)", "x", _N, 0
    )

    assert (decision.core, decision.queries, decision.failed) == ([1], [], False)
    assert llm.call_count == 1


# ── the parse helpers, as the exp decider has them ─────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        '{"core": [1], "next_queries": []}',
        '```json\n{"core": [1], "next_queries": []}\n```',
        'Sure. {"core": [1], "next_queries": []} -- done',
    ],
)
def test_parse_decision_finds_the_object(text: str) -> None:
    assert _parse_decision(text) == {"core": [1], "next_queries": []}


def test_a_stray_brace_in_prose_does_not_corrupt_the_parse() -> None:
    """The widest span fails to parse, so the first balanced object is taken."""
    text = 'Note {x} first. {"core": [3], "next_queries": ["a"]} and } more'
    assert _parse_decision(text) == {"core": [3], "next_queries": ["a"]}


@pytest.mark.parametrize("text", ["", "no json here", '{"core": [0', "[1, 2]"])
def test_no_object_is_an_error(text: str) -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        _parse_decision(text)


def test_core_coercion_takes_what_the_exp_decider_takes() -> None:
    """Ints and int-like strings, in range, once; bools, floats and the rest
    are dropped. ``isdecimal`` keeps a superscript digit from crashing ``int``."""
    raw = [0, "2", " 3 ", True, 2.0, -1, "-1", 50, "x", None, "²", 0, "٣"]
    assert _coerce_core_indices(raw, 50) == [0, 2, 3]


def test_a_non_list_core_is_no_core() -> None:
    assert _coerce_core_indices(3, 50) == []
    assert _coerce_core_indices({"0": 1}, 50) == []
