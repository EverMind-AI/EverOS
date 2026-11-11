"""``[decider].fit_to_window`` and ``[decider].salvage_truncated_core``.

Two ways a round of the multi-round decider is lost, both measured on BEAM with the
RL-tuned 27B decider (2026-11-10):

* the round prompt is longer than the decider's window (the core grows to ~50 items,
  each re-rendered in full every round): the endpoint answers HTTP 400 on every attempt
  and the round falls back;
* the reply stops right after the core list (``{"core": [0],``): it fails to parse on
  every attempt and the round falls back.

Both switches are off by default and, off, nothing changes. On, each acts only on a
round that would otherwise have failed: the fit only after an attempt was rejected for
its length, the salvage only after every attempt failed to parse.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from typing import Any

import httpx
import pytest
from everalgo.llm import LLMMessage
from everalgo.types import Candidate

from everos.component.llm import OpenAIProvider
from everos.component.llm.openai_provider import RawReply
from everos.config.search_runtime import SearchConfigSource
from everos.memory.ports import SearchHit
from everos.memory.search import llm_multiround as lmr
from everos.memory.search.llm_multiround import (
    _DECIDER_PROMPT,
    _PICTURE_MARK,
    LLMRoundDecider,
    PromptTokenCounter,
    RawChatClient,
    _build_fit_view,
    _is_context_overflow,
    _plan_fit,
    _render_blocks,
    _render_core_so_far,
    _render_fit_view,
    _salvage_core,
    _waterfill,
    search_episodes_llm_multiround,
)
from everos.memory.search.recall import RecallResult

from ._records import hit, recalled, stored_episode

_LIVE = SearchConfigSource()

# ── stubs ────────────────────────────────────────────────────────────────────


def _tokens(prompt: str) -> int:
    """The fake endpoint's tokenizer: its window check and its count both use it."""
    return len(prompt) // 4


class _OverflowError(Exception):
    pass


def _overflow_error(window: int) -> Exception:
    """What the decider sees from vLLM 0.32: the provider's error over the SDK's 400."""
    sdk = _OverflowError(
        "Error code: 400 - {'error': {'message': \"This model's maximum context "
        f"length is {window} tokens. However, your prompt contains at least "
        f"{window + 1} input tokens, which exceeds this limit even without any "
        "output tokens.\", 'type': 'BadRequestError', 'code': 400}}"
    )
    err = RuntimeError(str(sdk))
    err.__cause__ = sdk
    return err


def _text(messages: list[LLMMessage]) -> str:
    content = messages[0].content
    assert isinstance(content, str)
    return content


class _Endpoint:
    """A decider endpoint with a window, reached as ``achat_raw`` (+ /tokenize)."""

    def __init__(
        self, replies: list[str], *, window: int = 10**9, counts: bool = True
    ) -> None:
        self.replies = replies
        self.window = window
        self.counts = counts
        self.prompts: list[str] = []
        self.params: list[dict[str, Any]] = []
        self.counted: list[str] = []
        self.accepted = 0

    async def achat_raw(self, messages: list[LLMMessage], **params: Any) -> RawReply:
        prompt = _text(messages)
        self.prompts.append(prompt)
        self.params.append(dict(params))
        if _tokens(prompt) > self.window:
            raise _overflow_error(self.window)
        reply = self.replies[min(self.accepted, len(self.replies) - 1)]
        self.accepted += 1
        return RawReply(text=reply, finish_reason="stop", usage=None)

    async def achat(self, messages: list[LLMMessage], **params: Any) -> str:
        raise AssertionError("the decider must use achat_raw")

    async def acount_prompt_tokens(
        self, messages: list[LLMMessage], **extra: Any
    ) -> tuple[int, int] | None:
        prompt = _text(messages)
        self.counted.append(prompt)
        return (_tokens(prompt), self.window) if self.counts else None


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVEROS_LLMMR_DECIDER_BACKOFF_S", "0")


def _on(
    monkeypatch: pytest.MonkeyPatch, *, fit: bool = False, salvage: bool = False
) -> None:
    if fit:
        monkeypatch.setenv("EVEROS_DECIDER__FIT_TO_WINDOW", "true")
    if salvage:
        monkeypatch.setenv("EVEROS_DECIDER__SALVAGE_TRUNCATED_CORE", "true")


def _decider(llm: Any) -> LLMRoundDecider:
    return LLMRoundDecider(llm, tuning_source=_LIVE.tuning)


def _cand(i: int, text: str, *, picture: bool = False) -> Candidate:
    meta: dict[str, Any] = {
        "subject": f"subject {i}",
        "summary": "",
        "episode": text,
        "session_id": f"s{i}",
    }
    if picture:
        meta["kind"] = lmr.IMAGE_KIND
    return Candidate(
        id=f"alice__e{i}", score=1.0 / (i + 1), source="vector", metadata=meta
    )


def _view(core_texts: list[str], block_texts: list[list[str]]):
    cands = [_cand(100 + i, t) for i, t in enumerate(core_texts)]
    cand_by_id = {c.id: c for c in cands}
    core_order = [c.id for c in cands]
    core_source = {c.id: "original question" for c in cands}
    blocks: list[tuple[str, list[Candidate]]] = []
    i = 0
    for b, texts in enumerate(block_texts):
        bc = []
        for t in texts:
            bc.append(_cand(i, t))
            i += 1
        blocks.append((f"sub-query {b}", bc))
    rendered, global_cands, _ = _render_blocks(blocks)
    core = _render_core_so_far(core_order, core_source, cand_by_id)
    view = _build_fit_view(blocks, core_order, core_source, cand_by_id)
    return view, core, rendered, len(global_cands)


def _prompt(core: str, evidence: str) -> str:
    return _DECIDER_PROMPT.format(
        question="q",
        core_so_far=core or "(none)",
        evidence=evidence or "(no evidence retrieved this round)",
        max_sub=3,
    )


def _reply(core: list[int], nxt: list[str] | None = None) -> str:
    return json.dumps({"core": core, "next_queries": nxt or []})


def test_the_fake_endpoint_is_what_the_decider_detects() -> None:
    ep = _Endpoint([""])
    assert isinstance(ep, RawChatClient) and isinstance(ep, PromptTokenCounter)


# ── salvage: the parser ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("reply", "want"),
    [
        ('{"core": [0],', [0]),
        (
            '{"core": [0, 21, 2, 26, 4, 22, 1, 24, 8, 25, 12, 31, 14, 17],',
            [0, 21, 2, 26, 4, 22, 1, 24, 8, 25, 12, 31, 14, 17],
        ),
        ('{"core": [],', []),
        ('{"core": [0, 21, 2', [0, 21]),  # "2" may be the first digit of "26"
        ('{"core": ["3", 4],', ["3", 4]),
        ('Sure. {"core": [5, 6], "next_queries": ["a", "b', [5, 6]),
        ('{"core": [1, 2], "next_queries": [', [1, 2]),
        ('{"core":[7,8]}', [7, 8]),
    ],
)
def test_salvage_keeps_the_complete_indices(reply: str, want: list[object]) -> None:
    assert _salvage_core(reply) == want


@pytest.mark.parametrize(
    "reply", ["", "{", '{"co', '{"core"', '{"core": [', '{"core": [0', "no json"]
)
def test_salvage_refuses_when_no_index_is_complete(reply: str) -> None:
    assert _salvage_core(reply) is None


def test_a_reply_cut_at_any_point_never_yields_a_partial_index() -> None:
    """Every prefix of a valid reply salvages a prefix of its core, or nothing."""
    full = '{"core": [0, 12, 3, 45], "next_queries": ["x y", "z"]}'
    want = [0, 12, 3, 45]
    for cut in range(len(full) + 1):
        got = _salvage_core(full[:cut])
        if got is not None:
            assert got == want[: len(got)], (cut, full[:cut], got)


# ── salvage: the decider ────────────────────────────────────────────────────


async def test_salvage_off_a_truncated_reply_still_falls_back() -> None:
    ep = _Endpoint(['{"core": [0, 2],'])
    d = await _decider(ep)("q", "(none)", "ev", 5, 0)
    assert d.failed and d.core == [0, 1, 2] and d.salvaged is None
    assert len(ep.prompts) == 4  # 1 + retries, unchanged


async def test_salvage_on_keeps_the_emitted_core_and_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _on(monkeypatch, salvage=True)
    ep = _Endpoint(['{"core": [0, 2],'])
    d = await _decider(ep)("q", "(none)", "ev", 5, 0)
    assert not d.failed
    assert (d.core, d.queries, d.stop) == ([0, 2], [], True)
    assert d.salvaged == {"attempt": 0, "n_indices": 2}
    assert d.raw == '{"core": [0, 2],'
    assert len(ep.prompts) == 4  # the retries still run first


async def test_salvage_waits_for_the_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """A retry that parses wins over the salvage: only a failed round changes."""
    _on(monkeypatch, salvage=True)
    ep = _Endpoint(['{"core": [0, 2],', _reply([3], ["next"])])
    d = await _decider(ep)("q", "(none)", "ev", 5, 0)
    assert (d.core, d.queries, d.salvaged) == ([3], ["next"], None)
    assert len(ep.prompts) == 2


async def test_salvage_drops_indices_out_of_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _on(monkeypatch, salvage=True)
    d = await _decider(_Endpoint(['{"core": [0, 99, "1"],']))("q", "(none)", "e", 5, 0)
    assert d.core == [0, 1]


async def test_salvage_cannot_rescue_a_reply_without_a_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _on(monkeypatch, salvage=True)
    d = await _decider(_Endpoint(['{"next_queries": ["a"']))("q", "(none)", "e", 5, 0)
    assert d.failed and d.salvaged is None


@pytest.mark.parametrize(
    "reply",
    [
        _reply([0, 3], ["a", "b"]),
        _reply([], []),
        'Here: {"core": [1], "next_queries": []} done',
        '```json\n{"core": [2, 4], "next_queries": ["x"]}\n```',
    ],
)
async def test_valid_replies_are_unchanged_with_both_switches_on(
    monkeypatch: pytest.MonkeyPatch, reply: str
) -> None:
    off_ep = _Endpoint([reply])
    off = await _decider(off_ep)("q", "(none)", "ev", 5, 0)
    _on(monkeypatch, fit=True, salvage=True)
    view, core, ev, n = _view(["kept"], [["a", "b", "c", "d", "e"]])
    on_ep = _Endpoint([reply])
    on = await _decider(on_ep)("q", core, ev, n, 0, fit_view=view)
    assert (on.core, on.queries, on.stop, on.failed) == (
        off.core,
        off.queries,
        off.stop,
        off.failed,
    )
    assert (on.fit, on.salvaged) == (None, None)
    assert on_ep.params == off_ep.params
    assert on_ep.counted == []  # nothing was counted: nothing was rejected


# ── fit: detection and rendering ────────────────────────────────────────────


def test_the_overflow_is_recognised_through_the_wrapping_error() -> None:
    assert _is_context_overflow(_overflow_error(65536))
    assert _is_context_overflow(RuntimeError("code: context_length_exceeded"))
    assert not _is_context_overflow(TimeoutError("Request timed out."))
    assert not _is_context_overflow(ValueError("no JSON object in decider reply"))


def test_the_view_re_renders_exactly_what_the_loop_renders() -> None:
    long = "x" * 900 + "\nsecond line"
    cands = [
        _cand(i, long if i % 2 else f"short {i}", picture=i == 5) for i in range(7)
    ]
    blocks = [("original question", cands[:3]), ("empty", []), ("b2", cands[3:])]
    core_order = [cands[1].id, cands[4].id, "alice__missing"]
    core_source = {cands[1].id: "original question"}
    cand_by_id = {c.id: c for c in cands}
    view = _build_fit_view(blocks, core_order, core_source, cand_by_id)
    assert _render_fit_view(view) == (
        _render_core_so_far(core_order, core_source, cand_by_id),
        _render_blocks(blocks)[0],
    )
    # the picture mark is never cut, only the text before it
    _, ev = _render_fit_view(view, cand_cap=10)
    assert f"  5: subject 5 - {'x' * 10}{lmr._FIT_MARK}{_PICTURE_MARK}" in ev
    assert _render_fit_view(_build_fit_view([], [], {}, {})) == ("(none)", "")


def test_waterfill_cuts_only_the_longest_texts_to_one_length() -> None:
    lengths = [100, 1000, 3000, 50]
    assert _waterfill(lengths, 0, 10) == (None, 0)
    cap, removed = _waterfill(lengths, 1500, 10)
    assert removed >= 1500
    assert cap is not None and cap >= 1000  # the 1000-char text survives whole
    assert _waterfill([5, 6], 100, 10) == (None, 0)


def test_plan_cuts_core_first_then_candidates_then_drops_the_tail() -> None:
    view, *_ = _view(["c" * 2000] * 3, [["a" * 1000] * 4])
    core_cap, cand_cap, n_keep = _plan_fit(view, 1000)
    assert core_cap is not None and cand_cap is None and n_keep == 4
    core_cap, cand_cap, n_keep = _plan_fit(view, 3 * 1800 + 500)
    assert core_cap == lmr._FIT_CORE_FLOOR and cand_cap is not None and n_keep == 4
    assert _plan_fit(view, 10**6) == (lmr._FIT_CORE_FLOOR, lmr._FIT_CAND_FLOOR, 1)


# ── fit: the decider ────────────────────────────────────────────────────────


def _boundary_case(over: int):
    """A round whose prompt is ``over`` tokens past a window of the fake endpoint."""
    view, core, ev, n = _view(
        ["kept item " + "k" * 3000] * 6, [["cand " + "c" * 800] * 10, ["d" * 50] * 5]
    )
    return view, core, ev, n, _tokens(_prompt(core, ev)) - over


@pytest.mark.parametrize("over", [0, -1, -500])
async def test_a_prompt_within_the_window_is_never_touched(
    monkeypatch: pytest.MonkeyPatch, over: int
) -> None:
    """Exactly at the window (and under it) the endpoint accepts: nothing is fitted."""
    _on(monkeypatch, fit=True)
    view, core, ev, n, window = _boundary_case(over)
    ep = _Endpoint([_reply([0, 3])], window=window)
    d = await _decider(ep)("q", core, ev, n, 0, fit_view=view)
    assert ep.prompts == [_prompt(core, ev)]
    assert d.fit is None and ep.counted == [] and d.core == [0, 3]


async def test_one_token_over_the_window_is_fitted_under_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _on(monkeypatch, fit=True)
    monkeypatch.setenv("EVEROS_DECIDER__MAX_TOKENS", "64")
    view, core, ev, n, window = _boundary_case(1)
    ep = _Endpoint([_reply([0, 3], ["more"])], window=window)
    d = await _decider(ep)("q", core, ev, n, 0, fit_view=view)
    assert len(ep.prompts) == 2  # the rejected prompt, then the fitted one
    assert ep.prompts[0] == _prompt(core, ev)
    assert _tokens(ep.prompts[1]) <= window - 64
    assert (d.core, d.queries, d.failed) == ([0, 3], ["more"], False)
    fit = d.fit
    assert fit is not None and fit["fitted"] is True
    assert fit["tokens_before"] == window + 1 and fit["budget"] == window - 64
    assert fit["core_cap"] is not None and fit["cand_cap"] is None  # core first
    assert fit["n_shown"] == n and fit["dropped_indices"] == []
    assert lmr._FIT_MARK in ep.prompts[1]


async def test_the_fit_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    _on(monkeypatch, fit=True)
    view, core, ev, n, window = _boundary_case(4000)
    sent = []
    for _ in range(2):
        ep = _Endpoint([_reply([1])], window=window)
        await _decider(ep)("q", core, ev, n, 0, fit_view=view)
        sent.append(ep.prompts[-1])
    assert sent[0] == sent[1] and _tokens(sent[0]) <= window - 512


async def test_dropped_candidates_cannot_be_chosen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When even the floors do not fit, the last candidates go, and so do their
    indices: an index the sent prompt never showed is not the decider's choice."""
    _on(monkeypatch, fit=True)
    monkeypatch.setenv("EVEROS_DECIDER__MAX_TOKENS", "8")
    view, core, ev, n = _view([], [["x" * 600] * 40])
    window = _tokens(_prompt(core, ev)) // 2
    ep = _Endpoint([_reply([0, 1, 39])], window=window)
    d = await _decider(ep)("q", core, ev, n, 0, fit_view=view)
    fit = d.fit
    assert fit is not None and fit["fitted"] is True
    assert fit["cand_cap"] == lmr._FIT_CAND_FLOOR
    shown = fit["n_shown"]
    assert isinstance(shown, int) and 1 <= shown < n
    assert fit["dropped_indices"] == list(range(shown, n))
    assert d.core == [0, 1]  # 39 was dropped from the prompt
    assert f"  {shown}: " not in ep.prompts[-1]


async def test_fit_off_an_overflow_still_falls_back() -> None:
    _, core, ev, n, window = _boundary_case(1)
    ep = _Endpoint([_reply([0])], window=window)
    d = await _decider(ep)("q", core, ev, n, 0)
    assert d.failed and d.fit is None and len(ep.prompts) == 4 and ep.counted == []


async def test_no_token_count_means_no_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """An endpoint that cannot count leaves the round exactly as it was."""
    _on(monkeypatch, fit=True)
    view, core, ev, n, window = _boundary_case(1)
    ep = _Endpoint([_reply([0])], window=window, counts=False)
    d = await _decider(ep)("q", core, ev, n, 0, fit_view=view)
    assert d.failed and len(ep.prompts) == 4
    assert d.fit is not None and d.fit["fitted"] is False


async def test_a_view_that_does_not_match_is_not_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _on(monkeypatch, fit=True)
    view, core, ev, n, window = _boundary_case(1)
    ep = _Endpoint([_reply([0])], window=window)
    d = await _decider(ep)("q", core, ev + " ", n, 0, fit_view=view)
    assert d.failed and d.fit is None and ep.counted == []


# ── the loop ────────────────────────────────────────────────────────────────


def _hit(i: int, text: str) -> SearchHit[Any]:
    stored = stored_episode(f"e{i}")
    stored = dataclasses.replace(
        stored,
        ref=dataclasses.replace(stored.ref, id=f"alice__e{i}"),
        item=stored.item.model_copy(update={"episode": text}),
    )
    return hit(stored, 1.0 / (i + 1))


class _Recaller:
    def __init__(self, seed: list[SearchHit[Any]]) -> None:
        self._seed = seed

    async def sparse_recall(self, q: str, where: Any, *, limit: int) -> RecallResult:
        return recalled(*self._seed, source="keyword")

    async def dense_recall(
        self, v: Sequence[float], where: Any, *, limit: int
    ) -> RecallResult:
        return recalled(*self._seed)


class _Facts:
    async def facts_for_episodes(self, *a: Any, **k: Any) -> dict[str, list[Any]]:
        return {}


async def _embed(_q: str) -> list[float]:
    return [0.1, 0.2]


async def _search(llm: Any, seed: list[SearchHit[Any]], *, decider: Any = None):
    return await search_episodes_llm_multiround(
        "q",
        owner_id="alice_0",
        where="owner_id = 'alice'",
        episode_recaller=_Recaller(seed),
        atomic_fact_recaller=_Facts(),
        embed_query_fn=_embed,
        llm=llm,
        top_k=5,
        decider=decider,
        tuning_source=_LIVE.tuning,
        trace_path_source=_LIVE.trace_path,
    )


def _records(path: Any) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


async def test_both_switches_on_change_nothing_on_a_clean_search(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    seed = [_hit(i, f"text {i}") for i in range(6)]
    replies = [_reply([0, 2], ["follow up"]), _reply([1], [])]
    runs = []
    for on in (False, True):
        if on:
            _on(monkeypatch, fit=True, salvage=True)
        trace = tmp_path / f"trace_{on}.jsonl"
        monkeypatch.setenv("EVEROS_LLMMR_TRACE_DUMP", str(trace))
        ep = _Endpoint(replies)
        eps = await _search(ep, seed)
        recs = [
            {k: v for k, v in r.items() if k != "timing_s"} for r in _records(trace)
        ]
        runs.append(([e.id for e in eps], ep.prompts, ep.params, recs, ep.counted))
    assert runs[0] == runs[1]


async def test_the_trace_records_a_fit_and_a_salvage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _on(monkeypatch, fit=True, salvage=True)
    monkeypatch.setenv("EVEROS_DECIDER__MAX_TOKENS", "16")
    seed = [_hit(i, "t" * 4000) for i in range(6)]
    probe = _Endpoint([""])
    await _search(probe, seed)
    window = _tokens(probe.prompts[0]) - 100
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("EVEROS_LLMMR_TRACE_DUMP", str(trace))
    ep = _Endpoint(['{"core": [4, 1],'], window=window)
    await _search(ep, seed)
    (rnd, final) = _records(trace)
    assert rnd["decider_fit"]["fitted"] is True
    assert rnd["decider_salvaged"] == {"attempt": 0, "n_indices": 2}
    assert "decider_failed" not in rnd and rnd["core_indices"] == [4, 1]
    assert {x["id"] for x in final["injected"] if x["is_core"]} == {
        "alice__e1",
        "alice__e4",
    }


async def test_an_injected_decider_never_sees_the_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Phase-2 policy keeps the five-argument contract with the switch on."""
    _on(monkeypatch, fit=True)
    calls: list[tuple[Any, ...]] = []

    async def policy(*args: Any) -> lmr.RoundDecision:
        calls.append(args)
        return lmr.RoundDecision(stop=True, queries=[], core=[0])

    await _search(_Endpoint([""]), [_hit(i, "x") for i in range(3)], decider=policy)
    assert len(calls) == 1 and len(calls[0]) == 5


# ── the provider's count ────────────────────────────────────────────────────


async def test_the_provider_counts_through_tokenize_with_the_template_kwargs() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"count": 70012, "max_model_len": 65536})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = OpenAIProvider(
            model="policy",
            api_key="EMPTY",
            base_url="http://127.0.0.1:11320/v1",
            extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            http_client=http,
        )
        got = await provider.acount_prompt_tokens(
            [LLMMessage(role="user", content="hello")]
        )
    assert got == (70012, 65536)
    (req,) = seen
    assert str(req.url) == "http://127.0.0.1:11320/tokenize"
    body = json.loads(req.content)
    assert body["model"] == "policy" and body["add_generation_prompt"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"] == [{"role": "user", "content": "hello"}]


async def test_an_endpoint_without_tokenize_counts_nothing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Not Found"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = OpenAIProvider(
            model="m",
            api_key="k",
            base_url="https://example.invalid/v1",
            http_client=http,
        )
        assert (
            await provider.acount_prompt_tokens([LLMMessage(role="user", content="x")])
            is None
        )
    assert isinstance(provider, PromptTokenCounter)
