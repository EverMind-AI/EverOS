"""``[decider].dedupe_within_round``: each memory is listed once per round.

A round after the first shows the decider one block per sub-query, and a memory
that several sub-queries retrieved used to be listed in each of their blocks,
under an index of its own: 16.7% of the candidate slots of the recorded LoCoMo
searches (2026-11-13). Only the repeats inside one round are removed. A memory an
earlier round showed is shown again, in full: a decider that passed over an item
must still be able to choose it later.

What these pin:

- the copy kept is the one ranked highest in its block, a tie going to the
  earlier block; every block keeps its order and its header;
- the indices are numbered over what is listed, and a reply maps to the memories
  it was shown -- parsed, salvaged, the fallback core, a fitted prompt, an
  injected policy, the pictures attached;
- earlier rounds are not compared: what they showed is listed again;
- everything after the decider (core order, the guarantee, the fill, the
  ``top_k`` cap, the recall pools in the trace) is what it was, given the same
  memories chosen;
- switched off, every prompt is the one sent before the switch existed.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from everalgo.llm import LLMMessage
from everalgo.testing import FakeLLMClient
from everalgo.types import Candidate
from hypothesis import given
from hypothesis import strategies as st

from everos.component.llm.openai_provider import RawReply
from everos.config.search_runtime import SearchConfigSource
from everos.config.settings import DeciderSettings, load_settings
from everos.memory.ports import SearchHit
from everos.memory.search import llm_multiround as lmr
from everos.memory.search.llm_multiround import (
    _DECIDER_PROMPT,
    _PICTURE_MARK,
    RoundDecision,
    _build_fit_view,
    _decider_text,
    _in_round_repeats,
    _is_picture,
    _render_blocks,
    _render_fit_view,
    _round_pictures,
    search_episodes_llm_multiround,
)
from everos.memory.search.recall import RecallResult
from everos.memory.search.recall.image_memory import IMAGE_KIND

from ._records import hit, recalled, stored_episode

_WHERE = "owner_id = 'alice' AND owner_type = 'user'"

# Round 0 lists a, b, c and the decider keeps a. Round 1 asks two sub-queries: "b"
# is rank 1 under s1 and rank 0 under s2, "d" rank 0 under s1 and rank 2 under s2,
# and "a", already core, comes back under s1.
_ROUNDS: dict[str, list[str]] = {
    "q": ["a", "b", "c"],
    "s1": ["d", "b", "a", "e"],
    "s2": ["b", "f", "d"],
}
# What round 1 lists with the switch on: b in s2 (its best rank), d in s1.
_LISTED_ON = ["d", "a", "e", "b", "f"]
_LISTED_OFF = ["d", "b", "a", "e", "b", "f", "d"]


# ── stubs ────────────────────────────────────────────────────────────────────


def _ep(entry_id: str, text: str | None = None) -> SearchHit[Any]:
    stored = stored_episode(entry_id, session_id=f"s_{entry_id}")
    item = stored.item
    if text is not None:
        item = item.model_copy(update={"episode": text})
    keyed = dataclasses.replace(
        stored, ref=dataclasses.replace(stored.ref, id=f"alice__{entry_id}"), item=item
    )
    return hit(keyed, 1.0)


class _SparseOnly:
    """Recall by query text alone: a query's block is its list, in order.

    The query embeds to an empty vector, so the dense leg is skipped and the RRF
    score of the item at rank ``r`` is ``1 / (rrf_k + r + 1)``.
    """

    def __init__(
        self, per_query: Mapping[str, Sequence[str]], texts: Mapping[str, str]
    ) -> None:
        self._per_query = per_query
        self._texts = texts

    async def sparse_recall(
        self, query: str, where: str, *, limit: int
    ) -> RecallResult[Any]:
        return recalled(
            *[_ep(e, self._texts.get(e)) for e in self._per_query.get(query, [])]
        )

    async def dense_recall(
        self, vector: Sequence[float], where: str, *, limit: int
    ) -> RecallResult[Any]:
        return recalled()


async def _no_vector(_q: str) -> list[float]:
    return []


_LINE = re.compile(r"^  (\d+): subject (\S+) - ", re.MULTILINE)


def _section(prompt: str) -> str:
    """The round's own CANDIDATES THIS ROUND section (the template's examples come
    first)."""
    tail = prompt.rsplit("CANDIDATES THIS ROUND:\n", 1)[1]
    return tail.split("\n\nReply with ONLY", 1)[0]


def _listed(prompt: str) -> list[tuple[int, str]]:
    return [(int(gi), e) for gi, e in _LINE.findall(_section(prompt))]


def _reply(core: Sequence[int], nxt: Sequence[str] = ()) -> str:
    return json.dumps({"core": list(core), "next_queries": list(nxt)})


type Scripted = str | tuple[Sequence[str], Sequence[str]]


class _Decider:
    """The decider's model: records each prompt and replies from a script.

    A scripted reply is fixed text, or ``(memories, next_queries)``: the memories
    are found by subject in the prompt sent (their first listing), so the same
    memories are chosen whatever the numbering. The last reply repeats.
    """

    def __init__(self, *replies: Scripted) -> None:
        self.prompts: list[str] = []
        self._replies = replies

    def __call__(self, messages: Any, **_kwargs: Any) -> str:
        prompt = messages[0].content
        self.prompts.append(prompt)
        reply = self._replies[min(len(self.prompts), len(self._replies)) - 1]
        if isinstance(reply, str):
            return reply
        want, nxt = reply
        first: dict[str, int] = {}
        for gi, entry in _listed(prompt):
            first.setdefault(entry, gi)
        return _reply([first[w] for w in want if w in first], nxt)


async def _search(
    per_query: Mapping[str, Sequence[str]],
    llm: Any,
    *,
    top_k: int = 10,
    trace: Path | None = None,
    texts: Mapping[str, str] | None = None,
    policy: Any = None,
    **tuning: Any,
) -> list[str]:
    settings = DeciderSettings(**{"retry_backoff_seconds": 0, **tuning})
    found = await search_episodes_llm_multiround(
        "q",
        owner_id="alice",
        where=_WHERE,
        episode_recaller=_SparseOnly(per_query, texts or {}),
        embed_query_fn=_no_vector,
        llm=llm if not isinstance(llm, _Decider) else FakeLLMClient(handler=llm),
        top_k=top_k,
        decider=policy,
        tuning_source=lambda: settings,
        trace_path_source=lambda: str(trace) if trace else None,
    )
    return [e.id.removeprefix("alice__") for e in found]


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _line(gi: int, entry: str) -> str:
    return f"  {gi}: subject {entry} - summary {entry}"


# ── the setting ──────────────────────────────────────────────────────────────


def test_the_switch_is_on_by_default() -> None:
    assert DeciderSettings().dedupe_within_round is True
    assert load_settings().decider.dedupe_within_round is True
    assert SearchConfigSource().tuning().dedupe_within_round is True


def test_the_switch_reads_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVEROS_DECIDER__DEDUPE_WITHIN_ROUND", "false")
    assert load_settings().decider.dedupe_within_round is False
    assert SearchConfigSource().tuning().dedupe_within_round is False


# ── which copy is kept ───────────────────────────────────────────────────────


def _cand(entry: str, text: str | None = None) -> Candidate:
    (cand,) = recalled(_ep(entry, text)).candidates
    return cand


def _blocks(*lists: Sequence[str]) -> list[tuple[str, list[Candidate]]]:
    return [(f"s{b + 1}", [_cand(e) for e in ids]) for b, ids in enumerate(lists)]


def test_the_copy_ranked_highest_in_its_block_is_kept() -> None:
    # x: rank 2 in s1, rank 0 in s2 -> kept in s2, although s1 comes first.
    assert _in_round_repeats(_blocks(["a", "b", "x"], ["x", "c"])) == {(0, 2)}
    # y: rank 0 in s1, rank 1 in s2 -> kept in s1.
    assert _in_round_repeats(_blocks(["y", "a"], ["b", "y"])) == {(1, 1)}


def test_a_tie_goes_to_the_earlier_block() -> None:
    assert _in_round_repeats(_blocks(["a", "x"], ["b", "x"], ["c", "x"])) == {
        (1, 1),
        (2, 1),
    }


def test_a_round_without_repeats_leaves_nothing_out() -> None:
    assert _in_round_repeats(_blocks(["a", "b", "c"])) == frozenset()
    assert _in_round_repeats(_blocks(["a"], ["b"], [])) == frozenset()
    assert _in_round_repeats([]) == frozenset()


def test_the_listing_keeps_every_block_order_and_numbers_without_gaps() -> None:
    blocks = _blocks(["d", "b", "a", "e"], ["b", "f", "d"])
    rendered, cands, meta = _render_blocks(blocks, _in_round_repeats(blocks))
    assert rendered == "\n".join(
        [
            "[block: s1]",
            _line(0, "d"),
            _line(1, "a"),
            _line(2, "e"),
            "[block: s2]",
            _line(3, "b"),
            _line(4, "f"),
        ]
    )
    assert [c.id for c in cands] == [f"alice__{e}" for e in _LISTED_ON]
    # Block and rank are the recalled ones: the gaps say where a copy was left out.
    assert [(m["block_id"], m["rank_in_block"]) for m in meta] == [
        (0, 0),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 1),
    ]
    assert [m["rrf_score"] for m in meta] == [c.score for c in cands]


def test_a_block_whose_every_item_is_listed_elsewhere_keeps_its_header() -> None:
    blocks = _blocks(["b", "d", "e"], ["b", "d"])
    rendered, cands, _ = _render_blocks(blocks, _in_round_repeats(blocks))
    assert rendered.endswith("[block: s2]")
    assert len(cands) == 3


def test_nothing_omitted_renders_exactly_as_before() -> None:
    blocks = _blocks(["d", "b", "a", "e"], ["b", "f", "d"])
    assert _render_blocks(blocks, frozenset()) == _render_blocks(blocks)
    assert _render_blocks(blocks) == _render_blocks_0b8006f(blocks)


# ── the loop: what the decider is shown ──────────────────────────────────────


async def test_a_memory_two_blocks_retrieved_is_listed_once() -> None:
    llm = _Decider(_reply([0], ["s1", "s2"]), _reply([]))
    await _search(_ROUNDS, llm)
    assert [e for _, e in _listed(llm.prompts[1])] == _LISTED_ON
    assert [gi for gi, _ in _listed(llm.prompts[1])] == list(range(5))


async def test_switched_off_every_prompt_is_the_one_sent_before() -> None:
    llm = _Decider(_reply([0], ["s1", "s2"]), _reply([]))
    await _search(_ROUNDS, llm, dedupe_within_round=False)
    seed = "\n".join(
        ["[block: original question]", *(_line(i, e) for i, e in enumerate("abc"))]
    )
    round1 = "\n".join(
        [
            "[block: s1]",
            *(_line(i, e) for i, e in enumerate("dbae")),
            "[block: s2]",
            *(_line(i + 4, e) for i, e in enumerate("bfd")),
        ]
    )
    core = '  - [from "original question"] subject a - summary a'
    assert llm.prompts == [
        _DECIDER_PROMPT.format(
            question="q", core_so_far="(none)", evidence=seed, max_sub=3
        ),
        _DECIDER_PROMPT.format(
            question="q", core_so_far=core, evidence=round1, max_sub=3
        ),
    ]


async def test_a_round_without_repeats_sends_the_same_prompt_either_way() -> None:
    rounds = {"q": ["a", "b"], "s1": ["c", "a"], "s2": ["d"]}
    sent = []
    for on in (False, True):
        llm = _Decider(_reply([0], ["s1", "s2"]), _reply([1]))
        found = await _search(rounds, llm, dedupe_within_round=on)
        sent.append((llm.prompts, found))
    assert sent[0] == sent[1]


async def test_what_an_earlier_round_showed_is_listed_again() -> None:
    """Round 0 showed a and b; a is core. Round 1 lists both again (once each), and
    round 2 lists again what round 1 showed: no round is compared with another."""
    rounds = {**_ROUNDS, "s3": ["e", "d", "g"]}
    llm = _Decider(
        _reply([0], ["s1", "s2"]),
        (["b"], ["s3", "s1"]),
        _reply([]),
    )
    await _search(rounds, llm)
    round0, round1, round2 = (_listed(p) for p in llm.prompts)
    assert [e for _, e in round0] == ["a", "b", "c"]
    assert [e for _, e in round1] == _LISTED_ON
    # s3: e (rank 0, kept), d (rank 1; rank 0 under s1, kept there), g.
    assert [e for _, e in round2] == ["e", "g", "d", "b", "a"]
    # a is in CORE SO FAR as well: the core is never taken out of the listing.
    assert '[from "original question"] subject a - summary a' in llm.prompts[1]


# ── the loop: what a reply means ─────────────────────────────────────────────


async def test_a_reply_index_names_the_memory_listed_there(tmp_path: Path) -> None:
    trace = tmp_path / "t.jsonl"
    llm = _Decider(_reply([0], ["s1", "s2"]), _reply([3, 4]))
    found = await _search(_ROUNDS, llm, trace=trace)
    rnd = _records(trace)[1]
    assert [c["id"] for c in rnd["core_added"]] == ["alice__b", "alice__f"]
    assert [c["sub_query"] for c in rnd["core_added"]] == ["s2", "s2"]
    # Core first, by best RRF score: a and b tie at rank 0 and keep the decider's
    # order, f is rank 1 under s2.
    assert found[:3] == ["a", "b", "f"]


async def test_the_same_memories_chosen_give_the_same_result(tmp_path: Path) -> None:
    """Whatever the numbering, the result and the final trace record depend only on
    which memories the decider chose: under both core orders, and with a cap that
    keeps part of the core."""
    for order in ("score", "round"):
        for top_k in (2, 4, 10):
            runs = []
            for on in (False, True):
                trace = tmp_path / f"{order}_{top_k}_{on}.jsonl"
                llm = _Decider((["a"], ["s1", "s2"]), (["f", "d", "b"], []))
                found = await _search(
                    _ROUNDS,
                    llm,
                    top_k=top_k,
                    trace=trace,
                    core_order=order,
                    dedupe_within_round=on,
                )
                recs = _records(trace)
                runs.append((found, recs[-1], [r["recall"] for r in recs[:-1]]))
            assert runs[0] == runs[1], (order, top_k)
            assert len(runs[1][0]) == min(top_k, 6)


async def test_a_salvaged_core_names_the_memories_listed(tmp_path: Path) -> None:
    trace = tmp_path / "t.jsonl"
    llm = _Decider(_reply([0], ["s1", "s2"]), '{"core": [3, 1],')
    found = await _search(_ROUNDS, llm, trace=trace)
    rnd = _records(trace)[1]
    assert rnd["decider_salvaged"] == {"attempt": 0, "n_indices": 2}
    assert rnd["core_indices"] == [3, 1]
    # 3 is b; 1 is a, already core.
    assert [c["id"] for c in rnd["core_added"]] == ["alice__b"]
    assert set(found[:2]) == {"a", "b"}


async def test_the_fallback_core_is_the_first_memories_listed(tmp_path: Path) -> None:
    """Every attempt fails: the core is the first ``fallback_core`` of what the
    round listed -- d, a (already core), e -- never an index past it."""
    trace = tmp_path / "t.jsonl"
    llm = _Decider(_reply([0], ["s1", "s2"]), "no json here")
    await _search(_ROUNDS, llm, trace=trace)
    rnd = _records(trace)[1]
    assert rnd["decider_failed"] is True and rnd["core_indices"] == [0, 1, 2]
    assert [c["id"] for c in rnd["core_added"]] == ["alice__d", "alice__e"]
    # A round that lists fewer memories than the fallback size keeps them all.
    trace2 = tmp_path / "t2.jsonl"
    llm = _Decider(_reply([0], ["s1", "s2"]), "no json here")
    await _search(
        {"q": ["a"], "s1": ["b"], "s2": ["b"]}, llm, trace=trace2, fallback_core=3
    )
    rnd = _records(trace2)[1]
    assert rnd["core_indices"] == [0] and len(rnd["evidence"]) == 1


async def test_an_injected_policy_sees_the_listing_and_its_size() -> None:
    calls: list[tuple[Any, ...]] = []

    async def policy(*args: Any) -> RoundDecision:
        calls.append(args)
        if len(calls) == 1:
            return RoundDecision(stop=False, queries=["s1", "s2"], core=[0])
        return RoundDecision(stop=True, queries=[], core=[3])

    found = await _search(_ROUNDS, _Decider(_reply([])), policy=policy)
    _q, _core, evidence, n_candidates, round_idx = calls[1]
    assert (n_candidates, round_idx) == (5, 1)
    assert [e for _, e in _LINE.findall(evidence)] == _LISTED_ON
    assert set(found[:2]) == {"a", "b"}


# ── the loop: a fitted round ─────────────────────────────────────────────────


def _tokens(prompt: str) -> int:
    return len(prompt) // 4


class _OverflowError(Exception):
    pass


class _Window:
    """A decider endpoint with a context window, reached as ``achat_raw``, that
    counts prompts as /tokenize would (four characters a token)."""

    def __init__(self, replies: Sequence[str], window: int) -> None:
        self.replies = list(replies)
        self.window = window
        self.prompts: list[str] = []
        self.accepted = 0

    async def achat_raw(self, messages: list[LLMMessage], **params: Any) -> RawReply:
        prompt = messages[0].content
        assert isinstance(prompt, str)
        self.prompts.append(prompt)
        if _tokens(prompt) > self.window:
            raise _OverflowError(
                f"This model's maximum context length is {self.window}"
            )
        reply = self.replies[min(self.accepted, len(self.replies) - 1)]
        self.accepted += 1
        return RawReply(text=reply, finish_reason="stop", usage=None)

    async def achat(self, messages: list[LLMMessage], **params: Any) -> str:
        raise AssertionError("the decider must use achat_raw")

    async def acount_prompt_tokens(
        self, messages: list[LLMMessage], **extra: Any
    ) -> tuple[int, int] | None:
        prompt = messages[0].content
        assert isinstance(prompt, str)
        return _tokens(prompt), self.window


# Round 0 is short; round 1 carries three long memories, so only it overflows.
_LONG = {e: f"{e} " + "x" * 1500 for e in ("d", "e", "f")}
_FIT_REPLIES = [_reply([0], ["s1", "s2"]), _reply([0, 3, 4])]


async def _fit_run(window: int, trace: Path) -> tuple[list[str], _Window]:
    ep = _Window(_FIT_REPLIES, window)
    found = await _search(_ROUNDS, ep, trace=trace, texts=_LONG, max_tokens=8)
    return found, ep


async def test_a_fitted_round_keeps_the_listing_and_its_indices(
    tmp_path: Path,
) -> None:
    _, probe = await _fit_run(10**9, tmp_path / "probe.jsonl")
    trace = tmp_path / "t.jsonl"
    found, ep = await _fit_run(_tokens(probe.prompts[1]) - 50, trace)
    rnd = _records(trace)[1]
    fit = rnd["decider_fit"]
    # The view the fit re-renders is the de-duplicated one, or no fit is tried.
    assert fit["fitted"] is True and fit["n_candidates"] == 5
    assert fit["n_shown"] == 5 and fit["dropped_indices"] == []
    assert [e for _, e in _listed(ep.prompts[-1])] == _LISTED_ON
    assert [c["id"] for c in rnd["core_added"]] == ["alice__d", "alice__b", "alice__f"]
    assert {"d", "b", "f"} <= set(found)


def _round1_at_the_floor(n_keep: int) -> int:
    """Tokens of round 1 with every candidate text at the fit's floor and only the
    first ``n_keep`` listed memories kept."""
    blocks = [(q, [_cand(e, _LONG.get(e)) for e in _ROUNDS[q]]) for q in ("s1", "s2")]
    a = _cand("a")
    view = _build_fit_view(
        blocks,
        [a.id],
        {a.id: "original question"},
        {a.id: a},
        _in_round_repeats(blocks),
    )
    core, evidence = _render_fit_view(view, cand_cap=lmr._FIT_CAND_FLOOR, n_keep=n_keep)
    return _tokens(
        _DECIDER_PROMPT.format(
            question="q", core_so_far=core, evidence=evidence, max_sub=3
        )
    )


async def test_a_fit_drops_the_last_listed_memories(tmp_path: Path) -> None:
    trace = tmp_path / "t.jsonl"
    # Round 0 fits; round 1 fits only once its texts are at the floor and the last
    # two listed memories are dropped (max_tokens = 8 is the reply's room).
    window = _round1_at_the_floor(3) + 8
    assert window < _round1_at_the_floor(4) + 8
    _, ep = await _fit_run(window, trace)
    assert _tokens(ep.prompts[0]) <= window
    rnd = _records(trace)[1]
    fit = rnd["decider_fit"]
    assert fit["fitted"] is True and fit["n_candidates"] == 5
    shown = fit["n_shown"]
    assert 1 <= shown < 5 and fit["dropped_indices"] == list(range(shown, 5))
    assert [e for _, e in _listed(ep.prompts[-1])] == _LISTED_ON[:shown]
    # The reply named 0, 3, 4: only the indices the sent prompt showed count.
    want = [f"alice__{_LISTED_ON[i]}" for i in (0, 3, 4) if i < shown]
    assert [c["id"] for c in rnd["core_added"]] == want


# ── the trace ────────────────────────────────────────────────────────────────


async def test_the_trace_lists_what_was_shown_and_what_was_left_out(
    tmp_path: Path,
) -> None:
    traces = {}
    for on in (False, True):
        traces[on] = tmp_path / f"{on}.jsonl"
        llm = _Decider(_reply([0], ["s1", "s2"]), _reply([]))
        await _search(_ROUNDS, llm, trace=traces[on], dedupe_within_round=on)
    off, on = _records(traces[False]), _records(traces[True])
    assert [e["id"] for e in off[1]["evidence"]] == [f"alice__{e}" for e in _LISTED_OFF]
    assert [e["id"] for e in on[1]["evidence"]] == [f"alice__{e}" for e in _LISTED_ON]
    assert [e["global_index"] for e in on[1]["evidence"]] == list(range(5))
    assert [e["rank_in_block"] for e in on[1]["evidence"]] == [0, 2, 3, 0, 1]
    assert on[1]["decider_dedupe"] == {
        "left_out": [
            {"id": "alice__b", "block_id": 0, "rank_in_block": 1, "shown_as": 3},
            {"id": "alice__d", "block_id": 1, "rank_in_block": 2, "shown_as": 0},
        ]
    }
    # Present only on a round that left something out.
    assert "decider_dedupe" not in on[0]
    assert all("decider_dedupe" not in r for r in off)
    # The recall pools and the seed round are what they were.
    assert on[1]["recall"] == off[1]["recall"]
    strip = ("timing_s",)
    assert {k: v for k, v in on[0].items() if k not in strip} == {
        k: v for k, v in off[0].items() if k not in strip
    }


# ── pictures ─────────────────────────────────────────────────────────────────


class _Store:
    def __init__(self) -> None:
        self.read: list[str] = []

    async def read_derived(self, image_id: str, *, edge: int, quality: int) -> bytes:
        self.read.append(image_id)
        return b"thumb"


def _picture(entry_id: str, image_id: str) -> Candidate:
    return Candidate(
        id=f"alice__{entry_id}",
        score=0.5,
        source="vector",
        metadata={
            "kind": IMAGE_KIND,
            "image_id": image_id,
            "subject": "a latte",
            "summary": "a latte with a heart on it",
        },
    )


async def test_a_picture_two_blocks_retrieved_is_attached_once() -> None:
    """Listed once, a picture also takes one of the round's picture slots, not two."""
    one, two, three = "a" * 64, "b" * 64, "c" * 64
    blocks = [
        ("s1", [_picture("im_1", one), _picture("im_3", three)]),
        ("s2", [_picture("im_1", one), _picture("im_2", two)]),
    ]
    attached = {}
    for on in (False, True):
        omit = _in_round_repeats(blocks) if on else frozenset()
        _, cands, _ = _render_blocks(blocks, omit)
        store = _Store()
        picked = await _round_pictures(cands, store=store, k=3, edge=448, quality=85)
        attached[on] = ([p.label for p in picked], store.read)
    assert attached[False] == (["0", "1", "2"], [one, three, one])
    assert attached[True] == (["0", "1", "2"], [one, three, two])


# ── properties over random rounds ────────────────────────────────────────────


def _render_blocks_0b8006f(
    blocks: list[tuple[str, list[Candidate]]],
) -> tuple[str, list[Candidate], list[dict[str, object]]]:
    """``_render_blocks`` as it was before the switch (0b8006f), verbatim."""
    lines: list[str] = []
    global_cands: list[Candidate] = []
    index_meta: list[dict[str, object]] = []
    gi = 0
    for block_id, (sub_query, cands) in enumerate(blocks):
        lines.append(f"[block: {sub_query}]")
        for rank, c in enumerate(cands):
            subject = str(c.metadata.get("subject", ""))
            summary = _decider_text(c.metadata)
            if _is_picture(c):
                summary = f"{summary}{_PICTURE_MARK}"
            lines.append(f"  {gi}: {subject} - {summary}")
            global_cands.append(c)
            index_meta.append(
                {
                    "block_id": block_id,
                    "sub_query": sub_query,
                    "rank_in_block": rank,
                    "rrf_score": c.score,
                }
            )
            gi += 1
    return "\n".join(lines), global_cands, index_meta


_ROUND = st.lists(
    st.lists(st.sampled_from("abcdefgh"), unique=True, max_size=6),
    min_size=1,
    max_size=4,
)


@given(_ROUND)
def test_every_round_lists_each_memory_once_where_it_ranks_highest(
    lists: list[list[str]],
) -> None:
    blocks = _blocks(*lists)
    omit = _in_round_repeats(blocks)
    rendered, cands, meta = _render_blocks(blocks, omit)
    ids = [c.id for c in cands]
    assert len(ids) == len(set(ids))
    assert set(ids) == {c.id for _, b in blocks for c in b}
    for c, m in zip(cands, meta, strict=True):
        copies = [
            (r, b)
            for b, (_, bl) in enumerate(blocks)
            for r, x in enumerate(bl)
            if x.id == c.id
        ]
        assert (m["rank_in_block"], m["block_id"]) == min(copies)
    # Each block keeps its order; the index runs without gaps; every header stays.
    for b, (_, bl) in enumerate(blocks):
        kept = [c.id for c, m in zip(cands, meta, strict=True) if m["block_id"] == b]
        assert kept == [c.id for c in bl if c.id in kept]
    assert [int(gi) for gi in re.findall(r"^  (\d+): ", rendered, re.M)] == list(
        range(len(cands))
    )
    assert rendered.count("[block: ") == len(blocks)
    assert len(omit) == sum(len(bl) for _, bl in blocks) - len(cands)
    # Nothing omitted is exactly the renderer from before the switch.
    assert _render_blocks(blocks) == _render_blocks_0b8006f(blocks)


@given(_ROUND, st.lists(st.sampled_from("abcdefgh"), unique=True, max_size=3))
def test_the_fit_view_reproduces_the_listing(
    lists: list[list[str]], core: list[str]
) -> None:
    """The decider fits a prompt only through a view that re-renders it exactly."""
    blocks = _blocks(*lists)
    cand_by_id = {c.id: c for c in (_cand(e) for e in core)}
    core_order = list(cand_by_id)
    core_source = dict.fromkeys(core_order, "original question")
    for omit in (frozenset(), _in_round_repeats(blocks)):
        rendered, _, _ = _render_blocks(blocks, omit)
        view = _build_fit_view(blocks, core_order, core_source, cand_by_id, omit)
        core_text = lmr._render_core_so_far(core_order, core_source, cand_by_id)
        assert _render_fit_view(view) == (core_text, rendered)
