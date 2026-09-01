"""Unit tests for ``memory.search.llm_multiround`` (per-sub-query RRF blocks).

White-box: stubs the episode recaller, the query embedder, and the decider LLM
so the block loop runs without LanceDB or a real LLM. Covers the parse helpers,
the :class:`LLMRoundDecider` (JSON -> decision, retries, graceful stop), block
rendering + global indexing, core-so-far rendering, the search loop (early stop
/ sub-query expansion / max-rounds / empty seed), core-first + guarantee-then-fill
assembly, and per-round + final-injection trace completeness.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Sequence
from typing import Any, ClassVar

import pytest
from everalgo.llm.types import ChatMessage, ChatResponse
from everalgo.types import Candidate

from everos.memory.search import llm_multiround
from everos.memory.search.llm_multiround import (
    LLMRoundDecider,
    RoundDecision,
    _balanced_objects,
    _coerce_core_indices,
    _dataset_from_owner,
    _parse_decision,
    _render_blocks,
    _render_core_so_far,
    search_episodes_llm_multiround,
)

# ── Stubs ────────────────────────────────────────────────────────────────


def _ts() -> _dt.datetime:
    return _dt.datetime(2026, 1, 1, tzinfo=_dt.UTC)


def _ep(entry_id: str, score: float, *, session: str = "sess_a") -> Candidate:
    """Episode candidate keyed by LanceDB id ``<owner>__<entry_id>``."""
    return Candidate(
        id=f"alice__{entry_id}",
        score=score,
        source="vector",
        metadata={
            "entry_id": entry_id,
            "owner_id": "alice",
            "owner_type": "user",
            "session_id": session,
            "timestamp": _ts(),
            "sender_ids": ["alice"],
            "subject": f"subject {entry_id}",
            "summary": f"summary {entry_id}",
            "episode": f"body {entry_id}",
        },
    )


class _Recaller:
    """Episode recaller. ``per_query`` maps a sparse query -> its own seed, so a
    later sub-query can surface different candidates; otherwise ``seed`` is used.
    Dense recall returns ``seed`` (drive dense off by embedding to ``[]``)."""

    text_field: ClassVar[str] = "episode"

    def __init__(
        self,
        *,
        seed: list[Candidate],
        per_query: dict[str, list[Candidate]] | None = None,
    ) -> None:
        self._seed = seed
        self._per_query = per_query or {}
        self.sparse_queries: list[str] = []
        self.dense_vectors: list[list[float]] = []

    async def sparse_recall(
        self, query: str, where: str, *, limit: int
    ) -> list[Candidate]:
        self.sparse_queries.append(query)
        return list(self._per_query.get(query, self._seed))

    async def dense_recall(
        self, vector: Sequence[float], where: str, *, limit: int
    ) -> list[Candidate]:
        self.dense_vectors.append(list(vector))
        return list(self._seed)


class _FactRecaller:
    async def facts_for_episodes(self, *a: Any, **k: Any) -> dict[str, list[Any]]:
        return {}


class _ScriptLLM:
    """Decider LLM: replays ``replies[i]`` per ``.chat`` call (last repeats).

    ``error_first`` raises that many times before the first real reply (retry
    test). Each reply is the raw JSON string the decider parses.
    """

    def __init__(self, replies: list[str], *, error_first: int = 0) -> None:
        self._replies = replies
        self._error_first = error_first
        self.calls: list[list[ChatMessage]] = []
        self.kwargs: list[dict[str, Any]] = []
        """Every call's keyword arguments. Recorded because discarding them via
        ``**_`` is how ``max_tokens`` and ``extra`` could be added to
        ``DeciderSettings``, documented, and never sent, with the whole suite green:
        a stub that throws the request away cannot fail when the request is wrong."""

    async def chat(self, messages: list[ChatMessage], **kw: Any) -> ChatResponse:
        self.calls.append(messages)
        self.kwargs.append(dict(kw))
        if self._error_first > 0:
            self._error_first -= 1
            raise RuntimeError("transient decider failure")
        idx = min(len(self.calls) - 1, len(self._replies) - 1)
        return ChatResponse(content=self._replies[idx], model="stub")


async def _embed(_q: str) -> list[float]:
    return [0.1, 0.2, 0.3, 0.4]


async def _embed_empty(_q: str) -> list[float]:
    return []  # dense recall is skipped when the query vector is falsy


_WHERE = "owner_id = 'alice' AND owner_type = 'user'"


def _reply(indices: list[int], nxt: list[str]) -> str:
    return json.dumps({"core": indices, "next_queries": nxt})


def _run(
    recaller: _Recaller,
    *,
    llm: _ScriptLLM,
    embed: Any = _embed,
    top_k: int = 10,
    decider: Any = None,
) -> Any:
    return search_episodes_llm_multiround(
        "q",
        owner_id="alice",
        where=_WHERE,
        episode_recaller=recaller,
        atomic_fact_recaller=_FactRecaller(),
        embed_query_fn=embed,
        llm=llm,
        top_k=top_k,
        decider=decider,
    )


# ── parse helpers ──────────────────────────────────────────────────────────


def test_parse_decision_tolerates_prose() -> None:
    obj = _parse_decision('sure! {"core": [1], "next_queries": []} done')
    assert obj == {"core": [1], "next_queries": []}


def test_parse_decision_raises_without_json() -> None:
    with pytest.raises(ValueError):
        _parse_decision("no json here")


def test_balanced_objects_outermost_first() -> None:
    assert _balanced_objects('{"a": {"b": 1}} tail {"c": 2}') == [
        '{"a": {"b": 1}}',
        '{"c": 2}',
    ]


def test_coerce_core_indices_filters_and_dedups() -> None:
    assert _coerce_core_indices([0, "2", 2, 9, -1, True, "x"], n_evidence=3) == [0, 2]


def test_dataset_from_owner() -> None:
    assert _dataset_from_owner("longmemeval_42") == "longmemeval"
    assert _dataset_from_owner("no_index_here") == "no_index_here"


# ── decider ────────────────────────────────────────────────────────────────


async def test_decider_parses_reply() -> None:
    llm = _ScriptLLM([_reply([0, 2], ["gap one"])])
    d = await LLMRoundDecider(llm)("q?", "(none)", "[block: original question]", 3, 0)
    assert d.core == [0, 2]
    assert d.queries == ["gap one"]
    assert d.stop is False
    assert d.raw is not None


async def test_decider_stop_when_no_queries() -> None:
    d = await LLMRoundDecider(_ScriptLLM([_reply([0], [])]))("q", "(none)", "x", 1, 0)
    assert d.stop is True and d.queries == []


@pytest.fixture
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the retry backoff so retry-exhausting tests stay sub-second."""
    monkeypatch.setenv("EVEROS_LLMMR_DECIDER_BACKOFF_S", "0")


async def test_decider_falls_back_to_top_core_on_persistent_error(
    _no_backoff: None,
) -> None:
    """Exhausted retries must NOT silently yield an empty core.

    An empty core disables core-first injection for that question and is
    indistinguishable downstream from a decider that legitimately chose nothing,
    so the round falls back to the top-N candidates (evidence is in fused-score
    order) and flags itself via ``failed``.
    """
    llm = _ScriptLLM([_reply([0], [])], error_first=99)
    d = await LLMRoundDecider(llm)("q", "(none)", "x", 5, 0)
    assert d.stop is True and d.queries == []
    assert d.failed is True
    assert d.core == list(range(llm_multiround._tuning().fallback_core))


async def test_decider_fallback_core_clamped_to_candidates(_no_backoff: None) -> None:
    llm = _ScriptLLM([_reply([0], [])], error_first=99)
    d = await LLMRoundDecider(llm)("q", "(none)", "x", 1, 0)
    assert d.core == [0]  # never indexes past the candidates it was given


async def test_decider_retries_an_empty_completion() -> None:
    """A reasoning decider can return an empty completion (budget spent on
    reasoning tokens). That is a retryable failure, not a valid 'no decision'."""
    llm = _ScriptLLM(["", _reply([2], [])])
    d = await LLMRoundDecider(llm)("q", "(none)", "x", 3, 0)
    assert d.core == [2] and d.failed is False
    assert len(llm.calls) == 2  # empty reply consumed one attempt, then succeeded


async def test_decider_success_is_not_flagged_failed() -> None:
    d = await LLMRoundDecider(_ScriptLLM([_reply([0], [])]))("q", "(none)", "x", 1, 0)
    assert d.failed is False


async def test_decider_retries_then_succeeds(_no_backoff: None) -> None:
    # fails _DECIDER_RETRIES times, then the final attempt parses.
    llm = _ScriptLLM([_reply([1], [])], error_first=llm_multiround._tuning().retries)
    d = await LLMRoundDecider(llm)("q", "(none)", "x", 3, 0)
    assert d.core == [1]
    assert len(llm.calls) == llm_multiround._tuning().retries + 1


# ── rendering ──────────────────────────────────────────────────────────────


def test_render_blocks_global_index_across_blocks() -> None:
    blocks = [
        ("original question", [_ep("a", 0.9), _ep("b", 0.8)]),
        ("sub one", [_ep("c", 0.7)]),
    ]
    rendered, global_cands, index_meta = _render_blocks(blocks)
    assert "[block: original question]" in rendered
    assert "[block: sub one]" in rendered
    # continuous global index 0,1,2 across the two blocks
    assert [c.id for c in global_cands] == ["alice__a", "alice__b", "alice__c"]
    assert index_meta[2] == {
        "block_id": 1,
        "sub_query": "sub one",
        "rank_in_block": 0,
        "rrf_score": 0.7,
    }
    assert "  2: subject c - summary c" in rendered


def test_render_core_so_far_tags_source() -> None:
    assert _render_core_so_far([], {}, {}) == "(none)"
    cand = {"alice__a": _ep("a", 0.9)}
    out = _render_core_so_far(["alice__a"], {"alice__a": "sub one"}, cand)
    assert out == '  - [from "sub one"] subject a - summary a'


# ── search loop ────────────────────────────────────────────────────────────


async def test_early_stop_round0_one_llm_call() -> None:
    llm = _ScriptLLM([_reply([0], [])])  # stop immediately
    eps = await _run(_Recaller(seed=[_ep("a", 0.9), _ep("b", 0.5)]), llm=llm)
    assert len(llm.calls) == 1  # exactly one round
    assert eps[0].id == "alice__a"  # core pinned first


async def test_subquery_expansion_reembeds_and_two_rounds() -> None:
    rec = _Recaller(
        seed=[_ep("a", 0.9)],
        per_query={"gap two": [_ep("z", 0.6, session="s2")]},
    )
    llm = _ScriptLLM([_reply([0], ["gap two"]), _reply([0], [])])
    eps = await _run(rec, llm=llm, embed=_embed_empty)
    assert len(llm.calls) == 2  # round 0 expanded, round 1 stopped
    assert "gap two" in rec.sparse_queries  # the sub-query was re-recalled
    ids = {e.id for e in eps}
    assert {"alice__a", "alice__z"} <= ids  # both rounds' cores injected


async def test_max_rounds_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    # never stop (always a next query) + patience disabled -> exactly _MAX_ROUNDS.
    monkeypatch.setenv("EVEROS_LLMMR_PATIENCE", "99")
    llm = _ScriptLLM([_reply([0], ["again"])])  # last reply repeats forever
    await _run(_Recaller(seed=[_ep("a", 0.9)]), llm=llm, embed=_embed_empty)
    assert len(llm.calls) == llm_multiround._tuning().max_rounds


async def test_empty_seed_returns_empty_and_skips_decider() -> None:
    llm = _ScriptLLM([_reply([0], [])])
    eps = await _run(_Recaller(seed=[]), llm=llm, embed=_embed_empty)
    assert eps == []
    assert llm.calls == []  # decider never called on an empty seed


async def test_saturation_stop_after_no_new_core() -> None:
    # round 0 cores, round 1 adds no new core -> patience(=1) stops after round 1.
    llm = _ScriptLLM([_reply([0], ["again"]), _reply([], ["again"]), _reply([0], [])])
    rec = _Recaller(seed=[_ep("a", 0.9)])
    await _run(rec, llm=llm, embed=_embed_empty)
    assert len(llm.calls) == 2


# ── final injection: core-first + guarantee-then-fill ───────────────────────


async def test_core_first_pins_low_scored_core() -> None:
    # core the LOWEST-scored candidate; it must still land at rank 0.
    seed = [_ep("hi", 0.9), _ep("mid", 0.6), _ep("lo", 0.2)]
    llm = _ScriptLLM([_reply([2], [])])  # index 2 == "lo"
    eps = await _run(_Recaller(seed=seed), llm=llm, embed=_embed_empty)
    assert eps[0].id == "alice__lo"  # low-scored core pinned front
    assert {e.id for e in eps} == {"alice__hi", "alice__mid", "alice__lo"}


async def test_guarantee_gives_subquery_top1_a_slot() -> None:
    # top_k=2, round0 cores "a"; a later sub-query's rank-0 "z" must be guaranteed
    # a slot over the higher-scored non-core "b" from round 0.
    rec = _Recaller(
        seed=[_ep("a", 0.9), _ep("b", 0.8)],
        per_query={"gap": [_ep("z", 0.4, session="s2")]},
    )
    llm = _ScriptLLM([_reply([0], ["gap"]), _reply([], [])])
    eps = await _run(rec, llm=llm, embed=_embed_empty, top_k=2)
    ids = [e.id for e in eps]
    assert ids[0] == "alice__a"  # core first
    assert "alice__z" in ids  # sub-query top-1 guaranteed a slot despite low score


# ── trace completeness (schema C) ───────────────────────────────────────────

_PER_ROUND = {
    "dataset",
    "owner_id",
    "question_id",
    "question",
    "round_idx",
    "round_kind",
    "evidence",
    "core_indices",
    "core_session_ids",
    "core_source_subquery",
    "core_added",
    "core_carried_in",
    "stop",
    "next_queries",
    "recall",
    "timing_s",
    "decider",
}
_EVI = {
    "global_index",
    "block_id",
    "sub_query",
    "id",
    "session_id",
    "subject",
    "summary",
    "rrf_score",
    "rank_in_block",
}
_BLK = {
    "sub_query",
    "n_sparse",
    "n_dense",
    "topk_kept",
    "sparse",
    "dense",
    "rrf_ranked",
}
_RRK = {"id", "session_id", "rrf_score", "rank", "in_sparse", "in_dense"}
_FINAL = {
    "dataset",
    "owner_id",
    "question_id",
    "question",
    "round_idx",
    "injected",
    "assembly",
}
_INJ = {
    "rank",
    "id",
    "session_id",
    "timestamp",
    "is_core",
    "slot_source",
    "max_rrf_score",
    "max_rrf_subquery",
    "per_subquery_scores",
}
_ASM = {"n_core", "n_guaranteed", "n_filled", "top_k", "subqueries_seen"}


async def test_trace_records_every_schema_field(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("EVEROS_LLMMR_TRACE_DUMP", str(trace))
    rec = _Recaller(
        seed=[_ep("a", 0.9, session="s5"), _ep("b", 0.7, session="s8")],
        per_query={"gap": [_ep("z", 0.5, session="s2")]},
    )
    llm = _ScriptLLM([_reply([0], ["gap"]), _reply([0], [])])
    await search_episodes_llm_multiround(
        "the question?",
        owner_id="longmemeval_0",
        where=_WHERE,
        episode_recaller=rec,
        atomic_fact_recaller=_FactRecaller(),
        embed_query_fn=_embed_empty,
        llm=llm,
        top_k=20,
    )
    recs = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    per_round = [r for r in recs if r.get("round_idx") is not None]
    final = [r for r in recs if r.get("round_idx") is None]
    assert len(per_round) == 2 and len(final) == 1

    for r in per_round:
        assert set(r) >= _PER_ROUND, _PER_ROUND - set(r)
        assert set(r["decider"]) == {"tokens", "raw"}
        for e in r["evidence"]:
            assert set(e) >= _EVI, _EVI - set(e)
        for b in r["recall"]["blocks"]:
            assert set(b) >= _BLK, _BLK - set(b)
            for rr in b["rrf_ranked"]:
                assert set(rr) >= _RRK, _RRK - set(rr)

    assert per_round[0]["round_kind"] == "seed"
    assert per_round[1]["round_kind"] == "subquery"
    assert per_round[0]["core_source_subquery"] == ["original question"]

    fin = final[0]
    assert set(fin) >= _FINAL
    assert set(fin["assembly"]) >= _ASM
    for it in fin["injected"]:
        assert set(it) >= _INJ, _INJ - set(it)
        assert it["slot_source"] in {"core", "guarantee-top1", "maxscore-fill"}
    # "z" was surfaced only by the sub-query "gap": max_rrf_subquery reflects it.
    z = next(it for it in fin["injected"] if it["id"] == "alice__z")
    assert z["max_rrf_subquery"] == "gap"


async def test_trace_off_writes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVEROS_LLMMR_TRACE_DUMP", raising=False)
    # No dump env -> the loop must run and return without touching any file.
    llm = _ScriptLLM([_reply([0], [])])
    eps = await _run(_Recaller(seed=[_ep("a", 0.9)]), llm=llm, embed=_embed_empty)
    assert eps and eps[0].id == "alice__a"


def test_decider_text_defaults_to_summary_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default is the historical summary-only view so an in-flight comparison
    cannot silently change behaviour when the code is redeployed."""
    monkeypatch.setenv("EVEROS_LLMMR_DECIDER_FULL_TEXT", "0")
    meta = {"summary": "short 200-char preview", "episode": "the full body " * 50}
    assert llm_multiround._decider_text(meta) == "short 200-char preview"


def test_decider_text_full_mode_picks_the_longer_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stores disagree on which column holds the body: LoCoMo / DeepSeek / Gemini
    keep a 200-char prefix in ``summary`` and the body in ``episode``; the 27B and
    GPT LongMemEval stores keep the body in ``summary`` with ``episode`` empty.
    Picking the longer field is correct on both layouts without store detection.
    """
    monkeypatch.setenv("EVEROS_LLMMR_DECIDER_FULL_TEXT", "1")
    body = "the full body " * 50
    assert llm_multiround._decider_text({"summary": "prefix", "episode": body}) == body
    assert llm_multiround._decider_text({"summary": body, "episode": ""}) == body
    assert llm_multiround._decider_text({}) == ""


async def test_injection_never_exceeds_top_k_when_core_is_large() -> None:
    """Core-first must honour ``top_k``.

    The guarantee and fill stages already stop at ``top_k``; core-first used to be
    uncapped, so a decider that accumulated more core than the budget silently
    returned more episodes than the caller asked for (measured at 20.8% of
    SubtleMemory questions, up to 3.4x the budget), which breaks any same-budget
    comparison between retrieval methods.
    """
    seed = [_ep(f"e{i}", 1.0 - i / 100) for i in range(12)]
    # Round 0 selects 8 core, round 1 selects 8 more -> 16 core for a top_k of 5.
    llm = _ScriptLLM(
        [_reply(list(range(8)), ["more"]), _reply([*range(8, 12), 0, 1], [])]
    )
    out = await _run(_Recaller(seed=seed), llm=llm, top_k=5)
    assert len(out) == 5


async def test_core_overflow_env_restores_uncapped_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pre-fix behaviour stays reachable so a prior run can be reproduced."""
    monkeypatch.setenv("EVEROS_LLMMR_CORE_OVERFLOW", "1")
    seed = [_ep(f"e{i}", 1.0 - i / 100) for i in range(12)]
    llm = _ScriptLLM([_reply(list(range(8)), [])])
    out = await _run(_Recaller(seed=seed), llm=llm, top_k=5)
    assert len(out) == 8  # core kept beyond the budget, as before


async def test_injected_decider_drives_the_loop_instead_of_the_prompt_llm() -> None:
    """An injected decider must drive the loop, and the prompt-LLM must not run.

    The ``decider`` parameter and the RoundDecider protocol exist so a Phase-2 RL policy
    can BE the decider and reuse this loop verbatim as its environment. The body used to
    ignore the argument and always build LLMRoundDecider, which forced an RL environment
    to re-implement retrieval / block rendering / core accumulation / stop conditions --
    and re-implementing diverged: 25/25 sampled sub-queries retrieved a different
    candidate set (Jaccard median 0.538) because the HTTP search route dispatches to the
    hybrid hierarchy pipeline, not this file's per-sub-query rrf(sparse, dense)[:topk].
    """
    seen: list[dict[str, object]] = []

    async def policy(
        question: str,
        core_so_far: str,
        evidence: str,
        n_candidates: int,
        round_idx: int,
    ) -> RoundDecision:
        seen.append(
            {
                "round_idx": round_idx,
                "n_candidates": n_candidates,
                "core_so_far": core_so_far,
                "has_blocks": "[block:" in evidence,
            }
        )
        return RoundDecision(core=[0], queries=[], stop=True)

    llm = _ScriptLLM([_reply([0], [])])
    eps = await _run(_Recaller(seed=[_ep("a", 0.9)]), llm=llm, decider=policy)

    assert seen, "the injected decider was never called"
    assert llm.calls == [], "the prompt-LLM decider must not run when one is injected"
    assert seen[0]["round_idx"] == 0
    assert seen[0]["core_so_far"] == "(none)", "round 0 starts with an empty core"
    assert seen[0]["has_blocks"], "the decider must receive the blocked evidence view"
    assert eps, "the loop must still produce an injection"


# -- the settings that bound one round have to be in the outgoing request -----


async def test_the_decider_sends_its_max_tokens_and_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``[decider].max_tokens`` and ``.extra`` are per-call, not client defaults.

    ``extra`` is where a Qwen endpoint is told to stop thinking. Left on, an
    un-finetuned decider spends the whole deadline reasoning: 39.3% of rounds on
    Qwen3.5-0.8B returned nothing inside 60s, each costing ~729s of nested retries
    before the fixed top-3 fallback. The field was declared and documented while
    ``chat()`` was called with the messages and nothing else, and the suite could not
    notice because the stub discarded its keyword arguments.
    """
    monkeypatch.setenv("EVEROS_DECIDER__MAX_TOKENS", "256")
    monkeypatch.setenv(
        "EVEROS_DECIDER__EXTRA",
        '{"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}',
    )
    llm = _ScriptLLM([_reply([0], [])])
    await LLMRoundDecider(llm)("q", "(none)", "x", 1, 0)

    (kw,) = llm.kwargs
    assert kw["max_tokens"] == 256
    assert kw["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
