"""Episode LLM-guided multi-round retrieval (per-sub-query RRF blocks).

Genuinely *iterative* multi-round retrieval, unlike ``agentic`` (a fixed round1
+ single round2 fallback). Each round an injected **decider** looks at the
question + retrieved evidence and, in one shot:

  1. **selects** the *core* memories that carry information the answer needs,
  2. decides to **stop** (the core answers a self-contained question), else
  3. **expands** into one focused sub-query per still-missing aspect (multi-query
     gap coverage), grounded in the core carried forward.

Retrieval substrate — the design's defining choice. Every current sub-query is
recalled (BM25 + vector) and fused with RRF **independently**, so a facet-gold
ranked #1 for a single sub-query is never diluted by the other sub-queries'
consensus hits (the failure mode of merging every sub-query into one pool). The
decider sees one labelled block per sub-query (round 0 = a single
original-question block) with a single GLOBAL index, plus a CORE-SO-FAR section
tagging each kept item with the sub-query that surfaced it. There is **no
cross-encoder** anywhere — RRF is the only ranking (the CE was measured
net-negative: it scored gold below RRF order and buried much of it past rank 20).

The accumulated core also shapes the output: the final top-k is assembled
**core-first** (a hard guarantee), then each sub-query's top non-core candidate
is guaranteed a slot, then the rest is filled by MAX RRF score across sub-queries
(max, not sum — a specialist that is #1 for one sub-query keeps that score).
Evidence selection therefore genuinely shapes what is injected — the lever a
Phase-2 RL policy learns to control.

The decider is a *pluggable hook* — the whole point of the file:

* Phase 1 (default here): :class:`LLMRoundDecider` prompts an off-the-shelf
  LLM (no training).
* Phase 2 (RL): swap in a trained small policy with the same
  :class:`RoundDecider` interface. **This loop then IS the RL environment** —
  only the decider changes; the retrieval substrate is identical.

Retrieval explainability: setting ``EVEROS_LLMMR_TRACE_DUMP=<jsonl>`` appends one
per-round record (blocked evidence shown -> core selected -> stop / next
queries) plus a final-injection record (the assembled top-k with per-slot
provenance), so a real Phase-1 run yields the GRPO cold-start (behavior-cloning)
corpus. Unset by default — the dump adds nothing to the production path.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.rank.fusion import rrf
from everalgo.types import Candidate

from everos.config.settings import DeciderSettings, load_settings
from everos.core.context import mark_degraded
from everos.core.observability.logging import get_logger

from .dto import SearchEpisodeItem
from .shaper import shape_episode_from_candidate

logger = get_logger(__name__)

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

    from everos.component.rerank import RerankProvider
    from everos.memory.search.recall.atomic_fact import AtomicFactRecaller
    from everos.memory.search.recall.episode import EpisodeRecaller


# ── Retrieval hyperparameters (per-sub-query RRF blocks) ──────────────────────
# Each round fuses every current sub-query's BM25+vector recall with RRF
# INDEPENDENTLY and shows the decider one labelled block per sub-query (round 0 =
# a single original-question block). There is NO cross-encoder anywhere — RRF is
# the only ranking (the CE was measured net-negative: it scored gold below RRF
# order and buried much of it past rank 20). The final injection is core-first +
# each sub-query's top non-core guaranteed + max-RRF-score fill (MAX across
# sub-queries, not sum — max keeps a specialist that is #1 for one sub-query).
_SEED_CANDIDATES: int = 50
"""Sparse / dense episode recall pool size per sub-query per round."""
# ── Loop tuning ──────────────────────────────────────────────────────────────
# Resolved per search from ``[decider]``, not frozen at import.
#
# These were twelve module-level ``int(os.getenv("EVEROS_LLMMR_...", ...))`` reads. Two
# problems with that, and the second is the one that matters: nothing in the
# configuration named them, so an operator had no way to discover they existed; and a
# value read at import time cannot respond to a config reload, which is why the test
# suite -- whose conftest resets the settings cache per test -- could not exercise them
# without monkeypatching module attributes.
#
# The legacy env names still take precedence when set. Deployments and launch scripts
# that export them keep working unchanged, and an in-flight comparison cannot shift
# because this landed.
_LEGACY_ENV: dict[str, str] = {
    "max_rounds": "EVEROS_LLMMR_MAX_ROUNDS",
    "seed_topk": "EVEROS_LLMMR_SEED_TOPK",
    "subq_topk": "EVEROS_LLMMR_SUBQ_TOPK",
    "max_subqueries": "EVEROS_LLMMR_MAX_SUBQUERIES",
    "rrf_k": "EVEROS_LLMMR_RRF_K",
    "no_new_core_patience": "EVEROS_LLMMR_PATIENCE",
    "per_subquery_guarantee": "EVEROS_LLMMR_GUARANTEE",
    "retries": "EVEROS_LLMMR_DECIDER_RETRIES",
    "retry_backoff_seconds": "EVEROS_LLMMR_DECIDER_BACKOFF_S",
    "core_overflow": "EVEROS_LLMMR_CORE_OVERFLOW",
    "full_text": "EVEROS_LLMMR_DECIDER_FULL_TEXT",
    "fallback_core": "EVEROS_LLMMR_DECIDER_FALLBACK_CORE",
}

_SEED_CANDIDATES: int = 50
"""Sparse / dense episode recall pool size per sub-query per round."""


def _tuning() -> DeciderSettings:
    """The decider's loop settings, with the legacy env vars applied on top.

    Returns a copy rather than the live settings object so a legacy override cannot
    leak into anything else reading ``[decider]``.
    """
    cfg = load_settings().decider
    over: dict[str, Any] = {}
    for field, env in _LEGACY_ENV.items():
        raw = os.getenv(env, "").strip()
        if not raw:
            continue
        current = getattr(cfg, field)
        try:
            if isinstance(current, bool):
                over[field] = raw == "1"
            elif isinstance(current, int):
                over[field] = int(raw)
            else:
                over[field] = float(raw)
        except ValueError:
            # A malformed override is worth saying out loud rather than silently
            # falling back: the run would report a configuration it never used.
            logger.warning("llm_multiround_bad_env_override", env=env, value=raw[:40])
    return cfg.model_copy(update=over) if over else cfg


_TRACE_DUMP_ENV: str = "EVEROS_LLMMR_TRACE_DUMP"
"""Env var naming the JSONL file that receives one per-round decision record.

Unset / empty ⇒ tracing is off and the loop does zero extra work (production
default). Set to a path ⇒ each round appends a record whose schema mirrors
MemoryRL ``tasks/retrieval_policy/prepare_data.py`` (``load_phase1_traces`` /
``trace_to_sft_example``), so the trace feeds the Phase-2 GRPO cold-start
(behavior cloning) directly. This is a formal, first-class dump — read per call
(not at import) so a run or a test can toggle it via env without re-importing.

The search layer only knows what a ``/search`` request carries (query +
owner), so it emits every field it owns and leaves ``question_id`` (``None``
placeholder) plus the ``gold_session_ids`` / ``core_precision`` / ``core_recall``
labels to a downstream labeling step that has the eval ground truth."""


# ── Decider hook (pluggable: prompt-LLM in P1, trained policy in P2) ──────────


class RoundDecision(NamedTuple):
    """One round's decision.

    Attributes:
        stop: Evidence is sufficient — stop and answer.
        queries: Next queries to issue (when not stopping). The strategy is
            **multi-query gap coverage**, not a single narrow rewrite: the
            decider emits one focused sub-query per still-missing aspect, so
            multi-session / multi-hop questions fan out across threads instead
            of collapsing onto one. Empty list ⇒ stop.
        core: Indices (into the ``evidence`` list the decider was given) of the
            *core* episodes that actually matter — used to shrink the carried
            context and to ground the rewritten queries.
    """

    stop: bool
    queries: list[str]
    core: list[int]
    usage: dict[str, object] | None = (
        None  # decider LLM token usage (trace); None on failure
    )
    raw: str | None = None  # decider raw output before _parse_decision (trace)
    failed: bool = False
    """Every decider attempt failed; ``core`` is the deterministic fallback.

    Kept explicit so a degraded round is countable in the trace instead of
    looking like a decider that chose to stop with nothing."""


class RoundDecider(Protocol):
    """Each round: select core from the blocked view, then stop or expand.

    The Phase-1 default is :class:`LLMRoundDecider`; a Phase-2 policy
    implements this same signature so the loop becomes the RL environment — only
    the decider changes. ``core_so_far`` and ``evidence`` are pre-rendered
    strings (the accumulated core, and the per-sub-query candidate blocks with a
    single global index); ``n_candidates`` is the number of globally-indexed
    candidates the returned ``core`` indices address.
    """

    async def __call__(
        self,
        question: str,
        core_so_far: str,
        evidence: str,
        n_candidates: int,
        round_idx: int,
    ) -> RoundDecision: ...


_DECIDER_PROMPT = """You steer a multi-round memory search. Reply with ONLY a \
JSON object.
You are shown a RETRIEVED SUBSET of memory, never all of it. "Not shown here" \
does NOT mean "not in memory" — it may just not have been retrieved yet.

What you see each round:
- QUESTION: the user's original question (unchanged every round).
- CORE SO FAR: items you already selected in earlier rounds, each tagged with \
the sub-query that surfaced it ("original question" on round 0). These are \
ALREADY kept — never re-list them; use them only to judge what is still missing.
- CANDIDATES THIS ROUND: on round 0, ONE block retrieved for the original \
question; on later rounds, ONE BLOCK PER SUB-QUERY, each headed by that \
sub-query. Items carry a SINGLE global index across all blocks. The same memory \
may appear in several blocks — that just means several angles retrieved it.

Do two things:

1. CORE (required): global indices of every candidate shown THIS ROUND that \
could carry information the answer needs — favour RECALL over minimality.
   - A needed fact (an age, price, date, name, count) is often an INCIDENTAL \
detail inside an episode about a DIFFERENT topic. Judge by whether the item \
touches the same PERSON / ENTITY / PLACE / ACTIVITY / TIME-WINDOW the question \
is about — NOT by whether its headline topic matches.
   - SCAN EVERY BLOCK top to bottom; answer-bearing evidence is often ranked \
LOW in its block.
   - For a count / total / comparison / ordering, core EVERY item that \
contributes an instance — a multi-part answer usually needs several items from \
different sessions.
   - If the SAME memory appears in two blocks, core it ONCE (either index).
   - Return an empty list ONLY when nothing shown touches the question at all.

2. NEXT_QUERIES: KEEP SEARCHING (default) or STOP.
   - STOP (set "next_queries" to []) ONLY when the question is a SINGLE, \
self-contained fact AND the core you already hold answers it unambiguously.
   - OTHERWISE keep searching: one focused sub-query per still-missing aspect \
(up to {max_sub}). Each sub-query must NAME the specific entity / aspect no \
current block has covered — never paraphrase a sub-query already asked, never \
issue a broad topical query.

Illustrative examples (generic, invented — not real data):

Example 1 (round 0, single fact):
QUESTION: What breed is Nora's cat?
CORE SO FAR: (none)
CANDIDATES THIS ROUND:
[block: original question]
  0: Vet visit - Nora brought her cat Biscuit, a Ragdoll, in for shots.
  1: Weekend plans - Nora said she adopted a cat last spring.
Reply: {{"core": [0], "next_queries": []}}

Example 2 (later round, running total across sub-queries):
QUESTION: How many marathons has Devin run in total?
CORE SO FAR:
  - [from "Devin marathon 2021"] Devin finished the Lakeside Marathon in 2021.
CANDIDATES THIS ROUND:
[block: Devin marathon 2022 2023]
  0: Race recap - Devin ran the Harbor Marathon in 2022.
  1: Training log - Devin jogged 5 km most weekends.
[block: Devin other marathons]
  2: Trip photos - Devin flew to Berlin and ran its city marathon in 2023.
  3: Race recap - Devin ran the Harbor Marathon in 2022.
Reply: {{"core": [0, 2], "next_queries": ["Devin marathon before 2021"]}}

QUESTION:
{question}

CORE SO FAR:
{core_so_far}

CANDIDATES THIS ROUND:
{evidence}

Reply with ONLY this JSON object (no prose, no code fences); "core" values are \
GLOBAL indices into the candidates above:
{{"core": [<indices>], "next_queries": ["<q1>", "..."]}}"""


class LLMRoundDecider:
    """Prompt-LLM decider: select core across per-sub-query blocks by global index.

    What it is *shown* differs from what it returns. The evidence is pre-rendered
    by the caller as one labelled block per sub-query (round 0 = a single
    original-question block) with a single global
    index across all blocks, plus a CORE-SO-FAR section listing already-selected
    core tagged with the sub-query that surfaced each. It returns the same
    :class:`RoundDecision` (global core indices + next_queries) so the loop — and
    a future Phase-2 policy that learns the core-selection step — keep one
    contract. The ``core_so_far`` / ``evidence`` strings are built in
    :func:`_search_episodes_subq`; this class only formats + parses.

    Args:
        llm: everalgo LLM client.
        prompt: Template exposing ``{question}``, ``{core_so_far}``,
            ``{evidence}``, ``{max_sub}``. Defaults to :data:`_DECIDER_PROMPT`.
    """

    def __init__(self, llm: LLMClient, *, prompt: str | None = None) -> None:
        self._llm = llm
        self._prompt = prompt or _DECIDER_PROMPT

    async def __call__(
        self,
        question: str,
        core_so_far: str,
        evidence: str,
        n_candidates: int,
        round_idx: int,
    ) -> RoundDecision:
        tune = _tuning()
        prompt = self._prompt.format(
            question=question,
            core_so_far=core_so_far or "(none)",
            evidence=evidence or "(no evidence retrieved this round)",
            max_sub=tune.max_subqueries,
        )
        data: dict | None = None
        last_error: str = ""
        _usage: dict[str, object] | None = None  # decider token usage (trace)
        _raw: str | None = None  # decider raw output (trace)
        for attempt in range(tune.retries + 1):
            try:
                resp = await self._llm.chat(
                    messages=[LLMChatMessage(role="user", content=prompt)]
                )
                _raw = resp.content or ""
                _u = getattr(resp, "usage", None)
                _usage = _u.model_dump() if hasattr(_u, "model_dump") else None
                # Reasoning deciders intermittently return an EMPTY completion
                # (output budget spent on reasoning tokens). Name that case so the
                # log distinguishes "model said nothing" from "reply was unparseable".
                if not _raw.strip():
                    raise ValueError("empty decider reply")
                data = _parse_decision(_raw)
                break
            except Exception as err:
                last_error = f"{type(err).__name__}: {err}"
                logger.warning(
                    "llm_multiround_decider_error",
                    error=last_error[:200],
                    attempt=attempt,
                )
                if attempt < tune.retries and tune.retry_backoff_seconds > 0:
                    await asyncio.sleep(tune.retry_backoff_seconds * (2**attempt))
        if data is None:
            # Every attempt failed. Fall back to a deterministic core (the evidence
            # list is in fused-score order) rather than returning an empty one: an
            # empty core silently disables core-first injection for this question
            # and is invisible downstream. Flag it so the trace can count it.
            fallback = list(range(min(tune.fallback_core, max(n_candidates, 0))))
            # error, not warning: this is the multi-round mechanism not running at all.
            # A warning is what let twelve sweep arms report 87-93% while every decider
            # call 404'd -- the level said "noted", and the numbers were written up.
            logger.error(
                "llm_multiround_decider_fallback",
                round_idx=round_idx,
                attempts=tune.retries + 1,
                fallback_core=len(fallback),
                last_error=last_error[:200],
            )
            # And say so in the response. A caller cannot tell a degraded result from a
            # healthy one by looking at it: both are HTTP 200 with a full episode list.
            mark_degraded("decider_fallback")
            return RoundDecision(
                stop=True,
                queries=[],
                core=fallback,
                usage=_usage,
                raw=_raw,
                failed=True,
            )
        core = _coerce_core_indices(data.get("core", []), n_candidates)
        nxt_raw = data.get("next_queries")
        if not isinstance(nxt_raw, list):
            nxt_raw = []
        queries = [str(q).strip() for q in nxt_raw if str(q).strip()][
            : tune.max_subqueries
        ]
        return RoundDecision(
            stop=not queries, queries=queries, core=core, usage=_usage, raw=_raw
        )


def _parse_decision(text: str) -> dict:
    """Extract the JSON decision object from an LLM reply (tolerant of prose).

    Tries the widest ``{...}`` span first (the common single-object reply);
    if that fails to parse, falls back to scanning for the first brace-balanced
    object, so a stray ``{`` / ``}`` in surrounding prose cannot corrupt the
    parse (the greedy span would otherwise swallow it and raise). Raises
    ``ValueError`` only when no ``dict`` object can be recovered — the caller
    treats that as a safe stop with no core.
    """
    candidates: list[str] = []
    greedy = re.search(r"\{.*\}", text, re.DOTALL)
    if greedy:
        candidates.append(greedy.group())
    candidates.extend(_balanced_objects(text))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"no JSON object in decider reply: {text[:120]!r}")


def _balanced_objects(text: str) -> list[str]:
    """Yield brace-balanced ``{...}`` substrings, outermost first."""
    out: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(text[start : i + 1])
    return out


def _coerce_core_indices(raw: object, n_evidence: int) -> list[int]:
    """Coerce a decider's ``core`` field into valid, de-duplicated evidence indices.

    Tolerates ints and int-like strings (``"2"``), silently dropping anything
    out of range, non-integral, or duplicated. A non-list ``raw`` yields ``[]``
    (empty core ⇒ the caller falls back to the unfiltered / blind path).
    """
    if not isinstance(raw, list):
        return []
    seen: set[int] = set()
    core: list[int] = []
    for value in raw:
        if isinstance(value, bool):  # bool is an int subclass — reject explicitly
            continue
        if isinstance(value, int):
            idx = value
        elif isinstance(value, str) and value.strip().lstrip("-").isdigit():
            idx = int(value)
        else:
            continue
        if 0 <= idx < n_evidence and idx not in seen:
            seen.add(idx)
            core.append(idx)
    return core


# ── Main entry ──────────────────────────────────────────────────────────────


async def search_episodes_llm_multiround(
    query: str,
    *,
    owner_id: str,
    where: str,
    app_id: str = "default",
    project_id: str = "default",
    episode_recaller: EpisodeRecaller,
    atomic_fact_recaller: AtomicFactRecaller,
    embed_query_fn: Callable[[str], Awaitable[list[float]]],
    llm: LLMClient,
    top_k: int,
    reranker: RerankProvider | None = None,
    decider: RoundDecider | None = None,
) -> list[SearchEpisodeItem]:
    """LLM-guided iterative multi-round episode search (per-sub-query RRF blocks).

    Thin entry that delegates to :func:`_search_episodes_subq` — the only arm.
    Each round fuses every current sub-query's BM25+vector recall with RRF
    INDEPENDENTLY (round 0 = one original-question block), shows the decider the
    labelled blocks + accumulated core and selects core across them; the final
    injection is core-first + each sub-query's top-1 guarantee + max-RRF-score
    fill. No cross-encoder anywhere (RRF is the only ranking).

    Args:
        query: User query (also the round-0 query).
        owner_id: Owner whose memories are searched.
        where: Pre-compiled LanceDB filter (owner + request filters).
        app_id / project_id: Scope segments (parity; recall is owner-scoped via
            ``where``).
        episode_recaller: Episode sparse + dense recall.
        atomic_fact_recaller: Accepted for call-site parity; this scheme recalls at the
            episode level and does not drill facts.
        embed_query_fn: Async ``(str) -> vector`` query embedder.
        llm: LLM client for the decider.
        top_k: Maximum episodes to return.
        reranker: Accepted for parity but UNUSED — no cross-encoder stage.
        decider: Round-control hook. When given, it REPLACES the built-in
            :class:`LLMRoundDecider` and drives every round, which is how a
            Phase-2 RL policy reuses this loop as its environment.

    Returns:
        Ranked ``SearchEpisodeItem`` list, empty on an empty seed.
    """
    return await _search_episodes_subq(
        query,
        owner_id=owner_id,
        where=where,
        app_id=app_id,
        project_id=project_id,
        episode_recaller=episode_recaller,
        atomic_fact_recaller=atomic_fact_recaller,
        embed_query_fn=embed_query_fn,
        llm=llm,
        top_k=top_k,
        reranker=reranker,
        decider=decider,
    )


def _render_blocks(
    blocks: list[tuple[str, list[Candidate]]],
) -> tuple[str, list[Candidate], list[dict[str, object]]]:
    """Render per-sub-query blocks into the decider text + a global-index map.

    Returns ``(rendered, global_cands, index_meta)``: ``rendered`` is the
    ``[block: <sub-query>]`` header + numbered ``  <gi>: subject - summary`` lines
    with a SINGLE global index running across all blocks; ``global_cands[gi]`` is
    the Candidate at that index (blocks are concatenated in order, so the same id
    can appear at two indices when two sub-queries retrieved it — the decider is
    told to core it once); ``index_meta[gi]`` = block_id / sub_query /
    rank_in_block / rrf_score, used for the trace and the core source tag.
    """
    lines: list[str] = []
    global_cands: list[Candidate] = []
    index_meta: list[dict[str, object]] = []
    gi = 0
    for block_id, (sub_query, cands) in enumerate(blocks):
        lines.append(f"[block: {sub_query}]")
        for rank, c in enumerate(cands):
            subject = str(c.metadata.get("subject", ""))
            summary = _decider_text(c.metadata)
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


def _render_core_so_far(
    core_order: list[str],
    core_source: dict[str, str],
    cand_by_id: dict[str, Candidate],
) -> str:
    """Render accumulated core for the decider, each item tagged with the
    sub-query that first surfaced it. Empty core renders as ``(none)``."""
    out: list[str] = []
    for cid in core_order:
        c = cand_by_id.get(cid)
        if c is None:
            continue
        subject = str(c.metadata.get("subject", ""))
        summary = _decider_text(c.metadata)
        src = core_source.get(cid, "original question")
        out.append(f'  - [from "{src}"] {subject} - {summary}')
    return "\n".join(out) or "(none)"


def _decider_text(meta: dict) -> str:
    """Text shown to the decider for one candidate.

    Stores disagree on which column carries the full episode body (see
    ``[decider].full_text``), so prefer whichever is longer when full-text
    mode is on: that picks the body on both layouts without needing to know
    which store is loaded. Off (the default) keeps the historical
    ``summary``-only view so in-flight comparisons stay reproducible.
    """
    summary = str(meta.get("summary", "") or "")
    if not _tuning().full_text:
        return summary
    episode = str(meta.get("episode", "") or "")
    return episode if len(episode) > len(summary) else summary


def _build_round_trace(
    *,
    owner_id: str,
    question: str,
    round_idx: int,
    round_kind: str,
    global_cands: list[Candidate],
    index_meta: list[dict[str, object]],
    decision: RoundDecision,
    core_added: list[dict[str, object]],
    core_carried_in: list[dict[str, object]],
    block_meta: list[dict[str, object]],
    timing_s: dict[str, float],
) -> dict[str, object]:
    """Assemble one per-round trace record (schema C).

    Completeness principle — records everything the search layer owns this round:
    the decider's blocked view (``evidence``: every global index with its block /
    sub-query / RRF provenance), the action (``core_indices`` + resolved sessions
    + source sub-query), the core carried INTO this round, the full per-block
    recall pools (``recall.blocks``: sparse/dense pre-truncation + the RRF-ranked
    kept block with in_sparse/in_dense flags), the decider tokens + raw verdict,
    and per-stage timing. There is NO ``fused`` / ``reranked`` pool — this scheme
    has neither a merged fuse nor a cross-encoder. ``question_id`` and the gold labels
    are left for a downstream labeller (the search layer cannot know them).
    """
    evidence = [
        {
            "global_index": gi,
            "block_id": index_meta[gi]["block_id"],
            "sub_query": index_meta[gi]["sub_query"],
            "id": c.id,
            "session_id": c.metadata.get("session_id"),
            "subject": str(c.metadata.get("subject", "")),
            "summary": str(c.metadata.get("summary", "")),
            "rrf_score": index_meta[gi]["rrf_score"],
            "rank_in_block": index_meta[gi]["rank_in_block"],
        }
        for gi, c in enumerate(global_cands)
    ]
    core_indices = list(decision.core)
    core_session_ids = [
        global_cands[i].metadata.get("session_id")
        for i in core_indices
        if 0 <= i < len(global_cands)
    ]
    record: dict[str, object] = {
        "dataset": _dataset_from_owner(owner_id),
        "owner_id": owner_id,
        "question_id": None,
        "question": question,
        "round_idx": round_idx,
        "round_kind": round_kind,
        "evidence": evidence,
        "core_indices": core_indices,
        "core_session_ids": core_session_ids,
        "core_source_subquery": [m["sub_query"] for m in core_added],
        "core_added": core_added,
        "core_carried_in": core_carried_in,
        "stop": decision.stop,
        "next_queries": list(decision.queries),
        "recall": {"blocks": block_meta},
        "timing_s": timing_s,
    }
    if decision.usage is not None or decision.raw is not None:
        record["decider"] = {"tokens": decision.usage, "raw": decision.raw}
    if decision.failed:
        # Degraded round: every decider attempt failed and ``core_indices`` is the
        # deterministic fallback, not a decider choice. Recorded so analysis can
        # count or exclude these instead of silently mixing them with real rounds.
        record["decider_failed"] = True
    return record


async def _search_episodes_subq(
    query: str,
    *,
    owner_id: str,
    where: str,
    app_id: str = "default",
    project_id: str = "default",
    episode_recaller: EpisodeRecaller,
    atomic_fact_recaller: AtomicFactRecaller,
    embed_query_fn: Callable[[str], Awaitable[list[float]]],
    llm: LLMClient,
    top_k: int,
    reranker: RerankProvider | None,
    decider: RoundDecider | None,
) -> list[SearchEpisodeItem]:
    """Retrieval loop: per-sub-query RRF blocks + guarantee-then-fill, no cross-encoder.

    Each round: embed every current sub-query, recall (BM25 + vector) and fuse
    EACH sub-query's own pair with RRF INDEPENDENTLY (round 0 = one block for the
    original question, kept to ``[decider].seed_topk``; round>=1 = one block per
    sub-query, each kept to ``[decider].subq_topk``). The blocks are shown to the
    decider as labelled sections with a single GLOBAL index, alongside a
    CORE-SO-FAR section (each kept item tagged with its source sub-query). The
    decider returns core (global indices) + next sub-queries. Stopping:
    empty ``next_queries`` / ``[decider].no_new_core_patience`` saturated
    rounds (after >=1 follow-up) / ``[decider].max_rounds``.

    After the loop, :func:`_finalize_injection` assembles the top_k: core-first
    (hard guarantee) + each sub-query's top non-core + max-RRF-score fill.

    ``reranker`` is accepted for call-site parity but intentionally UNUSED — this
    scheme has no cross-encoder stage (the CE was net-negative). ``decider``, when
    supplied, replaces :class:`LLMRoundDecider` for every round: that is the seam an RL
    policy occupies so the loop it trains against is this loop rather than a copy.
    """
    # Honour an injected decider. The parameter and the RoundDecider protocol were
    # written for this ("a Phase-2 policy implements this same signature so the loop
    # becomes the RL environment — only the decider changes") but the body ignored the
    # argument and always built the prompt-LLM decider, so an RL environment had no way
    # to reuse this loop and had to re-implement retrieval, block rendering, core
    # accumulation and the stop conditions. Re-implementing diverged: measured against
    # this loop, a hand-built environment retrieved a different candidate set on 25/25
    # sampled sub-queries (Jaccard median 0.538): the HTTP search route it called
    # dispatches to the hierarchy/heap-expand hybrid pipeline, not this file's
    # per-sub-query rrf(sparse, dense)[:topk].
    decide = decider or LLMRoundDecider(llm)

    core_by_id: dict[str, Candidate] = {}  # accumulated core (best-score candidate)
    core_ids: set[str] = set()
    core_order: list[str] = []  # core insertion order (drives core-first)
    core_source: dict[str, str] = {}  # id -> sub-query that first cored it (#4)
    subq_hits: dict[str, dict[str, float]] = {}  # id -> {sub_query: rrf_score}
    cand_by_id: dict[str, Candidate] = {}  # id -> best-score candidate ever seen
    guarantee: list[tuple[str, str]] = []  # (sub_query, top id) per block, in order
    subqueries_seen: list[str] = []
    no_new_core_streak = 0
    cur_queries = [query]  # round 0 seeds with the user question

    tune = _tuning()
    for round_idx in range(tune.max_rounds):
        _ts_round_start = time.monotonic()
        round_kind = "seed" if round_idx == 0 else "subquery"
        block_topk = tune.seed_topk if round_idx == 0 else tune.subq_topk
        vecs = await asyncio.gather(*[embed_query_fn(q) for q in cur_queries])
        # Recall every sub-query concurrently (sparse + dense). The KEY design point:
        # each sub-query's pair is fused with RRF INDEPENDENTLY below, not
        # merged into one pool first — so a facet-gold ranked #1 for one sub-query
        # is not diluted by the other sub-queries' consensus hits.
        recalls = await asyncio.gather(
            *[
                asyncio.gather(
                    episode_recaller.sparse_recall(q, where, limit=_SEED_CANDIDATES),
                    episode_recaller.dense_recall(v, where, limit=_SEED_CANDIDATES)
                    if v
                    else _empty_candidates(),
                )
                for q, v in zip(cur_queries, vecs, strict=True)
            ]
        )
        _ts_recall_done = time.monotonic()
        blocks: list[tuple[str, list[Candidate]]] = []
        block_meta: list[dict[str, object]] = []
        for q, (r_sparse, r_dense) in zip(cur_queries, recalls, strict=True):
            # Block LABEL shown to the decider (and stored as each candidate's
            # source): round 0 labels its one block "original question" — the raw
            # question is already displayed above it and the few-shot uses this
            # label; later rounds label each block with its own sub-query text.
            label = "original question" if round_idx == 0 else q
            fused = rrf(r_sparse, r_dense, k=tune.rrf_k)  # ranked by RRF score
            block = fused[:block_topk]
            blocks.append((label, block))
            if label not in subqueries_seen:
                subqueries_seen.append(label)
            sparse_ids = {c.id for c in r_sparse}
            dense_ids = {c.id for c in r_dense}
            for c in block:
                subq_hits.setdefault(c.id, {})[label] = c.score
                cur = cand_by_id.get(c.id)
                if cur is None or c.score > cur.score:
                    cand_by_id[c.id] = c
            # Guarantee this block's top per_subquery_guarantee for the final fill.
            for c in block[: tune.per_subquery_guarantee]:
                guarantee.append((label, c.id))
            block_meta.append(
                {
                    "sub_query": label,
                    "n_sparse": len(r_sparse),
                    "n_dense": len(r_dense),
                    "topk_kept": len(block),
                    "sparse": _trace_cand_pool(r_sparse),
                    "dense": _trace_cand_pool(r_dense),
                    "rrf_ranked": [
                        {
                            "id": c.id,
                            "session_id": c.metadata.get("session_id"),
                            "rrf_score": c.score,
                            "rank": r,
                            "in_sparse": c.id in sparse_ids,
                            "in_dense": c.id in dense_ids,
                        }
                        for r, c in enumerate(block)
                    ],
                }
            )
        if round_idx == 0 and not any(b for _, b in blocks):
            return []  # empty seed — nothing to rank or decide over

        # Render the blocked decider view + CORE-SO-FAR (global index across blocks).
        rendered, global_cands, index_meta = _render_blocks(blocks)
        core_so_far = _render_core_so_far(core_order, core_source, cand_by_id)
        _ts_decide_start = time.monotonic()
        decision = await decide(
            query, core_so_far, rendered, len(global_cands), round_idx
        )

        # Accumulate new core (global index -> id, dedup against already-cored).
        n_before = len(core_ids)
        core_added: list[dict[str, object]] = []
        for gi in decision.core:
            if not (0 <= gi < len(global_cands)):
                continue
            cid = global_cands[gi].id
            if cid in core_ids:
                continue
            core_ids.add(cid)
            core_order.append(cid)
            core_by_id[cid] = cand_by_id.get(cid, global_cands[gi])
            core_source[cid] = str(index_meta[gi]["sub_query"])
            core_added.append(
                {
                    "global_index": gi,
                    "id": cid,
                    "sub_query": index_meta[gi]["sub_query"],
                }
            )
        added = len(core_ids) - n_before
        no_new_core_streak = 0 if added else no_new_core_streak + 1

        dump_path = _trace_dump_path()
        if dump_path:
            _append_round_trace(
                dump_path,
                _build_round_trace(
                    owner_id=owner_id,
                    question=query,
                    round_idx=round_idx,
                    round_kind=round_kind,
                    global_cands=global_cands,
                    index_meta=index_meta,
                    decision=decision,
                    core_added=core_added,
                    core_carried_in=[
                        {
                            "id": cid,
                            "sub_query": core_source.get(cid),
                            "subject": str(cand_by_id[cid].metadata.get("subject", "")),
                            "summary": str(cand_by_id[cid].metadata.get("summary", "")),
                        }
                        for cid in core_order
                        if cid in cand_by_id
                    ],
                    block_meta=block_meta,
                    timing_s={
                        "recall": round(_ts_recall_done - _ts_round_start, 3),
                        "decide": round(time.monotonic() - _ts_decide_start, 3),
                    },
                ),
            )
        logger.info(
            "llm_multiround_round",
            round=round_idx + 1,
            kind=round_kind,
            n_blocks=len(blocks),
            n_candidates=len(global_cands),
            n_core=len(core_ids),
            added_core=added,
            no_new_core_streak=no_new_core_streak,
            n_subqueries=len(decision.queries),
            next_queries=" | ".join(decision.queries)[:120],
            query=query[:80],
        )

        # Stop conditions, in the design's priority order.
        if not decision.queries:  # (1) decider is done (no gaps left)
            break
        # (2) coverage saturated — never before at least one follow-up round.
        if round_idx >= 1 and no_new_core_streak >= tune.no_new_core_patience:
            break
        cur_queries = decision.queries  # else continue; (3) the loop caps rounds

    if not cand_by_id and not core_by_id:
        return []
    return _finalize_injection(
        owner_id=owner_id,
        question=query,
        top_k=top_k,
        core_ids=core_ids,
        core_order=core_order,
        cand_by_id=cand_by_id,
        subq_hits=subq_hits,
        guarantee=guarantee,
        subqueries_seen=subqueries_seen,
    )


def _finalize_injection(
    *,
    owner_id: str,
    question: str,
    top_k: int,
    core_ids: set[str],
    core_order: list[str],
    cand_by_id: dict[str, Candidate],
    subq_hits: dict[str, dict[str, float]],
    guarantee: list[tuple[str, str]],
    subqueries_seen: list[str],
) -> list[SearchEpisodeItem]:
    """Assemble the final top_k and dump the final-injection trace (schema C).

    Assembly (post-loop, once):
      1. CORE first — every accumulated core, pinned to the front (a hard
         guarantee: kept even when ``len(core) > top_k``), ordered by max RRF
         score.
      2. GUARANTEE — each sub-query's top non-core candidate (its ``guarantee``
         entries) reserves a slot, so no facet loses coverage.
      3. FILL — remaining candidates by MAX RRF score across sub-queries. MAX, not
         sum: a specialist that is #1 for a single sub-query keeps that score
         instead of being diluted by consensus items (the RRF-dilution fix).

    The final-injection trace records, per injected episode, its slot_source
    (core / guarantee-top1 / maxscore-fill), the max RRF score + which sub-query
    gave it, and the full per-sub-query score map — so the whole assembly is
    reconstructable — plus an ``assembly`` summary count.
    """

    def max_score(cid: str) -> float:
        scores = subq_hits.get(cid)
        return max(scores.values()) if scores else 0.0

    def max_subquery(cid: str) -> str | None:
        scores = subq_hits.get(cid)
        return max(scores, key=lambda k: scores[k]) if scores else None

    slot_source: dict[str, str] = {}
    chosen: list[str] = []
    # 1. core-first, ordered by max RRF score across sub-queries. Capped at top_k:
    # steps 2 and 3 honour top_k, so leaving core uncapped silently returned MORE
    # than the caller asked for — measured at 316/1522 questions (20.8%) on
    # SubtleMemory, up to 68 items for a top_k of 20 (3.4x the budget). That breaks
    # the top_k contract and any same-budget comparison across methods. Core beyond
    # the budget is by definition the lowest-scored core, so it is what gets dropped.
    n_core_selected = 0
    for cid in sorted(core_order, key=max_score, reverse=True):
        if cid in slot_source:
            continue
        n_core_selected += 1
        if not _tuning().core_overflow and len(chosen) >= top_k:
            continue  # keep counting for the trace, but do not exceed the budget
        chosen.append(cid)
        slot_source[cid] = "core"
    n_core = len(chosen)
    if n_core_selected > n_core:
        logger.info(
            "llm_multiround_core_truncated",
            owner_id=owner_id,
            selected=n_core_selected,
            kept=n_core,
            top_k=top_k,
        )
    # 2. per-sub-query guarantee (top non-core, in block discovery order)
    n_guaranteed = 0
    for _sub_query, cid in guarantee:
        if len(chosen) >= top_k:
            break
        if cid in slot_source or cid in core_ids:
            continue
        chosen.append(cid)
        slot_source[cid] = "guarantee-top1"
        n_guaranteed += 1
    # 3. max-score fill to top_k
    n_filled = 0
    remaining = [cid for cid in cand_by_id if cid not in slot_source]
    for cid in sorted(remaining, key=max_score, reverse=True):
        if len(chosen) >= top_k:
            break
        chosen.append(cid)
        slot_source[cid] = "maxscore-fill"
        n_filled += 1

    ordered_cands = [cand_by_id[cid] for cid in chosen if cid in cand_by_id]
    episodes = [
        ep
        for ep in (shape_episode_from_candidate(c) for c in ordered_cands)
        if ep is not None
    ]

    dump_path = _trace_dump_path()
    if dump_path:
        _append_round_trace(
            dump_path,
            {
                "dataset": _dataset_from_owner(owner_id),
                "owner_id": owner_id,
                "question_id": None,
                "question": question,
                "round_idx": None,  # None ⇒ final-injection record, not a per-round one
                "injected": [
                    {
                        "rank": r,
                        "id": ep.id,
                        "session_id": ep.session_id,
                        "timestamp": str(ep.timestamp),
                        "is_core": ep.id in core_ids,
                        "slot_source": slot_source.get(ep.id),
                        "max_rrf_score": max_score(ep.id),
                        "max_rrf_subquery": max_subquery(ep.id),
                        "per_subquery_scores": subq_hits.get(ep.id, {}),
                    }
                    for r, ep in enumerate(episodes)
                ],
                "assembly": {
                    "n_core": n_core,
                    "n_guaranteed": n_guaranteed,
                    "n_filled": n_filled,
                    "top_k": top_k,
                    "subqueries_seen": subqueries_seen,
                },
            },
        )
    return episodes


def _trace_dump_path() -> str | None:
    """Return the trace JSONL path, or ``None`` when the dump is disabled.

    Read per call (not memoised at import) so a run or a test can toggle
    ``EVEROS_LLMMR_TRACE_DUMP`` via env without re-importing the module. An
    unset or whitespace-only value disables the dump.
    """
    path = os.getenv(_TRACE_DUMP_ENV, "").strip()
    return path or None


def _dataset_from_owner(owner_id: str) -> str:
    """Best-effort dataset tag from the eval ``owner_id`` convention.

    Eval owner ids follow ``"<dataset>_<conv_index>"`` (see MemoryRL
    ``prepare_data``), e.g. ``longmemeval_0`` ⇒ ``longmemeval``. The trailing
    numeric conv index is stripped; an id that does not match the convention
    (no ``_`` or a non-numeric tail) passes through unchanged.
    """
    head, sep, tail = owner_id.rpartition("_")
    if sep and tail.isdigit():
        return head
    return owner_id


def _trace_cand_pool(cands: object) -> list[dict[str, object]]:
    """Serialize a candidate pool (dict[id]->Candidate or list) for the full-retrieval
    attribution trace: id + session_id + score + source + metadata (session / entry_id /
    fact provenance). Lets a bad case be split into recall-miss vs rerank-miss vs
    decider-miss, and drilled to the atomic fact that surfaced each episode."""
    items = cands.values() if isinstance(cands, dict) else (cands or [])
    out: list[dict[str, object]] = []
    for c in items:
        md = c.metadata if isinstance(c.metadata, dict) else {}
        # Attribution-essential provenance only. Deliberately DROP the bulky episode
        # ``summary`` (recoverable from the store by entry_id/session_id; ~5-10x smaller
        # trace) and the per-owner constants (owner_id/app_id/...). The decider-view
        # ``evidence`` keeps its summary — that IS the decider's input.
        prov = {
            k: md[k]
            for k in (
                "entry_id",
                "session_id",
                "subject",
                "timestamp",
                "parent_id",
                "parent_type",
            )
            if md.get(k) is not None
        }
        out.append(
            {
                "id": c.id,
                "session_id": md.get("session_id"),
                "score": c.score,
                "source": c.source,
                "provenance": prov,
            }
        )
    return out


def _append_round_trace(path: str, record: dict[str, object]) -> None:
    """Append one trace ``record`` as a JSON line to ``path``.

    Diagnostic side-channel, reached only when ``EVEROS_LLMMR_TRACE_DUMP`` is
    set. The server event loop is single-threaded, so a small synchronous
    append cannot interleave across concurrent search coroutines (each
    ``write`` completes before the coroutine yields). A write failure is logged
    and swallowed — trace dumping must never break a live search.
    """
    try:
        # default=str: candidate metadata carries datetimes (episode timestamps) and
        # other non-JSON scalars; stringify them instead of raising. except Exception
        # (not just OSError): a trace-dump failure must NEVER break a live search.
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as err:
        logger.warning("llm_multiround_trace_dump_error", error=str(err)[:200])


async def _empty_candidates() -> list[Candidate]:
    return []
