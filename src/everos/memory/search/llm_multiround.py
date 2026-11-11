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
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    Protocol,
    runtime_checkable,
)

from everalgo.llm import (
    LLMMessage,
    StrictOutputModel,
    image_url_part_from_bytes,
)
from everalgo.rank.fusion import rrf
from everalgo.types import Candidate

from everos.config.settings import DeciderSettings
from everos.core.observability.logging import get_logger
from everos.memory.models import Episode, ImageMemory
from everos.memory.ports import Filter, Stored
from everos.memory.ports.images import ImageStore

from .dto import SearchEpisodeItem, SearchImageItem
from .recall import RecallResult
from .recall.base import to_candidate
from .recall.image_memory import IMAGE_KIND
from .shaper import shape_episodes, shape_images

logger = get_logger(__name__)

if TYPE_CHECKING:
    from everalgo.llm import LLMClient
    from pydantic import BaseModel

    from everos.component.llm import RawReply, TokenUsage
    from everos.component.rerank import RerankProvider
    from everos.memory.search.recall.atomic_fact import AtomicFactRecaller
    from everos.memory.search.recall.episode import EpisodeRecaller
    from everos.memory.search.recall.image_memory import ImageMemoryRecaller


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
# Configuration sources are called at each existing decision boundary.


# ── Decider hook (pluggable: prompt-LLM in P1, trained policy in P2) ──────────


class Picture(NamedTuple):
    """One picture the decider may look at this round.

    Attributes:
        label: The label the evidence text uses for it (``P3``), so the decider
            can tie an image part back to the numbered line it belongs to.
            Empty for a question picture, which belongs to no evidence line.
        data: Bytes of the thumbnail — the decider's copy, never the original.
        mime: MIME type of ``data``.
        question: Whether this picture came with the *question* rather than
            from the candidates. A question picture is shown on every round
            and is not charged against the per-round candidate cap: it is what
            is being asked about, not a candidate to select.
    """

    label: str
    data: bytes
    mime: str
    question: bool = False


_PICTURE_MARK = " (picture)"
"""Appended to a picture candidate's evidence line.

The decider has to know that a numbered item is a picture and not a narrative:
what it may expect to learn from one differs, and the model is told to judge a
picture by what it shows. The mark rides in the text so it survives a decider
that runs without images at all (``image_candidates_per_round = 0``).
"""


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
    usage: dict[str, object] | None = None
    """Token usage for the trace, keyed as the exp trace keys it
    (``prompt_tokens`` / ``completion_tokens``). The prompt-LLM decider fills it
    when its client returns a :class:`~everos.component.llm.RawReply`; a client
    reached through ``achat``, which returns text only, leaves it ``None``."""
    raw: str | None = None  # decider raw reply text, the last attempt's (trace)
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
        *,
        pictures: Sequence[Picture] = (),
    ) -> RoundDecision: ...


Coverage = Literal["complete", "partial", "none"]
"""How far a multi-round search got (docs/group-memory.md, section 8.5):
``complete`` -- the decider stopped with a core; ``partial`` -- the round
budget (or the no-new-core patience) ended the loop first; ``none`` -- no
core at all."""


class MultiroundResult(NamedTuple):
    """What one multi-round search returns besides its ranking.

    Attributes:
        episodes / images: The final injection, split by kind.
        coverage: See :data:`Coverage`.
        core: The decider's core, as candidate ids in the order they were
            first selected.
        records: The stored record behind every candidate the loop saw, by
            candidate id (what the constraint block reads the core from).
    """

    episodes: list[SearchEpisodeItem]
    images: list[SearchImageItem]
    coverage: Coverage
    core: list[str]
    records: dict[str, Stored[Episode] | Stored[ImageMemory]]


@runtime_checkable
class LinkExpander(Protocol):
    """Follows stored links from a round's new core (section 8.3).

    Called once per round that the loop continues past, with the records the
    decider just selected; returns the linked records, in priority order, each
    with the id of the anchor it was reached from. It reads through the read
    scope's predicate, so it can only return what the asker may see, and it
    never looks at the question's wording -- only ids and links.
    """

    async def __call__(
        self, anchors: Sequence[Stored[Episode] | Stored[ImageMemory]]
    ) -> list[tuple[str, Stored[Episode]]]: ...


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


class _DecisionReply(StrictOutputModel):
    """The reply shape the prompt's last line shows.

    Sent as ``response_format`` only when ``[decider].response_format_mode``
    asks for one, which by default it does not. Never used to judge a reply:
    :func:`_parse_decision` reads it the way the exp decider does.
    """

    core: list[int]
    next_queries: list[str]


@runtime_checkable
class RawChatClient(Protocol):
    """An LLM client that hands a reply back unchecked, for the decider to read.

    EverOS's ``OpenAIProvider`` is one (``achat_raw``), and the decider prefers
    it to the everalgo ``achat`` contract. ``achat`` validates the reply against
    the strict ``response_format`` model and fails a ``length`` finish even when
    the JSON in it is complete; the parity check against the exp tree
    (2026-11-10, deviation R2) measured both turning replies the exp decider
    uses -- fenced JSON, prose around the object, a string index, a missing
    ``next_queries`` -- into failed rounds: eight requests each, then the fixed
    fallback core, which leaves the question nothing to score core-only.

    A client without it (a test double, a policy's own client) is asked
    through ``achat``, and the text it returns is read the same tolerant way.
    """

    async def achat_raw(
        self,
        messages: list[LLMMessage],
        *,
        response_format: type[BaseModel] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **extra: Any,
    ) -> RawReply: ...


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
    :func:`_search_episodes_subq`; this class only formats the request and
    reads the reply, tolerantly, exactly as the exp decider the policy was
    trained and evaluated with reads it (:func:`_parse_decision`).

    Args:
        llm: everalgo LLM client.
        prompt: Template exposing ``{question}``, ``{core_so_far}``,
            ``{evidence}``, ``{max_sub}``. Defaults to :data:`_DECIDER_PROMPT`.
        tuning_source: Read current settings when this decider is invoked.
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        prompt: str | None = None,
        tuning_source: Callable[[], DeciderSettings],
    ) -> None:
        self._llm = llm
        self._tuning_source = tuning_source
        self._prompt = prompt or _DECIDER_PROMPT

    async def __call__(
        self,
        question: str,
        core_so_far: str,
        evidence: str,
        n_candidates: int,
        round_idx: int,
        *,
        pictures: Sequence[Picture] = (),
    ) -> RoundDecision:
        tune = self._tuning_source()
        prompt = self._prompt.format(
            question=question,
            core_so_far=core_so_far or "(none)",
            evidence=evidence or "(no evidence retrieved this round)",
            max_sub=tune.max_subqueries,
        )
        data: dict[str, Any] | None = None
        raw: str | None = None
        usage: dict[str, object] | None = None
        last_error = ""
        for attempt in range(tune.retries + 1):
            try:
                raw, usage = await self._ask(prompt, tune, pictures)
                # Reasoning deciders intermittently return an EMPTY completion
                # (output budget spent on reasoning tokens). Name that case so the
                # log tells "model said nothing" from "reply was unparseable".
                if not raw.strip():
                    raise ValueError("empty decider reply")
                data = _parse_decision(raw)
                break
            except Exception as err:
                # Any failure degrades this round to the deterministic fallback
                # below; the search request itself must never fail on the decider.
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
            return RoundDecision(
                stop=True,
                queries=[],
                core=fallback,
                usage=usage,
                raw=raw,
                failed=True,
            )
        core = _coerce_core_indices(data.get("core", []), n_candidates)
        # A missing, null or non-list ``next_queries`` is a stop, and a query that
        # is not a string is kept as its text -- as the exp decider reads them.
        next_raw = data.get("next_queries")
        if not isinstance(next_raw, list):
            next_raw = []
        queries = [str(q).strip() for q in next_raw if str(q).strip()][
            : tune.max_subqueries
        ]
        return RoundDecision(
            stop=not queries, queries=queries, core=core, usage=usage, raw=raw
        )

    async def _ask(
        self,
        prompt: str,
        tune: DeciderSettings,
        pictures: Sequence[Picture] = (),
    ) -> tuple[str, dict[str, object] | None]:
        """One decider call, one request: the reply text as sent, and its tokens.

        ``max_tokens``, ``temperature`` and ``extra`` come from ``[decider]``, not
        from the client's defaults; a key in ``extra`` wins over the two named ones.
        ``extra`` is where a Qwen endpoint gets
        ``chat_template_kwargs.enable_thinking=false``: left on, an un-finetuned
        decider spends the whole deadline reasoning and the round is lost to a
        timeout (39.3% of rounds on Qwen3.5-0.8B).

        Nothing here judges the reply; :meth:`__call__` parses it. A
        :class:`RawChatClient` is asked through ``achat_raw``, so the reply comes
        back whatever it holds and whatever the finish reason, from exactly one
        request -- the exp decider's request count. Any other client goes through
        ``achat`` with :class:`_DecisionReply` as the format.
        """
        params: dict[str, Any] = {
            "max_tokens": tune.max_tokens,
            "temperature": tune.temperature,
            **tune.extra,
        }
        messages = [LLMMessage(role="user", content=_content(prompt, pictures, tune))]
        if isinstance(self._llm, RawChatClient):
            reply = await self._llm.achat_raw(
                messages, response_format=_DecisionReply, **params
            )
            return reply.text, _trace_tokens(reply.usage)
        text = await self._llm.achat(messages, response_format=_DecisionReply, **params)
        return text, None


def _trace_tokens(usage: TokenUsage | None) -> dict[str, object] | None:
    """A completion's token counts as the trace records them (exp's keys)."""
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
    }


def _parse_decision(text: str) -> dict[str, Any]:
    """Extract the JSON decision object from a decider reply, tolerant of prose.

    Ported from the exp decider unchanged, because the decider policy was trained
    and evaluated against this reading: the parity check of 2026-11-10
    (deviation R2) found the strict model this replaced failing fenced JSON,
    prose before or after the object, a ``<think>`` prefix and an extra key --
    replies the exp decider takes as they are.

    Tries the widest ``{...}`` span first (the common single-object reply); if
    that fails to parse, falls back to the brace-balanced objects in order, so
    a stray ``{`` / ``}`` in surrounding prose cannot corrupt the parse (the
    greedy span would otherwise swallow it and raise).

    Raises:
        ValueError: No ``dict`` object can be recovered; the caller counts it as
            one failed attempt.
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
    """The brace-balanced ``{...}`` substrings of ``text``, outermost first."""
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

    The exp decider's rule (parity check 2026-11-10, deviation R2): ints and
    int-like strings (``"2"``) count; anything out of range, non-integral or
    repeated is dropped, a repeated index keeping its first position. ``bool``
    is an ``int`` subclass and is rejected explicitly, and a float is not an
    index even when integral. A non-list ``raw`` yields ``[]`` -- an empty
    core, not a failed round.
    """
    if not isinstance(raw, list):
        return []
    seen: set[int] = set()
    core: list[int] = []
    for value in raw:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            idx = value
        elif isinstance(value, str) and value.strip().lstrip("-").isdecimal():
            # isdecimal(), not isdigit(): "²".isdigit() is True while
            # int("²") raises, so a superscript index would crash the parse
            # instead of being ignored. isdecimal() is exactly what int() takes.
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
    where: Filter,
    app_id: str = "default",
    project_id: str = "default",
    episode_recaller: EpisodeRecaller,
    atomic_fact_recaller: AtomicFactRecaller,
    embed_query_fn: Callable[[str], Awaitable[list[float]]],
    llm: LLMClient,
    top_k: int,
    reranker: RerankProvider | None = None,
    decider: RoundDecider | None = None,
    tuning_source: Callable[[], DeciderSettings],
    trace_path_source: Callable[[], str | None],
    image_recaller: ImageMemoryRecaller | None = None,
    image_store: ImageStore | None = None,
) -> list[SearchEpisodeItem]:
    """Multi-round search, episodes only.

    The episode arm of :func:`search_episodes_and_images_llm_multiround`, for
    callers that never surface pictures: the RL environment, the ablations and
    the pre-picture API shape. The loop, the slot budget and the ranking are
    the same one; only the second returned array is dropped.
    """
    episodes, _images = await search_episodes_and_images_llm_multiround(
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
        tuning_source=tuning_source,
        trace_path_source=trace_path_source,
        image_recaller=image_recaller,
        image_store=image_store,
    )
    return episodes


async def search_episodes_and_images_llm_multiround(
    query: str,
    *,
    owner_id: str,
    where: Filter,
    app_id: str = "default",
    project_id: str = "default",
    episode_recaller: EpisodeRecaller,
    atomic_fact_recaller: AtomicFactRecaller,
    embed_query_fn: Callable[[str], Awaitable[list[float]]],
    llm: LLMClient,
    top_k: int,
    reranker: RerankProvider | None = None,
    decider: RoundDecider | None = None,
    tuning_source: Callable[[], DeciderSettings],
    trace_path_source: Callable[[], str | None],
    image_recaller: ImageMemoryRecaller | None = None,
    image_store: ImageStore | None = None,
    question_image_ids: Sequence[str] = (),
    visual_embed_query_fn: Callable[[str], Awaitable[list[float]]] | None = None,
) -> tuple[list[SearchEpisodeItem], list[SearchImageItem]]:
    """LLM-guided iterative multi-round search (per-sub-query RRF blocks).

    Thin entry that delegates to :func:`_search_episodes_subq` — the only arm.
    Each round fuses every current sub-query's BM25+vector recall with RRF
    INDEPENDENTLY (round 0 = one original-question block), shows the decider the
    labelled blocks + accumulated core and selects core across them; the final
    injection is core-first + each sub-query's top-1 guarantee + max-RRF-score
    fill. No cross-encoder anywhere (RRF is the only ranking).

    Args:
        query: User query (also the round-0 query).
        owner_id: Owner whose memories are searched.
        where: Backend-neutral predicate (owner + request filters).
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
        tuning_source: Read settings at each existing loop/decision boundary.
        trace_path_source: Read the optional trace destination at each write boundary.
        image_recaller: Picture recall; ``None`` searches episodes only.
        image_store: Reads the thumbnails the decider looks at; required with
            ``image_recaller``, and only consulted when
            ``[decider].image_candidates_per_round`` is above 0.

    Returns:
        ``(episodes, images)`` — one shared ranking split by kind, both empty
        on an empty seed. The split is a presentation detail: the two kinds
        were ranked together, so the slots one consumed are not available to
        the other.
    """
    result = await run_llm_multiround(
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
        tuning_source=tuning_source,
        trace_path_source=trace_path_source,
        image_recaller=image_recaller,
        image_store=image_store,
        question_image_ids=question_image_ids,
        visual_embed_query_fn=visual_embed_query_fn,
    )
    return result.episodes, result.images


async def run_llm_multiround(
    query: str,
    *,
    owner_id: str,
    where: Filter,
    app_id: str = "default",
    project_id: str = "default",
    episode_recaller: EpisodeRecaller,
    atomic_fact_recaller: AtomicFactRecaller,
    embed_query_fn: Callable[[str], Awaitable[list[float]]],
    llm: LLMClient,
    top_k: int,
    reranker: RerankProvider | None = None,
    decider: RoundDecider | None = None,
    tuning_source: Callable[[], DeciderSettings],
    trace_path_source: Callable[[], str | None],
    image_recaller: ImageMemoryRecaller | None = None,
    image_store: ImageStore | None = None,
    question_image_ids: Sequence[str] = (),
    visual_embed_query_fn: Callable[[str], Awaitable[list[float]]] | None = None,
    expander: LinkExpander | None = None,
    link_slots: int = 0,
) -> MultiroundResult:
    """The multi-round loop, with what a group search needs besides the ranking.

    The same loop as :func:`search_episodes_and_images_llm_multiround`; it
    also returns the coverage, the core and the records behind it. With an
    ``expander`` and ``link_slots`` above 0, each round's new core is followed
    along its stored links and at most ``link_slots`` linked records join the
    next round (section 8.3); without one the loop is exactly the plain one.
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
        tuning_source=tuning_source,
        trace_path_source=trace_path_source,
        image_recaller=image_recaller,
        image_store=image_store,
        question_image_ids=question_image_ids,
        visual_embed_query_fn=visual_embed_query_fn,
        expander=expander,
        link_slots=link_slots,
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


async def _decide(
    decide: RoundDecider,
    query: str,
    core_so_far: str,
    rendered: str,
    n_candidates: int,
    round_idx: int,
    pictures: Sequence[Picture],
) -> RoundDecision:
    """Call the decider, passing ``pictures`` only when there are some.

    A policy written against the five-argument signature — every decider that
    existed before pictures, and an RL policy compiling its own call — keeps
    working unchanged. Passing an empty ``pictures=`` to it would be a
    ``TypeError`` for no gain, and only a round that actually shows a picture
    needs the extra argument.
    """
    if not pictures:
        return await decide(query, core_so_far, rendered, n_candidates, round_idx)
    return await decide(
        query,
        core_so_far,
        rendered,
        n_candidates,
        round_idx,
        pictures=pictures,
    )


def _content(
    prompt: str, pictures: Sequence[Picture], tune: DeciderSettings
) -> str | list[object]:
    """The decider's user message: the text, then this round's pictures.

    A text-only round keeps the plain string shape it has always sent, so a
    deployment with ``image_candidates_per_round = 0`` sends byte-identical
    requests to what it sent before pictures existed -- and to what the exp
    decider sends -- which also matters because a ``response_format`` (when one
    is opted into) plus a multimodal content array is a combination not every
    endpoint accepts.

    Each picture is introduced by a text part naming its evidence index, so the
    decider can map "what does item 7 show" onto the right image rather than
    guessing by position.
    """
    if not pictures:
        return prompt
    parts: list[object] = [{"type": "text", "text": prompt}]
    for picture in pictures:
        caption = (
            "Picture attached to the question:"
            if picture.question
            else f"Picture for evidence item {picture.label}:"
        )
        parts.append({"type": "text", "text": caption})
        part = image_url_part_from_bytes(picture.data, picture.mime)
        parts.append(
            part.model_copy(
                update={
                    "image_url": part.image_url.model_copy(
                        update={"detail": tune.image_detail}
                    )
                }
            ).model_dump()
        )
    return parts


async def _round_pictures(
    global_cands: Sequence[Candidate],
    *,
    store: ImageStore | None,
    k: int,
    edge: int,
    quality: int,
    question_image_ids: Sequence[str] = (),
) -> list[Picture]:
    """The decider's pictures: the question's, then at most ``k`` candidates.

    Candidate pictures are picked in evidence order — the fused ranking already
    put the most relevant pictures first — so the cap costs the least relevant
    ones, and those stay in the evidence as text. Every failure to load one
    degrades that picture to text instead of failing the round: the decider's
    answer is still usable without the pixels, and a round that raises would
    lose the whole search.

    Question pictures come first and are not capped. They are what the question
    is *about*, so a cap that dropped one would leave the decider answering a
    question it cannot see; they are loaded even when ``k`` is 0, because unlike
    a candidate they were attached deliberately.

    Returns an empty list when the knob is 0 and the question carries no
    picture, which is the default: a decider that was never trained on picture
    candidates is not shown any.
    """
    picked: list[Picture] = []
    if store is not None:
        for image_id in question_image_ids:
            try:
                data = await store.read_derived(image_id, edge=edge, quality=quality)
            except (FileNotFoundError, OSError, ValueError) as err:
                logger.debug(
                    "llm_multiround_question_picture_unavailable",
                    image_id=image_id,
                    error=f"{type(err).__name__}: {err}"[:160],
                )
                continue
            picked.append(
                Picture(label="", data=data, mime="image/jpeg", question=True)
            )
    if k <= 0 or store is None:
        return picked
    limit = len(picked) + k
    for gi, cand in enumerate(global_cands):
        if len(picked) >= limit:
            break
        if not _is_picture(cand):
            continue
        image_id = str(cand.metadata.get("image_id", ""))
        if not image_id:
            continue
        try:
            data = await store.read_derived(image_id, edge=edge, quality=quality)
        except (FileNotFoundError, OSError, ValueError) as err:
            logger.debug(
                "llm_multiround_picture_unavailable",
                image_id=image_id,
                error=f"{type(err).__name__}: {err}"[:160],
            )
            continue
        picked.append(Picture(label=str(gi), data=data, mime="image/jpeg"))
    return picked


def _is_picture(cand: Candidate) -> bool:
    """Whether ``cand`` is a picture memory rather than an episode."""
    return cand.metadata.get("kind") == IMAGE_KIND


def _decider_text(meta: dict) -> str:
    """Text shown to the decider for one candidate: the full episode body.

    The decider has to judge the text the answering model will read. ``summary`` is
    not that on most stores -- everalgo falls back to the body's first 200 characters
    when the extractor writes no summary -- and deciding from that preview cost 9.3 pp
    of LoCoMo core-only accuracy (2026-08-06). There is deliberately no switch back to
    the preview (decided 2026-09-25): the training and evaluation code the decider
    policy was built with always shows the body. Stores also disagree on which column
    holds the body (``episode`` on some, ``summary`` with ``episode`` empty on others),
    so take the longer one: that is the body on both layouts without knowing which
    store is loaded. A picture memory has no ``episode``, so its description shows.
    """
    summary = str(meta.get("summary", "") or "")
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
    pictures: Sequence[Picture] = (),
    picture_candidates: int = 0,
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
    if pictures or picture_candidates:
        # What the decider actually *saw* as pixels, as opposed to as text.
        # Without this, "the decider read the thumbnails" and "the decider read
        # the descriptions" produce identical traces -- and the whole point of
        # the picture path is that they are different.
        record["pictures"] = {
            "attached": len(pictures),
            "question": sum(1 for picture in pictures if picture.question),
            "evidence_labels": [
                picture.label for picture in pictures if not picture.question
            ],
            "candidates": picture_candidates,
            "overflow_as_text": max(
                picture_candidates - sum(1 for p in pictures if not p.question), 0
            ),
            "bytes": [len(picture.data) for picture in pictures],
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
    where: Filter,
    app_id: str = "default",
    project_id: str = "default",
    episode_recaller: EpisodeRecaller,
    atomic_fact_recaller: AtomicFactRecaller,
    embed_query_fn: Callable[[str], Awaitable[list[float]]],
    llm: LLMClient,
    top_k: int,
    reranker: RerankProvider | None,
    decider: RoundDecider | None,
    tuning_source: Callable[[], DeciderSettings],
    trace_path_source: Callable[[], str | None],
    image_recaller: ImageMemoryRecaller | None = None,
    image_store: ImageStore | None = None,
    question_image_ids: Sequence[str] = (),
    visual_embed_query_fn: Callable[[str], Awaitable[list[float]]] | None = None,
    expander: LinkExpander | None = None,
    link_slots: int = 0,
) -> MultiroundResult:
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

    Relational expansion (section 8.3): when the loop continues past a round,
    ``expander`` follows the stored links of that round's new core, and at most
    ``link_slots`` of the linked records join the next round -- inside a block
    labelled with the sub-query that surfaced their anchor, rendered like any
    candidate, so the decider's input format does not change. A linked record
    already in the round, or already core, is not shown twice.
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
    decide = decider or LLMRoundDecider(llm, tuning_source=tuning_source)

    core_by_id: dict[str, Candidate] = {}  # accumulated core (best-score candidate)
    core_ids: set[str] = set()
    core_order: list[str] = []  # core insertion order (drives core-first)
    core_source: dict[str, str] = {}  # id -> sub-query that first cored it (#4)
    subq_hits: dict[str, dict[str, float]] = {}  # id -> {sub_query: rrf_score}
    cand_by_id: dict[str, Candidate] = {}  # id -> best-score candidate ever seen
    episodes_by_id: dict[str, Stored[Episode]] = {}  # id -> recalled record
    images_by_id: dict[str, Stored[ImageMemory]] = {}  # id -> recalled picture
    guarantee: list[tuple[str, str]] = []  # (sub_query, top id) per block, in order
    subqueries_seen: list[str] = []
    no_new_core_streak = 0
    cur_queries = [query]  # round 0 seeds with the user question
    linked: list[tuple[str, Stored[Episode]]] = []  # (anchor label, record)
    stop_reason = "budget"

    tune = tuning_source()
    for round_idx in range(tune.max_rounds):
        _ts_round_start = time.monotonic()
        round_kind = "seed" if round_idx == 0 else "subquery"
        block_topk = tune.seed_topk if round_idx == 0 else tune.subq_topk
        vecs = await asyncio.gather(*[embed_query_fn(q) for q in cur_queries])
        # Recall every sub-query concurrently (sparse + dense). The KEY design point:
        # each sub-query's pair is fused with RRF INDEPENDENTLY below, not
        # merged into one pool first — so a facet-gold ranked #1 for one sub-query
        # is not diluted by the other sub-queries' consensus hits.
        # Picture queries go to BOTH picture encoders when a vision embedder is
        # configured: the text model's space (description) and the vision
        # model's own (bytes). Their cosines are incomparable, so each is a
        # separate ranked list, fused by rank below.
        visual_vecs: list[list[float]] = []
        if visual_embed_query_fn is not None and image_recaller is not None:
            visual_vecs = await asyncio.gather(
                *[visual_embed_query_fn(q) for q in cur_queries]
            )
        recalls = await asyncio.gather(
            *[
                asyncio.gather(
                    episode_recaller.sparse_recall(q, where, limit=_SEED_CANDIDATES),
                    episode_recaller.dense_recall(v, where, limit=_SEED_CANDIDATES)
                    if v
                    else _no_recall(),
                    image_recaller.sparse_recall(q, where, limit=_SEED_CANDIDATES)
                    if image_recaller is not None
                    else _no_recall(),
                    image_recaller.dense_recall(v, where, limit=_SEED_CANDIDATES)
                    if image_recaller is not None and v
                    else _no_recall(),
                    image_recaller.visual_recall(
                        visual_vecs[i], where, limit=_SEED_CANDIDATES
                    )
                    if image_recaller is not None and visual_vecs
                    else _no_recall(),
                )
                for i, (q, v) in enumerate(zip(cur_queries, vecs, strict=True))
            ]
        )
        _ts_recall_done = time.monotonic()
        blocks: list[tuple[str, list[Candidate]]] = []
        block_meta: list[dict[str, object]] = []
        for q, (
            sparse_result,
            dense_result,
            image_sparse,
            image_dense,
            image_visual,
        ) in zip(cur_queries, recalls, strict=True):
            episodes_by_id.update(sparse_result.stored_by_id)
            episodes_by_id.update(dense_result.stored_by_id)
            images_by_id.update(image_sparse.stored_by_id)
            images_by_id.update(image_dense.stored_by_id)
            images_by_id.update(image_visual.stored_by_id)
            r_sparse = [*sparse_result.candidates, *image_sparse.candidates]
            # A picture has two vector routes; an episode has one. Fusing the
            # picture's two vector lists FIRST keeps the per-kind weight equal
            # — otherwise an image that both embedders liked would collect
            # three votes against an episode's two and win blocks by channel
            # count rather than by relevance. Inside the image pair, RRF is
            # the right operator: the two spaces are different models, so
            # their raw scores are not comparable.
            image_dense_fused = (
                rrf(image_dense.candidates, image_visual.candidates, k=tune.rrf_k)
                if image_visual.candidates
                else image_dense.candidates
            )
            r_dense = [*dense_result.candidates, *image_dense_fused]
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
                    "n_visual": len(image_visual.candidates),
                    "topk_kept": len(block),
                    "sparse": _trace_cand_pool(r_sparse),
                    "dense": _trace_cand_pool(r_dense),
                    "visual": _trace_cand_pool(image_visual.candidates),
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
            # Empty seed: nothing to rank or decide over.
            return MultiroundResult([], [], "none", [], {})
        n_linked = _join_linked(
            blocks,
            linked,
            core_ids=core_ids,
            subq_hits=subq_hits,
            cand_by_id=cand_by_id,
            episodes_by_id=episodes_by_id,
        )
        linked = []

        # Render the blocked decider view + CORE-SO-FAR (global index across blocks).
        rendered, global_cands, index_meta = _render_blocks(blocks)
        core_so_far = _render_core_so_far(core_order, core_source, cand_by_id)
        pictures = await _round_pictures(
            global_cands,
            store=image_store,
            k=tune.image_candidates_per_round,
            edge=tune.image_thumbnail_edge,
            quality=tune.image_quality,
            question_image_ids=question_image_ids,
        )
        _ts_decide_start = time.monotonic()
        decision = await _decide(
            decide,
            query,
            core_so_far,
            rendered,
            len(global_cands),
            round_idx,
            pictures,
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

        dump_path = trace_path_source()
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
                    pictures=pictures,
                    picture_candidates=sum(
                        1 for cand in global_cands if _is_picture(cand)
                    ),
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
            linked=n_linked,
        )

        # Stop conditions, in the design's priority order.
        if not decision.queries:  # (1) decider is done (no gaps left)
            stop_reason = "decider"
            break
        # (2) coverage saturated — never before at least one follow-up round.
        if round_idx >= 1 and no_new_core_streak >= tune.no_new_core_patience:
            stop_reason = "patience"
            break
        cur_queries = decision.queries  # else continue; (3) the loop caps rounds
        if expander is not None and link_slots > 0 and round_idx + 1 < tune.max_rounds:
            seen_records = {**images_by_id, **episodes_by_id}
            anchors = [
                seen_records[str(added["id"])]
                for added in core_added
                if str(added["id"]) in seen_records
            ]
            if anchors:
                linked = [
                    (core_source.get(anchor_id, "original question"), record)
                    for anchor_id, record in await expander(anchors)
                ][:link_slots]

    coverage: Coverage = (
        "none"
        if not core_ids
        else ("complete" if stop_reason == "decider" else "partial")
    )
    records: dict[str, Stored[Episode] | Stored[ImageMemory]] = {
        **images_by_id,
        **episodes_by_id,
    }
    if not cand_by_id and not core_by_id:
        return MultiroundResult([], [], coverage, list(core_order), records)
    episodes, images = _finalize_injection(
        owner_id=owner_id,
        question=query,
        top_k=top_k,
        core_ids=core_ids,
        core_order=core_order,
        cand_by_id=cand_by_id,
        episodes_by_id=episodes_by_id,
        images_by_id=images_by_id,
        subq_hits=subq_hits,
        guarantee=guarantee,
        subqueries_seen=subqueries_seen,
        tuning_source=tuning_source,
        trace_path_source=trace_path_source,
        image_thumbnail_edge=tune.image_thumbnail_edge,
    )
    return MultiroundResult(episodes, images, coverage, list(core_order), records)


def _join_linked(
    blocks: list[tuple[str, list[Candidate]]],
    linked: Sequence[tuple[str, Stored[Episode]]],
    *,
    core_ids: set[str],
    subq_hits: dict[str, dict[str, float]],
    cand_by_id: dict[str, Candidate],
    episodes_by_id: dict[str, Stored[Episode]],
) -> int:
    """Put last round's linked records into this round's blocks; how many joined.

    Each joins the block labelled with the sub-query that surfaced its anchor
    (a new block with that label when the round has none), after the block's
    own candidates. A record already in the round, or already core, is not
    shown again. A linked record has no RRF score of its own: it scores 0, so
    it reaches the final injection only when the decider makes it core.
    """
    seen = {c.id for _, block in blocks for c in block} | core_ids
    joined = 0
    for label, record in linked:
        if record.ref.id in seen:
            continue
        seen.add(record.ref.id)
        cand = to_candidate(record, score=0.0, source="other")  # type: ignore[arg-type]
        block = next((b for lbl, b in blocks if lbl == label), None)
        if block is None:
            block = []
            blocks.append((label, block))
        block.append(cand)
        subq_hits.setdefault(cand.id, {})[label] = 0.0
        cand_by_id.setdefault(cand.id, cand)
        episodes_by_id.setdefault(cand.id, record)
        joined += 1
    return joined


def _finalize_injection(
    *,
    owner_id: str,
    question: str,
    top_k: int,
    core_ids: set[str],
    core_order: list[str],
    cand_by_id: dict[str, Candidate],
    episodes_by_id: dict[str, Stored[Episode]],
    images_by_id: dict[str, Stored[ImageMemory]],
    subq_hits: dict[str, dict[str, float]],
    guarantee: list[tuple[str, str]],
    subqueries_seen: list[str],
    tuning_source: Callable[[], DeciderSettings],
    trace_path_source: Callable[[], str | None],
    image_thumbnail_edge: int = 448,
) -> tuple[list[SearchEpisodeItem], list[SearchImageItem]]:
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
        if not tuning_source().core_overflow and len(chosen) >= top_k:
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

    # The two kinds shared one ranking and one ``top_k`` budget; splitting them
    # here is a presentation step only. Each keeps its rank, so a caller that
    # interleaves the two arrays by score sees the order they competed in.
    ordered_cands = [cand_by_id[cid] for cid in chosen if cid in cand_by_id]
    episode_cands = [c for c in ordered_cands if not _is_picture(c)]
    image_cands = [c for c in ordered_cands if _is_picture(c)]
    episodes = shape_episodes(episode_cands, episodes_by_id)
    images = shape_images(
        image_cands,
        images_by_id,
        thumbnail_edge=image_thumbnail_edge,
    )

    dump_path = trace_path_source()
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
                    "n_images": len(images),
                    "subqueries_seen": subqueries_seen,
                },
            },
        )
    return episodes, images


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


async def _no_recall() -> RecallResult[Episode]:
    return RecallResult.empty()
