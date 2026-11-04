"""Rerank module — LLM rerank tool + unified facade pipeline + dispatch."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict

from everos.algo.llm import LLMMessage
from everos.algo.rank import fusion
from everos.algo.rank._replies import _call_llm_for_rerank
from everos.algo.rank.prompts.rerank_user import RERANK_USER_PROMPT
from everos.algo.types import Candidate, RankInput, RankOutput, ScoredItem
from everos.algo.utils import (
    InvalidInputError,
    append_custom_instructions,
    render_prompt,
)

if TYPE_CHECKING:
    from everos.algo.llm.protocols import LLMClient
    from everos.algo.rank._replies import _RankedCandidate

__all__ = [
    "DEFAULT_RANK_CONFIG",
    "FusionMode",
    "RankConfig",
    "_basic_arank",
    "arerank",
]


logger = logging.getLogger(__name__)

FusionMode = Literal["rrf", "lr", "vector_anchored"]
"""Top-level ranking strategy — parallel to enterprise ``method`` parameter.
"""


class RankConfig(BaseModel):
    """Rank pipeline configuration. Frozen value object."""

    model_config = ConfigDict(frozen=True)

    fusion_mode: FusionMode = "rrf"
    rrf_k: int = 60

    alpha: float = 1.0

    va_saturation_k: float = 5.0  # vector_anchored: saturation scale for BM25 scores
    va_alpha: float = 0.7  # vector_anchored: blend weight (1=vec-only, 0=bm25-only)


DEFAULT_RANK_CONFIG = RankConfig()


async def arerank(
    items: Sequence[Candidate],
    *,
    query: str,
    prompt: str,
    top_k: int,
    llm: LLMClient,
    custom_instructions: str | None = None,
) -> list[Candidate]:
    """LLM-driven rerank.

    Sends one ``[system, user]`` pair: ``prompt`` rendered as the system message, closed by the
    ``custom_instructions`` block when there is one, then the query and the candidates in
    :data:`~everos.algo.rank.prompts.rerank_user.RERANK_USER_PROMPT`.

    Args:
        items: Candidates produced by fusion (+ optional weighting).
        query: User query to rerank against, sent in the user message's ``<user_query>``.
        prompt: The system message: the rerank rules, whose only placeholder is ``{{top_k}}``. Callers typically
            pass one of the constants in ``everos.algo.rank.prompts.{en,zh}``.
        top_k: Maximum number of items to return.
        llm: LLM client (required — bound at the ranker instance level).
        custom_instructions: The caller's instructions, appended after the rendered ``prompt`` as a
            ``<custom_instructions>`` block, always the system message's last; it adds to the rules, never replaces
            them. Placed verbatim, never rendered as a template, so ``{{top_k}}`` in it stays literal. None or a blank
            string adds nothing.

    Returns:
        Up to ``top_k`` Candidates, sorted descending by the LLM-assigned score.
        Each result has ``.score`` overwritten with the LLM score; the original
        fusion score is preserved in ``metadata["fusion_score"]``.

    Raises:
        ValueError: ``prompt`` names a placeholder other than ``{{top_k}}``, or still uses the single-brace
            placeholder syntax.
        LLMOutputError: If the reply is not a valid rerank output. The client's own exceptions propagate unchanged.
    """
    if not items:
        return []

    candidates_payload = [
        {"id": item.id, "score": item.score, **item.metadata} for item in items
    ]
    messages = [
        LLMMessage(
            role="system",
            content=append_custom_instructions(
                render_prompt(prompt, None, top_k=top_k), custom_instructions
            ),
        ),
        LLMMessage(
            role="user",
            content=render_prompt(
                RERANK_USER_PROMPT,
                None,
                query=query,
                candidates_json=json.dumps(
                    candidates_payload, ensure_ascii=False, default=str
                ),
            ),
        ),
    ]
    reply = await _call_llm_for_rerank(llm, messages)
    return _apply_rerank_scores(items, reply.ranked, top_k)


def _apply_rerank_scores(
    items: Sequence[Candidate],
    ranked: list[_RankedCandidate],
    top_k: int,
) -> list[Candidate]:
    """Apply LLM-assigned scores to candidates, sort, truncate; an id the LLM made up is skipped."""
    by_id = {item.id: item for item in items}
    out: list[Candidate] = []
    for ranked_item in ranked:
        original = by_id.get(ranked_item.id)
        if original is None:
            continue
        out.append(
            original.model_copy(
                update={
                    "score": ranked_item.score,
                    "metadata": {**original.metadata, "fusion_score": original.score},
                }
            )
        )

    out.sort(key=lambda c: c.score, reverse=True)
    return out[:top_k]


type _AsyncRanker = Callable[..., Awaitable[RankOutput]]


class _RankerSpec(NamedTuple):
    """Per-facade config entry in ``_ALGO_REGISTRY`` — arank target, supported modes, rerank prompt, item label."""

    arank: _AsyncRanker
    modes: tuple[FusionMode, ...]
    rerank_prompt: str
    item_type: Literal["case", "skill", "episode", "profile"]


_ALGO_REGISTRY: dict[str, _RankerSpec] = {}


async def _basic_arank(
    rank_input: RankInput,
    *,
    config: RankConfig = DEFAULT_RANK_CONFIG,
    llm: LLMClient | None = None,
    custom_instructions: str | None = None,
    enable_rerank: bool = False,
    rerank_top_k: int | None = None,
) -> RankOutput:
    """Unified pipeline dispatched to by all facade rankers.

    Branches on ``config.fusion_mode``: ``rrf`` and ``lr`` are the two Phase-1 fusion strategies.
    Phase 5 LLM rerank is an optional shared final step.

    ``rerank_top_k`` lets the Phase-5 rerank narrow further than ``rank_input.top_k``:
    fusion produces a wider candidate pool (``top_k``), the LLM rescores all of them, and only the top
    ``rerank_top_k`` items survive. When ``None`` (default), rerank keeps ``rank_input.top_k`` items — the
    historical behaviour.

    ``custom_instructions`` reaches the rerank call only: the facade's bundled rerank rules, then the caller's
    ``<custom_instructions>`` block as the system message's last (see :func:`arerank`).

    Raises:
        KeyError: ``rank_input.memory_type`` not in ``_ALGO_REGISTRY``.
        InvalidInputError: ``enable_rerank`` is set without an ``llm``, or ``fusion_mode`` is unsupported for the
            facade.
    """
    spec = _ALGO_REGISTRY.get(rank_input.memory_type)
    if spec is None:
        raise KeyError(
            f"No ranker for memory_type={rank_input.memory_type!r}; registered: {sorted(_ALGO_REGISTRY)}"
        )
    if config.fusion_mode not in spec.modes:
        raise InvalidInputError(
            f"config.fusion_mode={config.fusion_mode!r} is not supported by the {rank_input.memory_type!r} facade: "
            f"set it to one of {list(spec.modes)}"
        )

    item_type = spec.item_type
    rerank_prompt = spec.rerank_prompt
    sparse, dense = rank_input.sparse_candidates, rank_input.dense_candidates

    scored: list[ScoredItem]
    if not sparse and not dense:
        return RankOutput(
            items=[], metadata={"stage": item_type, "stop_reason": "no_candidates"}
        )

    fused: list[Candidate]
    if not sparse:
        fused = list(dense)
    elif not dense:
        fused = list(sparse)
    elif config.fusion_mode == "lr":
        fused = fusion.lr(dense, sparse)
    elif config.fusion_mode == "vector_anchored":
        fused = fusion.vector_anchored(
            dense, sparse, saturation_k=config.va_saturation_k, alpha=config.va_alpha
        )
    else:  # rrf
        fused = fusion.rrf(dense, sparse, k=config.rrf_k)

    fused.sort(key=lambda c: c.score, reverse=True)
    fused = fused[: rank_input.top_k]

    scored = [
        ScoredItem(
            id=c.id, score=c.score, item_type=item_type, metadata=dict(c.metadata)
        )
        for c in fused
    ]
    meta: dict[str, Any] = {"stage": item_type, "fusion_mode": config.fusion_mode}

    if enable_rerank and scored:
        if llm is None:
            raise InvalidInputError(
                "enable_rerank=True was passed without an llm: pass the llm client, or set enable_rerank=False"
            )
        with_query = [
            Candidate(
                id=item.id,
                score=item.score,
                metadata={
                    **item.metadata,
                    "item_type": item.item_type,
                    "parent_episode_id": item.parent_episode_id,
                },
            )
            for item in scored
        ]
        effective_rerank_top_k = (
            rerank_top_k if rerank_top_k is not None else rank_input.top_k
        )
        reranked = await arerank(
            with_query,
            query=rank_input.query,
            prompt=rerank_prompt,
            top_k=effective_rerank_top_k,
            llm=llm,
            custom_instructions=custom_instructions,
        )
        by_id = {it.id: it for it in scored}
        scored = [
            by_id[c.id].model_copy(
                update={
                    "score": c.score,
                    "metadata": {**by_id[c.id].metadata, **c.metadata},
                }
            )
            for c in reranked
            if c.id in by_id
        ]
        meta = {**meta, "reranked": True, "rerank_top_k": effective_rerank_top_k}

    return RankOutput(items=scored, metadata=meta)


async def _arank(rank_input: RankInput, **kwargs: Any) -> RankOutput:
    """Dispatch by ``memory_type`` to the registered facade ranker.

    Raises:
        KeyError: ``rank_input.memory_type`` not in ``_ALGO_REGISTRY``.
    """
    spec = _ALGO_REGISTRY.get(rank_input.memory_type)
    if spec is None:
        raise KeyError(
            f"No ranker for memory_type={rank_input.memory_type!r}; registered: {sorted(_ALGO_REGISTRY)}"
        )
    return await spec.arank(rank_input, **kwargs)
