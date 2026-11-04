"""Episodic ranker facade — thin wrapper over ``rerank._basic_arank``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from everos.algo.rank.rerank import DEFAULT_RANK_CONFIG, RankConfig, _basic_arank

if TYPE_CHECKING:
    from everos.algo.llm.protocols import LLMClient
    from everos.algo.types import RankInput, RankOutput

__all__ = ["EpisodicRanker"]


class EpisodicRanker:
    """Class-style facade for the episodic ranker.

    The LLM client is bound to the instance at construction time.
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def arank(
        self,
        rank_input: RankInput,
        *,
        config: RankConfig = DEFAULT_RANK_CONFIG,
        custom_instructions: str | None = None,
        enable_rerank: bool = False,
        rerank_top_k: int | None = None,
    ) -> RankOutput:
        """Episodic ranker — see ``rerank._basic_arank`` for the pipeline body.

        Args:
            rank_input: Query + candidate sets for episodic memory ranking.
            config: Fusion mode and hyperparameters.
            custom_instructions: The caller's instructions for the rerank call, appended to the built-in rerank
                rules as a ``<custom_instructions>`` block, always the system message's last; it adds to the rules,
                never replaces them. Placed verbatim, never rendered as a template. None or a blank string adds
                nothing; no effect unless ``enable_rerank`` runs.
            enable_rerank: When ``True``, run the LLM rerank stage after fusion.
            rerank_top_k: When set, Phase-5 LLM rerank truncates to this count instead of
                ``rank_input.top_k`` — lets fusion produce a wider candidate pool that the LLM
                then narrows.

        Returns:
            Ranked and optionally LLM-reranked episodic items.
        """
        return await _basic_arank(
            rank_input,
            config=config,
            llm=self._llm,
            custom_instructions=custom_instructions,
            enable_rerank=enable_rerank,
            rerank_top_k=rerank_top_k,
        )


async def arank(
    rank_input: RankInput,
    *,
    config: RankConfig = DEFAULT_RANK_CONFIG,
    llm: LLMClient | None = None,
    custom_instructions: str | None = None,
    enable_rerank: bool = False,
    rerank_top_k: int | None = None,
) -> RankOutput:
    """Episodic module-level ranker — delegates to ``rerank._basic_arank``."""
    return await _basic_arank(
        rank_input,
        config=config,
        llm=llm,
        custom_instructions=custom_instructions,
        enable_rerank=enable_rerank,
        rerank_top_k=rerank_top_k,
    )
