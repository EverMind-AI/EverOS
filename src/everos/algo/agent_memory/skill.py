"""AgentSkillExtractor — incremental skill maintenance for one cluster.

Operation atoms (``_apply_add`` / ``_apply_update`` / ``_evaluate_maturity`` / formatters) live in
:mod:`everos.algo.agent_memory.skill_ops`; this module holds the public :class:`AgentSkillExtractor`.

Return contract: ``list[AgentSkill]`` — caller decodes add / update / retire via
``id ∈ existing_relevant_skills`` + ``confidence < retire_confidence``.
The caller pre-filters ``existing_relevant_skills`` to the relevant subset before passing in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from everos.algo.agent_memory._replies import (
    _call_llm_for_skill_extract,
    _SkillAddOp,
    _SkillUpdateOp,
)
from everos.algo.agent_memory.gate import admitted_case, should_extract_skill
from everos.algo.agent_memory.prompts.skill_failure import (
    AGENT_SKILL_FAILURE_EXTRACT_SYSTEM_PROMPT,
)
from everos.algo.agent_memory.prompts.skill_success import (
    AGENT_SKILL_ADMITTED_EXTRACT_SYSTEM_PROMPT,
    AGENT_SKILL_SUCCESS_EXTRACT_SYSTEM_PROMPT,
    SKILL_EXTRACT_USER_PROMPT,
)
from everos.algo.agent_memory.reasons import (
    OpOutcome,
    SkillExtractionResult,
    SkillSkipReason,
)
from everos.algo.agent_memory.skill_ops import (
    _apply_add,
    _apply_update,
    _format_cases,
    _format_existing_skills,
)
from everos.algo.llm import LLMMessage
from everos.algo.utils import render_prompt

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everos.algo.llm.protocols import LLMClient
    from everos.algo.types import AgentCase, AgentSkill

logger = logging.getLogger(__name__)


__all__ = [
    # Re-exported system prompts — monkey-patch at startup to override the rules the LLM is given
    "AGENT_SKILL_FAILURE_EXTRACT_SYSTEM_PROMPT",
    "AGENT_SKILL_SUCCESS_EXTRACT_SYSTEM_PROMPT",
    "AgentSkillExtractor",
]


@dataclass(frozen=True)
class _SkillCfg:
    """Internal threshold bag — packed from aextract kwargs; not part of the public API."""

    # Cases below this quality score short-circuit to [] without calling the LLM.
    skip_quality_threshold: float = 0.2
    # Maturity scoring is implemented but default-skipped: current retrieval doesn't consult
    # maturity_score. Set False to enable; useful for future retrieval paths that filter on it.
    skip_maturity_scoring: bool = True
    maturity_threshold: float = 0.6  # only consulted when skip_maturity_scoring=False
    retire_confidence: float = 0.1
    failure_quality_threshold: float = 0.5
    max_description_tokens: int = 400
    max_content_tokens: int = 5000
    # Below: change-ratio bands for maturity re-eval; inert when skip_maturity_scoring=True.
    maturity_trivial_change_ratio: float = 0.2
    maturity_reeval_change_ratio: float = 0.4


class AgentSkillExtractor:
    """Aggregate one new AgentCase into incremental skill operations for a cluster.

    Returns ``list[AgentSkill]`` with add / update / retire signals (§5.2 encoding); caller persists.
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def aextract(
        self,
        case: AgentCase,
        *,
        existing_relevant_skills: Sequence[AgentSkill],
        supporting_cases: Sequence[AgentCase],
        skip_quality_threshold: float = 0.2,
        skip_maturity_scoring: bool = True,
        maturity_threshold: float = 0.6,
        retire_confidence: float = 0.1,
        failure_quality_threshold: float = 0.5,
        max_description_tokens: int = 400,
        max_content_tokens: int = 5000,
        maturity_trivial_change_ratio: float = 0.2,
        maturity_reeval_change_ratio: float = 0.4,
    ) -> list[AgentSkill]:
        """Integrate ``case`` into the skill set; return add / update / retire entries.

        Thin wrapper over :meth:`aextract_with_reason` — use that one when an empty (or
        smaller-than-expected) result needs to be explained rather than merely observed.

        See :meth:`aextract_with_reason` for the full argument reference.

        Raises:
            LLMOutputError: If the skill extract reply, or a maturity reply when maturity scoring runs, is not valid
                for its call.
        """
        result = await self.aextract_with_reason(
            case,
            existing_relevant_skills=existing_relevant_skills,
            supporting_cases=supporting_cases,
            skip_quality_threshold=skip_quality_threshold,
            skip_maturity_scoring=skip_maturity_scoring,
            maturity_threshold=maturity_threshold,
            retire_confidence=retire_confidence,
            failure_quality_threshold=failure_quality_threshold,
            max_description_tokens=max_description_tokens,
            max_content_tokens=max_content_tokens,
            maturity_trivial_change_ratio=maturity_trivial_change_ratio,
            maturity_reeval_change_ratio=maturity_reeval_change_ratio,
        )
        return result.skills

    async def aextract_with_reason(
        self,
        case: AgentCase,
        *,
        existing_relevant_skills: Sequence[AgentSkill],
        supporting_cases: Sequence[AgentCase],
        skip_quality_threshold: float = 0.2,
        skip_maturity_scoring: bool = True,
        maturity_threshold: float = 0.6,
        retire_confidence: float = 0.1,
        failure_quality_threshold: float = 0.5,
        max_description_tokens: int = 400,
        max_content_tokens: int = 5000,
        maturity_trivial_change_ratio: float = 0.2,
        maturity_reeval_change_ratio: float = 0.4,
    ) -> SkillExtractionResult:
        """Integrate ``case`` into the skill set, reporting what happened to every proposed operation.

        Same algorithm as :meth:`aextract`. The difference is the return shape: one
        :class:`OpOutcome` per LLM-proposed operation (so ``len(outcomes) == len(operations)``),
        each holding either the resulting skill or the reason it was dropped, plus a separate
        ``pre_reason`` / ``pre_detail`` pair for the quality short-circuit that fires before any
        operation exists.

        Args:
            case: The newly extracted ``AgentCase`` to integrate.
            existing_relevant_skills: Pre-filtered by the caller (typically top-K cosine). Pass ``[]`` when none.
            supporting_cases: AgentCase records referenced by ``existing_relevant_skills[*].source_case_ids``;
                the algorithm builds an ``id → AgentCase`` dict and renders matched cases under each skill in
                the prompt. Pass ``[]`` when no supporting cases are available.
            skip_quality_threshold: Cases with ``quality_score < this`` short-circuit to ``[]`` without LLM call. Default 0.2.
            skip_maturity_scoring: Skip LLM maturity scoring (default True); set False to enable.
            maturity_threshold: Minimum score to mark a skill as mature (consulted when not skipping).
            retire_confidence: Confidence below which a skill is marked for retirement.
            failure_quality_threshold: Quality score below which the failure prompt branch is used.
            max_description_tokens: Token budget for skill description in the prompt.
            max_content_tokens: Token budget for skill content in the prompt.
            maturity_trivial_change_ratio: Change ratio below which maturity re-eval is skipped.
            maturity_reeval_change_ratio: Change ratio above which maturity is reset and re-evaluated.

        Raises:
            LLMOutputError: If the skill extract reply, or a maturity reply when maturity scoring runs, is not valid
                for its call.
        """
        cfg = _SkillCfg(
            skip_quality_threshold=skip_quality_threshold,
            skip_maturity_scoring=skip_maturity_scoring,
            maturity_threshold=maturity_threshold,
            retire_confidence=retire_confidence,
            failure_quality_threshold=failure_quality_threshold,
            max_description_tokens=max_description_tokens,
            max_content_tokens=max_content_tokens,
            maturity_trivial_change_ratio=maturity_trivial_change_ratio,
            maturity_reeval_change_ratio=maturity_reeval_change_ratio,
        )
        admission = admitted_case(case)
        if admission is None and not should_extract_skill(
            case.quality_score,
            skip_quality_threshold=cfg.skip_quality_threshold,
        ):
            logger.debug(
                "skipping skill extraction: case.quality_score=%.2f < skip_quality_threshold=%.2f",
                case.quality_score,
                cfg.skip_quality_threshold,
            )
            return SkillExtractionResult(
                SkillSkipReason.CASE_QUALITY_BELOW_THRESHOLD,
                {
                    "quality": case.quality_score,
                    "threshold": cfg.skip_quality_threshold,
                },
                [],
            )

        client = self._llm

        existing_list = list(existing_relevant_skills)

        # Format inputs for the LLM prompt
        new_case_json = _format_cases([case])
        existing_skills_json = _format_existing_skills(
            existing_list,
            supporting_cases=supporting_cases,
            max_description_tokens=cfg.max_description_tokens,
            max_content_tokens=cfg.max_content_tokens,
        )

        # Success-vs-failure prompt selection based on case.quality_score
        max_quality = case.quality_score
        if admission is not None:
            system = AGENT_SKILL_ADMITTED_EXTRACT_SYSTEM_PROMPT
        elif max_quality < cfg.failure_quality_threshold:
            system = AGENT_SKILL_FAILURE_EXTRACT_SYSTEM_PROMPT
            logger.debug(
                "using failure prompt (max_quality=%.2f < %.2f)",
                max_quality,
                cfg.failure_quality_threshold,
            )
        else:
            system = AGENT_SKILL_SUCCESS_EXTRACT_SYSTEM_PROMPT
            logger.debug(
                "using success prompt (max_quality=%.2f >= %.2f)",
                max_quality,
                cfg.failure_quality_threshold,
            )

        # Single LLM call; the host owns retries.
        user = render_prompt(
            SKILL_EXTRACT_USER_PROMPT,
            None,
            new_case_json=new_case_json,
            existing_skills_json=existing_skills_json,
        )
        logger.debug(
            "incremental extraction: case=%s, new_cases=1, existing_relevant_skills=%d",
            case.id,
            len(existing_list),
        )
        reply = await _call_llm_for_skill_extract(
            client,
            [
                LLMMessage(role="system", content=system),
                LLMMessage(role="user", content=user),
            ],
        )
        if reply.update_note:
            logger.debug("LLM update_note: %s", reply.update_note)

        # Apply ops in-memory — one OpOutcome per proposed op, so len(outcomes) == len(operations)
        outcomes: list[OpOutcome] = []
        processed_indices: set[int] = set()
        source_case_ids = [case.id] if case.id else []

        for op_index, op in enumerate(reply.operations):
            if isinstance(op, _SkillAddOp):
                outcomes.append(
                    await _apply_add(
                        op.data,
                        source_case_ids,
                        op_index=op_index,
                        client=client,
                        cfg=cfg,
                    )
                )
            elif isinstance(op, _SkillUpdateOp):
                outcomes.append(
                    await _apply_update(
                        op,
                        existing_list,
                        source_case_ids,
                        source_quality=max_quality,
                        op_index=op_index,
                        client=client,
                        cfg=cfg,
                        processed_indices=processed_indices,
                    )
                )
            else:
                outcomes.append(
                    OpOutcome(op_index, None, SkillSkipReason.OP_ACTION_NONE, {})
                )

        applied = sum(1 for o in outcomes if o.skill is not None)
        logger.info(
            "case=%s ops applied: total=%d of %d proposed (adds + updates + retires per §5.2 encoding)",
            case.id,
            applied,
            len(outcomes),
        )
        return SkillExtractionResult(None, {}, outcomes)
