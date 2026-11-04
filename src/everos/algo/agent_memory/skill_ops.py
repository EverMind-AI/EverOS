"""Operation atoms for :class:`AgentSkillExtractor` — add, update, maturity scoring, and formatting helpers."""

from __future__ import annotations

import json
import logging
import re
import uuid
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from everos.algo.agent_memory._replies import _call_llm_for_maturity
from everos.algo.agent_memory.prompts.skill_maturity import (
    AGENT_SKILL_MATURITY_SCORE_SYSTEM_PROMPT,
    AGENT_SKILL_MATURITY_SCORE_USER_PROMPT,
)
from everos.algo.agent_memory.reasons import OpOutcome, SkillSkipReason
from everos.algo.llm import LLMMessage
from everos.algo.types import AgentCase, AgentSkill
from everos.algo.utils import json_default, render_prompt, truncate_text

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everos.algo.agent_memory._replies import _SkillAddData, _SkillUpdateOp
    from everos.algo.agent_memory.skill import _SkillCfg
    from everos.algo.llm.protocols import LLMClient

logger = logging.getLogger(__name__)


__all__ = [
    "_apply_add",
    "_apply_update",
    "_content_change_ratio",
    "_evaluate_maturity",
    "_format_cases",
    "_format_existing_skills",
    "_is_hypothesis_promotion",
    "_is_skill_content_sufficient",
]


# ── Formatting helpers ──────────────────────────────────────────────────────────────────────────────


def _format_cases(case_records: Sequence[AgentCase]) -> str:
    """Serialize AgentCase instances to a JSON string for the LLM prompt."""
    formatted: list[dict[str, Any]] = []
    for rec in case_records:
        entry: dict[str, Any] = {
            "timestamp": rec.timestamp,
            "task_intent": rec.task_intent or "",
            "approach": rec.approach or "",
            "quality_score": rec.quality_score,
        }
        if rec.key_insight:
            entry["key_insight"] = rec.key_insight
        formatted.append(entry)
    return json.dumps(formatted, ensure_ascii=False, indent=2, default=json_default)


def _summarise_case_for_prompt(
    case_record: AgentCase, max_approach_tokens: int = 200
) -> dict[str, Any]:
    """Compact case summary dict for inclusion in the skill prompt."""
    entry: dict[str, Any] = {
        "task_intent": case_record.task_intent or "",
        "quality_score": case_record.quality_score,
    }
    if case_record.key_insight:
        entry["key_insight"] = case_record.key_insight
    if case_record.approach:
        entry["approach"] = truncate_text(
            case_record.approach, max_approach_tokens, suffix="... [omitted]"
        )
    return entry


def _format_existing_skills(
    existing_records: Sequence[AgentSkill],
    supporting_cases: Sequence[AgentCase],
    *,
    max_description_tokens: int,
    max_content_tokens: int,
    max_support_cases: int = 3,
    max_approach_tokens: int = 200,
) -> str:
    """Format existing skills with index numbers and supporting case summaries for the LLM prompt."""
    if not existing_records:
        return "(empty — no existing skills)"

    case_map: dict[str, AgentCase] = {}
    for case_rec in supporting_cases:
        cid = str(case_rec.id or "")
        if cid:
            case_map[cid] = case_rec

    lines: list[str] = []
    for idx, skill_rec in enumerate(existing_records):
        item: dict[str, Any] = {
            "index": idx,
            "name": skill_rec.name or "",
            "description": truncate_text(
                skill_rec.description or "",
                max_description_tokens,
                suffix="... [omitted]",
            ),
            "content": truncate_text(
                skill_rec.content or "", max_content_tokens, suffix="... [omitted]"
            ),
            "confidence": skill_rec.confidence,
        }

        if case_map:
            source_ids = list(skill_rec.source_case_ids or [])
            matched_ids = [sid for sid in source_ids if str(sid) in case_map]
            if matched_ids:
                recent_ids = matched_ids[-max_support_cases:]
                item["supporting_case_count"] = len(matched_ids)
                item["supporting_cases"] = [
                    _summarise_case_for_prompt(
                        case_map[str(sid)], max_approach_tokens=max_approach_tokens
                    )
                    for sid in recent_ids
                ]

        lines.append(json.dumps(item, ensure_ascii=False, default=json_default))
    return "[\n" + ",\n".join(lines) + "\n]"


# ── Content validation helpers ──────────────────────────────────────────────────────────────────────


def _is_skill_content_sufficient(
    content: str, min_lines: int = 5, min_length: int = 50
) -> bool:
    """Return ``True`` iff ``content`` meets the minimum line count and character length."""
    if not content:
        return False
    stripped = content.strip()
    if len(stripped) < min_length:
        return False
    non_empty_lines = [line for line in stripped.splitlines() if line.strip()]
    return len(non_empty_lines) >= min_lines


def _content_detail(
    content: str, min_lines: int = 5, min_length: int = 50
) -> dict[str, Any]:
    """Observed-vs-required numbers behind a content-sufficiency rejection, for ``OpOutcome.detail``."""
    stripped = content.strip()
    return {
        "lines": len([line for line in stripped.splitlines() if line.strip()]),
        "chars": len(stripped),
        "min_lines": min_lines,
        "min_chars": min_length,
    }


def _content_change_ratio(old: str, new: str) -> float:
    """Return ``1 - SequenceMatcher.ratio()`` as a 0.0-1.0 change fraction."""
    if not old and not new:
        return 0.0
    if not old or not new:
        return 1.0
    ratio = SequenceMatcher(None, old, new).ratio()
    return round(1.0 - ratio, 4)


_HYPOTHESIS_HEADER = re.compile(r"^##\s+Potential Steps", re.MULTILINE)
_VERIFIED_HEADER = re.compile(r"^##\s+Steps", re.MULTILINE)


def _is_hypothesis_promotion(old_content: str, new_content: str) -> bool:
    """Detect a ``## Potential Steps`` → ``## Steps`` header promotion."""
    old_has_potential = bool(_HYPOTHESIS_HEADER.search(old_content or ""))
    new_has_steps = bool(_VERIFIED_HEADER.search(new_content or ""))
    new_has_potential = bool(_HYPOTHESIS_HEADER.search(new_content or ""))
    return old_has_potential and new_has_steps and not new_has_potential


# ── Maturity scoring ────────────────────────────────────────────────────────────────────────────────


async def _evaluate_maturity(
    *,
    name: str,
    description: str,
    content: str,
    confidence: float,
    client: LLMClient,
    cfg: _SkillCfg,
) -> float:
    """4-dimension LLM scoring (completeness / executability / evidence / clarity) normalised to [0,1] via /20.

    Short-circuits to ``1.0`` when ``cfg.skip_maturity_scoring`` is ``True`` (default — current retrieval
    doesn't consult maturity_score; kept for forward compatibility).

    Raises:
        LLMOutputError: If the reply is not a valid maturity output.
    """
    if cfg.skip_maturity_scoring:
        logger.info("maturity scoring skipped by config, returning 1.0")
        return 1.0

    user = render_prompt(
        AGENT_SKILL_MATURITY_SCORE_USER_PROMPT,
        None,
        name=name or "",
        description=description or "",
        content=content or "",
        confidence=confidence,
    )
    reply = await _call_llm_for_maturity(
        client,
        [
            LLMMessage(role="system", content=AGENT_SKILL_MATURITY_SCORE_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user),
        ],
    )
    raw_total = (
        reply.completeness + reply.executability + reply.evidence + reply.clarity
    )
    score: float = max(0.0, min(1.0, raw_total / 20.0))
    logger.info(
        "maturity evaluation: name=%r, raw=%.1f, score=%.2f, threshold=%.2f, ready=%s",
        name,
        raw_total,
        score,
        cfg.maturity_threshold,
        score >= cfg.maturity_threshold,
    )
    return score


# ── add / update operation atoms ────────────────────────────────────────────────────────────────────


async def _apply_add(
    data: _SkillAddData,
    source_case_ids: Sequence[str],
    *,
    op_index: int,
    client: LLMClient,
    cfg: _SkillCfg,
) -> OpOutcome:
    """Apply an ``add`` op: validate content / name / description, clamp confidence, run maturity scoring.

    Returns an :class:`OpOutcome` carrying either a fresh :class:`AgentSkill` or the
    :class:`SkillSkipReason` that rejected the op. cluster_id is left empty; caller stamps it after
    extraction.

    Raises:
        LLMOutputError: If maturity scoring runs and its reply is not valid.
    """
    content = data.content
    name = data.name
    description = data.description

    if not content:
        logger.warning("add op empty content, skipping")
        return OpOutcome(op_index, None, SkillSkipReason.ADD_CONTENT_EMPTY, {})
    if not _is_skill_content_sufficient(content):
        logger.warning(
            "add op insufficient content (len=%d), skipping content=%r",
            len(content),
            content[:100],
        )
        return OpOutcome(
            op_index,
            None,
            SkillSkipReason.ADD_CONTENT_INSUFFICIENT,
            _content_detail(content),
        )
    if not name and not description:
        logger.warning("add op missing both name and description, skipping")
        return OpOutcome(op_index, None, SkillSkipReason.ADD_NAME_AND_DESC_EMPTY, {})

    description = truncate_text(description, cfg.max_description_tokens, suffix="...")
    confidence = max(0.0, min(1.0, data.confidence))

    score = await _evaluate_maturity(
        name=name,
        description=description,
        content=content,
        confidence=confidence,
        client=client,
        cfg=cfg,
    )

    return OpOutcome(
        op_index,
        AgentSkill(
            id=uuid.uuid4().hex,
            name=name,
            description=description,
            content=content,
            confidence=confidence,
            maturity_score=score,
            source_case_ids=list(source_case_ids),
        ),
        None,
        {},
    )


async def _apply_update(
    op: _SkillUpdateOp,
    existing_list: Sequence[AgentSkill],
    source_case_ids: Sequence[str],
    source_quality: float,
    *,
    op_index: int,
    client: LLMClient,
    cfg: _SkillCfg,
    processed_indices: set[int],
) -> OpOutcome:
    """Apply an ``update`` op; return an :class:`OpOutcome` with the modified skill or a drop reason.

    ``confidence < cfg.retire_confidence`` signals retire; ``confidence >=`` signals upsert.
    Drops on out-of-range / duplicate index, insufficient new content, or no changed fields.

    Note the two distinct indices in play: ``op_index`` is the op's position in the LLM's
    ``operations`` list, while ``op.index`` addresses ``existing_list``.

    Raises:
        LLMOutputError: If maturity is re-evaluated and its reply is not valid.
    """
    # ── index range + duplicate guard ─────────────────────────────────────────
    index = op.index
    if index < 0 or index >= len(existing_list):
        logger.warning(
            "update index %d out of range [0, %d)", index, len(existing_list)
        )
        return OpOutcome(
            op_index,
            None,
            SkillSkipReason.UPDATE_INDEX_OUT_OF_RANGE,
            {"index": index, "size": len(existing_list)},
        )
    if index in processed_indices:
        logger.warning("duplicate update on index %d, skipping", index)
        return OpOutcome(
            op_index, None, SkillSkipReason.UPDATE_DUPLICATE_INDEX, {"index": index}
        )
    processed_indices.add(index)

    prior = existing_list[index]
    data = op.data

    # ── extract new field values: null and "" both mean "unchanged" ───────────
    new_name = data.name or ""
    raw_new_description = data.description or ""
    new_description = (
        truncate_text(raw_new_description, cfg.max_description_tokens, suffix="...")
        if raw_new_description
        else ""
    )
    new_content = data.content or ""

    if new_content and not _is_skill_content_sufficient(new_content):
        logger.warning(
            "update[%d] insufficient content (len=%d), skipping",
            index,
            len(new_content),
        )
        return OpOutcome(
            op_index,
            None,
            SkillSkipReason.UPDATE_CONTENT_INSUFFICIENT,
            _content_detail(new_content),
        )

    parsed_confidence = (
        None if data.confidence is None else max(0.0, min(1.0, data.confidence))
    )
    confidence_changed = parsed_confidence is not None

    # ── change flags ──────────────────────────────────────────────────────────
    name_changed = bool(new_name) and new_name != (prior.name or "")
    desc_changed = bool(new_description) and new_description != (
        prior.description or ""
    )
    real_content_changed = bool(new_content) and new_content != (prior.content or "")

    if (
        not name_changed
        and not desc_changed
        and not real_content_changed
        and not confidence_changed
    ):
        logger.warning("update[%d] has no fields to change, skipping", index)
        return OpOutcome(
            op_index, None, SkillSkipReason.UPDATE_NO_FIELD_CHANGED, {"index": index}
        )

    # ── effective values ──────────────────────────────────────────────────────
    eff_name = new_name if new_name else (prior.name or "")
    eff_description = new_description if new_description else (prior.description or "")
    eff_content = new_content if new_content else (prior.content or "")
    eff_confidence: float = (
        parsed_confidence if parsed_confidence is not None else prior.confidence
    )

    # ── source_case_ids append (dedup) ────────────────────────────────────────
    existing_ids = list(prior.source_case_ids or [])
    for cid in source_case_ids:
        if cid and cid not in existing_ids:
            existing_ids.append(cid)

    # ── retire branch (DESIGN.md §5.2: confidence < retire_confidence) ────────
    if eff_confidence < cfg.retire_confidence:
        logger.warning(
            "retire skill[%d] (id=%s, name=%r): confidence=%.2f < %.2f",
            index,
            prior.id,
            prior.name or "",
            eff_confidence,
            cfg.retire_confidence,
        )
        return OpOutcome(
            op_index,
            prior.model_copy(
                update={
                    "confidence": eff_confidence,
                    "source_case_ids": existing_ids,
                }
            ),
            None,
            {},
        )

    # ── maturity re-eval decision (three-band) ────────────────────────────────
    new_maturity = prior.maturity_score
    content_changed = real_content_changed or name_changed or desc_changed
    if content_changed:
        change_ratio = _content_change_ratio(
            prior.content or "", new_content or prior.content or ""
        )
        hyp_promo = _is_hypothesis_promotion(prior.content or "", new_content or "")

        if change_ratio < cfg.maturity_trivial_change_ratio and not hyp_promo:
            logger.info(
                "skipping maturity re-eval for skill[%d]: trivial change_ratio=%.2f < %.2f",
                index,
                change_ratio,
                cfg.maturity_trivial_change_ratio,
            )
        elif change_ratio >= cfg.maturity_reeval_change_ratio or hyp_promo:
            reason = (
                "hypothesis promotion"
                if hyp_promo
                else f"major change ratio={change_ratio:.2f}"
            )
            logger.info("%s for skill[%d], re-evaluating maturity", reason, index)
            new_maturity = await _evaluate_maturity(
                name=eff_name,
                description=eff_description,
                content=eff_content,
                confidence=eff_confidence,
                client=client,
                cfg=cfg,
            )
        else:
            # moderate band 0.2 ~ 0.4
            old_score = prior.maturity_score or 0.0
            old_confidence = prior.confidence or 0.0
            confidence_dropping = eff_confidence < old_confidence

            if old_score >= cfg.maturity_threshold and (
                not confidence_dropping or eff_confidence >= 0.5
            ):
                logger.info(
                    "skipping maturity re-eval for skill[%d]: already mature (%.2f >= %.2f), "
                    "confidence=%.2f (dropping=%s), change_ratio=%.2f",
                    index,
                    old_score,
                    cfg.maturity_threshold,
                    eff_confidence,
                    confidence_dropping,
                    change_ratio,
                )
            elif old_score < cfg.maturity_threshold and source_quality < 0.3:
                logger.info(
                    "skipping maturity re-eval for skill[%d]: immature (%.2f) + low source quality "
                    "(%.2f < 0.3), change_ratio=%.2f",
                    index,
                    old_score,
                    source_quality,
                    change_ratio,
                )
            else:
                logger.info(
                    "moderate change for skill[%d]: score=%.2f, conf_drop=%s, source_quality=%.2f, "
                    "re-evaluating maturity",
                    index,
                    old_score,
                    confidence_dropping,
                    source_quality,
                )
                new_maturity = await _evaluate_maturity(
                    name=eff_name,
                    description=eff_description,
                    content=eff_content,
                    confidence=eff_confidence,
                    client=client,
                    cfg=cfg,
                )

    # ── update branch: emit AgentSkill (caller diffs name/description against prior to decide reembed) ──
    updated = prior.model_copy(
        update={
            "name": eff_name,
            "description": eff_description,
            "content": eff_content,
            "confidence": eff_confidence,
            "maturity_score": new_maturity,
            "source_case_ids": existing_ids,
        }
    )
    logger.info(
        "update skill[%d]: id=%s, fields_changed=name=%s/desc=%s/content=%s/conf=%s",
        index,
        prior.id,
        name_changed,
        desc_changed,
        real_content_changed,
        confidence_changed,
    )
    return OpOutcome(op_index, updated, None, {})
