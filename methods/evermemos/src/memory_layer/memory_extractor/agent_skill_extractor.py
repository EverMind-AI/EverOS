"""
AgentSkillExtractor for EverMemOS

Incrementally extracts reusable skills from new AgentCase records
via operation-based updates (add/update/none) on existing cluster skills.

Pipeline:
1. Format the NEW AgentCaseRecord(s) as JSON context
2. Format existing skills with index numbers for the LLM
3. Single LLM call: output incremental operations (add/update/none)
4. Apply each operation: embed changed skills, persist via targeted DB ops
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import numpy as np

from api_specs.memory_types import AgentCase
from common_utils.json_utils import parse_json_response

from memory_layer.llm.llm_provider import LLMProvider
from memory_layer.prompts import get_prompt_by
from core.observation.logger import get_logger
from core.observation.stage_timer import timed

logger = get_logger(__name__)


@dataclass
class SkillExtractionResult:
    """Result of an incremental skill extraction run.

    Attributes:
        added_records: Newly created skill records (need insert into search engines).
        updated_records: Existing skill records that were modified in MongoDB
            (need upsert/replace in search engines). The in-memory objects
            already reflect the updated field values.
        deleted_ids: String IDs of skill records that were soft-deleted in MongoDB
            (need removal from search engines).
    """

    added_records: List[Any] = field(default_factory=list)
    updated_records: List[Any] = field(default_factory=list)
    deleted_ids: List[str] = field(default_factory=list)


class AgentSkillExtractor:
    """
    Incrementally extracts reusable skills from a MemScene.

    For each new case added to a cluster, this extractor:
    - Takes only the NEW AgentCaseRecord(s)
    - Reads existing skills for the cluster (with index numbers)
    - Uses an LLM to produce incremental operations (add/update/none)
    - Applies each operation with targeted DB writes (unchanged skills are untouched)
    """

    # quality_score threshold that determines which extraction prompt to use
    FAILURE_QUALITY_THRESHOLD: float = 0.5

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        success_extract_prompt: Optional[str] = None,
        failure_extract_prompt: Optional[str] = None,
        maturity_threshold: float = 0.6,
        retire_confidence: float = 0.1,
        skip_maturity_scoring: bool = False,
    ):
        self.llm_provider = llm_provider
        self.success_extract_prompt = success_extract_prompt or get_prompt_by(
            "AGENT_SKILL_SUCCESS_EXTRACT_PROMPT"
        )
        self.failure_extract_prompt = failure_extract_prompt or get_prompt_by(
            "AGENT_SKILL_FAILURE_EXTRACT_PROMPT"
        )
        self.maturity_threshold = maturity_threshold
        self.retire_confidence = retire_confidence
        self.skip_maturity_scoring = skip_maturity_scoring
        self.maturity_prompt = get_prompt_by("AGENT_SKILL_MATURITY_SCORE_PROMPT")

    @staticmethod
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

    def _format_cases(self, case_records: List[AgentCase]) -> str:
        """Format new AgentCaseRecords as a concise JSON string for the LLM."""
        formatted = []
        for rec in case_records:
            entry = {
                "timestamp": rec.timestamp.isoformat() if rec.timestamp else None,
                "task_intent": getattr(rec, "task_intent", ""),
                "approach": getattr(rec, "approach", ""),
                "quality_score": getattr(rec, "quality_score", 0.5) or 0.5,
            }
            key_insight = getattr(rec, "key_insight", None)
            if key_insight:
                entry["key_insight"] = key_insight
            formatted.append(entry)
        return json.dumps(
            formatted, ensure_ascii=False, indent=2, default=self._json_default
        )

    @staticmethod
    def _truncate_text(text: str, max_chars: int = 800) -> str:
        """Truncate text to max_chars, appending '...' if truncated."""
        text = (text or "").strip()
        if len(text) <= max_chars:
            return text
        suffix = "... [omitted]"
        return text[: max_chars - len(suffix)].rstrip() + suffix

    def _summarize_case_for_prompt(
        self, case_record: Any, max_approach_chars: int = 800
    ) -> Dict[str, Any]:
        """Build a compact case summary dict for inclusion in the skill prompt."""
        entry: Dict[str, Any] = {
            "task_intent": getattr(case_record, "task_intent", ""),
            "quality_score": getattr(case_record, "quality_score", 0.5) or 0.5,
        }
        key_insight = getattr(case_record, "key_insight", None)
        if key_insight:
            entry["key_insight"] = key_insight
        approach = getattr(case_record, "approach", None)
        if approach:
            entry["approach"] = self._truncate_text(approach, max_chars=max_approach_chars)
        return entry

    def _format_existing_skills(
        self,
        existing_records: List[Any],
        case_history: Optional[List[Any]] = None,
        max_support_cases: int = 3,
        max_approach_chars: int = 800,
    ) -> str:
        """Format existing AgentSkillRecords with index numbers for the LLM.

        When case_history is provided, each skill's source_case_ids are looked up
        in case_history to attach up to max_support_cases supporting case summaries.
        """
        if not existing_records:
            return "(empty — no existing skills)"

        # Build lookup: case_id -> case_record
        case_map: Dict[str, Any] = {}
        for rec in (case_history or []):
            cid = str(getattr(rec, "id", "") or "")
            if cid:
                case_map[cid] = rec

        lines = []
        for idx, rec in enumerate(existing_records):
            item: Dict[str, Any] = {
                "index": idx,
                "name": rec.name,
                "description": self._truncate_text(rec.description, max_chars=1000),
                "content": self._truncate_text(rec.content, max_chars=20000),
                "confidence": rec.confidence,
            }

            # Attach supporting case summaries if available
            if case_map:
                source_ids = getattr(rec, "source_case_ids", None) or []
                matched_ids = [sid for sid in source_ids if str(sid) in case_map]
                if matched_ids:
                    recent_ids = matched_ids[-max_support_cases:]
                    item["supporting_case_count"] = len(matched_ids)
                    item["supporting_cases"] = [
                        self._summarize_case_for_prompt(
                            case_map[str(sid)], max_approach_chars=max_approach_chars
                        )
                        for sid in recent_ids
                    ]

            lines.append(json.dumps(item, ensure_ascii=False, default=self._json_default))
        return "[\n" + ",\n".join(lines) + "\n]"

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(dot / norm)

    async def _select_top_k_skills(
        self,
        existing_records: List[Any],
        new_case_records: List[AgentCase],
        top_k: int = 10,
    ) -> List[Any]:
        """Select the top-k most relevant existing skills by vector similarity."""
        query_parts = []
        for rec in new_case_records:
            intent = getattr(rec, "task_intent", "") or ""
            if intent:
                query_parts.append(intent)
        query_text = "\n".join(query_parts)
        if not query_text:
            return existing_records[:top_k]

        # Reuse the existing vector when there is exactly one new case with a vector
        query_vec = None
        if len(new_case_records) == 1:
            existing_vec = getattr(new_case_records[0], "vector", None)
            if existing_vec and len(existing_vec) > 0:
                query_vec = existing_vec

        if query_vec is None:
            query_embedding = await self._compute_embedding(query_text)
            if not query_embedding:
                logger.warning(
                    "[AgentSkillExtractor] Failed to compute query embedding for top-k selection, "
                    "falling back to first %d skills",
                    top_k,
                )
                return existing_records[:top_k]
            query_vec = query_embedding["embedding"]

        with_vec = []
        without_vec = []
        for rec in existing_records:
            if rec.vector and len(rec.vector) > 0:
                sim = self._cosine_similarity(query_vec, rec.vector)
                with_vec.append((sim, rec))
            else:
                without_vec.append(rec)

        with_vec.sort(key=lambda x: x[0], reverse=True)
        selected = [rec for _, rec in with_vec[:top_k]]

        remaining = top_k - len(selected)
        if remaining > 0 and without_vec:
            selected.extend(without_vec[:remaining])

        logger.info(
            "[AgentSkillExtractor] Top-k selection: %d/%d skills selected (top_k=%d)",
            len(selected),
            len(existing_records),
            top_k,
        )
        return selected

    async def _compute_embedding(self, text: str) -> Optional[Dict[str, Any]]:
        """Compute embedding for a skill item's name + description."""
        try:
            if not text:
                return None
            from agentic_layer.vectorize_service import get_vectorize_service

            vs = get_vectorize_service()
            vec = await vs.get_embedding(text)
            return {
                "embedding": vec.tolist() if hasattr(vec, "tolist") else list(vec),
                "vector_model": vs.get_model_name(),
            }
        except Exception as e:
            logger.error(f"[AgentSkillExtractor] Embedding failed: {e}")
            return None

    def _select_prompt(self, case_records: List[AgentCase]) -> str:
        """Select extraction prompt based on the max quality_score of new cases."""
        max_quality = (
            max((getattr(rec, "quality_score", 0.5) or 0.5) for rec in case_records)
            if case_records
            else 0.5
        )
        if max_quality < self.FAILURE_QUALITY_THRESHOLD:
            logger.debug(
                "[AgentSkillExtractor] Using failure prompt (max_quality=%.2f < %.2f)",
                max_quality,
                self.FAILURE_QUALITY_THRESHOLD,
            )
            return self.failure_extract_prompt
        logger.debug(
            "[AgentSkillExtractor] Using success prompt (max_quality=%.2f >= %.2f)",
            max_quality,
            self.FAILURE_QUALITY_THRESHOLD,
        )
        return self.success_extract_prompt

    async def _call_llm(
        self, new_case_json: str, existing_skills_json: str, prompt_template: str
    ) -> Optional[Dict[str, Any]]:
        """Single LLM call to produce incremental skill operations."""
        prompt = prompt_template.format(
            new_case_json=new_case_json, existing_skills_json=existing_skills_json
        )
        for attempt in range(3):
            try:
                resp = await self.llm_provider.generate(prompt)
                data = parse_json_response(resp)
                if data and isinstance(data.get("operations"), list):
                    return data
                logger.warning(
                    f"[AgentSkillExtractor] LLM retry {attempt + 1}/3: invalid format"
                )
            except Exception as e:
                logger.warning(f"[AgentSkillExtractor] LLM retry {attempt + 1}/3: {e}")
        return None

    async def _evaluate_maturity(
        self, name: str, description: str, content: str, confidence: float
    ) -> Optional[float]:
        """Evaluate maturity of a skill via LLM scoring.

        Scores the skill across 4 dimensions (1-5 each, total out of 20),
        then normalizes to 0.0-1.0.
        """
        if self.skip_maturity_scoring:
            logger.info("[AgentSkillExtractor] Maturity scoring skipped by config, returning 1.0")
            return 1.0
        try:
            prompt = self.maturity_prompt.format(
                name=name or "",
                description=description or "",
                content=content or "",
                confidence=confidence,
            )
            resp = await self.llm_provider.generate(prompt)
            data = parse_json_response(resp)
            dimensions = ["completeness", "executability", "evidence", "clarity"]
            if not data or not all(d in data for d in dimensions):
                logger.warning(
                    "[AgentSkillExtractor] Maturity evaluation returned invalid format"
                )
                return None

            raw_total = sum(float(data[d]) for d in dimensions)
            score = max(0.0, min(1.0, raw_total / 20.0))
            logger.info(
                "[AgentSkillExtractor] Maturity evaluation: name='%s', "
                "raw=%.1f, score=%.2f, threshold=%.2f, ready=%s, reason=%s",
                name,
                raw_total,
                score,
                self.maturity_threshold,
                score >= self.maturity_threshold,
                data.get("reason", ""),
            )
            return score
        except Exception as e:
            logger.warning("[AgentSkillExtractor] Maturity evaluation failed: %s", e)
            return None

    # Content change ratio below which maturity re-evaluation is always skipped
    MATURITY_TRIVIAL_CHANGE_RATIO: float = 0.2
    # Content change ratio above which maturity must be re-evaluated via LLM
    MATURITY_REEVAL_CHANGE_RATIO: float = 0.4

    @staticmethod
    def _is_hypothesis_promotion(old_content: str, new_content: str) -> bool:
        """Detect if an update promotes a hypothesis skill to a verified skill.

        Returns True when the old content had '## Potential Steps' but the new
        content has '## Steps' (not Potential), indicating the LLM promoted it.
        """
        old_has_potential = bool(
            re.search(r"^##\s+Potential Steps", old_content or "", re.MULTILINE)
        )
        new_has_steps = bool(re.search(r"^##\s+Steps", new_content or "", re.MULTILINE))
        new_has_potential = bool(
            re.search(r"^##\s+Potential Steps", new_content or "", re.MULTILINE)
        )
        return old_has_potential and new_has_steps and not new_has_potential

    @staticmethod
    def _content_change_ratio(old: str, new: str) -> float:
        """Return 0.0-1.0 indicating how much content changed.

        Uses SequenceMatcher to compute: 1 - (matching chars / max length).
        """
        if not old and not new:
            return 0.0
        if not old or not new:
            return 1.0
        ratio = SequenceMatcher(None, old, new).ratio()
        return round(1.0 - ratio, 4)

    @staticmethod
    def _is_skill_content_sufficient(
        content: str, min_lines: int = 5, min_length: int = 50
    ) -> bool:
        """Check if skill content has enough substance to be useful."""
        if not content:
            return False
        stripped = content.strip()
        if len(stripped) < min_length:
            return False
        non_empty_lines = [l for l in stripped.splitlines() if l.strip()]
        return len(non_empty_lines) >= min_lines

    async def _apply_add(
        self,
        op: Dict[str, Any],
        cluster_id: str,
        group_id: Optional[str],
        user_id: Optional[str],
        skill_repo: Any,
        source_case_ids: Optional[List[str]] = None,
    ) -> Optional[Any]:
        """Apply an 'add' operation: create and insert a new skill record."""
        data = op.get("data", {})
        content = data.get("content", "")
        if not content:
            logger.warning(
                "[AgentSkillExtractor] add operation has empty content, skipping"
            )
            return None

        if not self._is_skill_content_sufficient(content):
            logger.warning(
                "[AgentSkillExtractor] add operation has insufficient content "
                "(too short or no steps), skipping. content=%r",
                content[:100],
            )
            return None

        name = data.get("name", "")
        description = data.get("description", "")
        if not name and not description:
            logger.warning(
                "[AgentSkillExtractor] add operation has no name and no description, skipping"
            )
            return None

        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        except (ValueError, TypeError):
            confidence = 0.5

        embed_text = "\n".join(s for s in [name, description] if s)
        embedding_data = await self._compute_embedding(embed_text)

        from infra_layer.adapters.out.persistence.document.memory.agent_skill import (
            AgentSkillRecord,
        )

        score = await self._evaluate_maturity(
            name=name, description=description, content=content, confidence=confidence
        )

        record = AgentSkillRecord(
            cluster_id=cluster_id,
            user_id=user_id,
            group_id=group_id,
            name=name,
            description=description,
            content=content,
            confidence=confidence,
            maturity_score=score if score is not None else 0.6,
            vector=(embedding_data["embedding"] if embedding_data else None),
            vector_model=(embedding_data["vector_model"] if embedding_data else None),
            source_case_ids=source_case_ids or [],
        )
        saved = await skill_repo.save_skill(record)
        if saved:
            logger.info(
                f"[AgentSkillExtractor] ADD skill: name='{name}', cluster={cluster_id}"
            )
        return saved

    async def _rescore_maturity(
        self,
        updates: Dict[str, Any],
        new_name: str,
        new_description: str,
        new_content: str,
        record: Any,
    ) -> None:
        """Re-evaluate maturity via LLM and write into updates dict."""
        effective_name = new_name or record.name or ""
        effective_desc = new_description or record.description or ""
        effective_content = new_content or record.content or ""
        effective_confidence = updates.get("confidence", record.confidence)
        score = await self._evaluate_maturity(
            name=effective_name,
            description=effective_desc,
            content=effective_content,
            confidence=effective_confidence,
        )
        if score is not None:
            updates["maturity_score"] = score

    async def _apply_update(
        self,
        op: Dict[str, Any],
        existing_skill_records: List[Any],
        skill_repo: Any,
        result: SkillExtractionResult,
        source_case_ids: Optional[List[str]] = None,
        source_quality: float = 0.5,
    ) -> bool:
        """Apply an 'update' operation: modify an existing skill record in-place."""
        try:
            index = int(op.get("index", -1))
        except (ValueError, TypeError):
            logger.warning(
                f"[AgentSkillExtractor] update index is not a valid integer: {op.get('index')!r}, skipping"
            )
            return False
        data = op.get("data", {})

        if index < 0 or index >= len(existing_skill_records):
            logger.warning(
                f"[AgentSkillExtractor] update index {index} out of range "
                f"(valid: 0..{len(existing_skill_records) - 1} for {len(existing_skill_records)} skills), skipping"
            )
            return False

        record = existing_skill_records[index]
        record_id = record.id

        new_name = data.get("name", "")
        new_description = data.get("description", "")
        new_content = data.get("content", "")
        new_confidence = data.get("confidence")

        if new_content and not self._is_skill_content_sufficient(new_content):
            logger.warning(
                "[AgentSkillExtractor] update operation for index %d has insufficient content "
                "(too short or no steps), skipping. content=%r",
                index,
                new_content[:100],
            )
            return False

        updates: Dict[str, Any] = {}

        if new_name:
            updates["name"] = new_name
        if new_description:
            updates["description"] = new_description
        if new_content:
            updates["content"] = new_content
        if new_confidence is not None:
            try:
                clamped = max(0.0, min(1.0, float(new_confidence)))
                updates["confidence"] = clamped
            except (ValueError, TypeError):
                clamped = None

        # Append source case IDs for traceability
        if source_case_ids:
            existing_ids = list(getattr(record, "source_case_ids", None) or [])
            new_ids = [cid for cid in source_case_ids if cid not in existing_ids]
            if new_ids:
                existing_ids.extend(new_ids)
                updates["source_case_ids"] = existing_ids

        if not updates:
            logger.warning(
                f"[AgentSkillExtractor] update operation for index {index} has no fields to update, skipping"
            )
            return False

        # Retire skill when confidence drops below threshold.
        # The record stays in MongoDB (data preserved for audit/recovery)
        # but is removed from search engines and excluded from future extraction context.
        final_confidence = updates.get("confidence")
        if final_confidence is not None and final_confidence < self.retire_confidence:
            logger.warning(
                "[AgentSkillExtractor] Retiring skill[%d] (confidence=%.2f < %.2f): "
                "id=%s, name=%r",
                index,
                final_confidence,
                self.retire_confidence,
                record_id,
                getattr(record, "name", ""),
            )
            retire_updates: Dict[str, Any] = {"confidence": final_confidence}
            if "source_case_ids" in updates:
                retire_updates["source_case_ids"] = updates["source_case_ids"]
            success = await skill_repo.update_skill_by_id(
                record_id, retire_updates
            )
            if success:
                # Signal search-engine removal (ES / Milvus) — data stays in MongoDB
                result.deleted_ids.append(str(record_id))
            return success

        # Re-embed only if name or description actually changed
        name_changed = bool(new_name) and new_name != (record.name or "")
        desc_changed = bool(new_description) and new_description != (
            record.description or ""
        )
        if name_changed or desc_changed:
            effective_name = new_name or record.name or ""
            effective_desc = new_description or record.description or ""
            embed_text = "\n".join(s for s in [effective_name, effective_desc] if s)
            embedding_data = await self._compute_embedding(embed_text)
            if embedding_data:
                updates["vector"] = embedding_data["embedding"]
                updates["vector_model"] = embedding_data["vector_model"]

        # Re-evaluate maturity when content/name/description actually changed.
        #
        # Rules:
        # 1) change < 20%: trivial tweak, keep current score
        # 2) change >= 40% or hypothesis promotion: always re-score via LLM
        # 3) change 20~40%:
        #    - mature (>= threshold) AND confidence not dropping: skip
        #    - immature (< threshold) AND case quality < 0.3: skip (low-quality case won't help)
        #    - otherwise: re-score via LLM
        real_content_changed = bool(new_content) and new_content != (
            record.content or ""
        )
        content_changed = real_content_changed or name_changed or desc_changed
        if content_changed:
            change_ratio = self._content_change_ratio(
                record.content or "", new_content or record.content or ""
            )

            # 1) Trivial change (< 20%): keep current score
            if change_ratio < self.MATURITY_TRIVIAL_CHANGE_RATIO:
                logger.info(
                    "[AgentSkillExtractor] Skipping maturity re-evaluation for skill[%d]: "
                    "trivial change_ratio=%.2f < %.2f",
                    index,
                    change_ratio,
                    self.MATURITY_TRIVIAL_CHANGE_RATIO,
                )
            # 2) Major change (>= 40%) or hypothesis promotion: always LLM
            elif (
                change_ratio >= self.MATURITY_REEVAL_CHANGE_RATIO
                or self._is_hypothesis_promotion(
                    record.content or "", new_content or ""
                )
            ):
                reason = (
                    "hypothesis promotion"
                    if self._is_hypothesis_promotion(
                        record.content or "", new_content or ""
                    )
                    else f"major content change (ratio={change_ratio:.2f})"
                )
                logger.info(
                    "[AgentSkillExtractor] %s for skill[%d], using LLM maturity evaluation",
                    reason,
                    index,
                )
                await self._rescore_maturity(
                    updates, new_name, new_description, new_content, record
                )
            # 3) Moderate change (20~40%)
            else:
                old_score = record.maturity_score or 0.0
                old_confidence = record.confidence or 0.0
                new_confidence_val = updates.get("confidence", old_confidence)
                confidence_dropping = new_confidence_val < old_confidence

                if old_score >= self.maturity_threshold and (
                    not confidence_dropping or new_confidence_val >= 0.5
                ):
                    # Already mature and confidence stable or still strong (>= 0.5): skip
                    logger.info(
                        "[AgentSkillExtractor] Skipping maturity re-evaluation for skill[%d]: "
                        "already mature (%.2f >= %.2f), confidence=%.2f (dropping=%s), change_ratio=%.2f",
                        index,
                        old_score,
                        self.maturity_threshold,
                        new_confidence_val,
                        confidence_dropping,
                        change_ratio,
                    )
                elif old_score < self.maturity_threshold and source_quality < 0.3:
                    # Immature but low-quality case won't improve it: skip
                    logger.info(
                        "[AgentSkillExtractor] Skipping maturity re-evaluation for skill[%d]: "
                        "immature (%.2f) but low source quality (%.2f < 0.3), change_ratio=%.2f",
                        index,
                        old_score,
                        source_quality,
                        change_ratio,
                    )
                else:
                    # Re-score: immature skill with decent case, or mature but confidence dropping
                    logger.info(
                        "[AgentSkillExtractor] Moderate change for skill[%d]: "
                        "score=%.2f, confidence_dropping=%s, source_quality=%.2f, "
                        "using LLM maturity evaluation",
                        index,
                        old_score,
                        confidence_dropping,
                        source_quality,
                    )
                    await self._rescore_maturity(
                        updates, new_name, new_description, new_content, record
                    )

        success = await skill_repo.update_skill_by_id(record_id, updates)
        if success:
            from common_utils.datetime_utils import get_now_with_timezone

            for field_name, value in updates.items():
                setattr(record, field_name, value)
            record.updated_at = get_now_with_timezone()
            result.updated_records.append(record)
            logger.info(
                f"[AgentSkillExtractor] UPDATE skill[{index}]: id={record_id}, "
                f"fields={list(updates.keys())}"
            )
        return success

    async def _load_case_history(
        self,
        existing_skill_records: List[Any],
        max_cases: int = 9,
    ) -> List[Any]:
        """Load historical AgentCaseRecords referenced by existing skills.

        Collects all source_case_ids from existing skills, loads them from DB,
        sorts by quality_score (desc) then timestamp (desc), and returns top N.
        """
        all_case_ids: set = set()
        for rec in existing_skill_records:
            for cid in (getattr(rec, "source_case_ids", None) or []):
                if cid is None:
                    continue
                cid_str = str(cid).strip()
                if cid_str:
                    all_case_ids.add(cid_str)

        if not all_case_ids:
            return []

        try:
            from core.di.utils import get_bean_by_type
            from infra_layer.adapters.out.persistence.repository.agent_case_raw_repository import (
                AgentCaseRawRepository,
            )

            agent_case_repo = get_bean_by_type(AgentCaseRawRepository)
            records = await agent_case_repo.get_by_ids(list(all_case_ids))
            records.sort(
                key=lambda c: (c.quality_score or 0.0, c.timestamp or datetime.min),
                reverse=True,
            )
            logger.info(
                "[AgentSkillExtractor] Loaded case_history: %d/%d cases (max=%d)",
                min(len(records), max_cases), len(records), max_cases,
            )
            return records[:max_cases]
        except Exception as e:
            logger.warning("[AgentSkillExtractor] Failed to load case_history: %s", e)
            return []

    async def extract_and_save(
        self,
        cluster_id: str,
        group_id: Optional[str],
        new_case_records: List[AgentCase],
        existing_skill_records: List[Any],
        skill_repo: Any,
        user_id: Optional[str] = None,
        max_skills_in_prompt: int = 10,
        max_case_history: int = 9,
    ) -> SkillExtractionResult:
        """Incrementally extract skills via operation-based updates.

        Args:
            cluster_id: The MemScene cluster ID
            group_id: Group ID for scoping
            new_case_records: Only the NEW AgentCaseRecord(s) to integrate.
                Each record should have an `id` attribute (AgentCase ID) for traceability.
            existing_skill_records: Previously saved AgentSkillRecord for this cluster
            skill_repo: AgentSkillRawRepository instance
            user_id: User ID (agent owner)
            max_skills_in_prompt: Max existing skills to include in the LLM prompt.
            max_case_history: Max historical cases to load for supporting case summaries.

        Returns:
            SkillExtractionResult containing added, updated records and deleted IDs.
        """
        empty_result = SkillExtractionResult()

        if not new_case_records:
            logger.debug(
                f"[AgentSkillExtractor] No new cases for cluster={cluster_id}, skipping"
            )
            return empty_result

        # When too many existing skills, select top-k most relevant ones
        if len(existing_skill_records) > max_skills_in_prompt:
            logger.info(
                f"[AgentSkillExtractor] {len(existing_skill_records)} existing skills exceed "
                f"max_skills_in_prompt={max_skills_in_prompt}, selecting top-k"
            )
            with timed("select_top_k_skills"):
                existing_skill_records = await self._select_top_k_skills(
                    existing_skill_records, new_case_records, top_k=max_skills_in_prompt
                )

        # Load case history AFTER top-k selection so we only load cases
        # relevant to the skills that will actually appear in the prompt.
        case_history = await self._load_case_history(
            existing_skill_records, max_case_history
        )

        new_case_json = self._format_cases(new_case_records)
        existing_skills_json = self._format_existing_skills(
            existing_skill_records, case_history=case_history
        )
        prompt_template = self._select_prompt(new_case_records)

        logger.debug(
            f"[AgentSkillExtractor] Incremental extraction: cluster={cluster_id}, "
            f"new_cases={len(new_case_records)}, existing_skills={len(existing_skill_records)}"
        )

        with timed("extract_skill_ops"):
            llm_result = await self._call_llm(
                new_case_json, existing_skills_json, prompt_template
            )
        if not llm_result:
            logger.warning(
                f"[AgentSkillExtractor] LLM extraction failed for cluster={cluster_id}"
            )
            return empty_result

        operations = llm_result.get("operations", [])
        update_note = llm_result.get("update_note", "")

        # Collect all case IDs from new records for traceability
        source_case_ids = [
            str(getattr(rec, "id", "") or "")
            for rec in new_case_records
        ]
        source_case_ids = [cid for cid in source_case_ids if cid]

        result = SkillExtractionResult()
        update_count = 0
        processed_indices: set = set()

        with timed("apply_operations"):
            for op in operations:
                action = op.get("action", "none")

                if action == "add":
                    saved = await self._apply_add(
                        op,
                        cluster_id,
                        group_id,
                        user_id,
                        skill_repo,
                        source_case_ids=source_case_ids,
                    )
                    if saved:
                        result.added_records.append(saved)

                elif action == "update":
                    try:
                        index = int(op.get("index", -1))
                    except (ValueError, TypeError):
                        logger.warning(
                            f"[AgentSkillExtractor] update index is not a valid integer: {op.get('index')!r}, skipping"
                        )
                        continue
                    if index in processed_indices:
                        logger.warning(
                            f"[AgentSkillExtractor] Duplicate operation on index {index}, skipping update"
                        )
                        continue
                    processed_indices.add(index)
                    # Pass max quality_score from new cases for maturity decision
                    max_quality = (
                        max(
                            (getattr(rec, "quality_score", 0.5) or 0.5)
                            for rec in new_case_records
                        )
                        if new_case_records
                        else 0.5
                    )
                    success = await self._apply_update(
                        op,
                        existing_skill_records,
                        skill_repo,
                        result,
                        source_case_ids=source_case_ids,
                        source_quality=max_quality,
                    )
                    if success:
                        update_count += 1

                elif action == "none":
                    logger.debug(
                        f"[AgentSkillExtractor] No-op for cluster={cluster_id}"
                    )

                else:
                    logger.warning(
                        f"[AgentSkillExtractor] Unknown action '{action}', skipping"
                    )

        logger.info(
            f"[AgentSkillExtractor] cluster={cluster_id} operations applied: "
            f"added={len(result.added_records)}, updated={update_count}, "
            f"deleted={len(result.deleted_ids)}. note: {update_note}"
        )
        return result
