"""LanceDB ``agent_skill`` table schema.

Field set per 17_lancedb_tables_design.md §3.5. AgentSkill is a *named
entity* rather than a daily-log entry — PK is ``<owner_id>_<skill_name>``
(no date / seq), and same agent + same name is the same row (upsert).

``content`` is cascade-assembled from ``SKILL.md`` body plus every
``references/*.md`` sibling; ``scripts/`` is not indexed.
"""

from __future__ import annotations

from typing import ClassVar

from everos.core.persistence.lancedb import BaseLanceTable, Vector

_DIM = 1024


class AgentSkill(BaseLanceTable):
    """One agent skill indexed in LanceDB."""

    TABLE_NAME: ClassVar[str] = "agent_skill"
    BM25_FIELDS: ClassVar[list[str]] = ["description_tokens", "content_tokens"]

    id: str
    """PK = ``<owner_id>_<skill_name>``."""

    owner_id: str
    """The owning ``agent_id``."""

    owner_type: str
    """Fixed ``"agent"`` for this table."""

    app_id: str = "default"
    project_id: str = "default"
    """App / project scope (default ``"default"``); cascade fills from md path."""

    name: str
    """Skill identifier; half of the PK."""

    description: str
    """When-to-use / purpose — original surface form (Tier-1 ad copy)."""

    description_tokens: str
    """App-layer pre-tokenised ``description`` — BM25 main field
    (whitespace tokenizer); display goes through ``description``."""

    content: str
    """Cascade-assembled body: ``SKILL.md`` main text concatenated with
    every ``references/*.md`` sibling. ``scripts/`` files are excluded."""

    content_tokens: str
    """App-layer pre-tokenised ``content`` (secondary BM25 field).
    Tokenised by cascade when assembling ``content`` from md sources."""

    confidence: float
    """0.0–1.0; LLM-emitted confidence in the skill."""

    maturity_score: float
    """0.0–1.0; LLM-emitted maturity score. The retrieval-time threshold
    (``maturity_threshold``) lives in MemorizeConfig, not in this row."""

    source_case_ids: list[str]
    """AgentCase ids that fed into this skill's synthesis (lineage)."""

    cluster_id: str | None = None
    """Optional MemScene clustering tag."""

    md_path: str
    content_sha256: str
    """SHA-256 hex digest over the **content-bearing fields only** of
    the skill: ``name`` / ``description`` (frontmatter) + SKILL.md
    body + concatenated references content + ``confidence`` /
    ``maturity_score``. Cascade handler diffs by this digest to skip
    re-upsert + re-embed when neither retrieval-anchor text nor scores
    changed (e.g. the watcher fires for unrelated stat updates). See
    :attr:`AgentSkillHandler.content_change_keys`."""

    vector: Vector(_DIM)  # type: ignore[valid-type]
