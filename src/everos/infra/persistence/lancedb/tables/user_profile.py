"""LanceDB ``user_profile`` table schema.

Profile is a single-file kind: one ``users/<user_id>/user.md`` per
user, replaced wholesale on edit (mirrors ``AgentSkill`` for the
upsert/single-row contract). The LanceDB row is a typed projection
of the md frontmatter that the cascade keeps in sync; it carries no
vector / no BM25 because the recall surface is pure KV-by-owner
(``fetch(owner_id)``) — when query-aware profile lookup ships later
the schema will gain ``vector`` + ``*_tokens`` columns then.

``explicit_info`` / ``implicit_traits`` are heterogeneous LLM
emissions (mostly small dicts mixed with strings) — LanceDB has no
``list[dict]`` column type, so we stash them as JSON strings and
unpack at the recall boundary into ``profile_data`` of the DTO.
"""

from __future__ import annotations

from typing import ClassVar

from everos.core.persistence.lancedb import BaseLanceTable


class UserProfile(BaseLanceTable):
    """One ``users/<user_id>/user.md`` indexed in LanceDB."""

    TABLE_NAME: ClassVar[str] = "user_profile"
    # No BM25 columns: profile recall is KV-by-owner today.

    id: str
    """PK = ``owner_id`` (one row per user)."""

    owner_id: str
    owner_type: str
    """Always ``"user"`` for this schema; agent-side profiles would
    live in a sibling table once that schema lands."""

    app_id: str = "default"
    project_id: str = "default"
    """App / project scope (default ``"default"``); cascade fills from md path."""

    summary: str
    """Free-form one-paragraph user summary (retrieval anchor for the
    future query-aware lookup; today returned verbatim to the caller)."""

    explicit_info_json: str
    """JSON-serialised ``list[Any]`` — the algo's verbatim evidence
    bucket. Stored as a string because LanceDB has no
    ``list[dict]`` column type. The recaller json-decodes it back into
    ``profile_data['explicit_info']`` at the DTO boundary."""

    implicit_traits_json: str
    """Same shape as :attr:`explicit_info_json`, for the LLM-inferred
    preference bucket."""

    profile_timestamp_ms: int
    """Algo-emitted profile timestamp (ms epoch) — pinned to the
    timestamp of the freshest MemCell that fed into the synthesis.
    Mirrored from :attr:`UserProfileFrontmatter.profile_timestamp_ms`
    so downstream code can compare freshness without re-reading md."""

    md_path: str
    content_sha256: str
    """SHA-256 over the content-bearing frontmatter fields (summary +
    explicit_info_json + implicit_traits_json). Matches → cascade
    skips re-upsert. ``profile_timestamp_ms`` is intentionally not in
    the hash: it drifts with every synthesis even when the underlying
    content is identical, and the LanceDB row treats it as audit."""
