"""UserProfile cascade handler — md → LanceDB ``user_profile`` table.

Profile is a single-file kind (mirrors AgentSkill): one
``users/<user_id>/user.md`` per user, replaced wholesale on edit. No
entry markers, no per-entry diff. The LanceDB row carries the typed
projection of the frontmatter so a future query-aware lookup can run
off LanceDB; today the recaller is KV-by-owner.

md contract:

- frontmatter: :class:`UserProfileFrontmatter` (``user_id`` /
  ``summary`` / ``explicit_info`` / ``implicit_traits`` /
  ``profile_timestamp_ms``).
- body: free-form display text (not indexed; the structured payload
  the recaller returns is built from frontmatter alone).

``explicit_info`` / ``implicit_traits`` are heterogeneous LLM
emissions (list of small dicts mixed with strings). LanceDB has no
``list[dict]`` column type, so the handler json-encodes both into
``explicit_info_json`` / ``implicit_traits_json`` columns and the
recaller decodes them back at the DTO boundary.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from everos.core.persistence import MarkdownReader
from everos.infra.persistence.lancedb import UserProfile, user_profile_repo

from ..types import HandlerOutcome
from ._common import content_sha256 as compute_content_sha256
from ._common import resolve_scope
from .base import Handler


class UserProfileHandler(Handler):
    """Cascade handler for ``users/<user_id>/user.md``."""

    kind = "user_profile"
    lance_repo: ClassVar[Any] = user_profile_repo
    """Exposed for ``CascadeWorker._optimize_touched_kinds`` — the
    worker discovers the LanceDB repo to optimize via this attribute,
    mirroring the daily-log handlers that bind it through
    :class:`BaseDailyLogHandler`."""

    content_change_keys: ClassVar[tuple[str, ...]] = (
        "frontmatter:summary",
        "frontmatter:explicit_info_json",
        "frontmatter:implicit_traits_json",
    )
    """Retrieval-relevant fields. ``profile_timestamp_ms`` is treated
    as audit (changes with every synthesis even when content is
    identical) — it lands on the row but doesn't enter the hash, so a
    timestamp-only drift skips re-upsert."""

    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        absolute = self._deps.memory_root.root / md_path
        parsed = await MarkdownReader.read(absolute)
        fm = parsed.frontmatter

        owner_id = str(fm.get("user_id", ""))
        if not owner_id:
            raise ValueError(
                f"user_profile md missing required frontmatter user_id: {md_path}"
            )
        app_id, project_id = resolve_scope(md_path)

        summary = str(fm.get("summary", ""))
        explicit_info_json = _dump_json(fm.get("explicit_info", []))
        implicit_traits_json = _dump_json(fm.get("implicit_traits", []))
        profile_timestamp_ms = int(fm.get("profile_timestamp_ms", 0))

        digest = compute_content_sha256(
            {
                "frontmatter:summary": summary,
                "frontmatter:explicit_info_json": explicit_info_json,
                "frontmatter:implicit_traits_json": implicit_traits_json,
            }
        )

        row_id = owner_id
        prior = await user_profile_repo.get_by_id(row_id)
        if prior is not None and prior.content_sha256 == digest:
            return HandlerOutcome(
                md_path=md_path,
                kind=self.kind,
                upserted=0,
                deleted=0,
                skipped=1,
            )

        row = UserProfile(
            id=row_id,
            owner_id=owner_id,
            owner_type="user",
            app_id=app_id,
            project_id=project_id,
            summary=summary,
            explicit_info_json=explicit_info_json,
            implicit_traits_json=implicit_traits_json,
            profile_timestamp_ms=profile_timestamp_ms,
            md_path=md_path,
            content_sha256=digest,
        )
        await user_profile_repo.upsert([row])
        return HandlerOutcome(
            md_path=md_path,
            kind=self.kind,
            upserted=1,
            deleted=0,
            skipped=0,
        )

    async def handle_deleted(self, md_path: str) -> HandlerOutcome:
        deleted = await user_profile_repo.delete_by_md_path(md_path)
        return HandlerOutcome(
            md_path=md_path,
            kind=self.kind,
            upserted=0,
            deleted=deleted,
            skipped=0,
        )


def _dump_json(value: Any) -> str:
    """Canonical JSON encoding for list-style frontmatter buckets.

    ``sort_keys=True`` makes the digest stable across equivalent dicts;
    ``ensure_ascii=False`` keeps multibyte content readable in tooling
    that inspects the column without re-decoding.
    """
    return json.dumps(value, sort_keys=True, ensure_ascii=False)
