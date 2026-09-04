"""LanceDB ``principle`` table schema.

Principle is Meta Memory: one ``users/<user_id>/principles.md`` per
user, replaced wholesale on edit (same storage strategy as
``user.md``). Cascade explodes the frontmatter list into one Lance row
per principle. There is no vector and no BM25 — recall is KV-by-owner
(``include_principles``), not HYBRID.

``source_entry_ids`` is ``list[str]`` (Decision daily-log entry ids),
which Lance can store. The frontmatter's ``list[dict]`` of principles
never lands as a single column.
"""

from __future__ import annotations

from typing import ClassVar

from everos.core.persistence.lancedb import BaseLanceTable


class Principle(BaseLanceTable):
    """One engineering principle indexed in LanceDB."""

    TABLE_NAME: ClassVar[str] = "principle"
    # No BM25: principle recall is KV-by-owner, not keyword search.

    id: str
    """PK = ``<owner_id>_<principle_id>``."""

    principle_id: str
    """md-side id ``pr_<12hex>``, minted at write time."""

    owner_id: str
    owner_type: str
    """Always ``"user"`` for this schema."""

    app_id: str = "default"
    project_id: str = "default"
    """App / project scope (default ``"default"``); cascade fills from md path."""

    title: str
    statement: str
    source_entry_ids: list[str]
    """Decision daily-log entry ids this principle was synthesised from."""

    timestamp_ms: int
    """Algo-emitted principle timestamp (ms epoch). Audit only — not
    part of ``content_sha256``."""

    md_path: str
    content_sha256: str
    """SHA-256 over title + statement + source_entry_ids. Matches →
    cascade skips re-upsert of that row. ``timestamp_ms`` is not in
    the hash."""
