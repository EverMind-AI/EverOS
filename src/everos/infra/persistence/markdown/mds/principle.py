"""Principle frontmatter — single-file engineering principles per user.

Path: ``users/<user_id>/principles.md``.

Principle is Meta Memory synthesised from a Decision cluster, not a
product Memory Kind. Storage reuses the profile chassis (fixed-name
single-file rewrite via :class:`ProfileWriter` / :class:`ProfileReader`);
this schema only supplies ``PROFILE_FILENAME`` plus the structured
``principles`` list.

LanceDB has no ``list[dict]`` column, so the list stays in frontmatter
and the cascade handler explodes it into one ``principle`` row per item.
The markdown body is a human-readable list and is not indexed.
"""

from __future__ import annotations

import uuid
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from everos.core.persistence.markdown import ProfilePathMixin, UserScopedFrontmatter


def mint_principle_id() -> str:
    """Mint a fresh principle id (``pr_<12hex>``).

    EverAlgo ``Principle`` has no id; EverOS assigns one at write time
    so cascade can explode a stable Lance PK ``<owner_id>_<principle_id>``.
    """
    return f"pr_{uuid.uuid4().hex[:12]}"


class PrincipleItem(BaseModel):
    """One engineering principle in the ``principles.md`` frontmatter list.

    ``id`` is the EverOS-minted ``pr_<12hex>`` (EverAlgo ``Principle`` has
    no id). ``source_entry_ids`` point at Decision daily-log entry ids.
    """

    id: str
    title: str
    statement: str
    source_entry_ids: list[str] = Field(default_factory=list)
    timestamp_ms: int = 0


class PrincipleFrontmatter(ProfilePathMixin, UserScopedFrontmatter):
    """Frontmatter for ``users/<user_id>/principles.md``."""

    PROFILE_FILENAME: ClassVar[str] = "principles.md"

    type: Literal["principle"] = "principle"

    principles: list[PrincipleItem] = Field(default_factory=list)
    """Structured principle list. Cascade explodes each item into one
    Lance ``principle`` row (``id = <owner_id>_<principle_id>``)."""


def render_principles_body(items: list[PrincipleItem]) -> str:
    """Human-readable markdown list for the ``principles.md`` body.

    Display-only: cascade indexes the structured frontmatter list, not
    this body. Empty input yields the empty string (no trailing newline).
    """
    if not items:
        return ""
    lines = [f"- **{item.title}.** {item.statement}" for item in items]
    return "\n".join(lines) + "\n"
