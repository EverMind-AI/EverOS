"""Image memory frontmatter — daily-log markdown for user-scoped pictures.

Path: ``users/<scope_id>/images/image-<YYYY-MM-DD>.md``.

Deliberately *not* dot-prefixed, unlike ``.atomic_facts/``: an image memory
is material the user asked to have remembered, and its original is referenced
by a relative path so an editor renders the picture inline (see
:mod:`everos.component.images` for the layout the relative path resolves
against).

One entry per picture *occurrence*, not per file: the same picture sent in
two sessions is one file under ``figures/`` and two entries here, because the
time and provenance differ and both are what retrieval matches on.
"""

from __future__ import annotations

import datetime as _dt
from typing import ClassVar, Literal

from pydantic import Field

from everos.infra.persistence.markdown._base.frontmatter import (
    DailyLogPathMixin,
    UserScopedFrontmatter,
)


class ImageMemoryDailyFrontmatter(DailyLogPathMixin, UserScopedFrontmatter):
    """Frontmatter for ``users/<scope>/images/image-<YYYY-MM-DD>.md``."""

    ENTRY_ID_PREFIX: ClassVar[str] = "im"
    DIR_NAME: ClassVar[str] = "images"
    FILE_PREFIX: ClassVar[str] = "image"

    type: Literal["image_memory_daily"] = "image_memory_daily"
    file_type: Literal["image_memory_daily"] = "image_memory_daily"
    date: _dt.date
    entry_count: int = 0
    created_at: _dt.datetime | None = None
    last_appended_at: _dt.datetime | None = None
    deprecated_entries: dict[str, str] = Field(default_factory=dict)
