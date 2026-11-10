"""Image-memory cascade handler — md → LanceDB ``image_memory`` table.

Inherits the daily-log diff / dispatch loop and maps one picture entry into the
typed row. The md contract (what writers must produce and nothing else may
assume):

``inline`` block:

- ``owner_id``: ``user_id``, duplicated from the frontmatter so the index row
  is readable without re-reading the file.
- ``session_id``: the conversation the picture arrived in (optional).
- ``timestamp``: ISO-8601 string, the message time — not the time the memory
  was written. The question "when did she send this" is about the message.
- ``parent_type`` / ``parent_id``: the memcell the picture hung in; with
  ``image_id`` this is what a forget uses to find the segment to re-extract.
- ``image_id``: SHA-256 of the stored picture — stable across occurrences and
  what an episode body references.
- ``original_path``: root-relative path of the original.
- ``name`` (optional): what the client called the picture on this arrival. It
  is audit, not content: the digest does not cover it, so a renamed picture is
  not re-embedded.
- ``media_type`` / ``width`` / ``height``: what the file is, so a result can
  describe it without decoding it.

``sections``:

- ``Description``: what the picture shows — the BM25 source, and the text a
  result shows when the caller cannot take the picture itself.
- ``OCR`` (optional): text visible in the picture. Absent for a photograph of
  a scene, which is the common case rather than a defect.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from everos.core.observability.logging import get_logger
from everos.infra.persistence.index import VECTOR_PROJECTIONS, IndexRepository
from everos.infra.persistence.records import ImageMemory, ParentType

from ._common import (
    group_columns,
    optional_int,
    parse_inline_list,
    require_iso_timestamp,
)
from ._daily_log_base import BaseDailyLogHandler, ParsedEntry

logger = get_logger(__name__)


class ImageMemoryHandler(BaseDailyLogHandler):
    """Cascade handler for ``users/<u>/`` and ``groups/<g>/images/image-*.md``."""

    kind = "image_memory"

    @property
    def index_repo(self) -> IndexRepository[Any]:
        return self._deps.index.image_memory

    content_change_keys = (
        "section:Description",
        "section:OCR",
        "inline:image_id",
    )
    """All three participate in the digest: editing the description re-embeds
    and re-tokenises (correct); editing ``image_id`` repoints the row at
    another picture, which must rewrite everything about it."""

    def _embed_texts(self, entry: ParsedEntry) -> tuple[str, ...]:
        return VECTOR_PROJECTIONS[self.kind].texts(
            {"description": entry.structured.sections.get("Description", "").strip()}
        )

    async def _attach_side_vectors(self, rows: list[Any]) -> list[Any]:
        """Embed each picture's bytes into ``visual_vector``.

        The text batch already wrote ``vector`` (the description's embedding).
        This adds the second, independent embedding: the file itself, encoded
        by the vision model, so a question about what the picture *looks like*
        can reach it without the description having to name it.

        Two details are load-bearing:

        - The same picture recurs: one file, one row per appearance. Rows are
          grouped by ``image_id`` so a picture is embedded once per run and
          the vector is copied onto every occurrence.
        - A missing file is not a failed row. The row's description is still
          searchable, so the picture is written with ``visual_vector=None``
          and the next cascade pass over its md retries it — that is the same
          soft degradation a missing provider produces.
        """
        vision = self._deps.vision_embedding
        if vision is None or not rows:
            return rows
        root = self._deps.memory_root.root
        by_image: dict[str, tuple[bytes, str]] = {}
        for image in rows:  # the base hands back the rows it built: ImageMemory
            if image.visual_vector is not None or not image.original_path:
                continue
            if image.image_id in by_image:
                continue
            absolute = root / image.original_path
            try:
                data = await asyncio.to_thread(absolute.read_bytes)
            except OSError as exc:
                logger.warning(
                    "cascade_image_visual_embed_unreadable",
                    image_id=image.image_id,
                    original_path=image.original_path,
                    error=str(exc)[:200],
                )
                continue
            by_image[image.image_id] = (data, image.media_type or "image/png")
        if not by_image:
            return rows
        image_ids = list(by_image)
        vectors = await vision.embed_images([by_image[i] for i in image_ids])
        vec_by_id = dict(zip(image_ids, vectors, strict=True))
        return [
            row.model_copy(update={"visual_vector": vec_by_id[row.image_id]})
            if row.image_id in vec_by_id and row.visual_vector is None
            else row
            for row in rows
        ]

    async def _build_row(
        self,
        *,
        owner_id: str,
        owner_type: str,
        app_id: str = "default",
        project_id: str = "default",
        md_path: str,
        entry: ParsedEntry,
        vectors: Sequence[list[float] | None] = (),
    ) -> ImageMemory:
        s = entry.structured
        description = s.sections.get("Description", "").strip()
        ocr = s.sections.get("OCR", "").strip()
        timestamp = require_iso_timestamp(s.inline.get("timestamp"))
        group = group_columns(owner_type, s.inline, timestamp)
        if group:
            group["sender_ids"] = parse_inline_list(s.inline.get("sender_ids", ""))

        return ImageMemory(
            id=f"{owner_id}_{entry.entry_id}",
            entry_id=entry.entry_id,
            owner_id=owner_id,
            owner_type=owner_type,
            app_id=app_id,
            project_id=project_id,
            session_id=s.inline.get("session_id"),
            timestamp=timestamp,
            parent_type=s.inline.get("parent_type") or ParentType.MEMCELL.value,
            parent_id=s.inline.get("parent_id", ""),
            image_id=s.inline.get("image_id", ""),
            original_path=s.inline.get("original_path", ""),
            name=s.inline.get("name") or "",
            media_type=s.inline.get("media_type", ""),
            width=optional_int(s.inline.get("width")) or 0,
            height=optional_int(s.inline.get("height")) or 0,
            description=description,
            description_tokens=" ".join(self._deps.tokenizer.tokenize(description)),
            ocr=ocr,
            ocr_tokens=" ".join(self._deps.tokenizer.tokenize(ocr)) if ocr else "",
            md_path=md_path,
            content_sha256=entry.content_sha256,
            **group,
            **VECTOR_PROJECTIONS[self.kind].values(vectors),
        )
