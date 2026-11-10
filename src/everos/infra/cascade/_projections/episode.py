"""Episode cascade handler — md → LanceDB ``episode`` table.

Inherits :class:`BaseDailyLogHandler` for the diff / dispatch loop and
overrides :meth:`_build_row` to map an Episode entry's structured
body into the typed LanceDB row. Documented md contract (callers /
writers must match):

``inline`` block:

- ``owner_id``: ``user_id`` or ``agent_id`` — duplicates the
  frontmatter scope so the cascade can derive it without re-reading
  the frontmatter.
- ``session_id``: conversation scope.
- ``timestamp``: ISO-8601 string (``to_iso_format`` output).
- ``parent_type``: source kind label (currently always ``"memcell"``
  — :class:`ParentType` enum; explicit in md so future kinds plug in
  without a schema change).
- ``parent_id``: source memcell id.
- ``image_ids`` (optional): ``[<sha256>, ...]`` — the pictures the narrative
  refers to. Present only when the episode carried pictures; a handler that
  wrote before pictures existed has no such line, which reads as "none".
- ``sender_ids``: ``[u_a, u_b]`` rendered shape from
  ``_render_value`` (optional; defaults to empty list).
- ``source_start`` / ``subject_ids`` / ``origin`` (group episodes only, see
  :func:`._common.group_columns`): the segment's first message, whom it is
  about, whose messages it rests on.

``sections``:

- ``Subject`` (optional): one-line topic — embedded into
  ``subject_vector`` and appended to the BM25 tokenization source.
- ``Summary`` (optional): condensed narrative.
- ``Content``: full episode narrative — fed to the embedder AND the
  tokenizer for the ``episode_tokens`` BM25 field.

Embedding is a soft dependency: when unavailable, both ``vector`` and
``subject_vector`` are written as ``None`` and the row stays
BM25/scalar-searchable only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from everos.infra.persistence.index import VECTOR_PROJECTIONS, IndexRepository
from everos.infra.persistence.records import Episode, ParentType

from ._common import group_columns, parse_inline_list, require_iso_timestamp
from ._daily_log_base import BaseDailyLogHandler, ParsedEntry


class EpisodeHandler(BaseDailyLogHandler):
    """Cascade handler for ``users/<u>/`` and ``groups/<g>/episodes/episode-*.md``."""

    kind = "episode"

    @property
    def index_repo(self) -> IndexRepository[Any]:
        return self._deps.index.episode

    content_change_keys = (
        "section:Subject",
        "section:Summary",
        "section:Content",
    )
    """Subject / Summary / Content all participate in the digest:

    - Editing Content rewrites the embedding (correct).
    - Editing Subject / Summary doesn't change the embed text but still
      bumps the digest so the LanceDB ``subject`` / ``summary`` columns
      stay in sync with md. The slight overshoot (one wasted embed
      call on Subject edits) is accepted under the single-hash design
      (cascade Q2 discussion)."""

    def _embed_texts(self, entry: ParsedEntry) -> tuple[str, ...]:
        s = entry.structured
        text = s.sections.get("Content", "").strip()
        subject_text = s.sections.get("Subject", "").strip()
        return VECTOR_PROJECTIONS[self.kind].texts(
            {"episode": text, "subject": subject_text}
        )

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
    ) -> Episode:
        s = entry.structured
        text = s.sections.get("Content", "").strip()
        subject_text = s.sections.get("Subject", "").strip()

        # BM25 tokenization covers both body and subject keywords.
        tokenize_source = f"{text} {subject_text}" if subject_text else text
        tokens = self._deps.tokenizer.tokenize(tokenize_source)
        timestamp = require_iso_timestamp(s.inline.get("timestamp"))

        return Episode(
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
            sender_ids=parse_inline_list(s.inline.get("sender_ids", "")),
            image_ids=parse_inline_list(s.inline.get("image_ids", "")),
            subject=s.sections.get("Subject") or None,
            summary=s.sections.get("Summary") or None,
            episode=text,
            episode_tokens=" ".join(tokens),
            md_path=md_path,
            content_sha256=entry.content_sha256,
            **group_columns(owner_type, s.inline, timestamp),
            **VECTOR_PROJECTIONS[self.kind].values(vectors),
        )
