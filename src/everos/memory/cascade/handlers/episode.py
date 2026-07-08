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
- ``sender_ids``: ``[u_a, u_b]`` rendered shape from
  ``_render_value`` (optional; defaults to empty list).

``sections``:

- ``Subject`` (optional): one-line topic — embedded into
  ``subject_vector`` and appended to the BM25 tokenization source.
- ``Summary`` (optional): condensed narrative.
- ``Content``: full episode narrative — fed to the embedder AND the
  tokenizer for the ``episode_tokens`` BM25 field.
"""

from __future__ import annotations

import asyncio

from everos.infra.persistence.lancedb import Episode, ParentType, episode_repo

from ._common import parse_inline_list, require_iso_timestamp
from ._daily_log_base import BaseDailyLogHandler, ParsedEntry


class EpisodeHandler(BaseDailyLogHandler):
    """Cascade handler for ``users/<u>/episodes/episode-*.md``."""

    kind = "episode"
    lance_repo = episode_repo
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

    async def _build_row(
        self,
        *,
        owner_id: str,
        owner_type: str,
        app_id: str = "default",
        project_id: str = "default",
        md_path: str,
        entry: ParsedEntry,
    ) -> Episode:
        s = entry.structured
        text = s.sections.get("Content", "").strip()
        subject_text = s.sections.get("Subject", "").strip()

        # Embed content and subject concurrently; skip subject embed when absent.
        if subject_text:
            vector, subject_vector = await asyncio.gather(
                self._deps.embedder.embed(text),
                self._deps.embedder.embed(subject_text),
            )
        else:
            vector = await self._deps.embedder.embed(text)
            subject_vector = None

        # BM25 tokenization covers both body and subject keywords.
        tokenize_source = f"{text} {subject_text}" if subject_text else text
        tokens = self._deps.tokenizer.tokenize(tokenize_source)

        return Episode(
            id=f"{owner_id}_{entry.entry_id}",
            entry_id=entry.entry_id,
            owner_id=owner_id,
            owner_type=owner_type,
            app_id=app_id,
            project_id=project_id,
            session_id=s.inline.get("session_id"),
            timestamp=require_iso_timestamp(s.inline.get("timestamp")),
            parent_type=s.inline.get("parent_type") or ParentType.MEMCELL.value,
            parent_id=s.inline.get("parent_id", ""),
            sender_ids=parse_inline_list(s.inline.get("sender_ids", "")),
            subject=s.sections.get("Subject") or None,
            summary=s.sections.get("Summary") or None,
            episode=text,
            episode_tokens=" ".join(tokens),
            md_path=md_path,
            content_sha256=entry.content_sha256,
            vector=vector,
            subject_vector=subject_vector,
        )
