"""Backend-neutral declarations of an index record."""

from __future__ import annotations

from typing import ClassVar


class IndexRecordMeta:
    """What every derived-index backend needs to know about a record model.

    Each backend reads these through ``schema_for`` and translates them into
    its own physical form; none of them is a LanceDB or Milvus concept.
    ``BaseLanceTable`` mixes this in and adds only the Lance mechanics
    (Arrow schema, FTS / vector index upkeep).
    """

    TABLE_NAME: ClassVar[str] = ""
    """Logical table name. Business schemas must override (e.g.
    ``"episode"``); chassis / test schemas that construct their table inline
    leave it empty."""

    BM25_FIELDS: ClassVar[list[str]] = []
    """Columns to build a BM25 (full-text) index on.

    Each declared column must already exist as a ``str`` (or
    ``str | None``) field on the schema. Tokens are assumed to be
    **app-layer pre-tokenised** (space-joined) so segmentation is owned by
    the app layer (:class:`JiebaTokenizer`), which also owns stop-word
    filtering. The per-backend index knobs live with the backend — see
    :meth:`BaseLanceTable.ensure_fts_indexes` for LanceDB."""

    PRIMARY_KEY: ClassVar[tuple[str, ...]]
    """Columns that together identify one row — the write key on every
    backend. ``id`` is unique only within an ``(app_id, project_id)`` space,
    so business tables declare ``("app_id", "project_id", "id")``. No default
    on purpose: a table that forgets it fails at ``schema_for`` / ``upsert``
    instead of silently keying on ``id``.
    """
