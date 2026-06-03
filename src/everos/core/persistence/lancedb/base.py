"""Common LanceDB base for everos tables.

:class:`BaseLanceTable` adds ``created_at`` / ``updated_at`` columns and
the :attr:`BM25_FIELDS` declaration + :meth:`ensure_fts_indexes`
classmethod so each schema owns *both* its column shape **and** its
BM25 index spec — repos stay focused on queries.

Note:
    LanceDB has no SQL ``onupdate`` equivalent — the application must
    explicitly set ``updated_at = get_utc_now()`` before calling
    :meth:`AsyncTable.update` / :meth:`AsyncTable.merge_insert`. The
    convenience :func:`touch` helper does this in one call.

    **Every datetime column automatically carries ``tz=UTC`` in the
    Arrow schema.** LanceDB's Pydantic→PyArrow converter does not
    understand ``typing.Annotated`` metadata, so :data:`UtcDatetime`
    cannot be used as the field type annotation. Instead,
    :meth:`BaseLanceTable.to_arrow_schema` walks the inferred schema
    and rewrites every ``timestamp[us]`` (naive) column to
    ``timestamp[us, tz=UTC]``. PyArrow then auto-``astimezone(UTC)``
    aware inputs on write **and** returns aware UTC datetimes on read
    — no per-table configuration, no caller-side ``ensure_utc``.

    Subclasses just declare ``datetime`` fields normally::

        class Episode(BaseLanceTable):
            timestamp: dt.datetime
"""

from __future__ import annotations

import datetime as dt
from typing import ClassVar

import pyarrow as pa
from lancedb import AsyncTable
from lancedb.index import FTS
from lancedb.pydantic import LanceModel
from pydantic import Field

from everos.component.utils.datetime import get_utc_now


class BaseLanceTable(LanceModel):
    """Pydantic / LanceDB base with ``created_at`` / ``updated_at`` and
    schema-level LanceDB metadata (``TABLE_NAME`` / ``BM25_FIELDS``).

    The schema is the single source of truth for everything LanceDB
    needs to materialise the table: column shape, table name, vector
    dim (declared per-subclass), and which columns carry an FTS index.
    Repos read these ClassVars; they do not duplicate them.
    """

    TABLE_NAME: ClassVar[str] = ""
    """LanceDB table name. Business schemas must override (e.g.
    ``"episode"``). Left empty on chassis / test schemas that construct
    their table inline."""

    BM25_FIELDS: ClassVar[list[str]] = []
    """Columns to build LanceDB FTS (BM25) indexes on.

    Each declared column must already exist as a ``str`` (or
    ``str | None``) field on the schema. Tokens are assumed to be
    **app-layer pre-tokenised** (space-joined); the FTS index uses
    ``base_tokenizer="whitespace"`` so segmentation is owned by the
    app layer (:class:`JiebaTokenizer`). The same boundary owns stop-
    word filtering (English + Chinese); FTS-side ``remove_stop_words``
    is OFF. FTS *does* keep lightweight English-aware normalisation
    (``lower_case`` / ``stem`` / ``ascii_folding``) as a belt-and-
    braces layer on the same English tokens that survive jieba.
    See ``17_lancedb_tables_design.md`` §2.4.1 and
    :meth:`ensure_fts_indexes` below for the exact knobs."""

    created_at: dt.datetime = Field(default_factory=get_utc_now)
    updated_at: dt.datetime = Field(default_factory=get_utc_now)

    @classmethod
    def to_arrow_schema(cls) -> pa.Schema:
        """Patch the default Arrow schema: force every timestamp to ``tz=UTC``.

        The base ``LanceModel.to_arrow_schema()`` infers Arrow types from
        Pydantic field annotations and emits naive ``timestamp[us]`` for
        every :class:`datetime.datetime` column. We rewrite **every**
        timestamp column to ``timestamp[us, tz=UTC]``:

        * **on write** — PyArrow ``astimezone(UTC)``-s aware input
          automatically before serialising the i64 epoch micros.
        * **on read** — PyArrow returns aware UTC datetimes.

        Zero per-table configuration. The rewrite also **overrides any
        non-UTC tz** a subclass might have declared explicitly, because
        project convention is: storage is always UTC. Mixed-tz columns
        would violate the two-zone discipline (see
        ``docs/datetime.md``); enforcing UTC at the schema level closes
        that loophole.
        """
        base = super().to_arrow_schema()
        return pa.schema(
            [
                pa.field(f.name, pa.timestamp("us", tz="UTC"), nullable=f.nullable)
                if pa.types.is_timestamp(f.type)
                else f
                for f in base
            ]
        )

    @classmethod
    async def ensure_fts_indexes(cls, table: AsyncTable) -> None:
        """Create FTS indexes on every column in :attr:`BM25_FIELDS`.

        Idempotent: columns that already have an index are skipped, so
        this is safe to call on every startup. The FTS config is fixed
        to the app-layer pre-tokenisation + LanceDB normalisation
        convention (designed for **multilingual mixed content**):

        - ``base_tokenizer="whitespace"`` — split on the spaces our
          app-layer tokenizer provider already inserted between tokens.
        - ``lower_case=True`` — Unicode-aware case-fold (English A→a;
          no-op on CJK characters).
        - ``stem=True`` — Porter / Snowball English stemmer per
          ``language="English"`` (tantivy default). CJK tokens have no
          stemmer and pass through untouched.
        - ``remove_stop_words=False`` — **stop-word removal is owned by
          the app-layer** (:class:`JiebaTokenizer`), which already drops
          both Chinese and English stop-words before tokens reach the
          FTS index. Keeping FTS-side filtering off avoids double-
          filtering and a divided source of truth.
        - ``ascii_folding=True`` — strips diacritics (é→e) on Latin
          characters; no-op on CJK.
        - ``with_position=True`` — enables phrase queries.

        Subclasses normally do not need to override this — declaring
        :attr:`BM25_FIELDS` is enough.
        """
        if not cls.BM25_FIELDS:
            return
        indices = await table.list_indices()
        indexed_cols = {col for idx in indices for col in (idx.columns or [])}
        for field in cls.BM25_FIELDS:
            if field in indexed_cols:
                continue
            await table.create_index(
                column=field,
                config=FTS(
                    with_position=True,
                    base_tokenizer="whitespace",
                    lower_case=True,
                    stem=True,
                    remove_stop_words=False,
                    ascii_folding=True,
                ),
            )


def touch(record: BaseLanceTable) -> BaseLanceTable:
    """Set ``record.updated_at = now`` and return the record (chainable)."""
    record.updated_at = get_utc_now()
    return record
