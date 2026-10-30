"""LanceDB index upkeep for the business tables.

The table declarations (columns, ``BM25_FIELDS``, the vector columns) live in
:mod:`everos.infra.persistence.records` and reach every backend as an
:class:`~everos.infra.persistence.index.schema.IndexSchema`. Building and
maintaining LanceDB's own FTS and IVF indexes over that schema is LanceDB
runtime behaviour, so it lives here, next to the repository base that calls it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lancedb import AsyncTable
from lancedb.index import FTS, IvfFlat

if TYPE_CHECKING:
    from everos.infra.persistence.index.schema import IndexSchema

VECTOR_INDEX_ROWS_PER_PARTITION = 4096
"""IVF partition size at build time. Pinned so :data:`VECTOR_QUERY_NPROBES`
means something: lance's default partition count has changed across releases."""

VECTOR_INDEX_MAX_DELTAS = 16
"""Delta indices a vector column may accumulate before the heavy beat retrains
it. Each light beat with new rows adds one; probing 16 of them cost ~1-3 ms extra
at 27k-100k rows, while a retrain rewrites the whole index (107 MB at 27k x 1024,
391 MB at 100k), so a trickle writer must not pay that every 300 s."""

VECTOR_QUERY_NPROBES = 32
"""Partitions probed per vector query. 32 partitions of 4096 rows cover the whole
column, i.e. exact search, up to ~130k rows; past that the unprobed partitions
are skipped and recall degrades gradually. Every ``nearest_to`` in the tree sets
it, so the index and the query agree on what "exact" costs."""


async def ensure_fts_indexes(
    table: AsyncTable, schema: IndexSchema, *, replace: bool = False
) -> None:
    """Create FTS indexes on every column in ``schema.bm25_fields``.

    Idempotent: columns that already have an index are skipped, so
    this is safe to call on every startup.

    ``replace=True`` rebuilds each column's index in place instead of
    skipping it — used by :meth:`LanceRepoBase.rebuild_indexes`, which
    needs a fresh index but must never leave the column *without* one.
    Dropping first would do that, and a BM25 query in that window does not
    degrade — it raises ``Cannot perform full text search unless an
    INVERTED index has been created`` (measured; vector search does fall
    back to a flat scan, FTS does not). Since the recall legs are gathered
    without ``return_exceptions``, that window turns into a 500 on the
    whole search request. ``create_index(replace=True)`` is atomic: 49
    concurrent queries across 3 replaces saw 0 failures, and it collapses
    the live fragment set exactly as drop+create does (7 index files back
    to 4, measured).

    The FTS config is fixed
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
    - ``with_position=False`` — everos does OR-mode BM25 recall
      (``MatchQuery`` clauses; see ``search.recall.base.build_or_query``),
      never phrase queries, so token positions are never read.
      Building the position posting List is therefore pure overhead
      **and** triggers a ``Max offset exceeds length of values``
      offset-overflow crash inside lance's compaction once the
      position lists grow large (upstream lance-format/lance#7653).
      That crash blocks ``optimize()`` — including version cleanup —
      so the index dir grows unbounded until the disk fills. Keeping
      positions off avoids both the overhead and the crash.
    """
    if not schema.bm25_fields:
        return
    indices = await table.list_indices()
    indexed_cols = {col for idx in indices for col in (idx.columns or [])}
    for field in schema.bm25_fields:
        if field in indexed_cols and not replace:
            continue
        await table.create_index(
            column=field,
            replace=replace,
            config=FTS(
                with_position=False,
                base_tokenizer="whitespace",
                lower_case=True,
                stem=True,
                remove_stop_words=False,
                ascii_folding=True,
            ),
        )


async def ensure_vector_indexes(
    table: AsyncTable, schema: IndexSchema, *, min_rows: int
) -> list[str]:
    """Keep one IVF_FLAT (cosine) index per vector column that holds at
    least ``min_rows`` non-null vectors; return the columns touched.

    Without an index LanceDB answers ``nearest_to`` with a brute-force
    scan of the whole column — linear in rows and in bytes (27k rows of
    1024-dim float32 is 112 MB and ~0.6 s per query on a laptop SSD),
    and a hybrid search issues two or three of them. IVF_FLAT keeps
    exact distances inside the probed partitions; with
    :data:`VECTOR_INDEX_ROWS_PER_PARTITION` rows per partition and
    :data:`VECTOR_QUERY_NPROBES` probes the search stays exact up to
    ~130k rows. ``cosine`` matches the query side. Columns that are
    still all-null (a Tier 1 store has no embeddings) are skipped:
    there is nothing to train on.

    Two cases do work; everything else is a no-op:

    * no index yet and the column crossed ``min_rows`` -> build one;
    * the index has grown more than :data:`VECTOR_INDEX_MAX_DELTAS` delta
      indices -> retrain it in place. Every
      ``optimize()`` on a table with new rows appends one *delta* index
      instead of merging (``num_indices`` +1 per light beat, never
      collapsing on its own), and a query probes every delta, so latency
      climbs with the beats since the last rebuild: 27k x 1024 rows
      measured 5.9 ms at 0 deltas, 25.7 ms at 100, 303 ms at 400 —
      worse than the 24 ms scan the index replaces. The cascade's heavy
      beat (300 s) calls this, so at most ~30 deltas accumulate under
      sustained writes. The retrain is one atomic index swap (searches
      never see the column unindexed); a concurrent ``optimize()`` from
      another process can preempt it with a benign commit conflict, in
      which case the next heavy beat retries — the same exposure prune has.

    ponytail: retraining (``create_index(replace=True)``) rewrites the
    whole index once per heavy beat under load; merging the deltas
    instead needs pylance's ``optimize_indices``, which is not a
    dependency. Revisit when a table passes ~500k rows.
    """
    columns = [field.name for field in schema.vector_fields]
    if not columns:
        return []
    indices = {
        col: idx for idx in await table.list_indices() for col in (idx.columns or [])
    }
    touched: list[str] = []
    for column in columns:
        existing = indices.get(column)
        if existing is not None:
            stats = await table.index_stats(existing.name)
            if stats is None or stats.num_indices <= VECTOR_INDEX_MAX_DELTAS:
                continue
        rows = await table.count_rows(f"{column} IS NOT NULL")
        if rows < min_rows:
            continue
        await table.create_index(
            column,
            replace=existing is not None,
            config=IvfFlat(
                distance_type="cosine",
                num_partitions=max(1, rows // VECTOR_INDEX_ROWS_PER_PARTITION),
            ),
        )
        touched.append(column)
    return touched
