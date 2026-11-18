"""Re-embed derived-index rows written without a vector.

Rows written while no embedding provider was configured carry
``vector IS NULL``. :class:`ProjectionVectorBackfill` finds them in every
vector-carrying index table through the shared :data:`VECTOR_PROJECTIONS`,
and writes each row's vector back with a partial-column update, falling back
row by row when a batch fails. Confirmation, presentation and the phase's outcome are
the caller's; this module only reports progress through a callback.
"""

from __future__ import annotations

__all__ = [
    "ProgressCallback",
    "ProjectionVectorBackfill",
    "VectorBackfill",
    "VectorBackfillCounts",
    "VectorBacklog",
]

import dataclasses
import datetime as dt
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol

from everos.core.observability.logging import get_logger
from everos.memory.ports import primary_key

from .vector_projection import VECTOR_PROJECTIONS, VectorProjection

if TYPE_CHECKING:
    from everos.component.embedding import EmbeddingProvider
    from everos.component.tokenizer import Tokenizer

    from .maintenance import IndexMaintenance

logger = get_logger(__name__)

ProgressCallback = Callable[[str, int, int], None]
"""Called as ``(table_name, rows_done, rows_total)`` after every batch."""

_TABLE_SPECS = tuple(VECTOR_PROJECTIONS.values())


@dataclasses.dataclass
class VectorBackfillCounts:
    """Rows written back, rows left incomplete, and tokens embedded."""

    rows_processed: int = 0
    rows_failed: int = 0
    tokens_embedded: int = 0

    def add(self, other: VectorBackfillCounts) -> None:
        """Accumulate ``other`` into this tally."""
        self.rows_processed += other.rows_processed
        self.rows_failed += other.rows_failed
        self.tokens_embedded += other.tokens_embedded


_EMBED_BATCH_SIZE = 32
"""Rows grouped per ``embed_batch`` call — not the provider's own
request chunk size. The provider's configured ``batch_size`` /
``max_concurrent`` (``embedding.batch_size`` / ``embedding.max_concurrent``
in settings) govern the actual in-flight request shape from there; this
just bounds per-batch failure isolation + progress feedback granularity
for a bulk migration."""


@dataclasses.dataclass(frozen=True)
class _NullVectorRow:
    """One row missing its vector.

    ``tokens`` is the real token count (via the same tokenizer cascade
    uses for BM25), not an estimate — computed once here and never
    recomputed before the embed call.
    """

    id: str
    text: str
    tokens: int
    app_id: str = "default"
    project_id: str = "default"

    @property
    def key(self) -> tuple[str, str, str]:
        """The row's identity: the table's declared PRIMARY_KEY. ``id`` alone
        repeats across projects, so every per-batch map is keyed by this."""
        return (self.app_id, self.project_id, self.id)


@dataclasses.dataclass(frozen=True)
class _TableBacklog:
    spec: VectorProjection
    index: IndexMaintenance
    rows: list[_NullVectorRow]

    @property
    def table_name(self) -> str:
        return self.spec.table


@dataclasses.dataclass(frozen=True)
class VectorBacklog:
    """Every row that still needs embedding, grouped by table.

    ``scan_failed`` counts rows whose shape could not be read; they are
    logged and left out of ``tables``.
    """

    tables: tuple[_TableBacklog, ...]
    scan_failed: int = 0

    @property
    def total_rows(self) -> int:
        return sum(len(table.rows) for table in self.tables)

    @property
    def total_tokens(self) -> int:
        """The exact tokens the run would embed, as counted by the scan."""
        return sum(row.tokens for table in self.tables for row in table.rows)


def _extract_row(
    raw: dict[str, Any], spec: VectorProjection, tokenizer: Any
) -> _NullVectorRow | None:
    """Pull one raw row's embed-source text and token count.

    Returns ``None`` when the row's shape is broken (schema drift / bad
    row); it is logged and the caller tallies it as scan-failed.
    """
    try:
        text = spec.text_of(raw)
        row_id = raw["id"]
    except (KeyError, TypeError) as exc:
        logger.warning(
            "cascade_backfill_scan_extract_failed",
            table=spec.table,
            row_id=raw.get("id"),
            error=repr(exc),
        )
        return None
    return _NullVectorRow(
        id=row_id,
        text=text,
        tokens=len(tokenizer.tokenize(text)),
        app_id=str(raw.get("app_id", "default")),
        project_id=str(raw.get("project_id", "default")),
    )


async def _scan_null_vector_backlog(
    tables: Mapping[str, IndexMaintenance], tokenizer: Tokenizer
) -> VectorBacklog:
    """Fetch every row across the 6 business tables that still needs
    embedding.

    One query per table, tokenising each row's embed-source text on the
    way so the pre-run estimate and the later embed call read off the
    exact same text. Returns the backlog plus a count of rows whose
    extraction failed (bad row shape) — tallied by the caller into the
    phase's ``rows_failed``, never raised.

    No row cap: under concurrent server writes (Phase 1 is documented as
    concurrent-safe), a count taken first would let a fresh NULL-vector
    row push a legitimate one past it. The ``where`` filter bounds the
    scan naturally.
    """
    backlog: list[_TableBacklog] = []
    scan_failed = 0
    for spec in _TABLE_SPECS:
        index = tables[spec.table]
        records = await index.scan(spec.null_filter(), with_vectors=True)
        raw_rows = [record.model_dump(mode="python") for record in records]
        rows: list[_NullVectorRow] = []
        for raw in raw_rows:
            row = _extract_row(raw, spec, tokenizer)
            if row is None:
                scan_failed += 1
                continue
            rows.append(row)
        backlog.append(_TableBacklog(spec, index, rows))
    return VectorBacklog(tuple(backlog), scan_failed)


def _truncated_text_prefix(text: str, *, limit: int = 100) -> str:
    """Return the first ``limit`` chars of ``text``, with an ellipsis when
    truncated. Used for per-row failure logs — never log full text, it
    may be user data / PII."""
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


async def _embed_batch(
    provider: EmbeddingProvider, batch: list[_NullVectorRow], table_name: str
) -> list[list[float] | None]:
    """Embed the batch's text with per-row fallback.

    First attempts the batched ``provider.embed_batch(...)`` call — the
    happy path. On any exception (network glitch, one poison row whose
    text exceeds the provider's context limit and 400s the whole batch,
    etc.), logs the batch failure and falls back to per-row
    ``provider.embed(...)`` calls. Each per-row failure is logged with
    the row's id and a truncated text prefix (100 chars) so operators
    can identify poison rows and quarantine them.

    Returns a same-length list aligned with the input ``batch``: each
    entry is either the row's vector, or ``None`` for a row that failed
    even the per-row retry. The caller writes only rows with non-None
    vectors — failed rows keep ``vector IS NULL`` on disk and re-enter
    the next scan's NULL-vector backlog.
    """
    try:
        vectors = await provider.embed_batch([row.text for row in batch])
        return list(vectors)
    except Exception as exc:
        logger.warning(
            "cascade_backfill_batch_embed_failed_falling_back_per_row",
            table=table_name,
            batch_size=len(batch),
            error=repr(exc),
        )
    # Per-row fallback: isolate the poison row(s) so the rest advance.
    results: list[list[float] | None] = []
    for row in batch:
        try:
            vec = await provider.embed(row.text)
        except Exception as exc:
            logger.warning(
                "cascade_backfill_row_embed_failed",
                table=table_name,
                row_id=row.id,
                text_prefix=_truncated_text_prefix(row.text),
                error=repr(exc),
            )
            results.append(None)
            continue
        results.append(vec)
    return results


async def _backfill_table(
    backlog: _TableBacklog,
    provider: EmbeddingProvider,
    *,
    prune_retention: dt.timedelta,
    on_progress: ProgressCallback,
) -> VectorBackfillCounts:
    """Embed + write back every row in one table's backlog, in batches.

    Failure isolation:

    - A row whose embedding failed even the per-row retry keeps
      ``vector IS NULL`` on disk and re-enters the next scan. Counted as
      ``rows_failed`` on this pass.
    - Per-row write failing → tally the row as ``rows_failed`` and
      leave it for the next scan.

    Write-back stays per-row: LanceDB's partial-column ``update``
    takes a single-row predicate, so a failed write only fails that
    one row.
    """
    result = VectorBackfillCounts()
    rows = backlog.rows
    for start in range(0, len(rows), _EMBED_BATCH_SIZE):
        batch = rows[start : start + _EMBED_BATCH_SIZE]
        vectors = await _embed_batch(provider, batch, backlog.table_name)
        for row, vector in zip(batch, vectors, strict=True):
            if vector is None:
                result.rows_failed += 1
                continue
            try:
                await backlog.index.update(
                    backlog.spec.values([vector]),
                    where=primary_key(row.app_id, row.project_id, row.id),
                )
            except Exception:
                result.rows_failed += 1
                logger.warning(
                    "cascade_backfill_row_write_failed",
                    table=backlog.table_name,
                    row_id=row.id,
                    exc_info=True,
                )
                continue
            result.rows_processed += 1
            result.tokens_embedded += row.tokens

        done = min(start + _EMBED_BATCH_SIZE, len(rows))
        on_progress(backlog.table_name, done, len(rows))

    # Compact fragments accumulated during backfill. Every per-row
    # update opens a fresh LanceDB manifest version; without periodic
    # optimize() the on-disk directory grows unbounded (same failure
    # mode as v1.1.3's FTS optimize gap, #336 /
    # lance-format/lance#7653). Gate on rows_processed > 0 so a no-op
    # backlog doesn't pay the compact cost. Best-effort maintenance:
    # a failure here does not invalidate the writes.
    #
    # ``cleanup_older_than=timedelta(0)`` also physically prunes
    # older manifest versions right now — without it, ``optimize()``
    # only compacts fragments and leaves the pre-compaction manifest
    # chain on disk, so the ``.index/lancedb`` directory still grows
    # (round-4 review M2). Backfill has no reason to preserve older
    # manifest versions — pruning immediately mirrors the migration
    # path (see ``lancedb/__init__.py:140``).
    if result.rows_processed > 0:
        try:
            # optimize() compacts the per-row-update fragments; prune()
            # physically reclaims the superseded manifest versions. Split
            # after the repo API separated them (compact is lock-free, prune
            # runs under the write lock) and cross-process safe because prune
            # passes delete_unverified=False.
            #
            # Keep the daemon's retention window rather than reclaiming at
            # zero age: the window's job is to outlive an in-flight read (a
            # /search holding a version reference), and this runs in a
            # separate process where the write lock cannot fence one. Files
            # younger than the window are reclaimed by the next daemon prune.
            await backlog.index.optimize()
            await backlog.index.prune(prune_retention)
            logger.info(
                "cascade_backfill_table_optimized",
                table=backlog.table_name,
                rows_processed=result.rows_processed,
            )
        except Exception as exc:
            logger.warning(
                "cascade_backfill_table_optimize_failed",
                table=backlog.table_name,
                error=repr(exc),
            )
    return result


class VectorBackfill(Protocol):
    """Find and re-embed derived-index rows that lack their vector."""

    async def scan(self) -> VectorBacklog:
        """Collect every incomplete row across the vector-carrying tables."""
        ...

    async def run(
        self,
        backlog: VectorBacklog,
        embedding: EmbeddingProvider,
        *,
        on_progress: ProgressCallback,
    ) -> VectorBackfillCounts:
        """Embed and write back ``backlog``, one table after another.

        Failures are tallied in the returned counts, never raised.
        """
        ...


class ProjectionVectorBackfill:
    """:class:`VectorBackfill` over the tables :data:`VECTOR_PROJECTIONS` declares.

    Args:
        tables: Every derived-index table that carries a vector column, by
            table name.
        tokenizer: Counts each row's tokens for the estimate (the same
            tokenizer cascade uses for BM25).
        prune_retention: How old superseded index versions must be before a
            table's run reclaims them (the daemon's window).

    Raises:
        RuntimeError: ``tables`` and :data:`VECTOR_PROJECTIONS` name different
            tables -- a table added without a projection would be counted by
            the startup "unbackfilled rows" hint and silently skipped here.
    """

    def __init__(
        self,
        tables: Mapping[str, IndexMaintenance],
        *,
        tokenizer: Tokenizer,
        prune_retention: dt.timedelta,
    ) -> None:
        _check_tables(tables)
        self._tables = tables
        self._tokenizer = tokenizer
        self._prune_retention = prune_retention

    async def scan(self) -> VectorBacklog:
        return await _scan_null_vector_backlog(self._tables, self._tokenizer)

    async def run(
        self,
        backlog: VectorBacklog,
        embedding: EmbeddingProvider,
        *,
        on_progress: ProgressCallback,
    ) -> VectorBackfillCounts:
        counts = VectorBackfillCounts()
        for table in backlog.tables:
            if not table.rows:
                continue
            counts.add(
                await _backfill_table(
                    table,
                    embedding,
                    prune_retention=self._prune_retention,
                    on_progress=on_progress,
                )
            )
        return counts


def _check_tables(tables: Mapping[str, object]) -> None:
    covered = {spec.table for spec in _TABLE_SPECS}
    if covered != set(tables):
        raise RuntimeError(
            f"VECTOR_PROJECTIONS drift: the vector-carrying index tables are "
            f"{sorted(tables)}, VECTOR_PROJECTIONS covers {sorted(covered)}. "
            f"Declare a VectorProjection for every business table with a "
            f"nullable vector column."
        )
