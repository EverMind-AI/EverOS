"""Re-embed derived-index rows written without a vector.

Rows written while no embedding provider was configured carry
``vector IS NULL`` (and an episode may lack only its ``subject_vector``).
:class:`ProjectionVectorBackfill` finds them in every vector-carrying index
table through the shared :data:`VECTOR_PROJECTIONS`, and writes each row's
missing side(s) back with a partial-column update, falling back row by row
when a batch fails. Confirmation, presentation and the phase's outcome are
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

import asyncio
import dataclasses
import datetime as dt
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol

from everos.core.observability.logging import get_logger
from everos.memory.ports import Filter, primary_key

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
"""Rows grouped per ``embed_batch`` call (one for the primary text, one
more only if the group has subject rows) — not the provider's own
request chunk size. The provider's configured ``batch_size`` /
``max_concurrent`` (``embedding.batch_size`` / ``embedding.max_concurrent``
in settings) govern the actual in-flight request shape from there; this
just bounds per-batch failure isolation + progress feedback granularity
for a bulk migration."""


@dataclasses.dataclass(frozen=True)
class _NullVectorRow:
    """One row missing (at least one side of) its vector.

    ``tokens`` is the real token count (via the same tokenizer cascade
    uses for BM25), not an estimate — computed once here and never
    recomputed before the embed call.

    ``needs_primary`` / ``needs_subject`` mark which side(s) of the
    row's embedding are still NULL. Round-1 assumed "in the backlog ⇒
    primary vector is NULL"; round-2 widened the scan filter to also
    pick up rows where only ``subject_vector`` is NULL (Episode's
    orthogonal second embed can fail on its own — see
    :func:`_null_filter`). ``_backfill_table`` reads these flags to
    embed only the side that actually needs work, so a re-run after a
    partial failure doesn't waste an ``embed_batch`` call on the side
    that already succeeded.

    Both default to ``True`` so pre-existing test fixtures that build
    :class:`_NullVectorRow` directly keep their round-1 semantics
    (embed everything).
    """

    id: str
    text: str
    subject_text: str | None
    tokens: int
    needs_primary: bool = True
    needs_subject: bool = True
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
    """Every row that still needs (some) embedding, grouped by table.

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


def _null_filter(spec: VectorProjection) -> Filter:
    """Where-clause the scan uses to spot rows that need (some) embedding.

    Episode carries a ``subject_vector`` alongside ``vector`` — the two
    are orthogonal embeds (see :class:`Episode`), and either being NULL
    means the row is incompletely embedded. Round-1 filtered only on
    ``vector IS NULL``, which silently missed rows where the primary
    embed succeeded but ``_embed_subject_batch`` failed on its own
    (``subject_vector`` left NULL). Round-2 widens the filter to
    ``vector IS NULL OR subject_vector IS NULL`` so those rows
    re-enter the backlog on the next scan.

    Other tables have no subject column; adding the OR clause there
    would reference an unknown field and LanceDB would reject the
    query, so they keep the round-1 filter.
    """
    return spec.null_filter()


class _RowSkipped:
    """Sentinel returned by :func:`_extract_row` when the row shape is
    fine but nothing actually needs embedding (widened Episode filter
    matched on a legitimately-NULL ``subject_vector``). Distinct from
    ``None`` — which the caller tallies as a scan-failure — so the
    two paths never conflate."""


_ROW_SKIPPED = _RowSkipped()


def _extract_row(
    raw: dict[str, Any], spec: VectorProjection, tokenizer: Any
) -> _NullVectorRow | _RowSkipped | None:
    """Pull one raw LanceDB row's embed-source text, token count, and
    which side(s) still need embedding.

    Return values:

    - :class:`_NullVectorRow` — the row has real work; carries the
      ``needs_primary`` / ``needs_subject`` flags read off the raw
      vector columns.
    - :data:`_ROW_SKIPPED` — the row's shape is fine but nothing needs
      embedding (widened Episode filter matched on a legitimately-NULL
      ``subject_vector``). Silent; caller drops it.
    - ``None`` — the row's shape is broken (schema drift / bad row).
      Logged; caller tallies as scan-failed.
    """
    try:
        text = spec.text_of(raw)
        subject_text = spec.subject_of(raw) if spec.subject_of else None
        row_id = raw["id"]
    except (KeyError, TypeError) as exc:
        logger.warning(
            "cascade_backfill_scan_extract_failed",
            table=spec.table,
            row_id=raw.get("id"),
            error=repr(exc),
        )
        return None
    needs_primary, needs_subject = spec.missing_sides(raw, subject_text)
    if not needs_primary and not needs_subject:
        return _ROW_SKIPPED
    token_source = f"{text} {subject_text}" if needs_subject else text
    return _NullVectorRow(
        id=row_id,
        text=text,
        subject_text=subject_text,
        tokens=len(tokenizer.tokenize(token_source)),
        needs_primary=needs_primary,
        needs_subject=needs_subject,
        app_id=str(raw.get("app_id", "default")),
        project_id=str(raw.get("project_id", "default")),
    )


async def _scan_null_vector_backlog(
    tables: Mapping[str, IndexMaintenance], tokenizer: Tokenizer
) -> VectorBacklog:
    """Fetch every row across the 6 business tables that still needs
    (some) embedding.

    One query per table, tokenising each row's embed-source text on the
    way so the pre-run estimate and the later embed call read off the
    exact same text. Returns the backlog plus a count of rows whose
    extraction failed (bad row shape) — tallied by the caller into the
    phase's ``rows_failed``, never raised.

    Round-2 changes:

    - The Episode filter is widened to
      ``vector IS NULL OR subject_vector IS NULL`` via
      :func:`_null_filter` so a subject-only failure doesn't hide the
      row from subsequent backfill runs.
    - The prior ``count_rows`` + ``.limit(null_count)`` split is
      dropped. Under concurrent server writes (Phase 1 is documented
      as concurrent-safe), a fresh NULL-vector row inserted between
      the count and the query would push a legitimate row past the
      cap. LanceDB's ``.to_list()`` streams the whole predicate result;
      the ``where`` filter bounds it naturally.
    """
    backlog: list[_TableBacklog] = []
    scan_failed = 0
    for spec in _TABLE_SPECS:
        index = tables[spec.table]
        records = await index.scan(_null_filter(spec))
        raw_rows = [record.model_dump(mode="python") for record in records]
        rows: list[_NullVectorRow] = []
        for raw in raw_rows:
            row = _extract_row(raw, spec, tokenizer)
            if row is None:
                scan_failed += 1
                continue
            if isinstance(row, _RowSkipped):
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


async def _embed_primary_batch(
    provider: EmbeddingProvider, batch: list[_NullVectorRow], table_name: str
) -> list[list[float] | None]:
    """Embed the batch's primary text with per-row fallback.

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


async def _embed_subject_batch(
    provider: EmbeddingProvider, batch: list[_NullVectorRow], table_name: str
) -> dict[tuple[str, str, str], list[float]]:
    """Embed the batch's ``subject_text`` with per-row fallback (episode's
    second, independent embed — see ``EpisodeHandler._build_row``).

    A separate batch call because not every row has a subject; folding
    it into the primary call would either skip absent subjects (index
    mismatch) or force a dense-but-wrong text list.

    Returns a ``{row.key: vector}`` map. Rows whose per-row retry also failed
    are absent from the map, and their ``subject_vector`` stays NULL —
    the widened ``vector IS NULL OR subject_vector IS NULL`` scan filter
    (M1) picks them up on the next backfill run.
    """
    subject_rows = [row for row in batch if row.subject_text is not None]
    if not subject_rows:
        return {}
    try:
        vectors = await provider.embed_batch(
            [row.subject_text for row in subject_rows]  # type: ignore[misc]
        )
        return {row.key: vec for row, vec in zip(subject_rows, vectors, strict=True)}
    except Exception as exc:
        logger.warning(
            "cascade_backfill_subject_batch_embed_failed_falling_back_per_row",
            table=table_name,
            batch_size=len(subject_rows),
            error=repr(exc),
        )
    results: dict[tuple[str, str, str], list[float]] = {}
    for row in subject_rows:
        text = row.subject_text
        assert text is not None  # guarded by the filter above
        try:
            vec = await provider.embed(text)
        except Exception as exc:
            logger.warning(
                "cascade_backfill_row_subject_embed_failed",
                table=table_name,
                row_id=row.id,
                text_prefix=_truncated_text_prefix(text),
                error=repr(exc),
            )
            continue
        results[row.key] = vec
    return results


async def _backfill_table(
    backlog: _TableBacklog,
    provider: EmbeddingProvider,
    *,
    prune_retention: dt.timedelta,
    on_progress: ProgressCallback,
) -> VectorBackfillCounts:
    """Embed + write back every row in one table's backlog, in batches.

    Each batch is split by which side each row still needs — the
    orthogonal side (Episode's ``subject_vector``) is a separate
    ``embed_batch`` call, and a row that already has one side filled
    (e.g. primary succeeded on a prior run, subject failed) only pays
    the cost of the missing side. When both sides have work in the
    same batch the two calls run concurrently under ``asyncio.gather``
    (round-1 concurrency invariant preserved).

    Failure isolation:

    - Whole primary batch failing → those rows keep ``vector IS NULL``
      on disk; the widened scan filter picks them up on the next run.
      Counted as ``rows_failed`` on this pass.
    - Whole subject batch failing → the touched rows keep
      ``subject_vector IS NULL`` on disk; the widened scan filter
      picks them up on the next run (round-1 fix #10's silent
      corruption is closed here).
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
        primary_rows = [r for r in batch if r.needs_primary]
        subject_rows = [
            r for r in batch if r.needs_subject and r.subject_text is not None
        ]

        # Concurrent embed only for the sides that have work this
        # batch. ``asyncio.gather`` under both-need shape preserves the
        # round-1 wall-clock invariant; one-sided batches skip the
        # gather to avoid a wasted ``embed_batch({})`` round-trip.
        primary_vectors: list[list[float] | None]
        subject_vectors: dict[tuple[str, str, str], list[float]]
        if primary_rows and subject_rows:
            primary_vectors, subject_vectors = await asyncio.gather(
                _embed_primary_batch(provider, primary_rows, backlog.table_name),
                _embed_subject_batch(provider, subject_rows, backlog.table_name),
            )
        elif primary_rows:
            primary_vectors = await _embed_primary_batch(
                provider, primary_rows, backlog.table_name
            )
            subject_vectors = {}
        elif subject_rows:
            primary_vectors = []
            subject_vectors = await _embed_subject_batch(
                provider, subject_rows, backlog.table_name
            )
        else:
            # Purely defensive: the scan filters no-op rows out via
            # ``_ROW_SKIPPED``, so an empty split shouldn't happen in
            # practice. Still emit progress so the readout stays
            # monotonic.
            done = min(start + _EMBED_BATCH_SIZE, len(rows))
            on_progress(backlog.table_name, done, len(rows))
            continue

        # ``primary_vectors`` is a same-length list aligned with
        # ``primary_rows``: successful rows carry the vector, poison
        # rows (whose per-row retry also failed) carry ``None``. Count
        # the poison rows as ``rows_failed`` — they keep ``vector IS
        # NULL`` on disk and re-enter the next scan.
        primary_map: dict[tuple[str, str, str], list[float]] = {}
        for r, v in zip(primary_rows, primary_vectors, strict=True):
            if v is None:
                result.rows_failed += 1
            else:
                primary_map[r.key] = v
        # ``subject_vectors`` is already a ``{row.key: vec}`` dict; keys
        # absent mean the per-row subject retry also failed and the
        # row keeps ``subject_vector IS NULL``, re-entering the backlog
        # via the widened NULL scan next run.

        for row in batch:
            updates: dict[str, Any] = {}
            if row.key in primary_map:
                updates[backlog.spec.primary_column] = primary_map[row.key]
            if row.key in subject_vectors:
                updates[backlog.spec.subject_column] = subject_vectors[row.key]
            if not updates:
                continue
            # A row is only "fully processed" if every side it needed
            # actually made it into ``updates``. When the primary
            # succeeded but the subject batch failed (or per-row retry
            # dropped the subject), the row still carries a NULL side
            # on disk and the widened scan filter (``vector IS NULL OR
            # subject_vector IS NULL``) re-picks it next run —
            # automation must see a non-zero ``rows_failed`` for the
            # exit code to escalate to COMPLETED_WITH_FAILURES (4)
            # instead of falsely reporting SUCCESS (0). Two guards
            # tighten the check: (1) consult per-side ``needs_*`` so a
            # row where ``needs_primary=False`` (primary already on
            # disk from a prior pass) isn't counted as a gap; (2) gate
            # the subject-side arm on the SPEC having a subject_of
            # extractor at all — tables without a subject column
            # (e.g. atomic_fact) intentionally leave every row absent
            # from ``subject_vectors`` and must not be flagged.
            spec_has_subject = backlog.spec.subject_of is not None
            row_has_subject = row.subject_text is not None
            side_gap = (row.needs_primary and row.key not in primary_map) or (
                spec_has_subject
                and row.needs_subject
                and row_has_subject
                and row.key not in subject_vectors
            )
            try:
                await backlog.index.update(
                    updates, where=primary_key(row.app_id, row.project_id, row.id)
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
            # Partial-side gap: the write succeeded for at least one
            # side but some needed side is still NULL. Count against
            # ``rows_failed`` so the CLI's exit-4 (COMPLETED_WITH_FAILURES)
            # semantics fires instead of silently exit-0.
            if side_gap:
                result.rows_failed += 1
                logger.warning(
                    "cascade_backfill_row_partial_embed",
                    table=backlog.table_name,
                    row_id=row.id,
                    missing_side=(
                        "primary" if row.key not in primary_map else "subject"
                    ),
                )
                continue
            # A row is "processed" once every side it needed advanced
            # this pass.
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
    """Find and re-embed derived-index rows that lack (a side of) their vector."""

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
