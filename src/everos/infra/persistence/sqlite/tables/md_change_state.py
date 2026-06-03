"""``md_change_state`` — cascade work queue.

One row per markdown path. Both watcher (real-time fsevents) and
scanner (periodic sweep) UPSERT into this table; the worker consumes
``pending`` rows in ``lsn`` order, transitions them through an
internal ``processing`` claim state, and lands them in ``done`` or
``failed`` (with a ``retryable`` flag).

Schema sourced from ``12_cascade_design.md`` §4.1 + decisions DD-3 …
DD-12; the four indexes below are required by ``13_cascade_design.md``
§7 status / fix queries.
"""

from __future__ import annotations

from sqlalchemy import Index, text

from everos.component.utils.datetime import UtcDatetime, get_utc_now
from everos.core.persistence.sqlite import BaseTable, Field
from everos.core.persistence.sqlite.base import UtcDateTimeColumn


class MdChangeState(BaseTable, table=True):
    """One row per markdown path; UPSERT-driven work queue for cascade.

    The public state machine is the 3-tuple ``pending`` / ``done`` /
    ``failed`` (12 doc §6). ``processing`` is an internal claim state
    used by :meth:`MdChangeStateRepo.claim_one` and rolled back into
    ``pending`` for CLI / status output (16 doc §4.2 — DD-12 keeps the
    public surface clean).
    """

    __tablename__ = "md_change_state"  # type: ignore[assignment]
    __table_args__ = (
        # Worker scans pending rows in lsn order — partial index drops
        # done/failed rows from the b-tree and keeps it tight.
        Index(
            "idx_md_change_pending",
            "status",
            "lsn",
            sqlite_where=text("status = 'pending'"),
        ),
        # `cascade fix --apply` only ever touches failed + retryable=TRUE
        # rows — partial index makes that pass essentially O(retryable).
        Index(
            "idx_md_change_retryable",
            "status",
            "retryable",
            sqlite_where=text("status = 'failed' AND retryable = 1"),
        ),
        # Scanner reverse-reconcile (disk → state) compares mtime.
        Index("idx_md_change_mtime", "mtime"),
        # `cascade status` aggregates by kind.
        Index("idx_md_change_kind", "kind"),
    )

    md_path: str = Field(primary_key=True)
    """Path relative to the memory-root (e.g. ``users/u_jason/
    episodes/episode-2026-05-12.md``). Every reverse-link anchors here."""

    kind: str = Field(nullable=False, index=True)
    """Kind registry name (e.g. ``"episode"``); worker dispatches the
    matching handler."""

    change_type: str = Field(nullable=False)
    """``"added"`` | ``"modified"`` | ``"deleted"``. A hint for the
    worker — handler re-derives truth from the actual file state."""

    mtime: float = Field(default=0.0, nullable=False)
    """File mtime captured when the row was last UPSERTed. Scanner
    compares this against the on-disk mtime to identify dirty paths."""

    first_seen_at: UtcDatetime = Field(
        default_factory=get_utc_now, sa_type=UtcDateTimeColumn
    )
    """When the path was first enqueued."""

    last_changed_at: UtcDatetime = Field(
        default_factory=get_utc_now, sa_type=UtcDateTimeColumn
    )
    """Most recent UPSERT timestamp (re-stamped on every re-enqueue)."""

    lsn: int = Field(nullable=False, index=True)
    """Global monotonic sequence (``MAX(lsn) + 1`` per UPSERT). Worker
    processes pending rows in ascending lsn order; the gap between
    ``MAX(lsn)`` and the last processed lsn is the queue lag."""

    status: str = Field(default="pending", nullable=False, index=True)
    """Lifecycle:

    - ``"pending"`` — waiting for the worker.
    - ``"processing"`` — claimed by a worker (internal; CLI rolls into
      pending for display).
    - ``"done"`` — handler completed successfully.
    - ``"failed"`` — handler exhausted retries or hit an
      unrecoverable error (see :attr:`retryable`).
    """

    retryable: bool | None = Field(default=None)
    """Meaningful only when ``status='failed'``.

    - ``TRUE`` — RecoverableError exhausted MAX_RETRY; ``cascade fix
      --apply`` will re-enqueue this row (pending, retry_count reset).
    - ``FALSE`` — UnrecoverableError (malformed YAML, schema error
      etc.); requires editing the md and re-saving.
    - ``NULL`` — not a failed row (pending / processing / done).
    """

    last_attempt_at: UtcDatetime | None = Field(default=None, sa_type=UtcDateTimeColumn)
    """Timestamp of the most recent worker attempt (success or
    failure)."""

    retry_count: int = Field(default=0, nullable=False)
    """Number of retries the worker has *actually issued* (the first
    attempt does not count). Reaches MAX_RETRY (default 3) before the
    row transitions to ``failed`` with ``retryable=TRUE``."""

    error: str | None = Field(default=None)
    """Most recent failure message (truncated upstream if needed)."""
