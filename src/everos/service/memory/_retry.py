"""Segments whose extraction failed, extracted again by their session's next call.

A segment's memcell is committed, and its messages leave the buffer, before
its episode call runs. When that call fails (a provider's 5xx or timeout, an
unusable reply on every attempt) nothing else would ever extract it: the
buffer is empty, a group's ledger holds its messages, and a re-send is a
duplicate. So the failure is recorded on the memcell row
(:meth:`~everos.memory.ports.MemcellStore.record_extraction_failure`), and the
session's next add or flush extracts the segment again:

* **Never twice at once.** A segment is claimed under its session's lock and
  held by one call until that call's attempt ends; a segment whose first
  extraction is still running has no failure recorded, so no call claims it.
* **Never twice over.** A retry skips every owner whose episode of the segment
  is already written (a failure after a first copy's write), and a re-send of
  the segment's messages while it waits is a duplicate
  (:func:`held_positions`, by the ids
  :func:`~everos.memory.extract.ingest.gen_message_id` gives them; a group's
  ledger holds them too), so it is never cut into a second segment.
* **Bounded.** An add retries a segment once :data:`RETRY_AFTER_MS` passed
  since its last failure, a flush at once; after
  :data:`MAX_EXTRACTION_FAILURES` failures it is left, marked, and logged
  (``memorize_cell_extraction_abandoned``).
* **Forgotten stays forgotten.** A segment is not extracted while a forget of
  its conversation stands -- or, for a user's, a forget of everything or by
  condition, or for a group's, a member forget: they were recorded without
  it, and its episode would bring back what they took out. It stays marked,
  so an undo lets a later call extract it; meanwhile its messages are not
  held, so a conversation sent again after its delete is taken.
"""

from __future__ import annotations

__all__ = [
    "MAX_EXTRACTION_FAILURES",
    "RETRY_AFTER_MS",
    "SegmentRetries",
    "held_positions",
]

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import get_utc_now, to_timestamp_ms
from everos.memory.ports import FailedSegment, Scope, SessionKey

if TYPE_CHECKING:
    from everalgo.types import MemCell

    from everos.memory.ports import ForgetLogStore, MemcellStore, OwnerType

logger = get_logger("everos.service.memory")

MAX_EXTRACTION_FAILURES = 3
"""Failed extractions after which a segment is no longer tried again."""

RETRY_AFTER_MS = 30_000
"""How long after a failure an add tries the segment again; a flush does not
wait. A provider that is down fails every attempt, and a chat's adds come
seconds apart: the wait keeps one outage from using up every attempt."""


class SegmentRetries:
    """The failed segments of each session, claimed by one call at a time.

    Args:
        memcells: Where failures are recorded; ``None`` records and retries
            nothing (a build without the memcell store).
        forget_log: What a retry checks for a forget that stands over the
            segment; ``None`` checks nothing.
    """

    def __init__(
        self, memcells: MemcellStore | None, forget_log: ForgetLogStore | None
    ) -> None:
        self._memcells = memcells
        self._forget_log = forget_log
        self._claimed: set[str] = set()

    async def pending(self, session: SessionKey) -> list[FailedSegment]:
        """The session's failed segments still to extract: under
        :data:`MAX_EXTRACTION_FAILURES`, and no standing forget over them.

        Read under the session's lock. A re-send of one of their messages is
        a duplicate (:func:`held_positions`); the due ones are :meth:`claim` ed.
        """
        if self._memcells is None:
            return []
        found: list[FailedSegment] = []
        for segment in await self._memcells.failed_segments(session):
            if segment.failures >= MAX_EXTRACTION_FAILURES:
                continue
            if await self._held_by_forget(session, segment):
                continue
            found.append(segment)
        return found

    def claim(
        self, pending: Sequence[FailedSegment], *, is_final: bool
    ) -> list[FailedSegment]:
        """The ``pending`` segments due now, held by the caller.

        A flush takes every one, an add those whose last failure is
        :data:`RETRY_AFTER_MS` old; a segment another call holds is left to
        it. The caller extracts them and then :meth:`release` s them,
        whatever happened.
        """
        now = to_timestamp_ms(get_utc_now())
        due = [
            segment
            for segment in pending
            if segment.memcell_id not in self._claimed
            and (is_final or now - segment.failed_at_ms >= RETRY_AFTER_MS)
        ]
        self._claimed.update(segment.memcell_id for segment in due)
        return due

    def release(self, segments: Sequence[FailedSegment]) -> None:
        """Drop the caller's hold on ``segments``."""
        self._claimed.difference_update(segment.memcell_id for segment in segments)

    async def failed(
        self,
        *,
        memcell_id: str,
        cell: MemCell,
        sender_ids: Sequence[str],
        message_ids: Sequence[str],
        group_id: str | None,
        picture_names: Mapping[str, str],
        failures: int,
    ) -> None:
        """Record the segment's ``failures``-th failed extraction.

        Never raises: the extraction's own error is what the caller reports.
        """
        if self._memcells is None:
            return
        try:
            await self._memcells.record_extraction_failure(
                FailedSegment(
                    memcell_id=memcell_id,
                    cell=cell,
                    sender_ids=tuple(sender_ids),
                    message_ids=tuple(message_ids),
                    group_id=group_id,
                    picture_names=dict(picture_names),
                    failures=failures,
                    failed_at_ms=to_timestamp_ms(get_utc_now()),
                )
            )
        except Exception as exc:
            logger.error(
                "memorize_cell_failure_not_recorded",
                memcell_id=memcell_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            return
        if failures >= MAX_EXTRACTION_FAILURES:
            logger.error(
                "memorize_cell_extraction_abandoned",
                memcell_id=memcell_id,
                failures=failures,
            )

    async def extracted(self, segment: FailedSegment) -> None:
        """The segment is extracted: clear its failure."""
        if self._memcells is not None:
            await self._memcells.clear_extraction_failure(segment.memcell_id)

    async def _held_by_forget(
        self, session: SessionKey, segment: FailedSegment
    ) -> bool:
        """Whether a forget that stands covers the segment (see the module doc)."""
        if self._forget_log is None:
            return False
        if segment.group_id is not None:
            scopes = [_scope(session, segment.group_id, "group")]
        else:
            scopes = [_scope(session, h, "user") for h in _humans(segment.cell)]
        for scope in scopes:
            kinds = (
                ("session", "member")
                if scope.owner_type == "group"
                else ("session", "owner", "condition")
            )
            for kind in kinds:
                targets = await self._forget_log.live_targets(
                    target=kind,
                    app_id=scope.app_id,
                    project_id=scope.project_id,
                    owner_id=scope.owner_id,
                    owner_type=scope.owner_type,
                )
                if _covers(kind, targets, session, segment):
                    logger.info(
                        "memorize_cell_retry_held_by_forget",
                        memcell_id=segment.memcell_id,
                        owner_id=scope.owner_id,
                        forget=kind,
                    )
                    return True
        return False


def held_positions(
    ids: Sequence[str], pending: Sequence[FailedSegment]
) -> frozenset[int]:
    """The positions of a batch (by generated ``ids``) whose message a pending
    segment holds: sent again while the segment waits for its retry."""
    held = {mid for segment in pending for mid in segment.message_ids}
    return frozenset(i for i, mid in enumerate(ids) if mid in held)


def _covers(
    kind: str, targets: Sequence[str], session: SessionKey, segment: FailedSegment
) -> bool:
    if kind == "session":
        return session.session_id in targets
    if kind == "member":
        return any(sender in targets for sender in segment.sender_ids)
    return bool(targets)


def _scope(session: SessionKey, owner_id: str, owner_type: OwnerType) -> Scope:
    return Scope(
        app_id=session.app_id,
        project_id=session.project_id,
        owner_id=owner_id,
        owner_type=owner_type,
    )


def _humans(cell: MemCell) -> list[str]:
    """The cell's role ``user`` senders, first seen first: its owners."""
    found: list[str] = []
    for item in cell.items:
        sender = getattr(item, "sender_id", None)
        if getattr(item, "role", None) == "user" and sender and sender not in found:
            found.append(sender)
    return found
