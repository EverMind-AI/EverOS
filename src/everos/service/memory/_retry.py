"""Segments whose extraction did not finish, extracted again by a later call.

A segment's memcell is committed, and its messages leave the buffer, before
its episode call runs. When that call fails (a provider's 5xx or timeout, an
unusable reply on every attempt), or the process dies before the episode is
written, nothing else would ever extract it: the buffer is empty, and a group's
ledger holds its messages. So every memcell row is cut marked "to extract"
(``MemcellRecord.to_extract``); the mark is cleared when its extraction
succeeds and counts a failure when it fails
(:meth:`~everos.memory.ports.MemcellStore.record_extraction_failure`), and the
session's next add or flush -- in this process or after a restart -- extracts
a marked segment again:

* **Never twice at once.** A segment is held by the call that cut it, or the
  call that claimed it under its session's lock, until that call's attempt
  ends; no other call takes a held segment.
* **Never twice over.** A retry skips every owner whose episode of the segment
  is already written (a failure after a first copy's write). A message the
  client sends again while its segment waits is that segment's retry, not a
  new message: it is not taken again (:func:`resent_positions`) -- nor counted
  in ``duplicates`` -- and the segment is extracted again in that call, so it
  is never cut into a second segment.
* **Bounded.** An add retries a segment it does not carry once
  :data:`RETRY_AFTER_MS` passed since its last failure (or its cut), a flush
  at once; after :data:`MAX_EXTRACTION_FAILURES` failures it is left, marked,
  and logged (``memorize_cell_extraction_abandoned``), and its messages are
  then new again.
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
    "resent_positions",
]

from collections.abc import Collection, Iterable, Mapping, Sequence
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
"""How long after a failure (or the cut, before any) an add that does not send
the segment's messages again tries it again; a flush does not wait. A provider
that is down fails every attempt, and a chat's adds come seconds apart: the
wait keeps one outage from using up every attempt."""


class SegmentRetries:
    """The marked segments of each session, held by one call at a time.

    Args:
        memcells: Where the marks are kept; ``None`` retries nothing (a build
            without the memcell store).
        forget_log: What a retry checks for a forget that stands over the
            segment; ``None`` checks nothing.
    """

    def __init__(
        self, memcells: MemcellStore | None, forget_log: ForgetLogStore | None
    ) -> None:
        self._memcells = memcells
        self._forget_log = forget_log
        self._held: set[str] = set()

    def hold(self, memcell_ids: Iterable[str]) -> None:
        """Hold segments a call just cut, under the session's lock, until
        their extraction ends (:meth:`release`)."""
        self._held.update(memcell_ids)

    def release(self, memcell_ids: Iterable[str]) -> None:
        """Drop a call's hold on these segments."""
        self._held.difference_update(memcell_ids)

    async def pending(
        self, session: SessionKey
    ) -> tuple[list[FailedSegment], list[FailedSegment]]:
        """The session's marked segments still to extract -- under
        :data:`MAX_EXTRACTION_FAILURES`, and no standing forget over those no
        call holds -- and, of those, the ones no call held when the read
        began: only those may be claimed.

        A held segment's extraction may end while the marks are read -- its
        mark cleared, its hold dropped -- and the read still return it; taking
        it then would extract it twice. Read under the session's lock: a
        re-send of a pending segment's messages is its retry
        (:func:`resent_positions`), and :meth:`claim` takes the due ones.
        """
        if self._memcells is None:
            return [], []
        held = set(self._held)
        found: list[FailedSegment] = []
        for segment in await self._memcells.failed_segments(session):
            if segment.failures >= MAX_EXTRACTION_FAILURES:
                continue
            if segment.memcell_id not in held and await self._forgotten(
                session, segment
            ):
                continue
            found.append(segment)
        return found, [s for s in found if s.memcell_id not in held]

    def claim(
        self,
        unheld: Sequence[FailedSegment],
        *,
        is_final: bool,
        resent: Collection[str] = (),
    ) -> list[FailedSegment]:
        """The ``unheld`` segments (:meth:`pending`) due now, held by the caller.

        A flush takes every one, an add the ones whose messages it sent again
        (``resent``, memcell ids) and those whose last failure (or cut) is
        :data:`RETRY_AFTER_MS` old. The caller extracts them and then
        :meth:`release` s them, whatever happened.
        """
        now = to_timestamp_ms(get_utc_now())
        due = [
            segment
            for segment in unheld
            if segment.memcell_id not in self._held
            and (
                is_final
                or segment.memcell_id in resent
                or now - segment.failed_at_ms >= RETRY_AFTER_MS
            )
        ]
        self.hold(segment.memcell_id for segment in due)
        return due

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

    async def extracted(self, memcell_id: str) -> None:
        """The segment is extracted: clear its mark. Never raises (a mark
        left behind costs one retry that finds every copy written)."""
        if self._memcells is None:
            return
        try:
            await self._memcells.clear_extraction_failure(memcell_id)
        except Exception as exc:
            logger.warning(
                "memorize_cell_mark_not_cleared",
                memcell_id=memcell_id,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _forgotten(self, session: SessionKey, segment: FailedSegment) -> bool:
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


def resent_positions(
    ids: Sequence[str], pending: Sequence[FailedSegment]
) -> tuple[frozenset[int], set[str]]:
    """The positions of a batch (by generated ``ids``) whose message a pending
    segment holds -- sent again while the segment waits -- and those
    segments' memcell ids."""
    owner = {
        mid: segment.memcell_id for segment in pending for mid in segment.message_ids
    }
    positions = frozenset(i for i, mid in enumerate(ids) if mid in owner)
    return positions, {owner[ids[i]] for i in positions}


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
