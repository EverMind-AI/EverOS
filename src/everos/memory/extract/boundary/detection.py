"""Boundary detection: the session buffer cut into memcells.

Owns the buffer / merge / boundary / tail sequence that feeds the extraction
pipelines, and the ``memcell`` ledger: each cell gets exactly one row, and
the pipeline references that row's ``memcell_id``.

:func:`everalgo.boundary.adetect_boundaries` runs over one buffer track of
chat rows: tool rows are dropped before anything is buffered or cut
(:func:`~.messages.chat_rows`), and so is a tool row an older build left in
a session's buffer.

A buffered row a live forget names (``message_forget``) is read by no
model call: it stays buffered, out of every cut, while the forget stands,
and an undo makes it an ordinary row of the next cut.
"""

from __future__ import annotations

__all__ = ["BoundaryDetection", "BoundaryOutcome", "Status"]

import asyncio
import uuid
from collections.abc import Sequence
from typing import Literal, NamedTuple

from everalgo.boundary import DetectionResult, adetect_boundaries
from everalgo.llm import LLMClient, LLMOutputError
from everalgo.types import MemCell

from everos.config import BoundaryDetectionSettings
from everos.core.observability.logging import get_logger
from everos.core.observability.tracing import memory_span
from everos.core.utils.datetime import from_timestamp_ms
from everos.memory.custom_instructions import CustomInstructions
from everos.memory.extract.llm_retry import retry_llm_output
from everos.memory.models import CanonicalMessage, IngestResult, MemcellRecord
from everos.memory.ports import MessageForgetStore, SessionBufferStore, SessionKey

from .messages import (
    chat_rows,
    close_complete_tail,
    merge_dedupe_sort,
    picture_names,
    slice_tail,
    split_message_ids,
    to_conversation_item,
    unique_senders,
)
from .prompt_order import CountAfterMessages
from .quarantine import SessionQuarantine

logger = get_logger(__name__)

type Status = Literal["accumulated", "extracted", "skipped"]

_RAW_TYPE = "Conversation"
"""The memcell row's ``raw_type``: every cell is a slice of a chat."""


class BoundaryOutcome(NamedTuple):
    """What one call produced; the lists are parallel, index ``i`` is cell ``i``.

    Attributes:
        cells: The memcells cut this call.
        memcell_ids: Their ids, minted here.
        per_cell_message_ids: The buffered message ids each cell consumed.
        per_cell_all_senders: Each cell's distinct senders, first-seen order.
        status: ``extracted`` when cells were cut, ``accumulated`` when the
            messages wait in the buffer, ``skipped`` when there was nothing
            to do or no model to ask.
        message_count: Fresh chat rows this call contributed.
        tail_complete: The model's verdict that the uncut tail is already a
            complete episode; ``None`` when no model judged it. ``True`` on a
            non-final call has already closed the tail into the last cell.
        per_cell_picture_names: Each cell's ``image_id -> client file name``
            for the pictures it carries, which the extraction route needs and
            the cell's own text no longer holds. Defaulted so a caller that
            builds an outcome from its own cells does not have to know about
            pictures.
        duplicates: Fresh chat rows already buffered, and so skipped: a
            message sent again (:func:`~everos.memory.extract.ingest.gen_message_id`
            gives it the id it had).
    """

    cells: list[MemCell]
    memcell_ids: list[str]
    per_cell_message_ids: list[list[str]]
    per_cell_all_senders: list[list[str]]
    status: Status
    message_count: int
    tail_complete: bool | None = None
    per_cell_picture_names: Sequence[dict[str, str]] = ()
    duplicates: int = 0


class BoundaryDetection:
    """Cut a session's buffered messages into memcells.

    Args:
        llm: Boundary-detection model; ``None`` leaves messages buffered.
        instructions: Custom instructions for the boundary operator.
        session_buffer: The per-session buffer of uncut messages; it also
            records the memcells cut from it.
        limits: Token / message counts past which cells are cut without the
            model.
        attempt_timeout: Seconds one model attempt may take; ``None`` for no
            bound. It runs before any write, so a timeout changes nothing.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        instructions: CustomInstructions,
        session_buffer: SessionBufferStore,
        limits: BoundaryDetectionSettings,
        attempt_timeout: float | None = None,
        message_forget: MessageForgetStore | None = None,
    ) -> None:
        self._llm = llm
        self._instructions = instructions
        self._session_buffer = session_buffer
        self._limits = limits
        self._attempt_timeout = attempt_timeout
        self._message_forget = message_forget
        self._quarantine = SessionQuarantine(session_buffer)

    async def buffer(self, ingested: IngestResult) -> BoundaryOutcome:
        """Merge fresh messages into the session buffer without asking the model.

        The low-cost half of a deferred add: a later flush cuts the buffer.
        """
        fresh = chat_rows(ingested.messages)
        if not fresh:
            return _empty_outcome("skipped", message_count=0)
        session = _session_of(ingested)
        buffered = chat_rows(await self._session_buffer.list(session))
        fresh, duplicates = _new_only(fresh, buffered)
        if not fresh:
            return _empty_outcome("accumulated", message_count=0, duplicates=duplicates)
        merged = merge_dedupe_sort(buffered, fresh)
        await self._keep_buffered(session, merged)
        return _empty_outcome(
            "accumulated", message_count=len(fresh), duplicates=duplicates
        )

    async def detect(
        self, ingested: IngestResult, *, is_final: bool
    ) -> BoundaryOutcome:
        """Cut the buffer plus fresh messages into cells and keep the tail.

        A tail the model judges complete is closed into a last cell at once;
        one it does not waits for the next batch or the flush.

        Args:
            ingested: This call's messages.
            is_final: Flush: the algorithm closes the trailing segment too.

        Raises:
            LLMOutputError: Every attempt got an unusable reply (below the
                quarantine limit); nothing was written.
            ValueError: A message the operator refuses or a row that maps to
                no conversation item (below the quarantine limit); not retried.
            SessionBufferQuarantinedError: This failure reached the limit and
                the buffer was set aside.
            TimeoutError: A model attempt ran past ``attempt_timeout``.
        """
        session = _session_of(ingested)
        fresh = chat_rows(ingested.messages)
        if not fresh and not is_final:
            return _empty_outcome("skipped", message_count=0)
        buffered = await self._session_buffer.list(session)
        fresh, duplicates = _new_only(fresh, chat_rows(buffered))
        if not fresh and not is_final:
            # Every message was one the session holds: nothing new to cut.
            return _empty_outcome("accumulated", message_count=0, duplicates=duplicates)
        merged = merge_dedupe_sort(chat_rows(buffered), fresh)
        if not merged:
            return _empty_outcome("accumulated", message_count=0, duplicates=duplicates)
        held = await self._forgotten(session, merged)
        gone = {m.message_id for m in held}
        readable = [m for m in merged if m.message_id not in gone]
        if not readable or (
            not is_final and not any(m.role == "user" for m in readable)
        ):
            # Episode extraction needs a user message; wait for one.
            await self._keep_buffered(session, merged)
            return _empty_outcome(
                "accumulated", message_count=len(fresh), duplicates=duplicates
            )
        result = await self._segment_or_quarantine(
            session, readable, buffered=buffered, is_final=is_final
        )
        result = self._close_complete_tail(session, result, is_final=is_final)
        if not result.cells:
            await self._keep_buffered(session, merged)
            return _empty_outcome(
                "accumulated",
                message_count=len(fresh),
                tail_complete=result.tail_complete,
                duplicates=duplicates,
            )
        outcome = await self._commit_cells(
            session,
            readable,
            result,
            message_count=len(fresh),
            held=held,
            arrival=merged,
            group_id=ingested.group_id,
        )
        return outcome._replace(duplicates=duplicates)

    async def _forgotten(
        self, session: SessionKey, merged: Sequence[CanonicalMessage]
    ) -> list[CanonicalMessage]:
        """The rows a live forget names (docs/api.md, ``target.kind:
        "message"``): they stay buffered, out of every cut, and no model call
        reads them; an undo makes them ordinary rows of the next cut."""
        if self._message_forget is None:
            return []
        forgotten = await self._message_forget.forgotten(session)
        if not forgotten:
            return []
        return [m for m in merged if (m.client_message_id or m.message_id) in forgotten]

    @staticmethod
    def _close_complete_tail(
        session: SessionKey, result: DetectionResult, *, is_final: bool
    ) -> DetectionResult:
        closed = close_complete_tail(result, is_final=is_final)
        if closed is not result:
            logger.info(
                "boundary_tail_closed",
                session_id=session.session_id,
                items=len(result.tail),
            )
        return closed

    async def _segment_or_quarantine(
        self,
        session: SessionKey,
        merged: list[CanonicalMessage],
        *,
        buffered: Sequence[CanonicalMessage],
        is_final: bool,
    ) -> DetectionResult:
        """Segment ``merged``, counting a poisoned outcome towards the quarantine.

        Two outcomes say the session's input itself is bad and the next call
        would hit it again: an unusable reply on every attempt
        (``LLMOutputError``) and a message the operator refuses (everalgo's
        ``InvalidInputError``, a ``ValueError``). Only the first is retried.
        Any other error is infrastructure and propagates uncounted.
        """
        try:
            result = await self._segment(merged, is_final=is_final)
        except (LLMOutputError, ValueError):
            buffered_ids = {m.message_id for m in buffered}
            unsaved = [m for m in merged if m.message_id not in buffered_ids]
            await self._quarantine.record_failure(session, unsaved)
            raise
        self._quarantine.reset(session)
        return result

    async def _segment(
        self, merged: list[CanonicalMessage], *, is_final: bool
    ) -> DetectionResult:
        """One bounded-retry boundary call; each attempt has its own deadline.

        The request goes out with the window's count after the records
        (:mod:`.prompt_order`), so a session's consecutive calls share a
        provider cache prefix through every record they repeat.
        """
        llm = CountAfterMessages(self._llm)
        items = [to_conversation_item(m) for m in merged]
        custom_instructions = self._instructions.load("boundary_detection")

        async def attempt() -> DetectionResult:
            async with asyncio.timeout(self._attempt_timeout):
                return await adetect_boundaries(
                    items,
                    llm=llm,
                    is_final=is_final,
                    custom_instructions=custom_instructions,
                    hard_token_limit=self._limits.hard_token_limit,
                    hard_msg_limit=self._limits.hard_msg_limit,
                )

        # A generation span, so the call's token usage is costed (a span-typed
        # observation's usage is dropped).
        with memory_span("everos.memcell.boundary", observation_type="generation"):
            return await retry_llm_output(attempt, operator="boundary")

    async def _commit_cells(
        self,
        session: SessionKey,
        merged: list[CanonicalMessage],
        result: DetectionResult,
        *,
        message_count: int,
        held: Sequence[CanonicalMessage] = (),
        arrival: Sequence[CanonicalMessage] = (),
        group_id: str | None = None,
    ) -> BoundaryOutcome:
        """Record one memcell row per cell and keep only the tail buffered,
        in one transaction: a failure leaves the buffer as it was. ``held``
        (forgotten rows) stays buffered beside the tail, both in the order of
        ``arrival`` (the whole merged buffer). Each row is marked to be
        extracted (``MemcellRecord.to_extract``) until its extraction
        succeeds, with the group it was cut in (``group_id``) and its
        pictures' names, so a cell whose extraction never finishes is
        extracted again."""
        memcell_ids = [_mint_memcell_id() for _ in result.cells]
        message_ids = split_message_ids(merged, result.cells)
        senders = [unique_senders(cell) for cell in result.cells]
        names = _per_cell_picture_names(merged, result.cells)
        last_cell_ts = max(cell.timestamp for cell in result.cells)
        await self._session_buffer.cut(
            session,
            [
                MemcellRecord(
                    memcell_id=memcell_id,
                    cell=cell,
                    raw_type=_RAW_TYPE,
                    message_ids=ids,
                    sender_ids=cell_senders,
                    to_extract=True,
                    group_id=group_id,
                    picture_names=cell_names,
                )
                for memcell_id, cell, ids, cell_senders, cell_names in zip(
                    memcell_ids, result.cells, message_ids, senders, names, strict=True
                )
            ],
            tail=_in_arrival_order([*slice_tail(merged, result.tail), *held], arrival),
            last_memcell_ts=from_timestamp_ms(last_cell_ts) if last_cell_ts else None,
        )
        return BoundaryOutcome(
            cells=list(result.cells),
            memcell_ids=memcell_ids,
            per_cell_message_ids=message_ids,
            per_cell_all_senders=senders,
            per_cell_picture_names=names,
            status="extracted",
            message_count=message_count,
            tail_complete=result.tail_complete,
        )

    async def _keep_buffered(
        self, session: SessionKey, merged: list[CanonicalMessage]
    ) -> None:
        await self._session_buffer.keep(
            session, merged, last_message_ts=max(m.timestamp for m in merged)
        )


def _in_arrival_order(
    rows: Sequence[CanonicalMessage], arrival: Sequence[CanonicalMessage]
) -> list[CanonicalMessage]:
    """``rows`` in the order ``arrival`` has them (it is sorted by timestamp,
    so this is too); a row ``arrival`` lacks goes last."""
    position = {m.message_id: i for i, m in enumerate(arrival)}
    return sorted(rows, key=lambda m: position.get(m.message_id, len(position)))


def _new_only(
    fresh: list[CanonicalMessage], buffered: Sequence[CanonicalMessage]
) -> tuple[list[CanonicalMessage], int]:
    """``fresh`` without the messages already buffered, and how many those were.

    A message's id is made of its own sender, time and content
    (:func:`~everos.memory.extract.ingest.gen_message_id`), so a buffered id
    is that message sent again -- a client's retry -- and is taken once.
    """
    held = {m.message_id for m in buffered}
    new = [m for m in fresh if m.message_id not in held]
    return new, len(fresh) - len(new)


def _per_cell_picture_names(
    merged: Sequence[CanonicalMessage], cells: Sequence[MemCell]
) -> list[dict[str, str]]:
    """Each cell's picture names, sliced from ``merged`` in step with the cells.

    Rows and cell items correspond 1:1 and in order (see
    :func:`split_message_ids`), so the same walk gives each cell the content
    items its own messages carried — including the ``name`` a picture arrived
    under, which the rendered text does not keep.
    """
    names: list[dict[str, str]] = []
    start = 0
    for cell in cells:
        end = start + len(cell.items)
        names.append(picture_names(merged[start:end]))
        start = end
    return names


def _session_of(ingested: IngestResult) -> SessionKey:
    return SessionKey(
        app_id=ingested.app_id,
        project_id=ingested.project_id,
        session_id=ingested.session_id,
    )


def _mint_memcell_id() -> str:
    return f"mc_{uuid.uuid4().hex[:12]}"


def _empty_outcome(
    status: Status,
    *,
    message_count: int,
    tail_complete: bool | None = None,
    duplicates: int = 0,
) -> BoundaryOutcome:
    return BoundaryOutcome(
        cells=[],
        memcell_ids=[],
        per_cell_message_ids=[],
        per_cell_all_senders=[],
        per_cell_picture_names=[],
        status=status,
        message_count=message_count,
        tail_complete=tail_complete,
        duplicates=duplicates,
    )
