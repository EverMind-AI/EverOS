"""Boundary detection: the session buffer cut into memcells.

Owns the buffer / merge / boundary / tail sequence so the same cells feed
the user-memory pipeline and, in agent mode, the agent-memory pipeline. It
also owns the ``memcell`` ledger: each cell gets exactly one row, whatever
the mode, and both pipelines reference that row's ``memcell_id``.

Where to cut is the injected :class:`BoundaryOperator`'s decision (everalgo's
``BoundaryDetector``); this module only hands it the rows as conversation
items and slices the rows by the ``Cuts`` it returns, so a cell's message ids
are the ids at its positions (:func:`~.messages.apply_cuts`). Two strategies
(``settings.boundary_detection.strategy``), over one buffer track either way:

- ``batch`` hands the whole buffer to ``adetect_batch`` once per call. Chat
  mode first drops tool rows; agent mode keeps them, and the operator keeps
  them in the cells without showing them to the model.
- ``step`` hands ``adetect_step`` the history already judged plus one fresh
  message, once per message; the next history starts at the returned
  ``tail_start``. Chat messages only.

Switching mode needs a fresh process (``settings.memorize.mode``).
"""

from __future__ import annotations

__all__ = ["BoundaryDetection", "BoundaryOperator", "BoundaryOutcome", "Status"]

import asyncio
import uuid
from collections.abc import Sequence
from datetime import UTC, tzinfo
from typing import Literal, NamedTuple, Protocol

from everalgo.boundary import Cuts
from everalgo.llm import LLMOutputError
from everalgo.types import ChatMessage, ConversationItem, MemCell

from everos.config import BoundaryDetectionSettings
from everos.core.errors import BoundaryLLMError, LLMServiceError
from everos.core.observability.tracing import memory_span
from everos.core.utils.datetime import from_timestamp
from everos.memory.custom_instructions import CustomInstructions
from everos.memory.extract.llm_retry import retry_llm_output
from everos.memory.models import CanonicalMessage, IngestResult, MemcellRecord
from everos.memory.ports import SessionBufferStore, SessionKey

from .messages import (
    Mode,
    Segmentation,
    apply_cuts,
    filter_for_mode,
    merge_dedupe_sort,
    to_conversation_item,
    unique_senders,
)
from .quarantine import SessionQuarantine

type Status = Literal["accumulated", "extracted", "skipped"]

_RAW_TYPE_BY_MODE: dict[Mode, str] = {
    "chat": "Conversation",
    "agent": "AgentTrajectory",
}


class BoundaryOutcome(NamedTuple):
    """What one call produced; the lists are parallel, index ``i`` is cell ``i``.

    Attributes:
        cells: The memcells cut this call.
        memcell_ids: Their ids, minted here and shared by both pipelines.
        per_cell_message_ids: The buffered message ids each cell consumed.
        per_cell_all_senders: Each cell's distinct senders, first-seen order.
        status: ``extracted`` when cells were cut, ``accumulated`` when the
            messages wait in the buffer, ``skipped`` when there was nothing
            to do or no model to ask.
        message_count: Fresh rows this call contributed (after the mode filter).
    """

    cells: list[MemCell]
    memcell_ids: list[str]
    per_cell_message_ids: list[list[str]]
    per_cell_all_senders: list[list[str]]
    status: Status
    message_count: int


class BoundaryOperator(Protocol):
    """Where to cut a list of conversation items: everalgo's ``BoundaryDetector``.

    Both methods return ``Cuts`` over the list they were handed. Only the batch
    path's hard limits are passed; every other parameter is the operator's own
    default.
    """

    async def adetect_batch(
        self,
        items: Sequence[ConversationItem],
        *,
        is_final: bool,
        custom_instructions: str | None,
        tz: tzinfo,
        hard_token_limit: int,
        hard_message_limit: int,
    ) -> Cuts:
        """Every cut in one window; ``is_final`` closes the trailing segment too."""
        ...

    async def adetect_step(
        self,
        items: Sequence[ChatMessage],
        *,
        custom_instructions: str | None,
        tz: tzinfo,
    ) -> Cuts:
        """Whether the last item starts a new segment after the ones before it."""
        ...


class BoundaryDetection:
    """Cut a session's buffered messages into memcells.

    Args:
        detector: Places the cuts.
        instructions: Custom instructions for the boundary operator.
        session_buffer: The per-session buffer of uncut messages; it also
            records the memcells cut from it.
        settings: Which strategy cuts the buffer, and the batch path's hard
            limits.
        attempt_timeout: Seconds one model attempt may take; ``None`` for no
            bound. It runs before any write, so a timeout changes nothing.
        tz: The zone message times are shown to the model in.
    """

    def __init__(
        self,
        *,
        detector: BoundaryOperator,
        instructions: CustomInstructions,
        session_buffer: SessionBufferStore,
        settings: BoundaryDetectionSettings,
        attempt_timeout: float | None = None,
        tz: tzinfo = UTC,
    ) -> None:
        self._detector = detector
        self._instructions = instructions
        self._session_buffer = session_buffer
        self._settings = settings
        self._attempt_timeout = attempt_timeout
        self._tz = tz
        self._quarantine = SessionQuarantine(session_buffer)

    async def buffer(self, ingested: IngestResult, *, mode: Mode) -> BoundaryOutcome:
        """Merge fresh messages into the session buffer without asking the model.

        The low-cost half of a deferred add: a later flush cuts the buffer.
        """
        fresh = filter_for_mode(ingested.messages, mode)
        if not fresh:
            return _empty_outcome("skipped", message_count=0)
        session = _session_of(ingested)
        merged = merge_dedupe_sort(await self._session_buffer.list(session), fresh)
        await self._keep_buffered(session, merged)
        return _empty_outcome("accumulated", message_count=len(fresh))

    async def detect(
        self, ingested: IngestResult, *, mode: Mode, is_final: bool
    ) -> BoundaryOutcome:
        """Cut the buffer plus fresh messages into cells and keep the tail.

        The uncut tail stays buffered until a later batch cuts it or the flush
        closes it.

        Args:
            ingested: This call's messages.
            mode: ``chat`` or ``agent``.
            is_final: Flush: the algorithm closes the trailing segment too.

        Raises:
            LLMOutputError: Every attempt got an unusable reply (below the
                quarantine limit); nothing was written.
            ValueError: A message the operator refuses or a row that maps to
                no conversation item -- under ``step`` also a tool row or a
                message older than the buffer (below the quarantine limit); not
                retried.
            SessionBufferQuarantinedError: This failure reached the limit and
                the buffer was set aside.
            TimeoutError: A model attempt ran past ``attempt_timeout``.
        """
        session = _session_of(ingested)
        fresh = filter_for_mode(ingested.messages, mode)
        if not fresh and not is_final:
            return _empty_outcome("skipped", message_count=0)
        buffered = await self._session_buffer.list(session)
        merged = merge_dedupe_sort(buffered, fresh)
        if not merged:
            return _empty_outcome("accumulated", message_count=0)
        if not is_final and not any(m.role == "user" for m in merged):
            # Episode extraction needs a user message; wait for one.
            await self._keep_buffered(session, merged)
            return _empty_outcome("accumulated", message_count=len(fresh))
        result = await self._segment_or_quarantine(
            session, merged, buffered=buffered, mode=mode, is_final=is_final
        )
        if not result.cells:
            await self._keep_buffered(session, merged)
            return _empty_outcome("accumulated", message_count=len(fresh))
        return await self._commit_cells(
            session, result, mode=mode, message_count=len(fresh)
        )

    async def _segment_or_quarantine(
        self,
        session: SessionKey,
        merged: list[CanonicalMessage],
        *,
        buffered: Sequence[CanonicalMessage],
        mode: Mode,
        is_final: bool,
    ) -> Segmentation:
        """Segment ``merged``, counting a poisoned outcome towards the quarantine.

        Three outcomes say the session's input itself is bad and the next
        call would hit it again: an unusable reply on every attempt
        (``LLMOutputError``), a message the operator refuses (everalgo's
        ``InvalidInputError``, a ``ValueError``) and a row that maps to no
        conversation item (``ValueError``). Only the first is retried. Any
        other error is infrastructure and propagates uncounted.
        """
        try:
            if self._settings.strategy == "step":
                result = await self._segment_step(
                    merged, buffered=buffered, mode=mode, is_final=is_final
                )
            else:
                result = await self._segment_batch(merged, mode=mode, is_final=is_final)
        except (LLMOutputError, ValueError):
            buffered_ids = {m.message_id for m in buffered}
            unsaved = [m for m in merged if m.message_id not in buffered_ids]
            await self._quarantine.record_failure(session, unsaved)
            raise
        self._quarantine.reset(session)
        return result

    async def _segment_batch(
        self, merged: list[CanonicalMessage], *, mode: Mode, is_final: bool
    ) -> Segmentation:
        """One bounded-retry batch call; each attempt has its own deadline."""
        items = [to_conversation_item(m) for m in merged]
        custom_instructions = self._instructions.load("boundary_detection")

        async def attempt() -> Cuts:
            async with asyncio.timeout(self._attempt_timeout):
                return await self._detector.adetect_batch(
                    items,
                    is_final=is_final,
                    custom_instructions=custom_instructions,
                    tz=self._tz,
                    hard_token_limit=self._settings.hard_token_limit,
                    hard_message_limit=self._settings.hard_msg_limit,
                )

        # A generation span, so the call's token usage is costed (a span-typed
        # observation's usage is dropped).
        with memory_span("everos.memcell.boundary", observation_type="generation"):
            try:
                cuts = await retry_llm_output(attempt, operator="boundary", mode=mode)
            except TimeoutError as err:
                raise BoundaryLLMError("boundary detection timed out") from err
            except LLMServiceError as err:
                raise BoundaryLLMError(str(err)) from err
        return apply_cuts(merged, items, cuts)

    async def _segment_step(
        self,
        merged: list[CanonicalMessage],
        *,
        buffered: Sequence[CanonicalMessage],
        mode: Mode,
        is_final: bool,
    ) -> Segmentation:
        """Judge each fresh message against the history before it, in arrival order.

        The buffer is the history already judged, so the fresh rows must all
        sort after it. Each judgement hands the operator ``merged[start:end]``,
        the history plus the one fresh message, and moves ``start`` to the
        ``tail_start`` it returns. A flush closes whatever history is left.
        Nothing is written here: a failure on any message leaves the whole
        call unapplied, as for ``batch``.

        Raises:
            ValueError: A fresh message sorts before the buffered history, or
                a row is not a chat message.
        """
        held = len(buffered)
        if [m.message_id for m in merged[:held]] != [m.message_id for m in buffered]:
            raise ValueError(
                "step boundary detection: a fresh message is older than the "
                "buffered history"
            )
        items = [_chat_item(m) for m in merged]
        custom_instructions = self._instructions.load("boundary_detection")
        closed: list[Segmentation] = []
        start = 0
        for end in range(held + 1, len(merged) + 1):
            cuts = await self._judge_step(
                items[start:end], mode=mode, custom_instructions=custom_instructions
            )
            closed.append(apply_cuts(merged[start:end], items[start:end], cuts))
            start += cuts.tail_start
        if is_final and start < len(merged):
            closed.append(_close_all(merged[start:], items[start:]))
            start = len(merged)
        return _joined(closed, tail=merged[start:])

    async def _judge_step(
        self,
        items: Sequence[ChatMessage],
        *,
        mode: Mode,
        custom_instructions: str | None,
    ) -> Cuts:
        """One bounded-retry step call; each attempt has its own deadline."""

        async def attempt() -> Cuts:
            async with asyncio.timeout(self._attempt_timeout):
                return await self._detector.adetect_step(
                    items, custom_instructions=custom_instructions, tz=self._tz
                )

        with memory_span("everos.memcell.boundary", observation_type="generation"):
            try:
                return await retry_llm_output(attempt, operator="boundary", mode=mode)
            except TimeoutError as err:
                raise BoundaryLLMError("boundary detection timed out") from err
            except LLMServiceError as err:
                raise BoundaryLLMError(str(err)) from err

    async def _commit_cells(
        self,
        session: SessionKey,
        result: Segmentation,
        *,
        mode: Mode,
        message_count: int,
    ) -> BoundaryOutcome:
        """Record one memcell row per cell and keep only the tail buffered,
        in one transaction: a failure leaves the buffer as it was."""
        memcell_ids = [_mint_memcell_id() for _ in result.cells]
        senders = [unique_senders(cell) for cell in result.cells]
        raw_type = _RAW_TYPE_BY_MODE[mode]
        last_cell_ts = max(cell.timestamp for cell in result.cells)
        await self._session_buffer.cut(
            session,
            [
                MemcellRecord(
                    memcell_id=memcell_id,
                    cell=cell,
                    raw_type=raw_type,
                    message_ids=ids,
                    sender_ids=cell_senders,
                )
                for memcell_id, cell, ids, cell_senders in zip(
                    memcell_ids, result.cells, result.message_ids, senders, strict=True
                )
            ],
            tail=result.tail,
            last_memcell_ts=from_timestamp(last_cell_ts) if last_cell_ts else None,
        )
        return BoundaryOutcome(
            cells=result.cells,
            memcell_ids=memcell_ids,
            per_cell_message_ids=result.message_ids,
            per_cell_all_senders=senders,
            status="extracted",
            message_count=message_count,
        )

    async def _keep_buffered(
        self, session: SessionKey, merged: list[CanonicalMessage]
    ) -> None:
        await self._session_buffer.keep(
            session, merged, last_message_ts=max(m.timestamp for m in merged)
        )


def _session_of(ingested: IngestResult) -> SessionKey:
    return SessionKey(
        app_id=ingested.app_id,
        project_id=ingested.project_id,
        session_id=ingested.session_id,
    )


def _chat_item(message: CanonicalMessage) -> ChatMessage:
    """The row as the chat item ``adetect_step`` takes.

    Raises:
        ValueError: The row maps to a tool item, which ``step`` cannot judge.
    """
    item = to_conversation_item(message)
    if not isinstance(item, ChatMessage):
        raise ValueError(
            "step boundary detection handles chat messages only: "
            f"message_id={message.message_id!r} is a {type(item).__name__}"
        )
    return item


def _close_all(
    rows: Sequence[CanonicalMessage], items: Sequence[ChatMessage]
) -> Segmentation:
    """The flush under ``step``: the whole remaining history as one last cell."""
    return apply_cuts(rows, items, Cuts(ends=[len(rows)], tail_start=len(rows)))


def _joined(
    parts: Sequence[Segmentation], *, tail: Sequence[CanonicalMessage]
) -> Segmentation:
    """The cells of consecutive ``parts`` in order, with ``tail`` left buffered."""
    return Segmentation(
        cells=[cell for part in parts for cell in part.cells],
        message_ids=[ids for part in parts for ids in part.message_ids],
        tail=list(tail),
    )


def _mint_memcell_id() -> str:
    return f"mc_{uuid.uuid4().hex[:12]}"


def _empty_outcome(status: Status, *, message_count: int) -> BoundaryOutcome:
    return BoundaryOutcome(
        cells=[],
        memcell_ids=[],
        per_cell_message_ids=[],
        per_cell_all_senders=[],
        status=status,
        message_count=message_count,
    )
