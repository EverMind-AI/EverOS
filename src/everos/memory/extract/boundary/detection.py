"""Boundary detection: the session buffer cut into memcells.

Owns the buffer / merge / boundary / tail sequence so the same cells feed
the user-memory pipeline and, in agent mode, the agent-memory pipeline. It
also owns the ``memcell`` ledger: each cell gets exactly one row, whatever
the mode, and both pipelines reference that row's ``memcell_id``.

Both modes run :func:`everalgo.boundary.adetect_boundaries` over one buffer
track. Chat mode first drops tool rows; agent mode keeps them, and the
algorithm keeps them in the cells without showing them to the model.
Switching mode needs a fresh process (``settings.memorize.mode``).
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

from everos.component.utils.datetime import from_timestamp
from everos.config import BoundaryDetectionSettings
from everos.core.observability.logging import get_logger
from everos.core.observability.tracing import memory_span
from everos.memory.custom_instructions import CustomInstructions
from everos.memory.extract.llm_retry import retry_llm_output
from everos.memory.models import CanonicalMessage, IngestResult, MemcellRecord
from everos.memory.ports import MemcellStore, SessionBufferStore, SessionKey

from .messages import (
    Mode,
    filter_for_mode,
    merge_dedupe_sort,
    slice_tail,
    split_message_ids,
    to_conversation_item,
    unique_senders,
)
from .quarantine import SessionQuarantine

logger = get_logger(__name__)

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
        tail_complete: The model's verdict that the uncut tail is already a
            complete episode; ``None`` when no model judged it. Recorded, not
            acted on.
    """

    cells: list[MemCell]
    memcell_ids: list[str]
    per_cell_message_ids: list[list[str]]
    per_cell_all_senders: list[list[str]]
    status: Status
    message_count: int
    tail_complete: bool | None = None


class BoundaryDetection:
    """Cut a session's buffered messages into memcells.

    Args:
        llm: Boundary-detection model; ``None`` leaves messages buffered.
        instructions: Custom instructions for the boundary operator.
        session_buffer: The per-session buffer of uncut messages.
        memcells: The memcell ledger.
        limits: Token / message counts past which cells are cut without the
            model.
        attempt_timeout: Seconds one model attempt may take; ``None`` for no
            bound. It runs before any write, so a timeout changes nothing.
    """

    def __init__(
        self,
        *,
        llm: LLMClient | None,
        instructions: CustomInstructions,
        session_buffer: SessionBufferStore,
        memcells: MemcellStore,
        limits: BoundaryDetectionSettings,
        attempt_timeout: float | None = None,
    ) -> None:
        self._llm = llm
        self._instructions = instructions
        self._session_buffer = session_buffer
        self._memcells = memcells
        self._limits = limits
        self._attempt_timeout = attempt_timeout
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

        Args:
            ingested: This call's messages.
            mode: ``chat`` or ``agent``.
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
        if self._llm is None:
            await self._session_buffer.replace(session, merged)
            logger.warning(
                "memorize_no_llm_client",
                session_id=session.session_id,
                buffered=len(merged),
            )
            return _empty_outcome("skipped", message_count=len(fresh))

        result = await self._segment_or_quarantine(
            session, merged, buffered=buffered, mode=mode, is_final=is_final
        )
        if not result.cells:
            await self._keep_buffered(session, merged)
            return _empty_outcome(
                "accumulated",
                message_count=len(fresh),
                tail_complete=result.tail_complete,
            )
        return await self._commit_cells(
            session, merged, result, mode=mode, message_count=len(fresh)
        )

    async def _segment_or_quarantine(
        self,
        session: SessionKey,
        merged: list[CanonicalMessage],
        *,
        buffered: Sequence[CanonicalMessage],
        mode: Mode,
        is_final: bool,
    ) -> DetectionResult:
        """Segment ``merged``, counting a poisoned outcome towards the quarantine.

        Three outcomes say the session's input itself is bad and the next
        call would hit it again: an unusable reply on every attempt
        (``LLMOutputError``), a message the operator refuses (everalgo's
        ``InvalidInputError``, a ``ValueError``) and a row that maps to no
        conversation item (``ValueError``). Only the first is retried. Any
        other error is infrastructure and propagates uncounted.
        """
        try:
            result = await self._segment(merged, mode=mode, is_final=is_final)
        except (LLMOutputError, ValueError):
            buffered_ids = {m.message_id for m in buffered}
            unsaved = [m for m in merged if m.message_id not in buffered_ids]
            await self._quarantine.record_failure(session, unsaved)
            raise
        self._quarantine.reset(session)
        return result

    async def _segment(
        self, merged: list[CanonicalMessage], *, mode: Mode, is_final: bool
    ) -> DetectionResult:
        """One bounded-retry boundary call; each attempt has its own deadline."""
        assert self._llm is not None  # detect returns before without one
        llm = self._llm
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
            return await retry_llm_output(attempt, operator="boundary", mode=mode)

    async def _commit_cells(
        self,
        session: SessionKey,
        merged: list[CanonicalMessage],
        result: DetectionResult,
        *,
        mode: Mode,
        message_count: int,
    ) -> BoundaryOutcome:
        """Record one memcell row per cell, then keep only the tail buffered."""
        memcell_ids = [_mint_memcell_id() for _ in result.cells]
        message_ids = split_message_ids(merged, result.cells)
        senders = [unique_senders(cell) for cell in result.cells]
        raw_type = _RAW_TYPE_BY_MODE[mode]
        await self._memcells.insert_many(
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
                    memcell_ids, result.cells, message_ids, senders, strict=True
                )
            ],
        )
        last_cell_ts = max(cell.timestamp for cell in result.cells)
        if last_cell_ts:
            await self._session_buffer.touch_last_memcell(
                session, from_timestamp(last_cell_ts)
            )
        await self._session_buffer.replace(session, slice_tail(merged, result.tail))
        return BoundaryOutcome(
            cells=list(result.cells),
            memcell_ids=memcell_ids,
            per_cell_message_ids=message_ids,
            per_cell_all_senders=senders,
            status="extracted",
            message_count=message_count,
            tail_complete=result.tail_complete,
        )

    async def _keep_buffered(
        self, session: SessionKey, merged: list[CanonicalMessage]
    ) -> None:
        await self._session_buffer.replace(session, merged)
        await self._session_buffer.touch_last_message(
            session, max(m.timestamp for m in merged)
        )


def _session_of(ingested: IngestResult) -> SessionKey:
    return SessionKey(
        app_id=ingested.app_id,
        project_id=ingested.project_id,
        session_id=ingested.session_id,
    )


def _mint_memcell_id() -> str:
    return f"mc_{uuid.uuid4().hex[:12]}"


def _empty_outcome(
    status: Status, *, message_count: int, tail_complete: bool | None = None
) -> BoundaryOutcome:
    return BoundaryOutcome(
        cells=[],
        memcell_ids=[],
        per_cell_message_ids=[],
        per_cell_all_senders=[],
        status=status,
        message_count=message_count,
        tail_complete=tail_complete,
    )
