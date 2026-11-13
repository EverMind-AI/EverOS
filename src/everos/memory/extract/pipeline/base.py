"""Pipeline contract: one boundary cell in, the cell's outcome and events out.

The service layer walks the cells the boundary stage cut and hands each one
to the conversation pipeline. A pipeline does the cell's
in-request work (Episode extraction and its markdown write) and returns the
events that announce it; it never publishes them. The service publishes
each cell's events as soon as that cell is done, so a cell that fails later
cannot take an earlier cell's events down with it.
"""

from __future__ import annotations

__all__ = ["CellOutcome", "Pipeline", "PipelineOutcome", "PreparedCell"]

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from everos.memory.models import IngestResult, MemCell
from everos.memory.ports import Event

type Track = Literal["conversation"]
"""Which pipeline a :class:`PipelineOutcome` sums up."""


@dataclass(frozen=True)
class PreparedCell:
    """One memcell the boundary stage cut, with what the pipelines need of it.

    Args:
        cell: The memcell.
        memcell_id: Its id in the memcell store.
        sender_ids: Every sender in the cell (episodes record them).
        picture_names: ``image_id -> client file name`` for the pictures the
            cell carries. Read off the ingested content items, because the
            cell's text keeps only the hash; empty when no picture had a name.
        retry: An earlier extraction of the cell failed. Its owners whose
            episode of it was written before the failure are skipped, so no
            one gets a second copy.
    """

    cell: MemCell
    memcell_id: str
    sender_ids: Sequence[str] = ()
    picture_names: Mapping[str, str] = field(default_factory=dict)
    retry: bool = False


@dataclass(frozen=True)
class CellOutcome:
    """What one pipeline produced for one cell.

    Args:
        message_count: Messages the cell contributed.
        events: Events to publish for the cell, in order.
        md_paths: Markdown files written for the cell.
    """

    message_count: int = 0
    events: Sequence[Event] = ()
    md_paths: Sequence[str] = ()


@dataclass(frozen=True)
class PipelineOutcome:
    """One pipeline's result over every cell of an add cycle.

    Args:
        track: Which pipeline.
        status: ``"extracted"`` if any cell arrived, ``"accumulated"`` if
            none did.
        message_count: Messages across the extracted cells.
        extracted_md_paths: Markdown files written across the cells.
    """

    track: Track
    status: Literal["accumulated", "extracted"]
    message_count: int
    extracted_md_paths: Sequence[str] = ()

    @classmethod
    def from_cells(
        cls, track: Track, outcomes: Sequence[CellOutcome]
    ) -> PipelineOutcome:
        """Summarise the per-cell outcomes of one pipeline."""
        return cls(
            track=track,
            status="extracted" if outcomes else "accumulated",
            message_count=sum(o.message_count for o in outcomes),
            extracted_md_paths=[p for o in outcomes for p in o.md_paths],
        )


class Pipeline(Protocol):
    """One extraction track, run cell by cell."""

    async def run_cell(
        self, ingested: IngestResult, prepared: PreparedCell
    ) -> CellOutcome: ...
