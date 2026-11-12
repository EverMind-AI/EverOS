"""Reading a group segment without what was forgotten (docs/group-memory.md, 9.4).

A forgotten message stays in its segment's memcell payload (``system.db``),
so that an undo has nothing to copy back; every reader of the segment skips
it instead. That is the one rule this module holds, for every reader alike --
the group episode's re-extraction, the member-in-group and group profiles,
and the quotes a search hands back:

* :func:`without_forgotten` drops the chat messages a live forget names (the
  ids :meth:`~everos.memory.ports.MessageForgetStore.forgotten_in_segments`
  returns for the segment);
* :func:`readable_segments` is the two steps a strategy takes: fetch the
  forgotten ids of its segments, read each segment without them.

A forgotten picture is dropped from a segment by
:func:`~everos.memory.extract.pipeline.pictures.without_pictures`, which the
one-to-one profile reads with too.

Pure functions over the cell: the payload itself is never changed.
"""

from __future__ import annotations

__all__ = ["readable_segments", "without_forgotten"]

from collections.abc import Collection, Mapping

from everalgo.types import ChatMessage

from everos.memory.models import MemCell
from everos.memory.ports import MessageForgetStore


def without_forgotten(cell: MemCell, forgotten: Collection[str]) -> MemCell:
    """``cell`` without the chat messages whose id is in ``forgotten``.

    ``forgotten`` may hold the ids of the whole session (what the port
    returns); ids that are not the cell's are ignored. Items other than chat
    messages carry no id and stay. The cell's own ``timestamp`` is kept: it is
    the segment's, and a re-extraction dates its episode by the messages it
    reads.
    """
    if not forgotten:
        return cell
    gone = set(forgotten)
    items = [
        item
        for item in cell.items
        if not (isinstance(item, ChatMessage) and item.id in gone)
    ]
    if len(items) == len(cell.items):
        return cell
    return cell.model_copy(update={"items": items})


async def readable_segments(
    store: MessageForgetStore, cells: Mapping[str, MemCell]
) -> dict[str, MemCell]:
    """Each segment of ``cells`` (by memcell id) without its forgotten messages.

    One read of the forgotten ids for all of them; a segment with none is
    returned as it is.
    """
    if not cells:
        return {}
    forgotten = await store.forgotten_in_segments(list(cells))
    return {
        memcell_id: without_forgotten(cell, forgotten.get(memcell_id, ()))
        for memcell_id, cell in cells.items()
    }
