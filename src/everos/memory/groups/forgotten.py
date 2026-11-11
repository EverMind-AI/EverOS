"""Reading a group segment without what was forgotten (docs/group-memory.md, 9.4).

A forgotten message stays in its segment's memcell payload (``system.db``),
so that an undo has nothing to copy back; every reader of the segment skips
it instead. That is the one rule this module holds, for every reader alike --
the group episode's re-extraction, the member-in-group and group profiles,
and the quotes a search hands back:

* :func:`without_forgotten` drops the chat messages a live forget names (the
  ids :meth:`~everos.memory.ports.MessageForgetStore.forgotten_in_segments`
  returns for the segment);
* :func:`without_pictures` drops a forgotten picture from the messages that
  carried it (its ``[IMAGE: <hash>]`` reference), and a message left with no
  text at all;
* :func:`readable_segments` is the two steps a strategy takes: fetch the
  forgotten ids of its segments, read each segment without them.

Pure functions over the cell: the payload itself is never changed.
"""

from __future__ import annotations

__all__ = ["readable_segments", "without_forgotten", "without_pictures"]

from collections.abc import Collection, Mapping

from everalgo.types import ChatMessage

from everos.memory.models import MemCell
from everos.memory.ports import MessageForgetStore

_IMAGE_REF = "[IMAGE: {}]"


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


def without_pictures(cell: MemCell, image_ids: Collection[str]) -> MemCell:
    """``cell`` with every reference to ``image_ids`` taken out of its messages.

    A picture is in a message as ``[IMAGE: <hash>]``, which is what the
    extraction route finds; removing the reference removes the picture from
    the call. A message that held nothing but forgotten pictures is dropped
    rather than kept as an empty line.
    """
    if not image_ids:
        return cell
    references = [_IMAGE_REF.format(image_id) for image_id in image_ids]
    items: list[object] = []
    changed = False
    for item in cell.items:
        content = str(getattr(item, "content", "") or "")
        if not isinstance(item, ChatMessage) or not any(
            ref in content for ref in references
        ):
            items.append(item)
            continue
        changed = True
        for ref in references:
            content = content.replace(ref, "")
        text = "\n".join(line.rstrip() for line in content.splitlines() if line.strip())
        if text.strip():
            items.append(item.model_copy(update={"content": text}))
    if not changed:
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
