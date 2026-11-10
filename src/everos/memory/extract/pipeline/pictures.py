"""A cell's pictures: read, labelled, and attributed to whoever posted them.

Shared by the one-to-one and the group pipelines, which read a cell's pictures
the same way and route what a picture says about a person the same way
(docs/group-memory.md, section 7.3 and D17):

* every ``[IMAGE: <hash>]`` reference in the cell's text is one picture,
  labelled per cell (``img-1``, ``img-2``) in conversation order -- a local
  name a reply can refer to, deliberately distinct from the content hash that
  identifies the file;
* a picture's **sender** is the sender of the message it first appears in;
  its **poster**, the one its profile items may reach, is that sender only
  when it is a user: an assistant's (or a tool's) picture is evidence about no
  user at all;
* :func:`ops_by_poster` routes each profile operation to the poster of the
  picture it names, and drops one that cannot be pinned to a single poster --
  giving it to everyone would hand one person's photo to another.
"""

from __future__ import annotations

__all__ = [
    "Picture",
    "image_ids_in",
    "memcell_alive",
    "ops_by_poster",
    "pictures_of",
]

from collections.abc import Sequence

from everos.component.images import ImageLimits
from everos.core.observability.logging import get_logger
from everos.memory.extract.pipeline import profile_delta
from everos.memory.models import MemCell
from everos.memory.ports import MemcellStore, SourceCheck
from everos.memory.ports.images import ImageStore

logger = get_logger(__name__)

_IMAGE_REF_PREFIX = "[IMAGE: "


class Picture:
    """One picture of a cell, ready for the model.

    Args:
        label: Local name the reply refers to it by (``img-1``).
        image_id: Content hash — what the memory records and a forget takes.
        data: Bytes of the model-scaled copy.
        mime: MIME type of those bytes.
        poster: The user who posted it; empty when an assistant or a tool did.
        sender_id: Whoever posted it, assistants included.
        at: Epoch ms of the message it first appears in.
    """

    __slots__ = ("at", "data", "image_id", "label", "mime", "poster", "sender_id")

    def __init__(
        self,
        *,
        label: str,
        image_id: str,
        data: bytes,
        mime: str,
        poster: str = "",
        sender_id: str = "",
        at: int = 0,
    ) -> None:
        self.label = label
        self.image_id = image_id
        self.data = data
        self.mime = mime
        self.poster = poster
        self.sender_id = sender_id
        self.at = at


async def pictures_of(
    cell: MemCell, *, images: ImageStore | None, limits: ImageLimits | None
) -> list[Picture]:
    """The cell's readable pictures, in conversation order, labelled.

    A picture that cannot be read (forgotten, or a store rebuilt without its
    cache) is skipped with a warning: the conversation is still worth
    remembering, and its text keeps the reference.
    """
    if images is None or limits is None:
        return []
    pictures: list[Picture] = []
    seen: dict[str, str] = {}
    for item in cell.items:
        text = str(getattr(item, "content", "") or "")
        sender = str(getattr(item, "sender_id", "") or "")
        poster = sender if getattr(item, "role", None) == "user" else ""
        for image_id in image_ids_in(text):
            if image_id in seen:
                continue
            label = f"img-{len(seen) + 1}"
            seen[image_id] = label
            try:
                data = await images.read_derived(
                    image_id, edge=limits.model_edge, quality=limits.cache_quality
                )
            except (FileNotFoundError, OSError, ValueError) as err:
                logger.warning(
                    "memory.images.unreadable",
                    image_id=image_id,
                    error=f"{type(err).__name__}: {err}"[:160],
                )
                continue
            pictures.append(
                Picture(
                    label=label,
                    image_id=image_id,
                    data=data,
                    mime="image/jpeg",
                    poster=poster,
                    sender_id=sender,
                    at=int(getattr(item, "timestamp", 0) or 0),
                )
            )
    return pictures


def ops_by_poster(
    ops: Sequence[profile_delta.ProfileItemOp], pictures: Sequence[Picture]
) -> dict[str, list[profile_delta.ProfileItemOp]]:
    """Each picture-derived profile operation, under the picture's poster (D17).

    An operation names its picture by content hash (``source_image_id`` after
    :func:`profile_delta.retarget`); it goes to whoever posted that picture.
    One that names none was read off the cell's pictures together: it goes to
    their poster when they all have one, and to nobody when they do not.
    """
    poster_of = {picture.image_id: picture.poster for picture in pictures}
    posters = {picture.poster for picture in pictures}
    routed: dict[str, list[profile_delta.ProfileItemOp]] = {}
    dropped = 0
    for op in ops:
        source = getattr(op, "source_image_id", "")
        if source in poster_of:
            owner = poster_of[source]
        elif len(posters) == 1:
            (owner,) = posters
        else:
            dropped += 1
            continue
        routed.setdefault(owner, []).append(op)
    if dropped:
        logger.info("memory.images.profile_unattributed", dropped=dropped)
    return routed


def image_ids_in(text: str) -> list[str]:
    """Every ``[IMAGE: <id>]`` reference in ``text``, in order."""
    found: list[str] = []
    start = 0
    while True:
        index = text.find(_IMAGE_REF_PREFIX, start)
        if index < 0:
            return found
        end = text.find("]", index)
        if end < 0:
            return found
        found.append(text[index + len(_IMAGE_REF_PREFIX) : end])
        start = end + 1


def memcell_alive(memcell_store: MemcellStore, memcell_id: str) -> SourceCheck:
    """The ``only_if`` shared by every write derived from one cell."""

    async def check() -> bool:
        return bool(await memcell_store.find_by_ids([memcell_id]))

    return check
