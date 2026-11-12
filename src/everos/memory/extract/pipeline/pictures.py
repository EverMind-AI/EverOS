"""A cell's pictures: read, labelled, and attributed to whoever posted them.

Shared by the one-to-one and the group pipelines, which read a cell's pictures
the same way and route what a picture says about a person the same way
(docs/group-memory.md, section 7.3 and D17):

* every ``[IMAGE: <hash>]`` reference in the cell's text is one picture,
  labelled per cell (``img-1``, ``img-2``) in conversation order -- a local
  name a reply can refer to, deliberately distinct from the content hash that
  identifies the file;
* a picture's **sender** is the sender of the message it first appears in;
  its **poster**, the one whose profile it may reach, is that sender only
  when it is a user: an assistant's (or a tool's) picture is evidence about no
  user at all;
* :func:`profile_pictures` hands a profile update the pictures one person
  posted in its segments, and nobody else's: a picture says something about
  its poster only (D17);
* :func:`without_pictures` takes forgotten pictures out of a cell's text, for
  every reader alike: the one-to-one profile, and a group's re-extraction and
  profiles.
"""

from __future__ import annotations

__all__ = [
    "Picture",
    "image_ids_in",
    "memcell_alive",
    "pictures_of",
    "profile_pictures",
    "without_pictures",
]

from collections.abc import Collection, Sequence

from everalgo.llm import image_url_part_from_bytes
from everalgo.types import ChatMessage
from everalgo.user_memory.profile import ProfilePicture

from everos.component.images import ImageLimits
from everos.core.observability.logging import get_logger
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


async def profile_pictures(
    cells: Sequence[MemCell],
    owner_id: str,
    *,
    images: ImageStore | None,
    limits: ImageLimits | None,
    forgotten: Collection[str] = (),
    detail: str = "high",
) -> tuple[list[MemCell], list[ProfilePicture]]:
    """The cells as a profile update reads them, and the pictures ``owner_id`` posted.

    Each picture the owner posted is labelled once for the whole batch
    (``img-1``, ``img-2`` in conversation order) and its ``[IMAGE: <hash>]``
    reference in the text becomes ``[IMAGE: <label>]``, the name the update
    and its reply use; the hash is the id the lineage reports back. A picture
    someone else posted, or one that cannot be read (a store rebuilt without
    its cache), is not attached, and its reference becomes a bare ``[IMAGE]``:
    the update sees that a picture was sent, never what it shows. A picture in
    ``forgotten`` (a forget no undo reversed names it) is taken out of the
    text (:func:`without_pictures`), as every group read takes it out: its
    bytes stay on disk for the undo, so whether they can be read says
    nothing. ``detail`` is the image parts' hint, as the episode call sends.

    Without a picture store or limits (the extraction model reads no images),
    no picture is attached.
    """
    if forgotten:
        cells = [without_pictures(cell, forgotten) for cell in cells]
    if images is None or limits is None:
        return list(cells), []
    labels: dict[str, str] = {}
    pictures: list[ProfilePicture] = []
    for cell in cells:
        for picture in await pictures_of(cell, images=images, limits=limits):
            if picture.poster != owner_id or picture.image_id in labels:
                continue
            label = f"img-{len(labels) + 1}"
            labels[picture.image_id] = label
            part = image_url_part_from_bytes(picture.data, picture.mime)
            if part.image_url.detail is None and detail:
                part = part.model_copy(
                    update={
                        "image_url": part.image_url.model_copy(
                            update={"detail": detail}
                        )
                    }
                )
            pictures.append(ProfilePicture(label=label, id=picture.image_id, part=part))
    return [_relabelled(cell, labels) for cell in cells], pictures


def _relabelled(cell: MemCell, labels: dict[str, str]) -> MemCell:
    """``cell`` with each picture reference as the update names it."""

    def text(content: str) -> str:
        for image_id in image_ids_in(content):
            label = labels.get(image_id)
            reference = f"{_IMAGE_REF_PREFIX}{image_id}]"
            content = content.replace(
                reference, f"{_IMAGE_REF_PREFIX}{label}]" if label else "[IMAGE]"
            )
        return content

    items = [
        item.model_copy(update={"content": text(item.content)})
        if isinstance(getattr(item, "content", None), str)
        and _IMAGE_REF_PREFIX in item.content
        else item
        for item in cell.items
    ]
    return cell.model_copy(update={"items": items})


def without_pictures(cell: MemCell, image_ids: Collection[str]) -> MemCell:
    """``cell`` with every reference to ``image_ids`` taken out of its messages.

    A picture is in a message as ``[IMAGE: <hash>]``, which is what the
    extraction route finds; removing the reference removes the picture from
    the call. A message that held nothing but forgotten pictures is dropped
    rather than kept as an empty line. The cell itself is never changed.
    """
    if not image_ids:
        return cell
    references = [f"{_IMAGE_REF_PREFIX}{image_id}]" for image_id in image_ids]
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
