"""Ranked candidates -> public DTO items: the outbound mapper of the search path.

The multi-round loop speaks everalgo ``Candidate``; the response speaks
:mod:`everos.memory.search.dto`. This module is the only place the two meet.
A ranked item contributes its id and score; every other DTO field comes from
the stored domain record the recaller returned next to it (``stored_by_id``,
keyed by ``Candidate.id``), so no column name of any index is read here.

* Episodes -- :func:`shape_episodes` emits one DTO per candidate.
* Pictures are shaped twice, from the same record: as a ranked result
  (:func:`shape_images`), and as the *content of an episode's references*
  (:func:`attach_referenced_pictures`). The second exists because an episode
  body names a picture by id and never restates it — without resolving the
  reference, an episode selected without its picture would reach the answering
  model as a dangling id and nothing else.

A candidate whose record is missing or belongs to the wrong owner type is
dropped with a warning rather than failing the request.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping
from typing import Any

from everalgo.types import Candidate

from everos.component.images import derived_relpath
from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import from_timestamp_ms, to_display_tz
from everos.memory.models import Episode, ImageMemory
from everos.memory.ports import Stored

from .dto import SearchEpisodeItem, SearchImageItem, SearchSource

logger = get_logger(__name__)


# ── Episodes ────────────────────────────────────────────────────────────


def shape_episodes(
    ranked: Iterable[Candidate], stored_by_id: Mapping[str, Stored[Episode]]
) -> list[SearchEpisodeItem]:
    """One episode DTO per ranked candidate, in order; unknown ids dropped."""
    return [
        item
        for item in (_episode(r.id, r.score, stored_by_id) for r in ranked)
        if item is not None
    ]


def shape_images(
    ranked: Iterable[Candidate],
    stored_by_id: Mapping[str, Stored[ImageMemory]],
    *,
    thumbnail_edge: int,
) -> list[SearchImageItem]:
    """One picture DTO per ranked candidate, in order; unknown ids dropped.

    ``thumbnail_edge`` is the configured decider size, echoed into the path so
    a caller resolving ``thumbnail_path`` lands on the copy the decider was
    actually shown rather than on whatever size the cache happens to hold.
    """
    return [
        item
        for item in (
            _image(r.id, r.score, stored_by_id, thumbnail_edge=thumbnail_edge)
            for r in ranked
        )
        if item is not None
    ]


def attach_referenced_pictures(
    episodes: list[SearchEpisodeItem],
    *,
    image_ids: Iterable[str],
    stored_images: Mapping[str, Stored[ImageMemory]],
    selected_ids: Collection[str],
    thumbnail_edge: int,
) -> list[SearchImageItem]:
    """Resolve the pictures an episode's body refers to; returns the extra ones.

    An episode carries ``image_ids`` because its body refers to a picture
    rather than restating it. That reference has to resolve for whoever reads
    the result, or an episode kept without its picture memory is worth less
    than the 1.0 text that inlined the picture's content.

    Each selected episode gets the pictures *it* refers to, attached as
    :attr:`~everos.memory.search.dto.SearchEpisodeItem.images`. A picture that
    is also selected as core memory is **not** copied in — it is already at the
    top level of the response, and duplicating it would show the reader the
    same memory twice and pay for it twice in the answer context. A referenced
    id that resolves to nothing (forgotten, superseded, or never extracted)
    attaches nothing: the reference stays, which is exactly what a forget
    means.

    Args:
        episodes: The selected episodes, mutated in place.
        image_ids: Every id any of them refers to.
        stored_images: The pictures fetched for those ids, by content hash.
        selected_ids: Ids of the pictures already selected as core results.
        thumbnail_edge: Configured decider size, echoed into the path.

    Returns:
        The pictures that were attached, once each — the caller reports them
        alongside the result so a client can see what the episodes resolved to
        without re-reading every episode.
    """
    wanted = {image_id for image_id in image_ids if image_id in stored_images}
    if not wanted:
        return []
    attached: dict[tuple[str, str], SearchImageItem] = {}
    for episode in episodes:
        labels = _labels_in(episode.episode)
        for image_id in episode.image_ids:
            if image_id not in wanted or image_id in selected_ids:
                continue
            label = labels.get(image_id, "")
            picture = attached.get((image_id, label))
            if picture is None:
                stored = stored_images[image_id]
                picture = _image(
                    stored.ref.id,
                    0.0,
                    {stored.ref.id: stored},
                    thumbnail_edge=thumbnail_edge,
                )
                if picture is None:
                    continue
                picture = picture.model_copy(update={"label": label})
                attached[(image_id, label)] = picture
            episode.images.append(picture)
    return list(attached.values())


_LABEL_REFERENCE = re.compile(r"\((img-\d+): ([0-9a-f]{64})\)")
"""(img-N: <hash>) — the reference shape the picture route writes into a body."""


def _labels_in(body: str) -> dict[str, str]:
    """``image_id -> img-N`` for every reference in one episode body.

    Read back from the text because the body is where the label lives: it is a
    per-cell name the model used, deliberately not stored, and a reader needs
    it to tie an attached description to the sentence that named it.
    """
    return {image_id: label for label, image_id in _LABEL_REFERENCE.findall(body)}


def _image(
    candidate_id: str,
    score: float,
    stored_by_id: Mapping[str, Stored[ImageMemory]],
    *,
    thumbnail_edge: int,
) -> SearchImageItem | None:
    stored = _lookup(
        candidate_id, stored_by_id, kind="image_memory", owner_type=_MEMORY_OWNERS
    )
    if stored is None:
        return None
    image = stored.item
    thumbnail = derived_relpath(image.image_id, thumbnail_edge)
    group = stored.scope.owner_type == "group"
    return SearchImageItem(
        id=stored.ref.id,
        entry_id=stored.ref.entry_id,
        image_id=image.image_id,
        user_id=None if group else image.owner_id,
        group_id=image.owner_id if group else None,
        app_id=stored.scope.app_id,
        project_id=stored.scope.project_id,
        session_id=image.session_id,
        timestamp=to_display_tz(from_timestamp_ms(image.timestamp)),
        sender_ids=list(image.sender_ids),
        subject_ids=list(image.subject_ids),
        origin=image.origin,
        source=(
            SearchSource.model_validate(
                {
                    "memcell_id": image.parent_id or None,
                    "from": image.timestamp,
                    "to": image.timestamp,
                }
            )
            if group
            else None
        ),
        description=image.description,
        ocr=image.ocr,
        name=image.name,
        original_path=image.original_relpath,
        thumbnail_path=thumbnail,
        media_type=image.media_type,
        width=image.width,
        height=image.height,
        status=image.status,
        score=float(score),
    )


def _episode(
    candidate_id: str,
    score: float,
    stored_by_id: Mapping[str, Stored[Episode]],
) -> SearchEpisodeItem | None:
    stored = _lookup(
        candidate_id, stored_by_id, kind="episode", owner_type=_MEMORY_OWNERS
    )
    if stored is None:
        return None
    episode = stored.item
    group = stored.scope.owner_type == "group"
    return SearchEpisodeItem(
        id=stored.ref.id,
        user_id=None if group else episode.owner_id,
        owner_type="group" if group else None,
        group_id=episode.owner_id if group else None,
        source=group_source(episode) if group else None,
        app_id=stored.scope.app_id,
        project_id=stored.scope.project_id,
        session_id=episode.session_id,
        timestamp=to_display_tz(from_timestamp_ms(episode.timestamp)),
        sender_ids=episode.sender_ids,
        subject_ids=list(episode.subject_ids),
        origin=episode.origin,
        image_ids=episode.image_ids,
        summary=episode.summary or "",
        subject=episode.subject or "",
        episode=episode.episode,
        type="Scene" if episode.parent_type == "scene" else "Conversation",
        score=float(score),
    )


def group_source(episode: Episode) -> SearchSource:
    """A group record's segment and span: first and last message, epoch ms."""
    return SearchSource(
        memcell_id=episode.parent_id if episode.parent_type == "memcell" else None,
        from_=episode.source_start
        if episode.source_start is not None
        else episode.timestamp,
        to=episode.timestamp,
    )


# ── Lookup ──────────────────────────────────────────────────────────────


_MEMORY_OWNERS = frozenset({"user", "group"})
"""Owners of episodes and picture memories. Which of them a read may see was
decided by the scope predicate the recall ran with; shaping only maps."""


def _lookup[T](
    candidate_id: str,
    stored_by_id: Mapping[str, Stored[T]],
    *,
    kind: str,
    owner_type: Any,
) -> Stored[T] | None:
    stored = stored_by_id.get(candidate_id)
    if stored is None:
        logger.warning("shape_missing_record", kind=kind, id=candidate_id)
        return None
    allowed = owner_type if isinstance(owner_type, frozenset) else {owner_type}
    if stored.scope.owner_type not in allowed:
        logger.warning(
            "shape_unexpected_owner_type",
            kind=kind,
            id=candidate_id,
            owner_type=stored.scope.owner_type,
        )
        return None
    return stored
