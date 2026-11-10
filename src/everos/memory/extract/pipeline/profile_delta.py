"""The profile half of the picture extraction call — what a photo says about its sender.

A photo is evidence about the person who sent it: the breed on a collar, the city
on a sign, the kitchen of a home, a hobby spread across a table. The design
requires that reading to reach the profile **at write time**, in the same call
that writes the episode and the picture memories, rather than waiting for the
next text-driven profile update — a conversation that is mostly pictures would
otherwise never move the portrait.

Why this module lives in EverOS rather than everalgo
----------------------------------------------------
`everalgo.user_memory.ProfileExtractor` is the profile pipeline, and it stays the
profile pipeline: it owns INIT / UPDATE / COMPACT / REGROUP, the item caps, the
dedupe, the history handling, and its output shape. What it cannot do is *see a
picture*: its inputs are ``MemCell`` s of text messages, and its prompts are
modelled on a conversation. The algorithm library cannot carry image content
yet, so the one call that does read pictures emits its profile observations
here, in **everalgo's own UPDATE operation shape** so that anything reading a
profile cannot tell which route wrote an item.

When everalgo can take picture content (an image-carrying extraction route, or a
profile stage that accepts an image), this module's prompt block and its reply
model move into everalgo and the merge below collapses into the existing UPDATE
path. Nothing else in EverOS changes: the call site already hands over the
saved profile.

What is deliberately *not* here
-------------------------------
Only ``add`` and ``none`` are accepted. everalgo's UPDATE contract also has
``update`` and ``delete`` by index, and those resolve against a rendered stored
profile — which this call never sees, because it runs before the episode is
written and must not spend the request's tokens on the whole portrait. An index
the model cannot see would be a guess, and a guessed ``delete`` on a profile is
the single worst failure this path could have. Contradictions and retractions
stay where they belong: the text-driven UPDATE, which does see the profile.
"""

from __future__ import annotations

__all__ = [
    "PICTURE_PROFILE_RULES",
    "SOURCE_IMAGE_KEYS",
    "ProfileItemOp",
    "adds_anything",
    "apply_ops",
    "forget_images",
    "item_source_images",
    "normalised",
    "restore_items",
    "retarget",
    "write_picture_profile",
]

from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

from everalgo.user_memory._language import OutputLanguage
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from everos.core.observability.logging import get_logger
from everos.memory._partition_locks import get_partition_lock
from everos.memory.models import ExplicitInfoItem, ImplicitTraitItem, Profile
from everos.memory.ports import ProfileStore, Scope

logger = get_logger(__name__)

SOURCE_IMAGE_KEYS = "source_image_ids"
"""The host key each picture-derived item carries, holding the content hashes it
was read from.

Everalgo's item models are ``extra="allow"`` and its merge keeps a host's own
keys on a stored item, which is what makes "forget this picture" able to reach
the profile item it established. The key is a list because two photos of the
same cat establish one fact between them, and the fact survives forgetting
either one.

It is EverOS's key, not everalgo's: an item everalgo later re-adds from text
carries none, which is correct — the fact then has a text source too, and
forgetting a picture must not remove a fact the conversation stated.
"""

PICTURE_PROFILE_RULES = """\
Photos also say something about the person who sent them, and <profile> is where
you record it. Report only what a photo *shows* about its sender, as background
facts that will still be true next time you meet them:
- A pet (its species and any name or breed the photo makes visible), a home (the
  city on a sign, a recurring room, a garden), a hobby or a collection, a
  workplace, a vehicle, a strongly personal possession.
- Do NOT report the moment: not "sent a photo of a latte", not what they were
  doing that afternoon. A single event is not a portrait fact.
- Do NOT infer from an object in the background what the person is like
  ("interested in design", "wealthy"); report the visible fact.
- Prefer what only the photo establishes: a year, a name or brand on a sign, a
  price board, an address, a model number, a breed on a tag. Those are the
  facts the text update can never find, and the reason this call reports them.
- Do NOT restate what the conversation's own text already establishes. If the
  only thing a photo shows is something the messages already say, it adds no
  item — but when it shows one extra legible detail the messages never mention,
  that detail is the item, even if it sits beside familiar content.
- A photo that shows nothing about its sender contributes nothing, and an empty
  list is the right answer then.

Each entry is one operation in the profile update format, and only "add" is
available -- there is no stored profile here to point an index at. Facts already
on file are skipped for you, so add a fact whenever you read one; do not try to
guess whether it is already there.

{"action": "add", "type": "explicit_info", "source_image_id": "img-1",
 "data": {"category": "pets", "description": "Has a cat named Biscuit",
          "evidence": "2026-10-30: the collar tag in the photo reads Biscuit"}}
When the messages already mention the cafe and its prices and the photo's sign
also reads "EST. 2019", the item is the year: "The Dou She cafe has been open
since 2019" — not a sentence repeating the name and prices the text gave.
- "explicit_info" for a fact the photo makes legible: a name on a tag, a city on
  a sign, a breed, a model number.
- "implicit_traits" for a disposition the photo evidences through a choice: a
  shelf of hand tools maintained in reach, a kitchen organised for daily cooking.
  It needs a "basis" naming the visible signal, exactly as the profile rules
  require; a photo alone rarely supports one, and an empty trait list is normal.
- "source_image_id" is the label of the picture the fact was read off ("img-1",
  as the conversation text and the <pictures> section spell it), so forgetting
  that picture can reach the item. Omit it only when no single picture shows it.
- "evidence" quotes only what the photo actually shows, dated with the message
  time, and names no one."""


class _ExplicitData(BaseModel):
    """An ``explicit_info`` item, in everalgo's own shape (grounding first)."""

    model_config = ConfigDict(extra="forbid")

    evidence: str
    description: str
    category: str


class _TraitData(BaseModel):
    """An ``implicit_traits`` item, in everalgo's own shape."""

    model_config = ConfigDict(extra="forbid")

    basis: str
    description: str
    trait: str


class _AddOp(BaseModel):
    """One ``add``, addressed by a bucket name rather than an index.

    ``action`` / ``type`` / ``data`` are everalgo's UPDATE reply fields byte for
    byte, so a caller holding one of those can read this one. ``source_image_id``
    is the one addition: which picture the fact was read from, so a forget can
    take the item with it. It is optional — an untagged add is still valid.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    action: Literal["add"]
    bucket: Literal["explicit_info", "implicit_traits"] = Field(alias="type")
    data: _ExplicitData | _TraitData
    source_image_id: str = ""


class _NoOp(BaseModel):
    """The documented no-op: nothing in these photos is about their sender."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["none"]


type ProfileItemOp = Annotated[_AddOp | _NoOp, Field(discriminator="action")]

_PROFILE_OP = TypeAdapter(ProfileItemOp)
"""Validates one operation from either a reply model or a raw dict."""


def _validated(ops: Sequence[object]) -> list[ProfileItemOp]:
    """Coerce a reply's operations into the reply models, models or dicts alike.

    ``parse_reply`` hands over validated models in production; a caller replaying
    a stored reply, or a test, hands over dicts. One coercion here keeps both on
    the same code path rather than a second branch in the merge.
    """
    return [
        op if isinstance(op, _AddOp | _NoOp) else _PROFILE_OP.validate_python(op)
        for op in ops
    ]


def adds_anything(ops: Sequence[ProfileItemOp]) -> bool:
    """Whether any of ``ops`` is an ``add`` (the only operation that writes)."""
    return any(isinstance(op, _AddOp) for op in _validated(ops))


def normalised(text: str) -> str:
    """The identity of a profile item: its description, case- and space-folded.

    Everalgo's dedupe compares descriptions the same way, so an item this module
    declines to add is one everalgo's own merge would also have found.
    """
    return " ".join(text.split()).casefold()


def apply_ops(
    existing: Profile | None,
    ops: Sequence[ProfileItemOp],
    *,
    owner_id: str,
    observed_at: int,
    image_ids: Sequence[str],
) -> tuple[Profile, int]:
    """Merge the picture call's additions into ``existing``; returns the profile.

    The merge is deliberately narrow: every accepted ``add`` appends one item,
    stamped with the observation time and with the pictures it was read from,
    unless an item with the same description is already on file. Nothing is
    removed or rewritten — that is everalgo's UPDATE, and doing it here would
    be the second profile-writing path the design forbids.

    Args:
        existing: The stored profile, or ``None`` for an owner with none yet.
        ops: The reply's operations, already validated.
        owner_id: Whose profile this is.
        observed_at: Epoch ms of the cell that carried the pictures.
        image_ids: Content hashes the call read, recorded on each added item.

    Returns:
        ``(profile, added)`` — the profile to write, and how many items the
        merge actually added (zero when every fact was already on file).
    """
    profile = (
        existing.model_copy(deep=True)
        if existing is not None
        else Profile(owner_id=owner_id, summary="", timestamp=observed_at)
    )
    known = {
        normalised(item.description)
        for item in (*profile.explicit_info, *profile.implicit_traits)
        if item.description
    }
    sources = list(dict.fromkeys(image_ids))
    added = 0
    for op in _validated(ops):
        if not isinstance(op, _AddOp):
            continue
        identity = normalised(op.data.description)
        if not identity or identity in known:
            continue
        known.add(identity)
        sources_here = (
            [op.source_image_id] if op.source_image_id in sources else sources
        )
        item = _item_of(op, observed_at=observed_at, image_ids=sources_here)
        if isinstance(item, ExplicitInfoItem):
            profile.explicit_info.append(item)
        else:
            profile.implicit_traits.append(item)
        added += 1
    if added:
        profile.timestamp = max(profile.timestamp, observed_at)
    return profile, added


def _item_of(
    op: _AddOp, *, observed_at: int, image_ids: Sequence[str]
) -> ExplicitInfoItem | ImplicitTraitItem:
    """The domain item one ``add`` becomes, carrying its picture provenance."""
    source = {SOURCE_IMAGE_KEYS: list(image_ids)}
    if isinstance(op.data, _TraitData):
        return ImplicitTraitItem(
            description=op.data.description,
            trait=op.data.trait,
            basis=op.data.basis,
            observed_at=observed_at,
            **source,
        )
    return ExplicitInfoItem(
        description=op.data.description,
        category=op.data.category,
        evidence=op.data.evidence,
        observed_at=observed_at,
        **source,
    )


async def write_picture_profile(
    *,
    store: ProfileStore,
    scope: Scope,
    ops: Sequence[ProfileItemOp],
    observed_at: int,
    image_ids: Sequence[str],
    language: OutputLanguage | None,
) -> int:
    """Read, merge and write the profile — under the profile strategy's lock.

    The lock is the one :func:`everos.memory.strategies.extract_user_profile`
    holds around its own read → LLM → write. Without it the two paths could
    each read the same profile and the later write would silently drop the
    other's items; sharing the key makes the same call site serialise them
    without either knowing about the other.

    Args:
        store: The profile document (md is the truth).
        scope: The owner whose profile is updated.
        ops: What the picture call added, already validated.
        observed_at: Epoch ms of the cell that carried the pictures.
        image_ids: Content hashes the call read.
        language: The reply's language, frozen onto a profile that has none.

    Returns:
        How many items were added (0 when the call found nothing new).
    """
    if not any(isinstance(op, _AddOp) for op in ops):
        return 0
    partition = f"{scope.app_id}:{scope.project_id}:{scope.owner_id}"
    async with get_partition_lock("extract_user_profile", partition):
        existing = await store.read(scope)
        merged, added = apply_ops(
            existing,
            ops,
            owner_id=scope.owner_id,
            observed_at=observed_at,
            image_ids=image_ids,
        )
        if not added:
            return 0
        if merged.primary_language is None and language is not None:
            merged.primary_language = str(language)
        await store.write(scope, merged)
    logger.info(
        "memory.images.profile_updated",
        app_id=scope.app_id,
        project_id=scope.project_id,
        owner_id=scope.owner_id,
        added=added,
        photos=len(image_ids),
    )
    return added


def forget_images(
    existing: Profile | None, image_ids: Sequence[str]
) -> tuple[Profile | None, list[tuple[str, BaseModel]], int]:
    """Drop every profile item that rests on forgotten pictures alone.

    An item carries the pictures it was read from. Forgetting one of them
    retires the item only when **every** picture behind it is gone: a fact two
    photos established between them survives forgetting either one, and an
    item with no picture source at all (a text-derived one) is never touched.
    That edge is the whole reason the key exists — a broader rule would delete
    profile facts the user never asked to forget.

    Args:
        existing: The stored profile, or ``None`` when the owner has none.
        image_ids: The pictures being forgotten.

    Returns:
        ``(profile, removed, count)`` — the profile to write back (``None``
        when the owner had none), the removed ``(bucket, item)`` pairs for an
        undo, and how many items went.
    """
    if existing is None:
        return None, [], 0
    gone = set(image_ids)
    kept = existing.model_copy(deep=True)
    removed: list[tuple[str, BaseModel]] = []

    def survives(item: BaseModel) -> bool:
        sources = item_source_images(item)
        return not sources or bool(set(sources) - gone)

    explicit = [item for item in kept.explicit_info if survives(item)]
    traits = [item for item in kept.implicit_traits if survives(item)]
    for item in kept.explicit_info:
        if item not in explicit:
            removed.append(("explicit_info", item))
    for item in kept.implicit_traits:
        if item not in traits:
            removed.append(("implicit_traits", item))
    if not removed:
        return existing, [], 0
    kept.explicit_info = explicit
    kept.implicit_traits = traits
    return kept, removed, len(removed)


def restore_items(
    existing: Profile | None,
    removed: Sequence[tuple[str, BaseModel]],
    *,
    owner_id: str,
    timestamp: int,
) -> tuple[Profile, int]:
    """Put items a forget removed back into the profile, once.

    An item is identified by its folded description, the same identity the
    merge uses, so an undo after the item was re-established from another
    source does not duplicate it. The item comes back as it was stored, with
    its ``observed_at`` and its picture provenance intact.

    Args:
        existing: The current profile, or ``None``.
        removed: ``(bucket, item)`` pairs an earlier :func:`forget_images`
            returned.
        owner_id: Whose profile this is.
        timestamp: Epoch ms to stamp a profile that had to be created.

    Returns:
        ``(profile, restored_count)``.
    """
    profile = (
        existing.model_copy(deep=True)
        if existing is not None
        else Profile(owner_id=owner_id, summary="", timestamp=timestamp)
    )
    known = {
        normalised(item.description)
        for item in (*profile.explicit_info, *profile.implicit_traits)
        if item.description
    }
    restored = 0
    for bucket, item in removed:
        description = (
            str(item.get("description") or "")
            if isinstance(item, dict)
            else getattr(item, "description", "") or ""
        )
        identity = normalised(description)
        if not identity or identity in known:
            continue
        known.add(identity)
        if bucket == "implicit_traits":
            profile.implicit_traits.append(ImplicitTraitItem.model_validate(item))
        else:
            profile.explicit_info.append(ExplicitInfoItem.model_validate(item))
        restored += 1
    return profile, restored


def retarget(
    ops: Sequence[ProfileItemOp], labels: Mapping[str, str]
) -> list[ProfileItemOp]:
    """Replace per-cell picture labels with the pictures' durable hashes.

    The reply names a picture the way the conversation text does (``img-1``),
    because that is the only name the model was shown; the store, a forget and
    a later profile update know the content hash. A label the cell does not
    define is dropped rather than kept: an unresolvable label would produce an
    item nothing can ever take back.
    """
    retargeted: list[ProfileItemOp] = []
    for op in _validated(ops):
        if not isinstance(op, _AddOp):
            retargeted.append(op)
            continue
        image_id = labels.get(op.source_image_id, "") if op.source_image_id else ""
        retargeted.append(op.model_copy(update={"source_image_id": image_id}))
    return retargeted


def item_source_images(item: BaseModel) -> list[str]:
    """The picture hashes an item was read from; empty for a text-derived one."""
    raw = (item.model_extra or {}).get(SOURCE_IMAGE_KEYS)
    return [str(value) for value in raw] if isinstance(raw, list) else []
