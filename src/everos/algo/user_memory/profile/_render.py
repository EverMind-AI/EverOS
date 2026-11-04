"""What the model is shown: the batch selection, the conversation, the narratives and the stored items.

A narrative or item that spells out one of the prompts' own closing tags would end its element early, so every
narrative and every item's JSON goes through :func:`_defuse_closing_tags`. Items are shown with their model-facing
fields only (``_FIELD_ORDER``); a host's own keys stay in storage and out of the prompt.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, cast

from everos.algo.types import ChatMessage, Episode, MemCell, Profile
from everos.algo.user_memory._render import chat_messages
from everos.algo.user_memory.profile._items import (
    _OBSERVED_AT,
    _dated_buckets,
    _Item,
    _item_label,
    _normalize,
)
from everos.algo.utils import (
    InvalidInputError,
    format_utc_time,
    render_chat_line,
    speaker,
)

# The fields an item shows the model, in this order: the declared order of both output item models, merged, then
# the date. A host's own keys are not among them.
_FIELD_ORDER = ("evidence", "basis", "description", "category", "trait", _OBSERVED_AT)
# The prompts' own closing tags, in any case: the only strings in unescaped text that could end an element early.
_CLOSING_TAGS = re.compile(
    r"</(?:narrative|episode_narratives|item|explicit_info|implicit_traits|stored_profile|group)>",
    re.IGNORECASE,
)
_FULLWIDTH_LESS_THAN = "\uff1c"


def _resolve_episode_target(owner_id: str, owner_name: str | None) -> str:
    """Resolve the one literal owner reference used to select Episode narratives."""
    if not owner_id.strip():
        raise InvalidInputError(
            "owner_id is blank: pass the owner's non-blank id, the literal the episodes name the owner by "
            "when no owner_name is given"
        )
    return owner_name.strip() if owner_name and owner_name.strip() else owner_id.strip()


def _select_episodes(
    episodes: Sequence[object], *, target_user: str
) -> tuple[Episode, ...]:
    """Validate the batch, keep the Episodes whose narrative references the target user, order them by date."""
    if isinstance(episodes, (str, bytes)) or not episodes:
        raise InvalidInputError(
            "episodes must be a non-empty sequence of Episode, got an empty one or a bare str/bytes: "
            "pass a list of Episode, or skip the call when there are none"
        )

    normalized_target = _normalize(target_user)
    selected: list[Episode] = []
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Episode):
            raise TypeError(f"episodes[{index}] must be an Episode")
        if not episode.episode.strip():
            raise InvalidInputError(
                f"episodes[{index}].episode is blank: every Episode needs its narrative text; "
                "drop or fill the Episode before calling"
            )
        if normalized_target in _normalize(episode.episode):
            selected.append(episode)

    if not selected:
        raise InvalidInputError(
            f"no episode narrative mentions target user {target_user!r}: the narratives must name the owner "
            "by owner_name (or owner_id when there is none); pass the matching name or episodes that mention the owner"
        )
    return tuple(sorted(selected, key=lambda episode: episode.timestamp))


def _normalize_available_categories(categories: object) -> tuple[str, ...]:
    """Validate and normalize a caller-provided category snapshot without changing its order."""
    if categories is None:
        return ()
    if isinstance(categories, (str, bytes)) or not isinstance(categories, Sequence):
        raise TypeError("categories must be a sequence of strings or None")

    normalized: list[str] = []
    seen: set[str] = set()
    category_values = cast("Sequence[object]", categories)
    for index, category in enumerate(category_values):
        if not isinstance(category, str):
            raise TypeError(f"categories[{index}] must be a string")
        stripped = category.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        normalized.append(stripped)
    return tuple(normalized)


def _render_available_categories(categories: Sequence[str]) -> str:
    """Render the normalized category snapshot as deterministic JSON for prompt injection."""
    return json.dumps(list(categories), ensure_ascii=False, indent=2)


def _render_episodes(episodes: Sequence[Episode]) -> str:
    """Render narratives oldest to newest as ``<narrative time="2024-04-10 14:00 UTC">text</narrative>``, one per line."""
    return "\n".join(_render_narrative(episode) for episode in episodes)


def _render_narrative(episode: Episode) -> str:
    """One narrative element, its text stripped and its closing-tag look-alikes defused."""
    text = _defuse_closing_tags(episode.episode.strip())
    return f'<narrative time="{format_utc_time(episode.timestamp)}">{text}</narrative>'


def _user_senders(memcells: Sequence[MemCell]) -> set[str]:
    """Return sender_ids of human (``role == "user"``) chat messages across all memcells."""
    return {
        m.sender_id
        for cell in memcells
        for m in chat_messages(cell)
        if m.role == "user"
    }


def _sender_display_name(memcells: Sequence[MemCell], sender_id: str) -> str:
    """Human-readable label for ``sender_id``; the id itself when the conversation carries no name."""
    for cell in memcells:
        for m in chat_messages(cell):
            if m.sender_id == sender_id and m.sender_name and m.sender_name.strip():
                return m.sender_name.strip()
    return sender_id


def _render_conversation(memcells: Sequence[MemCell]) -> str:
    """Render ChatMessage items as :func:`~everos.algo.utils.render_chat_line` lines, the speaker tagged with its user_id.

    A line reads ``[2024-04-10 14:00 UTC (Wednesday)] Alice(user_id:u1): hello``: the prompts locate the target user's
    lines by that tag. Text-less messages and tool items are skipped.

    Raises:
        InvalidInputError: If a message with text carries no timestamp (0).
    """
    lines = [
        render_chat_line(_tag_speaker_with_user_id(message))
        for cell in memcells
        for message in chat_messages(cell)
        if message.content
    ]
    return "\n".join(lines) if lines else "(no prior MemCells in the cluster)"


def _tag_speaker_with_user_id(message: ChatMessage) -> ChatMessage:
    """The message with its speaker label followed by ``(user_id:<sender_id>)``, the label the MemCell prompts read."""
    return message.model_copy(
        update={"sender_name": f"{speaker(message)}(user_id:{message.sender_id})"}
    )


def _render_profile_for_update(profile: Profile) -> str:
    """Render the existing profile for the UPDATE prompt: label inventory first, then indexed JSON."""
    explicit_info, implicit_traits = _dated_buckets(profile)
    return "\n".join(
        (
            _render_label_inventory(explicit_info, implicit_traits),
            _render_indexed_items(explicit_info, implicit_traits),
        )
    )


def _render_profile_for_episode_update(profile: Profile) -> str:
    """Render both buckets as XML, without listing stored category names as preferred classification choices."""
    explicit_info, implicit_traits = _dated_buckets(profile)
    return "\n".join(
        (
            _render_bucket("explicit_info", explicit_info),
            _render_bucket("implicit_traits", implicit_traits),
        )
    )


def _render_label_inventory(
    explicit_info: Sequence[_Item], implicit_traits: Sequence[_Item]
) -> str:
    """List the category / trait labels already in use, deduplicated, in first-seen order."""
    return "\n".join(
        (
            "=== categories already in use (reuse one when the new fact belongs to that dimension) ===",
            _render_labels(explicit_info),
            "=== trait labels already in use (one disposition = one trait; add evidence to one of these rather than naming a synonym) ===",
            _render_labels(implicit_traits),
        )
    )


def _render_labels(items: Sequence[_Item]) -> str:
    """Comma-joined distinct non-empty labels, or a sentinel when the bucket carries none."""
    seen: dict[str, None] = {}
    for item in items:
        if (label := _item_label(item)) is not None:
            seen.setdefault(label, None)
    return ", ".join(seen) if seen else "(none yet)"


def _render_indexed_items(
    explicit_info: Sequence[_Item], implicit_traits: Sequence[_Item]
) -> str:
    """Render both buckets as ``[i] {json}`` lines for the MemCell UPDATE prompt; its ops resolve against them."""
    parts: list[str] = ["=== explicit_info ==="]
    for i, item in enumerate(explicit_info):
        parts.append(f"[{i}] {_render_item(item)}")
    parts.append("=== implicit_traits ===")
    for i, item in enumerate(implicit_traits):
        parts.append(f"[{i}] {_render_item(item)}")
    return "\n".join(parts)


def _render_bucket(bucket: str, items: Sequence[_Item]) -> str:
    """Render one bucket as ``<bucket>`` holding ``<item index="i">{json}</item>`` lines; ops name the index."""
    lines = [
        f'<item index="{i}">{_render_item(item)}</item>' for i, item in enumerate(items)
    ]
    return "\n".join((f"<{bucket}>", *lines, f"</{bucket}>"))


def _render_profile_for_compact(profile: Profile) -> str:
    """Render both buckets as one indented JSON object for COMPACT, its items undated."""
    text = json.dumps(
        {
            "explicit_info": _undated_shown_fields(profile.explicit_info),
            "implicit_traits": _undated_shown_fields(profile.implicit_traits),
        },
        ensure_ascii=False,
        indent=2,
    )
    return _defuse_closing_tags(text)


def _render_group(items: Sequence[_Item]) -> str:
    """Render one group's items as an indented JSON list for REGROUP, undated as for COMPACT."""
    return _defuse_closing_tags(
        json.dumps(_undated_shown_fields(items), ensure_ascii=False, indent=2)
    )


def _render_item(item: _Item) -> str:
    """One item as JSON of its shown fields, ``observed_at`` as a UTC minute; a field never set stays out."""
    fields = _shown_fields(item.model_dump(exclude_unset=True))
    if item.observed_at is not None:
        fields[_OBSERVED_AT] = format_utc_time(item.observed_at)
    return _defuse_closing_tags(json.dumps(fields, ensure_ascii=False))


def _shown_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """The names of ``fields`` that ``_FIELD_ORDER`` lists, in its order; any other key is left out."""
    return {name: fields[name] for name in _FIELD_ORDER if name in fields}


def _undated_shown_fields(items: Sequence[_Item]) -> list[dict[str, Any]]:
    """Each item's shown fields without ``observed_at``, for prompts that rewrite rather than date."""
    return [
        _shown_fields(item.model_dump(exclude_unset=True, exclude={_OBSERVED_AT}))
        for item in items
    ]


def _defuse_closing_tags(text: str) -> str:
    """``text`` with the ``<`` of each of the prompts' closing tags made fullwidth (U+FF1C); all else unchanged."""
    return _CLOSING_TAGS.sub(
        lambda match: _FULLWIDTH_LESS_THAN + match.group()[1:], text
    )
