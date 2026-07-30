"""Hermes-agnostic formatting helpers for the EverOS plugin.

Turns ``SearchData`` and other API shapes into the strings the Hermes agent
sees (prefetch block, tool-result JSON, mirror messages). Stdlib + local
``_constants`` / ``_types`` only.
"""

from __future__ import annotations

import json
import logging

from ._constants import _MAX_PREFETCH_CHARS
from ._types import MessageItem, SearchData, SearchEpisodeItem, SearchProfileItem

logger = logging.getLogger(__name__)


def _truncate(text: str, max_chars: int) -> str:
    """Truncate ``text`` to ``max_chars`` at the nearest preceding word boundary.

    Appends ``" …"`` when truncation actually happens. Returns the original
    string unchanged when it already fits.
    """
    if len(text) <= max_chars:
        return text
    if max_chars <= 0:
        return " …"
    # Reserve room for the ellipsis token.
    budget = max_chars - len(" …")
    if budget <= 0:
        return " …"
    slice_end = budget
    # Walk back to a whitespace boundary so we do not split a word.
    boundary = text.rfind(" ", 0, slice_end)
    if boundary > 0:
        slice_end = boundary
    return text[:slice_end].rstrip() + " …"


def _profile_one_line(profile: SearchProfileItem) -> str:
    """Render a profile dict as a single summary line."""
    data = profile.get("profile_data") or {}
    if isinstance(data, dict) and data:
        # Prefer a small set of common keys, fall back to the first kv pair.
        for key in ("name", "summary", "bio", "description", "title"):
            if key in data and data[key]:
                return str(data[key]).replace("\n", " ").strip()
        first_key = next(iter(data))
        return f"{first_key}: {data[first_key]}".replace("\n", " ").strip()
    uid = profile.get("user_id") or profile.get("id") or "user"
    return f"profile for {uid}"


def _format_episode(ep: SearchEpisodeItem) -> str:
    """Render one episode block (subject + truncated episode + atomic facts)."""
    subject = ep.get("subject") or "(no subject)"
    episode_text = (ep.get("episode") or "").strip()
    facts = ep.get("atomic_facts") or []

    lines = [f"- **Episode**: {subject}"]
    if episode_text:
        lines.append(episode_text)
    for fact in facts:
        content = (fact.get("content") or "").strip()
        if content:
            lines.append(f"  - {content}")
    return "\n".join(lines)


def format_prefetch(
    query: str,
    search_data: SearchData,
    *,
    max_chars: int = _MAX_PREFETCH_CHARS,
) -> str:
    """Build the markdown prefetch block injected into the agent context.

    Episodes are sorted by ``score`` descending. Atomic facts are nested
    under their parent episode (may be empty). The whole block is truncated
    to ``max_chars`` at a word boundary, appending ``" …"`` when it spills.

    Returns ``""`` when there are no episodes and no profiles.
    """
    episodes = list(search_data.get("episodes") or [])
    profiles = list(search_data.get("profiles") or [])

    if not episodes and not profiles:
        return ""

    header = "## EverOS Memory"
    sections: list[str] = [header]

    if profiles:
        # One-line summary from the highest-scoring profile (or the only one).
        ranked = sorted(
            profiles,
            key=lambda p: p.get("score") if p.get("score") is not None else -1.0,
            reverse=True,
        )
        sections.append(f"- **Profile**: {_profile_one_line(ranked[0])}")

    for ep in sorted(episodes, key=lambda e: e.get("score", 0.0), reverse=True):
        sections.append(_format_episode(ep))

    body = "\n".join(sections)
    truncated = _truncate(body, max_chars)
    # Keep the query out of the visible block — it is only used as context
    # for logging/debugging and is intentionally not surfaced to the agent
    # here, matching mem0's prefetch contract.
    logger.debug("format_prefetch query=%r len=%d", query, len(truncated))
    return truncated


def format_tool_result(data: object) -> str:
    """Serialize a tool result payload as JSON (the inner serializer)."""
    return json.dumps(data, ensure_ascii=False)


def format_memory_write_message(
    content: str, user_id: str, timestamp_ms: int
) -> MessageItem:
    """Build a user-role ``MessageItem`` for mirroring into EverOS."""
    item: MessageItem = {
        "sender_id": user_id,
        "role": "user",
        "timestamp": timestamp_ms,
        "content": content,
    }
    return item
