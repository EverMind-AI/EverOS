"""Name each message's sender the way that sender last named themselves.

A message's ``sender_name`` labels its chat line wherever the line is
rendered -- the boundary and episode prompts, so the episode narratives and
the profile read from them. A client that names a sender once and later
sends only ``sender_id`` would otherwise split one person into two labels.

Rules, per ``(app_id, project_id, sender_id)``:

- A message's own ``sender_name`` is used stripped; a blank one counts as
  absent.
- The last name a sender carries in the batch is recorded, replacing the
  stored one, before any message is named -- so a message earlier in the
  batch than its sender's first named line is named too.
- A message without a name takes its sender's: the batch's when the sender
  was named in it, else the stored one; with neither it stays ``None`` and
  its line falls back to ``sender_id``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from everos.memory.ports import SenderNameStore


async def resolve_sender_names(
    messages: Sequence[Mapping[str, Any]],
    *,
    app_id: str,
    project_id: str,
    store: SenderNameStore,
) -> dict[str, str]:
    """Record the batch's names and return the name each sender goes by."""
    own = [(message["sender_id"], _own_name(message)) for message in messages]
    named = {sender_id: name for sender_id, name in own if name is not None}
    if named:
        await store.upsert_many(app_id, project_id, named)
    unnamed = {sender_id for sender_id, name in own if name is None} - named.keys()
    known = await store.get_many(app_id, project_id, unnamed) if unnamed else {}
    return {**known, **named}


def sender_name_of(message: Mapping[str, Any], names: Mapping[str, str]) -> str | None:
    """The message's own name, else the one its sender goes by (if any)."""
    return _own_name(message) or names.get(message["sender_id"])


def _own_name(message: Mapping[str, Any]) -> str | None:
    name = message.get("sender_name")
    return (name.strip() or None) if isinstance(name, str) else None
