"""A group takes each message once: the group side of one batch's dedupe.

docs/group-memory.md, section 5.2. A message is a **duplicate** when its group
already holds it -- in any session of the group, buffered or cut into a
segment, forgotten or not. Its key (:func:`message_key`) is the client's
``message_id`` when it sent one, else the sender, the time in epoch ms and the
SHA-256 of the content as sent. A duplicate is skipped silently, so a retry,
or a second client forwarding the same chat, is safe; the batch's answer
counts what was skipped.

Rules the implementation keeps:

* keys come from the batch as sent, before ingest: a duplicate costs no
  ingest, no boundary call and no extraction -- dedupe only removes calls;
* a key is recorded (:meth:`Claim.commit`) only once its batch is stored and
  the batch's own cells are extracted: a batch that fails records nothing, so
  its retry is taken in full -- a cell whose extraction failed is extracted
  again from the retry, as before dedupe;
* a forget never removes a key: a forgotten message sent again stays
  forgotten, whichever session it comes back in;
* a ``/memory/delete`` of a session (a forget of the whole session whose
  reason is ``delete``, while no undo reversed it) releases the session's
  keys: the conversation sent again is taken as it was before dedupe existed,
  and its key moves to the session that took it;
* batches of one group claim their keys under the group's lock; a batch
  whose messages another batch of the group is storing waits until that one
  has stored them -- they are then duplicates -- or failed -- they are then
  its own (one process owns a memory root, so the claims are process
  state);
* within one batch nothing is a duplicate of its own messages: a batch is
  taken as sent, as before dedupe.

External usage::

    dedupe = GroupDedupe(stores.group_messages, forget_log=stores.forget_log)
    claim = await dedupe.claim(scope, session, payload["messages"])
    try:
        ...  # ingest the batch without claim.duplicates, store it
        await claim.commit()
    finally:
        claim.release()
"""

from __future__ import annotations

__all__ = ["Claim", "GroupDedupe", "message_key"]

import asyncio
import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from everos.core.errors import SessionBusyError
from everos.core.locks import LOCKS, owner_key
from everos.core.utils.datetime import from_timestamp, to_timestamp_ms
from everos.memory.ports import ForgetLogStore, GroupMessageStore, Scope, SessionKey

_DELETE_REASON = "delete"
"""The reason ``/memory/delete`` records on the forget it runs (docs/api.md)."""


def message_key(message: Mapping[str, Any]) -> str | None:
    """The dedupe key of one batch message (a ``/memory/add`` message dict).

    ``id:<message_id>`` when the client gave the message an id; otherwise
    ``fp:<sender_id>:<epoch ms>:<sha256>`` over the content as sent (a string,
    or the content items without their unset fields, as canonical JSON).
    ``None`` for a row a chat memory never stores -- a tool row, an assistant
    message that calls a tool -- which is never a duplicate.
    """
    if message.get("role") not in ("user", "assistant") or message.get("tool_calls"):
        return None
    client_id = message.get("message_id")
    if client_id:
        return f"id:{client_id}"
    content = json.dumps(
        _plain(message.get("content")),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    at = to_timestamp_ms(from_timestamp(int(message["timestamp"])))
    return f"fp:{message.get('sender_id')}:{at}:{digest}"


def _plain(content: Any) -> Any:
    """``content`` as plain data: items as dicts without their unset fields."""
    if isinstance(content, str) or content is None:
        return content
    items: list[Any] = []
    for item in content:
        data = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        items.append({k: v for k, v in data.items() if v is not None})
    return items


class Claim:
    """One batch's keys, claimed: which of its messages are duplicates.

    Attributes:
        duplicates: Positions in the batch of the messages the group already
            holds (or another batch is storing); they are not taken.
    """

    def __init__(
        self,
        owner: GroupDedupe,
        scope: Scope,
        session: SessionKey,
        *,
        duplicates: frozenset[int],
        fresh: frozenset[str],
        done: asyncio.Event,
    ) -> None:
        self._owner = owner
        self._scope = scope
        self._session = session
        self.duplicates = duplicates
        self._fresh = fresh
        self._done = done
        self._released = False

    async def commit(self) -> None:
        """Record the batch's other keys as held by its session.

        Called once the batch is stored and its own cells extracted; a key
        recorded twice is the same row.
        """
        await self._owner.store.record(self._scope, self._session, self._fresh)

    def release(self) -> None:
        """Drop the claim (after :meth:`commit`, or when the batch failed).

        The batches waiting on these keys look again: committed, they are
        duplicates for them; not, they are theirs to take.
        """
        if not self._released:
            self._released = True
            self._owner.unclaim(self._scope, self._fresh, self._done)
            self._done.set()


class GroupDedupe:
    """Which messages of a batch a group already holds, claimed per batch.

    Args:
        store: The groups' ledgers.
        forget_log: Where a ``/memory/delete`` of a session is found, which
            releases the session's keys; ``None`` releases none.
        wait_seconds: How long a batch waits for another batch of its group
            that is storing the same messages; ``None`` waits until it is
            done.
    """

    def __init__(
        self,
        store: GroupMessageStore,
        *,
        forget_log: ForgetLogStore | None = None,
        wait_seconds: float | None = None,
    ) -> None:
        self.store = store
        self._forget_log = forget_log
        self._wait_seconds = wait_seconds
        self._claimed: dict[tuple[str, str, str], dict[str, asyncio.Event]] = {}

    async def claim(
        self,
        scope: Scope,
        session: SessionKey,
        messages: Sequence[Mapping[str, Any]],
    ) -> Claim:
        """Claim the keys of ``messages`` for ``session`` of the group ``scope``.

        Under the group's lock: the keys the group holds make their messages
        duplicates; the rest are this batch's until it releases the claim. A
        key another batch of the group has claimed is waited for first.

        Raises:
            SessionBusyError: Another batch still stores some of these
                messages after ``wait_seconds`` (503: retry later).
        """
        keys = [message_key(m) for m in messages]
        wanted = frozenset(k for k in keys if k is not None)
        group = _group_of(scope)
        while True:
            async with LOCKS.hold("group", _lock_key(scope)):
                claimed = self._claimed.get(group, {})
                storing = {claimed[k] for k in wanted if k in claimed}
                if not storing:
                    taken = await self._held(scope, wanted)
                    fresh = wanted - taken
                    done = asyncio.Event()
                    if fresh:
                        self._claimed.setdefault(group, {}).update(
                            dict.fromkeys(fresh, done)
                        )
                    break
            # Another batch of the group is storing some of these messages:
            # once it has stored them (or failed), look again.
            try:
                async with asyncio.timeout(self._wait_seconds):
                    await asyncio.gather(*(event.wait() for event in storing))
            except TimeoutError:
                raise SessionBusyError(
                    "another batch of the group is still storing these messages; "
                    "retry later"
                ) from None
        duplicates = frozenset(i for i, key in enumerate(keys) if key in taken)
        return Claim(
            self, scope, session, duplicates=duplicates, fresh=fresh, done=done
        )

    def unclaim(self, scope: Scope, keys: Collection[str], done: asyncio.Event) -> None:
        """Release the ``keys`` one claim (``done``) holds for the group ``scope``."""
        group = _group_of(scope)
        claimed = self._claimed.get(group)
        if claimed is None:
            return
        for key in keys:
            if claimed.get(key) is done:
                del claimed[key]
        if not claimed:
            del self._claimed[group]

    async def _held(self, scope: Scope, keys: Collection[str]) -> set[str]:
        """The keys the group holds, less those a live delete released."""
        if not keys:
            return set()
        rows = await self.store.held(scope, keys)
        if not rows:
            return set()
        deleted = await self._deleted(scope, set(rows.values()))
        return {key for key, session in rows.items() if session not in deleted}

    async def _deleted(self, scope: Scope, sessions: Collection[str]) -> set[str]:
        """The ``sessions`` a ``/memory/delete`` took out, while no undo reversed it."""
        if self._forget_log is None:
            return set()
        deleted: set[str] = set()
        for session_id in sorted(sessions):
            entry = await self._forget_log.latest_live(
                target="session",
                target_id=session_id,
                app_id=scope.app_id,
                project_id=scope.project_id,
                owner_id=scope.owner_id,
                owner_type="group",
            )
            if entry is not None and entry.reasons == _DELETE_REASON:
                deleted.add(session_id)
        return deleted


def _group_of(scope: Scope) -> tuple[str, str, str]:
    return (scope.app_id, scope.project_id, scope.owner_id)


def _lock_key(scope: Scope) -> str:
    return owner_key(scope.app_id, scope.project_id, "group", scope.owner_id)
