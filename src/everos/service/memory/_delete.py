"""``POST /memory/delete`` and ``/delete_by_ids``: EverOS 2.0 has no hard delete.

Both routes stay, as thin aliases of a forget, because callers send them (the
evaluation harnesses' ``--replace-sessions`` sends ``/memory/delete`` with a
``session_id``): nothing leaves markdown, every memory a delete reaches is
marked forgotten -- out of every default read and of the index at once -- and
the answer carries the forget's ``forget_id``, which ``/memory/forget/undo``
takes. What a conversation's segments and buffer hold in ``system.db`` stays
too; sending the conversation again writes new memories beside the forgotten
ones, and the profile reads them (it skips only what is forgotten).

* ``user_id`` -- the user's whole memory (target ``owner``);
* ``user_id`` and ``session_id`` -- the user's memory of one conversation
  (target ``session``);
* ``session_id`` alone -- that conversation in the memory of every user that
  holds it, one forget each;
* ``/delete_by_ids`` -- the episodes by id, per owner (target ``record``); an id
  that names nothing is skipped, as before.

A group's delete is already a forget (``MemoryService.delete_in_group``).
"""

from __future__ import annotations

__all__ = ["DeleteAlias", "DeleteResult"]

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from everos.core.observability.logging import get_logger
from everos.memory.access import Identity
from everos.memory.ports import Scope

from ._forget_models import ForgetResult, ForgetTarget

if TYPE_CHECKING:
    from everos.memory.deps import Stores

    from ._forget import ForgetManager

logger = get_logger("everos.service.memory")

_MEMORY_ID = re.compile(r"(?P<owner>.+)_(?P<entry>ep_\d{8}_\d+)")

_NOBODY = Identity(asker_id=None)
"""A delete names no asker: the forget log records none."""


@dataclass(frozen=True)
class DeleteResult:
    """What a delete (a forget, in 2.0) did.

    Attributes:
        filters: The scope fields the request set.
        count: Memories taken out of default reads.
        forget_ids: The forgets it ran, one per user it reached; each is an
            undo token.
    """

    filters: list[str]
    count: int
    forget_ids: tuple[str, ...] = ()


class DeleteAlias:
    """``/memory/delete`` and ``/delete_by_ids`` as forgets.

    Args:
        stores: Where the owners of a conversation are found.
        forgetter: The forget engine, which does the work.
    """

    def __init__(self, *, stores: Stores, forgetter: ForgetManager) -> None:
        self._stores = stores
        self._forgetter = forgetter

    async def delete(
        self,
        *,
        app_id: str,
        project_id: str,
        user_id: str | None,
        session_id: str | None,
    ) -> DeleteResult:
        """Forget one user's memory, one conversation of it, or a conversation
        everywhere.

        Raises:
            SessionBusyError: The conversation stays busy (503).
        """
        if user_id is None:
            if session_id is None:
                raise ValueError("a deletion needs user_id or session_id")
            owners = await self._stores.owner_dir.owners_with_session(
                app_id, project_id, session_id
            )
            filters = ["session_id"]
        else:
            owners = [
                Scope(
                    app_id=app_id,
                    project_id=project_id,
                    owner_id=user_id,
                    owner_type="user",
                )
            ]
            filters = ["user_id"] if session_id is None else ["user_id", "session_id"]
        target = (
            ForgetTarget(kind="owner")
            if session_id is None
            else ForgetTarget(kind="session", session_id=session_id)
        )
        done = [
            await self._forgetter.forget(
                scope=scope,
                identity=_NOBODY,
                target=target,
                reason="delete",
                dry_run=False,
            )
            for scope in owners
        ]
        result = DeleteResult(
            filters=filters,
            count=sum(_count(r) for r in done),
            forget_ids=tuple(r.forget_id for r in done),
        )
        logger.info(
            "memory_delete_forgotten",
            app_id=app_id,
            project_id=project_id,
            filters=filters,
            count=result.count,
            forget_ids=list(result.forget_ids),
        )
        return result

    async def delete_by_ids(
        self,
        *,
        app_id: str,
        project_id: str,
        memory_type: str,
        ids: Sequence[str],
        reason_code: str | None,
    ) -> tuple[str, ...]:
        """Forget episodes by memory id (``<owner_id>_<entry_id>``), per owner.

        Returns the forgets' ids. The caller's free-text note never reaches
        here, so it is never logged.
        """
        if memory_type != "episode":
            raise ValueError(f"unsupported memory_type {memory_type!r}")
        by_owner: dict[str, list[str]] = {}
        for memory_id in dict.fromkeys(ids):
            match = _MEMORY_ID.fullmatch(memory_id)
            if match is None:
                raise ValueError(f"not an episode memory id: {memory_id!r}")
            by_owner.setdefault(match["owner"], []).append(match["entry"])
        forget_ids: list[str] = []
        for owner_id, entry_ids in by_owner.items():
            done = await self._forgetter.forget(
                scope=Scope(
                    app_id=app_id,
                    project_id=project_id,
                    owner_id=owner_id,
                    owner_type="user",
                ),
                identity=_NOBODY,
                target=ForgetTarget(kind="record", record_ids=tuple(entry_ids)),
                reason=reason_code or "delete",
                dry_run=False,
                strict=False,
            )
            forget_ids.append(done.forget_id)
        logger.info(
            "memory_delete_by_ids_forgotten",
            app_id=app_id,
            project_id=project_id,
            reason=reason_code,
            forget_ids=forget_ids,
        )
        return tuple(forget_ids)


def _count(result: ForgetResult) -> int:
    """Memories one forget took out of default reads."""
    return (
        len(result.episodes)
        + len(result.image_memories)
        + len(result.scenes)
        + len(result.profile_items)
    )
