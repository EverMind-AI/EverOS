"""Group roster cascade handler -- ``group.md`` -> the roster mirror in SQLite.

``group.md`` is the truth for a group's meta, roster and session bindings; the
``group_meta`` / ``group_member`` / ``session_binding`` rows are a projection of
it. The write path keeps them in step synchronously (the group store writes
both under one lock); this handler is what re-derives them afterwards -- after
a hand edit of the file, and on ``everos cascade rebuild`` -- through the same
projection, so a rebuilt mirror reads exactly as the written one did.

No LanceDB table, no embedding: this kind writes SQLite only. A log that no
longer folds (a hand edit introduced a second open interval, say) raises
``ValueError``, which the worker records as unrecoverable instead of retrying.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from everos.infra.persistence.groups import GroupRosterStore
from everos.memory.ports import Scope

from ..models import HandlerOutcome
from ._common import resolve_scope
from .base import Handler


class GroupRosterHandler(Handler):
    """Cascade handler for ``groups/<group_id>/group.md``."""

    kind = "group_roster"

    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        record = await self._store().remirror(_scope(md_path))
        return HandlerOutcome(
            md_path=md_path,
            kind=self.kind,
            upserted=1 if record is not None else 0,
            deleted=0 if record is not None else 1,
            skipped=0,
        )

    async def handle_deleted(self, md_path: str) -> HandlerOutcome:
        scope = _scope(md_path)
        await self._repo().remove(scope.app_id, scope.project_id, scope.owner_id)
        return HandlerOutcome(
            md_path=md_path, kind=self.kind, upserted=0, deleted=1, skipped=0
        )

    def _repo(self):
        repo = self._deps.group_roster
        if repo is None:
            raise RuntimeError("cascade was built without the group roster mirror")
        return repo

    def _store(self) -> GroupRosterStore:
        return GroupRosterStore(self._deps.memory_root, self._repo())


def _scope(md_path: str) -> Scope:
    """``<app>/<project>/groups/<group_id>/group.md`` -> the group's scope."""
    parts = PurePosixPath(md_path).parts
    if len(parts) < 5 or parts[2] != "groups":
        raise ValueError(f"not a group.md path: {md_path!r}")
    app_id, project_id = resolve_scope(md_path)
    return Scope(
        app_id=app_id, project_id=project_id, owner_id=parts[3], owner_type="group"
    )
