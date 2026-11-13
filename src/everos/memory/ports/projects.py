"""Project links port — which groups belong to a project.

Chat memory writes group memory only; a project adds no owner of its own here
(its project memory is extracted by working memory). What a project-scope read
pools -- the ``project_scope`` search, ``GET /projects/{project_id}`` and its
``/profile`` -- is the memory of the project's groups, and which groups those
are is this port's answer (docs/group-memory.md, section 4.4); a personal read
with ``include_groups`` covers its owner's groups of the project alike. The
relation is many-to-many: a project may link several groups and a group
several projects. A separate project-link feature maintains it and plugs in
its own adapter here; the default adapter answers by the partition rule, a
project's groups being the groups stored in its ``(app_id, project_id)``
space.

Invariants a caller relies on:

* Each group is answered by its own storage scope (``owner_type`` ``group``),
  the address :class:`~everos.memory.ports.GroupStore` reads it by: a group
  linked to several projects is still stored in one space only.
* The answer says nothing about membership or visibility. A group the asker
  is not in is still the project's group; the access policy reads each
  roster and decides what the asker sees.

The reverse direction -- the projects a group belongs to -- is reserved for
the project-link feature. Nothing reads it yet, so it is not a method.
"""

from __future__ import annotations

__all__ = ["ProjectLinkStore"]

from typing import Protocol, runtime_checkable

from .query import Scope


@runtime_checkable
class ProjectLinkStore(Protocol):
    """Which groups belong to a project, addressed by ``(app_id, project_id)``."""

    async def groups_of(self, app_id: str, project_id: str) -> list[Scope]:
        """The groups of the project, each by its scope, sorted by space and id.

        A project with no group answers ``[]``.
        """
        ...
