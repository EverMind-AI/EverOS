"""Project reads: the groups of a project an asker is in (docs/group-memory.md, 8.7).

A project adds no owner: chat memory writes group memory only, and a project
read pools the memory of the project's groups. Which groups those are is the
project links' answer (section 4.4; by default the groups stored in the
project's ``(app_id, project_id)`` space), which the access policy asks. Both
reads run over the groups of the project the asker is a current member of,
through the access policy:

* :meth:`ProjectUseCases.read` (``GET /projects/{project_id}``) -- those
  groups and their current members. Only the asker's groups are listed: a
  roster is read by the group's current members only (section 9.1), so the
  members of a group the asker is not in are not handed out.
* :meth:`ProjectUseCases.profile` (``GET /projects/{project_id}/profile``) --
  the visible current group-profile items of those groups, each tagged with
  its group, oldest first; duplicates across groups are not merged. The read
  scope's predicate is in the index query, as for every group read.

An asker in none of the project's groups is refused (403).
"""

from __future__ import annotations

__all__ = ["ProjectMember", "ProjectProfile", "ProjectUseCases", "ProjectView"]

from dataclasses import dataclass, field

from everos.memory.access import AccessPolicy, ScopeRequest, resolve_identity
from everos.memory.extract.group_profile import is_current
from everos.memory.ports import GroupProfileIndex
from everos.memory.search import (
    GroupProfileEntry,
    shape_profile_item,
    visible_profile_items,
)
from everos.memory.search.audience import with_audience


@dataclass(frozen=True)
class ProjectMember:
    """One current member of the asker's groups, with the groups they share."""

    member_id: str
    display_name: str
    kind: str
    group_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProjectView:
    """A project as ``GET /projects/{project_id}`` returns it."""

    app_id: str
    project_id: str
    groups: list[dict[str, str]] = field(default_factory=list)
    """``[{group_id, name}]`` of the groups the asker is in, by id."""
    members: list[ProjectMember] = field(default_factory=list)


@dataclass(frozen=True)
class ProjectProfile:
    """The project roll-up of the group profiles the asker may see."""

    app_id: str
    project_id: str
    group_ids: list[str]
    items: list[GroupProfileEntry] = field(default_factory=list)


class ProjectUseCases:
    """The project reads over one memory root.

    Args:
        access: The access policy (which groups, which windows, which
            rosters).
        profile: The group-profile index; ``None`` leaves the roll-up empty.
    """

    def __init__(self, access: AccessPolicy, profile: GroupProfileIndex | None) -> None:
        self._access = access
        self._profile = profile

    async def read(
        self,
        *,
        app_id: str,
        project_id: str,
        asker_id: str | None,
        asserted_asker: str | None,
    ) -> ProjectView:
        """The asker's groups in the project and their current members.

        Raises:
            InvalidInputError: No asker.
            ForbiddenError: The asker is in none of the project's groups.
        """
        scope = await self._access.read_scope(
            resolve_identity(header=asserted_asker, body_asker=asker_id),
            ScopeRequest(app_id=app_id, project_id=project_id, project_scope=True),
        )
        members: dict[str, ProjectMember] = {}
        groups: list[dict[str, str]] = []
        for view in scope.groups:
            groups.append({"group_id": view.group_id, "name": view.group.name})
            for interval in view.roster.current_members():
                found = members.setdefault(
                    interval.member_id,
                    ProjectMember(
                        member_id=interval.member_id,
                        display_name=interval.display_name,
                        kind=interval.kind,
                    ),
                )
                found.group_ids.append(view.group_id)
        return ProjectView(
            app_id=app_id,
            project_id=project_id,
            groups=groups,
            members=sorted(members.values(), key=lambda m: m.member_id),
        )

    async def profile(
        self,
        *,
        app_id: str,
        project_id: str,
        asker_id: str | None,
        asserted_asker: str | None,
    ) -> ProjectProfile:
        """Every visible current group-profile item of the asker's groups.

        Raises:
            InvalidInputError: No asker.
            ForbiddenError: The asker is in none of the project's groups.
        """
        scope = await self._access.read_scope(
            resolve_identity(header=asserted_asker, body_asker=asker_id),
            ScopeRequest(app_id=app_id, project_id=project_id, project_scope=True),
        )
        stored = (
            await visible_profile_items(self._profile, scope)
            if self._profile is not None
            else []
        )
        current = sorted(
            (s for s in stored if is_current(s.item)),
            key=lambda s: (s.item.source_end, s.scope.owner_id, s.ref.entry_id),
        )
        return ProjectProfile(
            app_id=app_id,
            project_id=project_id,
            group_ids=list(scope.owner_ids),
            items=with_audience(scope, [shape_profile_item(s) for s in current]),
        )
