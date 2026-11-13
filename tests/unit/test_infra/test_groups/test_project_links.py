"""``PartitionProjectLinkStore``: a project's groups are the groups of its space.

Real ``group.md`` files and a real ``system.db``. What these pin
(docs/group-memory.md, section 4.4):

- the default answer to "which groups belong to project P" is every group
  stored in P's ``(app_id, project_id)`` space, in group-id order, each by its
  group scope in that space;
- a group of another project, or of the same project id under another app,
  is not P's; a project without groups answers ``[]``;
- membership plays no part: a group nobody is in any more is still listed
  (the access policy reads the rosters).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from everos.config import load_settings
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.groups import (
    GroupRosterStore,
    PartitionProjectLinkStore,
)
from everos.infra.persistence.sqlite import SqliteDatabase, SqliteRepos
from everos.memory.models import Group, RosterEvent
from everos.memory.ports import ProjectLinkStore, Scope


def _group(app_id: str, project_id: str, group_id: str) -> Scope:
    return Scope(
        app_id=app_id, project_id=project_id, owner_id=group_id, owner_type="group"
    )


@pytest.fixture
async def groups(tmp_path: Path) -> AsyncIterator[GroupRosterStore]:
    root = MemoryRoot(tmp_path)
    root.ensure()
    database = SqliteDatabase(root, load_settings().sqlite)
    await database.create_schema()
    yield GroupRosterStore(root, SqliteRepos.over(database).group_roster)
    await database.dispose()


async def _save(store: GroupRosterStore, scope: Scope, *events: RosterEvent) -> None:
    await store.save(
        scope, group=Group(group_id=scope.owner_id, created_at=0), events=events
    )


async def test_a_projects_groups_are_the_groups_of_its_space(
    groups: GroupRosterStore,
) -> None:
    joined = RosterEvent(op="join", member_id="u_chen", at=0)
    left = RosterEvent(op="leave", member_id="u_chen", at=10)
    await _save(groups, _group("default", "p", "ops"), joined)
    await _save(groups, _group("default", "p", "build"), joined)
    await _save(groups, _group("default", "p", "archive"), joined, left)
    await _save(groups, _group("default", "elsewhere", "travel"), joined)
    await _save(groups, _group("other-app", "p", "design"), joined)
    links = PartitionProjectLinkStore(groups)

    assert isinstance(links, ProjectLinkStore)
    assert await links.groups_of("default", "p") == [
        _group("default", "p", "archive"),
        _group("default", "p", "build"),
        _group("default", "p", "ops"),
    ]
    assert await links.groups_of("default", "elsewhere") == [
        _group("default", "elsewhere", "travel")
    ]
    assert await links.groups_of("other-app", "p") == [
        _group("other-app", "p", "design")
    ]
    assert await links.groups_of("default", "empty") == []
