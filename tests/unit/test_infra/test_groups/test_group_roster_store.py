"""``GroupRosterStore``: ``group.md`` as the roster log, mirrored into SQLite.

Real markdown under a temporary root and a real ``system.db``. What these pin
(docs/group-memory.md, section 14, task 1):

- ``group.md`` round-trips: meta in the frontmatter, one ``gr_`` entry per
  event, in the shape section 6.1 documents;
- the mirror changes in the same operation as the file, and a conflicting
  batch changes neither;
- a session is bound once and the binding is also written to ``group.md``;
- a rebuild from a wiped ``.index/`` (the cascade handler over ``group.md``)
  restores identical roster rows, bindings included;
- a group whose rows are missing re-mirrors from its file on first read.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from everos.core.errors import RosterConflictError
from everos.infra.cascade._projections.group_roster import GroupRosterHandler
from everos.infra.cascade.registry import match_kind
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.groups import GroupRosterStore
from everos.infra.persistence.markdown import MarkdownReader
from everos.infra.persistence.sqlite import SqliteDatabase, SqliteRepos
from everos.memory.models import Group, RosterEvent
from everos.memory.ports import Scope, SessionKey
from tests.fakes import handler_deps

_T0 = 1_772_323_200_000  # 2026-03-01T00:00:00Z
_SCOPE = Scope(
    app_id="default",
    project_id="spring-offsite",
    owner_id="trip-team",
    owner_type="group",
)
_GROUP = Group(
    group_id="trip-team",
    name="Trip team",
    description="Planning the spring offsite",
    created_at=_T0,
    created_by="u_chen",
)


def _roster() -> list[RosterEvent]:
    return [
        RosterEvent(
            op="join",
            member_id="u_chen",
            display_name="Chen",
            role="owner",
            title="Organiser",
            at=_T0,
        ),
        RosterEvent(op="join", member_id="u_betty", display_name="Betty", at=_T0),
        RosterEvent(
            op="join",
            member_id="bot_ava",
            kind="assistant",
            display_name="Ava",
            role="assistant",
            at=_T0,
        ),
    ]


async def _open(root: Path) -> tuple[SqliteDatabase, SqliteRepos]:
    from everos.config import load_settings

    database = SqliteDatabase(MemoryRoot(root), load_settings().sqlite)
    await database.create_schema()
    return database, SqliteRepos.over(database)


@pytest.fixture
async def store(tmp_path: Path) -> Any:
    root = MemoryRoot(tmp_path)
    root.ensure()
    database, repos = await _open(tmp_path)
    yield GroupRosterStore(root, repos.group_roster), repos
    await database.dispose()


def _rows(repos_rows: Any) -> dict[str, Any]:
    """Business columns only: ``updated_at`` is when the row was written."""
    meta, members, bindings = repos_rows

    def plain(row: Any) -> dict[str, Any]:
        return {
            k: v
            for k, v in row.model_dump().items()
            if k not in {"updated_at", "created_at"}
        }

    return {
        "meta": plain(meta),
        "members": [plain(m) for m in members],
        "bindings": [plain(b) for b in bindings],
    }


async def test_group_md_round_trips_and_the_mirror_follows(
    store: Any, tmp_path: Path
) -> None:
    groups, _ = store
    record = await groups.save(_SCOPE, group=_GROUP, events=_roster())

    path = tmp_path / "default_app/spring-offsite/groups/trip-team/group.md"
    parsed = await MarkdownReader.read(path)
    assert parsed.frontmatter["id"] == "group_trip-team"
    assert parsed.frontmatter["type"] == "group"
    assert parsed.frontmatter["track"] == "group"
    assert parsed.frontmatter["name"] == "Trip team"
    assert parsed.frontmatter["created_at"] == "2026-03-01T00:00:00+00:00"
    assert parsed.frontmatter["entry_count"] == 3
    assert [e.id for e in parsed.entries] == [
        "gr_20260301_00000001",
        "gr_20260301_00000002",
        "gr_20260301_00000003",
    ]
    first = parsed.entries[0].as_structured().inline
    assert first == {
        "op": "join",
        "member_id": "u_chen",
        "display_name": "Chen",
        "role": "owner",
        "title": "Organiser",
        "at": "2026-03-01T00:00:00+00:00",
        "source": "api",
    }

    assert [i.member_id for i in record.roster.current_members()] == [
        "bot_ava",
        "u_betty",
        "u_chen",
    ]
    fetched = await groups.get(_SCOPE)
    assert fetched is not None
    assert fetched.group == _GROUP
    assert fetched.roster == record.roster


async def test_a_conflicting_batch_writes_nothing(store: Any, tmp_path: Path) -> None:
    groups, _ = store
    await groups.save(_SCOPE, group=_GROUP, events=_roster())
    path = tmp_path / "default_app/spring-offsite/groups/trip-team/group.md"
    before = path.read_text(encoding="utf-8")

    with pytest.raises(RosterConflictError):
        await groups.save(
            _SCOPE,
            group=_GROUP,
            events=[
                RosterEvent(op="join", member_id="u_evan", at=_T0 + 1),
                RosterEvent(op="join", member_id="u_betty", at=_T0 + 2),  # open
            ],
        )

    assert path.read_text(encoding="utf-8") == before
    fetched = await groups.get(_SCOPE)
    assert fetched is not None
    assert not fetched.roster.intervals_of("u_evan")


async def test_a_leave_is_mirrored_before_save_returns(store: Any) -> None:
    groups, _ = store
    await groups.save(_SCOPE, group=_GROUP, events=_roster())
    await groups.save(
        _SCOPE,
        group=_GROUP,
        events=[RosterEvent(op="leave", member_id="u_betty", at=_T0 + 60_000)],
    )

    fetched = await groups.get(_SCOPE)
    assert fetched is not None
    assert not fetched.roster.is_current_member("u_betty")
    assert fetched.roster.windows_of("u_betty") == [(_T0, _T0 + 60_000)]


async def test_a_session_binds_once_and_the_binding_is_in_group_md(
    store: Any, tmp_path: Path
) -> None:
    groups, _ = store
    await groups.save(_SCOPE, group=_GROUP, events=_roster())
    session = SessionKey(
        app_id="default", project_id="spring-offsite", session_id="trip-team-main"
    )

    assert await groups.binding(session) is None
    assert await groups.bind(session, "trip-team", at=_T0 + 5) == "trip-team"
    # Never rebound: the earlier binding answers.
    assert await groups.bind(session, "other-team", at=_T0 + 6) == "trip-team"
    assert await groups.binding(session) == "trip-team"

    path = tmp_path / "default_app/spring-offsite/groups/trip-team/group.md"
    sessions = (await MarkdownReader.read(path)).frontmatter["sessions"]
    assert sessions == [
        {"session_id": "trip-team-main", "bound_at": "2026-03-01T00:00:00.005000+00:00"}
    ]


async def test_a_rebuild_from_a_wiped_index_restores_identical_rows(
    store: Any, tmp_path: Path
) -> None:
    groups, repos = store
    await groups.save(_SCOPE, group=_GROUP, events=_roster())
    await groups.save(
        _SCOPE,
        group=_GROUP,
        events=[
            RosterEvent(op="leave", member_id="u_betty", at=_T0 + 10),
            RosterEvent(op="join", member_id="u_betty", at=_T0 + 20, title="PM"),
        ],
    )
    await groups.bind(
        SessionKey(app_id="default", project_id="spring-offsite", session_id="s1"),
        "trip-team",
        at=_T0 + 30,
    )
    written = _rows(
        await repos.group_roster.load("default", "spring-offsite", "trip-team")
    )

    # Wipe .index/ (system.db with it) and rebuild through the cascade handler.
    shutil.rmtree(tmp_path / ".index")
    MemoryRoot(tmp_path).ensure()
    database, rebuilt_repos = await _open(tmp_path)
    try:
        rel = "default_app/spring-offsite/groups/trip-team/group.md"
        spec = match_kind(rel)
        assert spec is not None and spec.name == "group_roster"
        handler = spec.handler_factory(
            handler_deps(
                memory_root=MemoryRoot(tmp_path),
                tokenizer=None,
                group_roster=rebuilt_repos.group_roster,
            )
        )
        assert isinstance(handler, GroupRosterHandler)
        await handler.handle_added_or_modified(rel)
        rebuilt = _rows(
            await rebuilt_repos.group_roster.load(
                "default", "spring-offsite", "trip-team"
            )
        )
    finally:
        await database.dispose()

    assert rebuilt == written


async def test_missing_rows_re_mirror_from_the_file_on_read(
    store: Any, tmp_path: Path
) -> None:
    groups, repos = store
    await groups.save(_SCOPE, group=_GROUP, events=_roster())
    await repos.group_roster.remove("default", "spring-offsite", "trip-team")

    fetched = await groups.get(_SCOPE)

    assert fetched is not None
    assert fetched.group.name == "Trip team"
    assert len(fetched.roster.current_members()) == 3
    assert await repos.group_roster.load("default", "spring-offsite", "trip-team")


async def test_an_unknown_group_reads_as_none(store: Any) -> None:
    groups, _ = store
    assert await groups.get(_SCOPE) is None
