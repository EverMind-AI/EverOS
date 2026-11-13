"""A hand-broken markdown entry costs that entry only.

Real markdown, ``system.db`` and LanceDB (``tests/helpers/groups.py``); each
file is projected by its own cascade handler, as the worker runs it. Every
case breaks one entry among good ones, the way a hand edit does. What these
pin:

- an episode daily log: an entry whose timestamp no longer parses, and one
  whose close marker was deleted, are skipped -- logged as
  ``markdown_entry_unreadable`` with the file, the entry id, its line and the
  reason, and counted in the projection's ``unreadable`` -- while every other
  entry of the file is indexed, searchable, listed by get and has its
  history; a row the index already held for the broken entry is kept, not
  deleted;
- a profile file (``user.md``): an item that is no longer an item is dropped
  from get and search (``profile_entry_dropped`` names the file and the
  item's position) and counted; the other items are read;
- a group file (``group.md``): an unreadable roster entry is skipped and
  counted; every other member's events fold, and the member it names is left
  out of the roster -- an unreadable ``leave`` never lets someone who left
  read on; an entry that names no member still fails the file; a roster write
  refuses while the log holds an entry it cannot read;
- a group profile log: an entry the parser refuses is skipped; the rest is
  indexed and listed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import structlog.testing

from everos.component.tokenizer import build_tokenizer
from everos.core.errors import ForbiddenError, NotFoundError
from everos.infra.cascade.models import HandlerOutcome
from everos.infra.cascade.registry import match_kind
from everos.memory.models import ExplicitInfoItem, Profile
from everos.service import GetRequest, SearchRequest
from tests.fakes import handler_deps
from tests.helpers.groups import APP, PROJECT, GroupWorld, join, leave

T0 = 1_772_323_200_000


async def _project(world: GroupWorld, path: Path) -> HandlerOutcome:
    """One file through its cascade handler, as the worker runs it."""
    rel = path.relative_to(world.root.root).as_posix()
    spec = match_kind(rel)
    assert spec is not None, rel
    deps = handler_deps(
        memory_root=world.root,
        tokenizer=build_tokenizer(),
        index=world.storage.repos,
        embedding=world.embedding,
        group_roster=world.storage.sqlite.group_roster,
    )
    outcome = await spec.handler_factory(deps).handle_added_or_modified(rel)
    await world.storage.index.ensure_business_indexes()
    return outcome


def _file(world: GroupWorld, pattern: str) -> Path:
    (path,) = sorted(world.root.root.glob(pattern))
    return path


def _ids(path: Path) -> list[str]:
    return re.findall(r"<!-- entry:(\S+) -->", path.read_text(encoding="utf-8"))


def _set_inline(path: Path, entry_id: str, key: str, value: str) -> None:
    """Rewrite one inline field of one entry, as a hand edit would."""
    text = path.read_text(encoding="utf-8")
    start = text.index(f"<!-- entry:{entry_id} -->")
    end = text.index(f"<!-- /entry:{entry_id} -->", start)
    field = re.compile(rf"^\*\*{re.escape(key)}\*\*: .*$", re.MULTILINE)
    body, n = field.subn(f"**{key}**: {value}", text[start:end])
    assert n == 1, (entry_id, key)
    path.write_text(text[:start] + body + text[end:], encoding="utf-8")


def _drop_inline(path: Path, entry_id: str, key: str) -> None:
    """Delete one inline line of one entry, as a hand edit would."""
    text = path.read_text(encoding="utf-8")
    start = text.index(f"<!-- entry:{entry_id} -->")
    end = text.index(f"<!-- /entry:{entry_id} -->", start)
    field = re.compile(rf"^\*\*{re.escape(key)}\*\*: .*\n", re.MULTILINE)
    body, n = field.subn("", text[start:end])
    assert n == 1, (entry_id, key)
    path.write_text(text[:start] + body + text[end:], encoding="utf-8")


def _drop_close_marker(path: Path, entry_id: str) -> None:
    text = path.read_text(encoding="utf-8")
    marker = f"<!-- /entry:{entry_id} -->\n"
    assert marker in text
    path.write_text(text.replace(marker, "", 1), encoding="utf-8")


def _warnings(captured: list[dict[str, Any]], event: str) -> list[dict[str, Any]]:
    return [e for e in captured if e["event"] == event]


def _line_of(path: Path, entry_id: str) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines.index(f"<!-- entry:{entry_id} -->") + 1


# ── an episode daily log ─────────────────────────────────────────────────


async def _amy(world: GroupWorld) -> Path:
    for i, text in enumerate(("Lisbon tram", "Lisbon hotel", "Lisbon tiles")):
        await world.personal_episode("u_amy", text, at=T0 + i * 1_000)
    return _file(world, "*/*/users/u_amy/episodes/episode-*.md")


async def _amy_episode_ids(world: GroupWorld) -> list[str]:
    got = await world.memory.get(
        GetRequest(
            app_id=APP,
            project_id=PROJECT,
            user_id="u_amy",
            memory_type="episode",
            page_size=50,
        )
    )
    return sorted(e.id for e in got.data.episodes)


async def _amy_search_ids(world: GroupWorld) -> list[str]:
    found = await world.memory.search(
        SearchRequest(
            app_id=APP, project_id=PROJECT, user_id="u_amy", query="Lisbon", top_k=10
        )
    )
    return sorted(e.id for e in found.data.episodes)


async def test_an_episode_with_a_broken_timestamp_costs_that_entry_only(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        log = await _amy(world)
        first, broken, last = _ids(log)
        _set_inline(log, broken, "timestamp", "yesterday-ish")
        md_path = log.relative_to(world.root.root).as_posix()

        with structlog.testing.capture_logs() as captured:
            outcome = await _project(world, log)

        assert (outcome.upserted, outcome.unreadable) == (2, 1)
        assert [
            (w["path"], w["entry_id"], w["line"], w["kind"])
            for w in _warnings(captured, "markdown_entry_unreadable")
        ] == [(md_path, broken, _line_of(log, broken), "episode")]
        assert (
            "yesterday-ish"
            in _warnings(captured, "markdown_entry_unreadable")[0]["reason"]
        )
        good = sorted(f"u_amy_{e}" for e in (first, last))
        assert await _amy_episode_ids(world) == good
        assert await _amy_search_ids(world) == good
        for entry_id in (first, last):
            history = await world.memory.memory_history(
                app_id=APP,
                project_id=PROJECT,
                record_id=f"u_amy_{entry_id}",
                user_id="u_amy",
            )
            assert history.current == f"u_amy_{entry_id}"
        with pytest.raises(NotFoundError):
            await world.memory.memory_history(
                app_id=APP,
                project_id=PROJECT,
                record_id=f"u_amy_{broken}",
                user_id="u_amy",
            )


async def test_an_episode_without_its_close_marker_costs_that_entry_only(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        log = await _amy(world)
        first, broken, last = _ids(log)
        _drop_close_marker(log, broken)

        outcome = await _project(world, log)

        # The entry after the broken one is read: the scan goes on past it.
        assert (outcome.upserted, outcome.unreadable) == (2, 1)
        good = sorted(f"u_amy_{e}" for e in (first, last))
        assert await _amy_episode_ids(world) == good
        assert await _amy_search_ids(world) == good


async def test_a_row_the_index_holds_for_a_broken_entry_is_kept(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        log = await _amy(world)
        await _project(world, log)
        everything = await _amy_episode_ids(world)
        _drop_close_marker(log, _ids(log)[1])

        outcome = await _project(world, log)

        assert (outcome.deleted, outcome.skipped, outcome.unreadable) == (0, 2, 1)
        assert await _amy_episode_ids(world) == everything


async def test_a_well_formed_log_reports_nothing_unreadable(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        log = await _amy(world)

        with structlog.testing.capture_logs() as captured:
            outcome = await _project(world, log)

        assert (outcome.upserted, outcome.unreadable) == (3, 0)
        assert not _warnings(captured, "markdown_entry_unreadable")


# ── a profile file ───────────────────────────────────────────────────────


async def test_a_broken_profile_item_costs_that_item_only(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        await world.stores.profile.write(
            world.personal("u_amy"),
            Profile(
                owner_id="u_amy",
                summary="Amy plans a Lisbon trip",
                timestamp=1,
                explicit_info=[
                    ExplicitInfoItem(description=f"Lisbon fact {i}", category="trip")
                    for i in range(3)
                ],
            ),
        )
        user_md = _file(world, "*/*/users/u_amy/user.md")
        text = user_md.read_text(encoding="utf-8")
        assert text.count("description: Lisbon fact 1") == 1
        user_md.write_text(
            text.replace("description: Lisbon fact 1", "description: [a, list]"),
            encoding="utf-8",
        )

        with structlog.testing.capture_logs() as captured:
            outcome = await _project(world, user_md)

        assert (outcome.upserted, outcome.unreadable) == (1, 1)
        dropped = _warnings(captured, "profile_entry_dropped")
        assert [(d["md_path"], d["bucket"], d["index"]) for d in dropped] == [
            (user_md.relative_to(world.root.root).as_posix(), "ExplicitInfoItem", 1)
        ]
        assert dropped[0]["reason"].startswith("description:")
        got = await world.memory.get(
            GetRequest(
                app_id=APP,
                project_id=PROJECT,
                user_id="u_amy",
                memory_type="profile",
                page_size=50,
            )
        )
        found = await world.memory.search(
            SearchRequest(
                app_id=APP,
                project_id=PROJECT,
                user_id="u_amy",
                query="Lisbon",
                top_k=10,
                include_profile=True,
            )
        )
        for profile in (got.data.profiles[0], found.data.profiles[0]):
            items = profile.profile_data["explicit_info"]
            assert [i["description"] for i in items] == [
                "Lisbon fact 0",
                "Lisbon fact 2",
            ]


# ── a group file ─────────────────────────────────────────────────────────


async def _trip(world: GroupWorld) -> Path:
    await world.roster(
        "trip",
        [
            join("u_chen", T0),
            join("u_betty", T0),
            join("u_dan", T0),
            leave("u_dan", T0 + 5_000),
        ],
    )
    await world.episode("trip", "Lisbon venue booked", start=T0 + 1_000, end=T0 + 2_000)
    await world.episode("trip", "Lisbon hotel booked", start=T0 + 6_000, end=T0 + 7_000)
    await world.project()
    return _file(world, "*/*/groups/trip/group.md")


async def _members(world: GroupWorld) -> list[str]:
    record = await world.stores.groups.get(world.scope("trip"))
    assert record is not None
    return sorted({i.member_id for i in record.roster.intervals})


async def _search_as(world: GroupWorld, asker: str) -> list[str]:
    found = await world.memory.search(
        SearchRequest(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id=asker,
            query="Lisbon",
            top_k=10,
        )
    )
    return sorted(e.id for e in found.data.episodes)


async def test_a_broken_roster_entry_leaves_out_only_its_member(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        group_md = await _trip(world)
        chen_sees = await _search_as(world, "u_chen")
        dans_leave = _ids(group_md)[-1]
        _set_inline(group_md, dans_leave, "at", "not a time")

        with structlog.testing.capture_logs() as captured:
            outcome = await _project(world, group_md)

        assert (outcome.upserted, outcome.unreadable) == (1, 1)
        assert [
            (w["entry_id"], w["line"], w["kind"])
            for w in _warnings(captured, "markdown_entry_unreadable")
        ] == [(dans_leave, _line_of(group_md, dans_leave), "group_roster")]
        assert [
            w["member_id"] for w in _warnings(captured, "group_roster_member_left_out")
        ] == ["u_dan"]
        # Every other member's events fold; Dan, whose leave cannot be read,
        # is left out instead of being a member again.
        assert await _members(world) == ["u_betty", "u_chen"]
        assert await _search_as(world, "u_chen") == chen_sees
        assert len(chen_sees) == 2
        with pytest.raises(ForbiddenError):
            await _search_as(world, "u_dan")


async def test_a_roster_entry_without_its_close_marker_leaves_out_its_member(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        group_md = await _trip(world)
        bettys_join = _ids(group_md)[1]
        _drop_close_marker(group_md, bettys_join)

        outcome = await _project(world, group_md)

        assert outcome.unreadable == 1
        assert await _members(world) == ["u_chen", "u_dan"]
        assert len(await _search_as(world, "u_chen")) == 2


async def test_a_roster_entry_naming_no_member_still_fails_the_file(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        group_md = await _trip(world)
        entry_id = _ids(group_md)[1]
        _set_inline(group_md, entry_id, "at", "not a time")
        _drop_inline(group_md, entry_id, "member_id")

        with pytest.raises(ValueError, match="names no member"):
            await _project(world, group_md)


async def test_a_roster_write_refuses_while_an_entry_is_unreadable(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        group_md = await _trip(world)
        _drop_close_marker(group_md, _ids(group_md)[1])

        with pytest.raises(ValueError, match="unreadable"):
            await world.roster("trip", [join("u_eve", T0 + 9_000)])
        assert "u_eve" not in group_md.read_text(encoding="utf-8")


# ── a group profile log ──────────────────────────────────────────────────


async def test_a_broken_group_profile_entry_costs_that_entry_only(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        await world.roster("trip", [join("u_chen", T0)])
        refs = [
            await world.profile_item(
                "trip", f"Lisbon decision {i}", start=T0 + i, end=T0 + i
            )
            for i in range(3)
        ]
        log = _file(world, "*/*/groups/trip/profile/group_profile-*.md")
        _set_inline(log, refs[1].entry_id, "kind", "nonsense")

        outcome = await _project(world, log)

        assert (outcome.upserted, outcome.unreadable) == (2, 1)
        got = await world.memory.get(
            GetRequest(
                app_id=APP,
                project_id=PROJECT,
                group_id="trip",
                asker_id="u_chen",
                memory_type="group_profile",
                page_size=50,
            )
        )
        assert sorted(i.item_id for i in got.data.group_profile) == sorted(
            r.entry_id for r in (refs[0], refs[2])
        )
