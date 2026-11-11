"""Group-profile items through the cascade into ``group_profile_item`` and back.

docs/group-memory.md, sections 6.3 and 7.5. A real stack (:class:`GroupWorld`):
items are written by the markdown store, projected by the cascade handler into
a real LanceDB and read through the index port. What these pin:

- one row per entry: the owner, link and window columns, ``statement_tokens``
  for BM25, ``payload_json`` without the read-side fields, and no vector;
- the access policy's read-scope predicate applies unchanged and returns only
  the items whose sources lie in the asker's window;
- superseded and forgotten rows leave the scope (``deprecated_by``) and come
  back marked as such when a read does not filter on it, also when the mark
  arrives after the rows were indexed;
- every link column filters by its domain name; ``contains`` finds an item by
  one of its segments, BM25 by its words, ``find_by_entry_ids`` by its id, and
  ``find`` lists newest first;
- an edit of a window field reaches the row, and an entry the parser refuses
  fails the file instead of indexing half an item.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from everos.component.tokenizer import build_tokenizer
from everos.core.utils.datetime import to_timestamp_ms, today_with_timezone
from everos.infra.persistence.markdown import GroupProfileWriter
from everos.memory.access import AccessPolicy, ScopeRequest, resolve_identity
from everos.memory.models import FORGOTTEN_MARK
from everos.memory.ports import (
    Filter,
    GroupProfileIndex,
    NoSourceCheck,
    all_of,
    contains,
    eq,
    primary_key,
)
from tests.helpers.groups import GroupWorld, join


async def _read_scope(world: GroupWorld, asker: str) -> Filter:
    """The predicate the access policy compiles for ``asker`` reading ``trip``."""
    scope = await AccessPolicy(world.stores.groups).read_scope(
        resolve_identity(header=None, body_asker=asker),
        ScopeRequest(project_id="p", group_ids=("trip",)),
    )
    return scope.predicate


async def _ids(world: GroupWorld, where: Filter) -> list[str]:
    return [s.ref.entry_id for s in await world.deps.indexes.group_profile.find(where)]


async def test_an_item_projects_into_one_row(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        ref = await world.profile_item(
            "trip",
            "Book the hotel near the station",
            start=2_000,
            end=3_000,
            topic_id="tp_1",
            decided_by=["u_chen"],
            agreed_by=["u_betty"],
        )
        await world.project()

        repo = world.storage.repos.group_profile_item
        row = await repo.find_one_where(primary_key("default", "p", ref.id))
        assert row is not None
        assert (row.owner_type, row.owner_id, row.entry_id, row.md_path) == (
            "group",
            "trip",
            ref.entry_id,
            ref.md_path,
        )
        assert (row.kind, row.topic_id, row.statement) == (
            "decision",
            "tp_1",
            "Book the hotel near the station",
        )
        assert row.statement_tokens.split() == ["book", "hotel", "near", "station"]
        assert row.source_memcell_ids == ["mc_trip_2000"]
        assert (row.decided_by, row.agreed_by, row.sender_ids, row.origin) == (
            ["u_chen"],
            ["u_betty"],
            ["u_chen"],
            "member",
        )
        assert [
            to_timestamp_ms(at)
            for at in (row.source_start, row.source_end, row.observed_at)
        ] == [2_000, 3_000, 3_000]
        assert row.deprecated_by is None
        payload = json.loads(row.payload_json)
        assert not {"item_id", "status", "superseded_by"} & set(payload)
        assert payload["sources"][0]["quote"] == "Book the hotel near the station"
        assert repo.schema.vector_columns() == []


async def test_a_read_scope_sees_only_the_items_inside_the_askers_window(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        await world.roster("trip", [join("u_chen", 0), join("u_betty", 1_000)])
        before = await world.profile_item("trip", "Pick the venue", start=200, end=400)
        straddles = await world.profile_item(
            "trip", "Fix the budget", start=900, end=1_500
        )
        inside = await world.profile_item(
            "trip", "Depart on April 30", start=2_000, end=3_000
        )
        await world.project()

        assert await _ids(world, await _read_scope(world, "u_betty")) == [
            inside.entry_id
        ]
        assert await _ids(world, await _read_scope(world, "u_chen")) == [
            inside.entry_id,
            straddles.entry_id,
            before.entry_id,
        ]


async def test_replaced_and_forgotten_rows_leave_the_scope_with_their_marks(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        await world.roster("trip", [join("u_chen", 0)])
        first = await world.profile_item("trip", "Depart on May 1", start=100, end=200)
        dropped = await world.profile_item(
            "trip", "Chen books the flights", start=300, end=400
        )
        await world.project()
        # The marks arrive after the rows were indexed: a frontmatter-only
        # change, which the projection carries into ``deprecated_by``.
        second = await world.profile_item(
            "trip",
            "Depart on April 30",
            start=500,
            end=600,
            supersedes=first.entry_id,
        )
        assert await world.stores.group_profile.deprecate(
            world.scope("trip"),
            entries=[(dropped.md_path, dropped.entry_id)],
            replaced_by=FORGOTTEN_MARK,
            only_if=NoSourceCheck(),
        )
        await world.project()

        assert await _ids(world, await _read_scope(world, "u_chen")) == [
            second.entry_id
        ]
        every = await world.deps.indexes.group_profile.find(
            world.scope("trip").to_filter()
        )
        assert {
            s.ref.entry_id: (s.item.item_id, s.item.status, s.item.superseded_by)
            for s in every
        } == {
            first.entry_id: (first.entry_id, "superseded", second.entry_id),
            dropped.entry_id: (dropped.entry_id, "forgotten", None),
            second.entry_id: (second.entry_id, "current", None),
        }


async def test_every_link_column_filters_by_its_domain_name(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        topic = await world.profile_item(
            "trip",
            "Departure date",
            start=100,
            end=200,
            kind="topic",
            topic_status="consensus",
        )
        decision = await world.profile_item(
            "trip", "Depart on April 30", start=300, end=400, topic_id="tp_100"
        )
        dissent = await world.profile_item(
            "trip",
            "Betty would rather leave on May 2",
            start=500,
            end=600,
            kind="dissent",
            speaker="u_betty",
            held_by=["u_betty"],
            about=decision.entry_id,
        )
        role = await world.profile_item(
            "trip",
            "Chen makes the call",
            start=700,
            end=800,
            kind="role",
            member_id="u_chen",
            role="decision_maker",
            confidence="strong",
        )
        newer = await world.profile_item(
            "trip",
            "Depart on April 29",
            start=900,
            end=1_000,
            topic_id="tp_100",
            supersedes=decision.entry_id,
        )
        await world.project()

        every = world.scope("trip").to_filter()
        assert await _ids(world, all_of(every, eq("kind", "topic"))) == [topic.entry_id]
        assert await _ids(world, all_of(every, eq("topic_id", "tp_100"))) == [
            newer.entry_id,
            decision.entry_id,
            topic.entry_id,
        ]
        assert await _ids(world, all_of(every, eq("about", decision.entry_id))) == [
            dissent.entry_id
        ]
        chen_decides = all_of(
            every, eq("member_id", "u_chen"), eq("role", "decision_maker")
        )
        assert await _ids(world, chen_decides) == [role.entry_id]
        assert await _ids(
            world, all_of(every, eq("supersedes", decision.entry_id))
        ) == [newer.entry_id]
        assert await _ids(world, all_of(every, contains("held_by", "u_betty"))) == [
            dissent.entry_id
        ]
        statuses = {
            s.ref.entry_id: s.item.status
            for s in await world.deps.indexes.group_profile.find(every)
        }
        assert statuses == {
            topic.entry_id: "consensus",
            decision.entry_id: "superseded",
            dissent.entry_id: "open",
            role.entry_id: "active",
            newer.entry_id: "current",
        }


async def test_items_are_found_by_segment_by_words_and_by_id(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        await world.roster("trip", [join("u_chen", 0)])
        hotel = await world.profile_item(
            "trip", "Betty books the hotel", start=100, end=200, memcell_id="mc_a"
        )
        flight = await world.profile_item(
            "trip",
            "Chen books the morning flight",
            start=300,
            end=400,
            memcell_id="mc_b",
        )
        await world.project()
        index = world.deps.indexes.group_profile
        assert isinstance(index, GroupProfileIndex)
        visible = await _read_scope(world, "u_chen")

        assert await _ids(
            world, all_of(visible, contains("source_memcell_ids", "mc_b"))
        ) == [flight.entry_id]
        hits = await index.sparse_search(
            build_tokenizer().tokenize("hotel"), visible, limit=5
        )
        assert [hit.stored.ref.entry_id for hit in hits] == [hotel.entry_id]
        assert hits[0].score is not None and hits[0].score > 0
        assert hits[0].stored.item.status == "current"
        assert await _ids(world, visible) == [flight.entry_id, hotel.entry_id]
        assert [s.ref.entry_id for s in await index.find(visible, limit=1)] == [
            flight.entry_id
        ]
        by_id = await index.find_by_entry_ids(world.scope("trip"), [hotel.entry_id])
        assert [(s.item.item_id, s.item.statement, s.scope) for s in by_id] == [
            (hotel.entry_id, "Betty books the hotel", world.scope("trip"))
        ]


async def test_an_edit_of_a_window_field_reaches_the_row(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        ref = await world.profile_item(
            "trip", "Depart on April 30", start=2_000, end=3_000
        )
        await world.project()
        _edit(
            world.root.root / ref.md_path,
            "**source_start**: 1970-01-01T00:00:02+00:00",
            "**source_start**: 1970-01-01T00:00:01+00:00",
        )

        await world.project()

        row = await world.storage.repos.group_profile_item.find_one_where(
            primary_key("default", "p", ref.id)
        )
        assert row is not None
        assert to_timestamp_ms(row.source_start) == 1_000


async def test_an_entry_the_parser_refuses_fails_the_file(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        await GroupProfileWriter(world.root).append_entries(
            "trip",
            [({"kind": "decision"}, {"Statement": "No times at all"})],
            app_id="default",
            project_id="p",
        )
        assert (
            world.root.root
            / "default_app/p/groups/trip/profile"
            / f"group_profile-{today_with_timezone().isoformat()}.md"
        ).is_file()

        with pytest.raises(ValueError, match=r"has no 'source_start'"):
            await world.project()
        assert await world.storage.repos.group_profile_item.count() == 0


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")
