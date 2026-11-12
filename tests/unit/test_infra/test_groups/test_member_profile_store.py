"""Member-in-group profiles at the storage level (docs/group-memory.md, 6.1, 6.3).

The one profile store and handler, in a group's space. Real markdown under a
temporary root and a real LanceDB. What these pin:

- a member's profile is ``groups/<g>/members/<member_id>.md`` and round-trips
  with its items' provenance keys, the pending segments and the history, in
  the file's own key order;
- ``people`` lists the members that have a profile document;
- the scope is the space: a group's needs the member, a user's space holds
  only the user's own profile (``users/<u>/user.md``);
- the cascade projects it into ``user_profile`` as ``owner_type = group``,
  ``owner_id = <g>``, ``subject_id = <m>``, under a row id no user id can
  equal; the index reads it back as the member's profile, and a personal read
  of the member's own id never sees it;
- deleting the file removes the row.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.component.tokenizer import build_tokenizer
from everos.infra.cascade.registry import match_kind
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.index import primary_key
from everos.infra.persistence.index.adapters import ProfileIndexAdapter
from everos.infra.persistence.markdown import MarkdownProfileStore, MarkdownReader
from everos.memory.models import ExplicitInfoItem, Profile
from everos.memory.ports import Scope
from tests.fakes import handler_deps
from tests.fakes.storage import make_storage

_GROUP = Scope(app_id="default", project_id="p", owner_id="trip", owner_type="group")
_PERSONAL = Scope(
    app_id="default", project_id="p", owner_id="u_betty", owner_type="user"
)


def _profile() -> Profile:
    item = ExplicitInfoItem(
        description="Books the hotel for the team",
        category="commitments",
        evidence="Betty: I'll book the hotel from April 30.",
        observed_at=1_772_442_300_000,
        item_id="pi_1",
        source_memcell_ids=["mc_1"],
        source_start=1_772_442_120_000,
        source_end=1_772_442_480_000,
    )
    return Profile(
        owner_id="u_betty",
        summary="Betty handles the hotel.",
        timestamp=1_772_442_480_000,
        explicit_info=[item],
        primary_language="English",
        pending_memcell_ids=["mc_2"],
        history=[{"description": "Undecided on dates", "replaced_at": 1}],
    )


async def test_a_member_profile_round_trips_under_members(tmp_path: Path) -> None:
    root = MemoryRoot(tmp_path)
    store = MarkdownProfileStore(root)
    ref = await store.write(_GROUP, _profile())

    assert ref.md_path == "default_app/p/groups/trip/members/u_betty.md"
    assert ref.id == "trip:u_betty"
    back = await store.read(_GROUP, "u_betty")
    assert back == _profile()
    assert await store.people(_GROUP) == ["u_betty"]
    assert await store.read(_GROUP, "u_chen") is None
    # Nothing reached the member's personal tree.
    assert not (tmp_path / "default_app" / "p" / "users").exists()
    parsed = await MarkdownReader.read(tmp_path / ref.md_path)
    assert list(parsed.frontmatter) == [
        "id",
        "type",
        "schema_version",
        "group_id",
        "track",
        "member_id",
        "summary",
        "explicit_info",
        "implicit_traits",
        "profile_timestamp_ms",
        "primary_language",
        "pending_memcell_ids",
        "history",
    ]


async def test_the_scope_names_the_space(tmp_path: Path) -> None:
    store = MarkdownProfileStore(MemoryRoot(tmp_path))
    with pytest.raises(ValueError, match="one member"):
        await store.read(_GROUP)
    with pytest.raises(ValueError, match="own profile"):
        await store.read(_PERSONAL, "u_chen")
    # The user's space: their own user.md, never a file under a group.
    ref = await store.write(_PERSONAL, _profile().model_copy(update={"history": []}))
    assert ref.md_path == "default_app/p/users/u_betty/user.md"
    assert ref.id == "u_betty"
    assert await store.people(_PERSONAL) == ["u_betty"]
    assert await store.people(_GROUP) == []


async def test_the_cascade_indexes_it_as_the_members_profile_in_the_group(
    tmp_path: Path,
) -> None:
    root = MemoryRoot(tmp_path)
    root.ensure()
    ref = await MarkdownProfileStore(root).write(_GROUP, _profile())
    storage = make_storage(root)
    await storage.create_schema()
    try:
        spec = match_kind(ref.md_path)
        assert spec is not None and spec.name == "group_member_profile"
        handler = spec.handler_factory(
            handler_deps(
                memory_root=root, tokenizer=build_tokenizer(), index=storage.repos
            )
        )
        await handler.handle_added_or_modified(ref.md_path)

        index = ProfileIndexAdapter(storage.repos.user_profile)
        found = await index.member(_GROUP, "u_betty")
        assert found is not None
        assert found.scope == _GROUP
        assert found.item.owner_id == "u_betty"
        item = found.item.explicit_info[0]
        assert item.model_extra is not None
        assert item.model_extra["source_memcell_ids"] == ["mc_1"]
        assert item.model_extra["source_start"] == 1_772_442_120_000
        row = await storage.repos.user_profile.find_one_where(
            primary_key("default", "p", "trip:u_betty")
        )
        assert (row.owner_type, row.owner_id, row.subject_id) == (
            "group",
            "trip",
            "u_betty",
        )
        # The member's personal profile is another document: none here.
        assert await index.get(_PERSONAL) is None
        assert await index.member(_GROUP, "u_chen") is None

        (tmp_path / ref.md_path).unlink()
        await handler.handle_deleted(ref.md_path)
        assert await index.member(_GROUP, "u_betty") is None
    finally:
        await storage.aclose()
