"""Value objects and filter vocabulary of ``everos.memory.ports.query``."""

from __future__ import annotations

import dataclasses

import pytest

from everos.memory.models import Episode
from everos.memory.ports import (
    All,
    Comparison,
    Filter,
    IndexHit,
    MemoryRef,
    Page,
    PageResult,
    Scope,
    SessionKey,
    Stored,
    all_of,
    eq,
    primary_key,
)


def _scope() -> Scope:
    return Scope(app_id="a", project_id="p", owner_id="u1", owner_type="user")


def test_scope_to_filter_pins_the_four_ownership_fields() -> None:
    where = _scope().to_filter()

    assert isinstance(where, All)
    assert where.children == (
        Comparison("owner_id", "eq", "u1"),
        Comparison("owner_type", "eq", "user"),
        Comparison("app_id", "eq", "a"),
        Comparison("project_id", "eq", "p"),
    )


def test_scope_filter_composes_with_more_conditions() -> None:
    where = all_of(_scope().to_filter(), eq("parent_type", "memcell"))

    assert isinstance(where, All)
    assert len(where.children) == 5


@pytest.mark.parametrize(
    "value",
    [
        _scope(),
        SessionKey(app_id="a", project_id="p", session_id="s"),
        Page(sort_by="timestamp"),
        MemoryRef(id="u1_ep_1", entry_id="ep_1", md_path="x.md"),
        IndexHit(id="u1_ep_1", score=None, fields={}),
    ],
)
def test_value_objects_are_frozen(value: object) -> None:
    field = dataclasses.fields(value)[0].name  # type: ignore[arg-type]
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(value, field, "changed")


def test_page_defaults_to_first_page_descending() -> None:
    page = Page(sort_by="timestamp")

    assert (page.number, page.size, page.descending) == (1, 20, True)


def test_stored_carries_identity_beside_the_domain_object() -> None:
    episode = Episode(owner_id="u1", episode="text", timestamp=1, parent_id="mc_1")
    ref = MemoryRef(id="u1_ep_1", entry_id="ep_1", md_path="users/u1/e.md")

    stored = Stored(ref=ref, scope=_scope(), item=episode)
    page = PageResult(items=[stored], total=7)

    assert page.items[0].item.parent_type == "memcell"
    assert page.items[0].ref.id == "u1_ep_1"
    assert page.total == 7


def test_primary_key_addresses_a_record_by_space_and_id() -> None:
    where = primary_key("a", "p", "u1")

    assert isinstance(where, Filter)
    assert isinstance(where, All)
    assert [c.field for c in where.children] == ["app_id", "project_id", "id"]  # type: ignore[attr-defined]
