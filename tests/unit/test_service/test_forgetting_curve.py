"""The forgetting curve and asking about the past, through the memory service.

docs/configuration.md (``[search]``) and docs/api.md (``include_past``,
``asked_at``). Real markdown, ``system.db`` and LanceDB (``GroupWorld``):

* off by default -- no writer, no strength row, no new field in a response;
* on, a search reinforces the decider's own core and a ``get`` what its page
  returned, after the response (the strength writer); a round that fell back
  to the ranking reinforces nothing, and neither does a read-only server
  (``[cascade] enabled = false``) or ``reinforce = false``;
* the curve ranks an old memory below a fresh one, and a reinforced one back
  up; ``asked_at`` is the time ages are measured to;
* ``include_past`` brings back, in the user's own memory, the superseded
  versions and the forgotten memories, each with its ``status`` (and its
  successor and that one's time); in a group, the superseded versions inside
  the asker's windows, never a forgotten record -- with the group cases of the
  leak suite (``test_group_leak_suite.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient
from pydantic import ValidationError

from everos.config import Settings
from everos.core.utils.datetime import from_timestamp_ms, to_display_tz
from everos.memory.search.forgetting import DAY_MS
from everos.service import (
    ForgetTarget,
    GetRequest,
    MemoryChanges,
    SearchRequest,
)
from tests.helpers.groups import APP, PROJECT, GroupWorld, join

T0 = 1_772_323_200_000
"""2026-03-01T00:00:00Z: every record time below is ``T0`` plus an offset."""


def _settings(**search: Any) -> Settings:
    base = Settings()
    cascade = search.pop("cascade_enabled", True)
    return base.model_copy(
        update={
            "search": base.search.model_copy(update=search),
            "cascade": base.cascade.model_copy(update={"enabled": cascade}),
        }
    )


def _curve(**extra: Any) -> Settings:
    return _settings(forgetting_curve=True, **extra)


def _keep(*core: int) -> FakeLLMClient:
    """A decider that keeps ``core`` (global indices) and stops."""
    reply = json.dumps({"core": list(core), "next_queries": []})
    return FakeLLMClient(handler=lambda *_a, **_k: reply)


def _search(user: str = "amy", **fields: Any) -> SearchRequest:
    values: dict[str, Any] = {
        "app_id": APP,
        "project_id": PROJECT,
        "user_id": user,
        "query": "trip marker Lisbon",
        "top_k": 20,
        **fields,
    }
    return SearchRequest(**values)


async def _rows(world: GroupWorld) -> list[Any]:
    return await world.storage.sqlite.memory_strength.list_all()


async def _seed_two(world: GroupWorld) -> tuple[str, str]:
    """An old and a fresh episode of amy's that the same words recall."""
    old = await world.personal_episode(
        "amy", "trip marker Lisbon hotel booked", at=T0 - 400 * DAY_MS
    )
    new = await world.personal_episode(
        "amy", "trip marker Lisbon hotel booked", at=T0 - 5 * DAY_MS
    )
    await world.project()
    return old, new


# ── off by default ────────────────────────────────────────────────────────────


async def test_off_by_default_nothing_is_written_and_nothing_changes(
    tmp_path: Path,
) -> None:
    assert Settings().search.forgetting_curve is False
    async with GroupWorld.open(tmp_path) as world:
        await _seed_two(world)
        await world.memory.start()
        found = await world.memory.search(_search(asked_at=T0))
        await world.memory.get(
            GetRequest(app_id=APP, project_id=PROJECT, user_id="amy",
                       memory_type="episode")
        )  # fmt: skip
        await world.memory.aclose()
        assert await _rows(world) == []
        for item in found.data.model_dump(mode="json")["episodes"]:
            assert {"status", "superseded_by", "valid_until"}.isdisjoint(item)


# ── reinforcement ─────────────────────────────────────────────────────────────


async def test_a_search_reinforces_the_deciders_core_after_answering(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path, settings=_curve(), decider=_keep(0)) as world:
        await _seed_two(world)
        await world.memory.start()
        found = await world.memory.search(_search(asked_at=T0))
        await world.memory.aclose()
        rows = await _rows(world)
        assert len(rows) == 1  # the one core item, not the filled one
        row = rows[0]
        assert (row.owner_type, row.owner_id) == ("user", "amy")
        assert row.reinforcements == 1
        assert row.reinforced_at_ms == T0
        assert row.strength_days > 30.0
        core = found.data.episodes[0]
        assert row.memory_key.startswith("ep:memcell:")
        assert core.id.startswith("amy_")


async def test_a_round_that_fell_back_reinforces_nothing(tmp_path: Path) -> None:
    broken = FakeLLMClient(handler=lambda *_a, **_k: "no json here")
    settings = _curve()
    settings = settings.model_copy(
        update={"decider": settings.decider.model_copy(update={"retries": 0})}
    )
    async with GroupWorld.open(tmp_path, settings=settings, decider=broken) as world:
        await _seed_two(world)
        await world.memory.start()
        found = await world.memory.search(_search(asked_at=T0))
        await world.memory.aclose()
        assert found.data.degraded  # the round fell back to the ranking
        assert await _rows(world) == []


async def test_get_reinforces_what_its_page_returned(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path, settings=_curve()) as world:
        await _seed_two(world)
        await world.memory.start()
        await world.memory.get(
            GetRequest(app_id=APP, project_id=PROJECT, user_id="amy",
                       memory_type="episode")
        )  # fmt: skip
        await world.memory.get(
            GetRequest(app_id=APP, project_id=PROJECT, user_id="amy",
                       memory_type="profile")
        )  # fmt: skip
        await world.memory.aclose()
        rows = await _rows(world)
        assert len(rows) == 2
        assert {r.reinforcements for r in rows} == {1}


@pytest.mark.parametrize(
    "settings",
    [
        _curve(cascade_enabled=False),  # a read-only server
        _curve(reinforce=False),
    ],
    ids=["read-only", "reinforce-off"],
)
async def test_a_read_only_server_never_reinforces(
    tmp_path: Path, settings: Settings
) -> None:
    async with GroupWorld.open(tmp_path, settings=settings, decider=_keep(0)) as world:
        await _seed_two(world)
        await world.memory.start()
        first = await world.memory.search(_search(asked_at=T0))
        await world.memory.get(
            GetRequest(app_id=APP, project_id=PROJECT, user_id="amy",
                       memory_type="episode")
        )  # fmt: skip
        again = await world.memory.search(_search(asked_at=T0))
        await world.memory.aclose()
        assert await _rows(world) == []
        assert first.data.model_dump() == again.data.model_dump()


# ── ranking ───────────────────────────────────────────────────────────────────


async def test_the_curve_ranks_an_old_memory_below_a_fresh_one(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path, settings=_curve()) as world:
        old, new = await _seed_two(world)
        found = await world.memory.search(_search(asked_at=T0))
        ids = [e.id for e in found.data.episodes]
        assert ids.index(new) < ids.index(old)
        scores = {e.id: e.score for e in found.data.episodes}
        assert scores[old] < scores[new]


async def test_asked_at_is_the_time_ages_are_measured_to(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path, settings=_curve()) as world:
        old, new = await _seed_two(world)
        # Asked right after the old one happened, the other one is still to come
        # (its clock reads zero) and both keep their whole score.
        early = await world.memory.search(_search(asked_at=T0 - 400 * DAY_MS))
        late = await world.memory.search(_search(asked_at=T0 + 3650 * DAY_MS))
        early_scores = {e.id: e.score for e in early.data.episodes}
        late_scores = {e.id: e.score for e in late.data.episodes}
        assert late_scores[old] < early_scores[old]
        assert late_scores[new] < early_scores[new]


async def test_a_reinforced_memory_climbs_back(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path, settings=_curve(), decider=_keep()) as world:
        old, new = await _seed_two(world)
        await world.memory.start()
        # Read the old one by itself (a get of its session), then search later.
        session = (await world.memory.get(
            GetRequest(app_id=APP, project_id=PROJECT, user_id="amy",
                       memory_type="episode", filters={"session_id": "s_amy"},
                       page_size=1, sort_order="asc")
        )).data.episodes  # fmt: skip
        assert [e.id for e in session] == [old]
        await world.memory.aclose()
        found = await world.memory.search(_search())
        ids = [e.id for e in found.data.episodes]
        assert ids.index(old) < ids.index(new)


def test_asked_at_takes_epoch_ms_or_iso_and_refuses_anything_else() -> None:
    assert _search(asked_at=T0).asked_at_ms == T0
    assert _search(asked_at="2026-03-01T00:00:00Z").asked_at_ms == T0
    assert _search().asked_at_ms is None
    for bad in ("yesterday", True, "2026-13-01"):
        with pytest.raises(ValidationError, match="asked_at"):
            _search(asked_at=bad)


# ── asking about the past: one's own memory ───────────────────────────────────


async def test_the_past_brings_back_ones_superseded_and_forgotten_versions(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        edited = await world.personal_episode(
            "amy", "trip marker Lisbon hotel was the Alfama inn", at=T0
        )
        forgotten = await world.personal_episode(
            "amy", "trip marker Lisbon secret detour", at=T0 + 1
        )
        await world.project()
        update = await world.memory.update_memory(
            app_id=APP, project_id=PROJECT, user_id="amy", record_id=edited,
            changes=MemoryChanges(content="trip marker Lisbon hotel is the Baixa inn"),
        )  # fmt: skip
        await world.memory.forget(
            app_id=APP, project_id=PROJECT, user_id="amy",
            target=ForgetTarget(kind="record", record_ids=(forgotten,)),
            dry_run=False,
        )  # fmt: skip
        await world.project()

        present = await world.memory.search(_search())
        assert {e.id for e in present.data.episodes} == {update.id}
        assert "status" not in present.data.model_dump(mode="json")["episodes"][0]

        past = await world.memory.search(_search(include_past=True))
        by_id = {e.id: e for e in past.data.episodes}
        assert set(by_id) == {edited, update.id, forgotten}
        assert by_id[update.id].status == "current"
        assert by_id[edited].status == "superseded"
        assert by_id[edited].superseded_by == update.id
        assert by_id[edited].valid_until == to_display_tz(from_timestamp_ms(T0))
        assert by_id[forgotten].status == "forgotten"
        assert by_id[forgotten].superseded_by is None
        # The present version outranks what it replaced (state factor 0.3).
        assert by_id[update.id].score > by_id[edited].score


async def test_the_past_of_two_people_stays_each_ones_own(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        ann = await world.personal_episode("u_ann", "trip marker Lisbon ann", at=T0)
        bob = await world.personal_episode("u_bob", "trip marker Lisbon bob", at=T0)
        await world.project()
        for user, mine in (("u_ann", ann), ("u_bob", bob)):
            await world.memory.forget(
                app_id=APP, project_id=PROJECT, user_id=user,
                target=ForgetTarget(kind="record", record_ids=(mine,)),
                dry_run=False,
            )  # fmt: skip
        await world.project()
        for user, mine, other in (("u_ann", ann, bob), ("u_bob", bob, ann)):
            past = await world.memory.search(_search(user, include_past=True))
            ids = {e.id for e in past.data.episodes}
            assert ids == {mine}, user
            assert other not in json.dumps(past.data.model_dump(mode="json"))


# ── asking about the past: a group ────────────────────────────────────────────


async def _group(world: GroupWorld) -> dict[str, str]:
    await world.roster(
        "trip",
        [
            join("u_chen", T0, role="owner"),
            join("u_evan", T0 + 5_000),
        ],
    )
    early = await world.episode(
        "trip", "trip marker Lisbon early plan", start=T0 + 1_000, end=T0 + 2_000
    )
    late = await world.episode(
        "trip", "trip marker Lisbon late plan", start=T0 + 6_000, end=T0 + 7_000
    )
    gone = await world.episode(
        "trip", "trip marker Lisbon dropped plan", start=T0 + 8_000, end=T0 + 9_000
    )
    await world.project()
    edits: dict[str, str] = {}
    for record in (early, late):
        changes = MemoryChanges(content=f"trip marker Lisbon edited {record}")
        update = await world.memory.update_memory(
            app_id=APP, project_id=PROJECT, group_id="trip", asker_id="u_chen",
            record_id=record, changes=changes,
        )  # fmt: skip
        edits[record] = update.id
    await world.memory.forget_in_group(
        app_id=APP, project_id=PROJECT, group_id="trip", asker_id="u_chen",
        target=ForgetTarget(kind="record", record_ids=(gone,)),
    )  # fmt: skip
    await world.project()
    return {"early": early, "late": late, "gone": gone, **{
        f"{k}_edit": edits[v] for k, v in (("early", early), ("late", late))
    }}  # fmt: skip


def _group_search(asker: str, **fields: Any) -> SearchRequest:
    return SearchRequest(
        app_id=APP, project_id=PROJECT, group_id="trip", asker_id=asker,
        query="trip marker Lisbon", top_k=20, **fields,
    )  # fmt: skip


async def test_a_groups_past_is_its_superseded_versions_inside_the_window(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        ids = await _group(world)
        chen = await world.memory.search(_group_search("u_chen", include_past=True))
        seen = {e.id: e for e in chen.data.episodes}
        assert set(seen) == {
            ids["early"],
            ids["early_edit"],
            ids["late"],
            ids["late_edit"],
        }
        assert seen[ids["early"]].status == "superseded"
        assert seen[ids["early"]].superseded_by == ids["early_edit"]
        # The late joiner: the late versions only, never the early ones.
        evan = await world.memory.search(_group_search("u_evan", include_past=True))
        assert {e.id for e in evan.data.episodes} == {ids["late"], ids["late_edit"]}
        # A forgotten record of the group is in nobody's past.
        for asker in ("u_chen", "u_evan"):
            found = await world.memory.search(_group_search(asker, include_past=True))
            blob = json.dumps(found.data.model_dump(mode="json"))
            assert "dropped plan" not in blob, asker


async def test_include_groups_reads_the_groups_past_by_the_same_rules(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        ids = await _group(world)
        own = await world.personal_episode("u_evan", "trip marker Lisbon own", at=T0)
        await world.project()
        found = await world.memory.search(
            _search("u_evan", include_groups=True, include_past=True)
        )
        assert {e.id for e in found.data.episodes} == {
            own,
            ids["late"],
            ids["late_edit"],
        }
        blob = json.dumps(found.data.model_dump(mode="json"))
        assert "early plan" not in blob and "dropped plan" not in blob
