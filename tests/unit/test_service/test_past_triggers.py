"""Asking about the past by its wording, through the memory service.

docs/configuration.md (``[search] past_triggers``) and docs/api.md
(``include_past``, ``past_trigger``). Real markdown, ``system.db`` and
LanceDB (``GroupWorld``):

* off by default: a question full of past-time wording is answered as it was,
  with no ``past_trigger`` anywhere in the response;
* on, a question that holds a trigger and leaves ``include_past`` unset is
  answered exactly as the same question with ``include_past = true``, and the
  response names the trigger; a date trigger is read against ``asked_at``;
* an ``include_past`` the request sets, true or false, wins over any trigger;
* only the question as asked is read: a decider's sub-query that holds a
  trigger switches nothing;
* in a group, a triggered search reads the past through the same windows as
  an explicit one -- a late joiner gets no version from before they joined,
  nobody a forgotten record; the RL seam reads the trigger too;
* over HTTP, ``past_trigger`` is in the body only when a trigger fired.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from everalgo.testing import FakeLLMClient
from httpx import ASGITransport, AsyncClient

from everos.config import Settings
from everos.core.utils.datetime import from_iso_format, to_timestamp_ms
from everos.entrypoints.api.app import create_app
from everos.memory.search.llm_multiround import RoundDecision
from everos.service import ForgetTarget, MemoryChanges, SearchRequest
from tests.fakes import make_application
from tests.helpers.groups import APP, PROJECT, GroupWorld, join

T0 = 1_772_323_200_000
"""2026-03-01T00:00:00Z: every record time below is ``T0`` plus an offset."""

ASKED_2026 = to_timestamp_ms(from_iso_format("2026-11-13T00:00:00Z"))
ASKED_2025 = to_timestamp_ms(from_iso_format("2025-11-13T00:00:00Z"))

PLAIN = "trip marker Lisbon hotel"
PAST = "trip marker Lisbon hotel back then"
SUB_QUERY = "what hotel I used to book back then"


def _on() -> Settings:
    base = Settings()
    return base.model_copy(
        update={"search": base.search.model_copy(update={"past_triggers": True})}
    )


def _search(query: str, user: str = "amy", **fields: Any) -> SearchRequest:
    values: dict[str, Any] = {
        "app_id": APP,
        "project_id": PROJECT,
        "user_id": user,
        "query": query,
        "top_k": 20,
        "asked_at": ASKED_2026,
        **fields,
    }
    return SearchRequest(**values)


def _json(response: Any) -> dict[str, Any]:
    return response.data.model_dump(mode="json")


def _but_the_trigger(response: Any) -> dict[str, Any]:
    """The response as JSON, without ``past_trigger``."""
    return {k: v for k, v in _json(response).items() if k != "past_trigger"}


async def _seed(world: GroupWorld) -> dict[str, str]:
    """An edited episode of amy's and a forgotten one."""
    edited = await world.personal_episode(
        "amy", "trip marker Lisbon hotel was the Alfama inn", at=T0
    )
    forgotten = await world.personal_episode(
        "amy", "trip marker Lisbon hotel secret detour", at=T0 + 1
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
    return {"edited": edited, "forgotten": forgotten, "current": update.id}


# ── off by default ────────────────────────────────────────────────────────────


async def test_off_by_default_a_past_question_is_answered_as_before(
    tmp_path: Path,
) -> None:
    assert Settings().search.past_triggers is False
    async with GroupWorld.open(tmp_path) as world:
        ids = await _seed(world)
        found = await world.memory.search(_search(PAST))
        assert {e.id for e in found.data.episodes} == {ids["current"]}
        assert found.data.past_trigger is None
        dumped = _json(found)
        assert "past_trigger" not in dumped
        assert all("status" not in e for e in dumped["episodes"])


# ── on ────────────────────────────────────────────────────────────────────────


async def test_a_past_question_reads_the_past_as_include_past_does(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path, settings=_on()) as world:
        ids = await _seed(world)
        triggered = await world.memory.search(_search(PAST))
        explicit = await world.memory.search(_search(PAST, include_past=True))

        assert triggered.data.past_trigger == "back then"
        by_id = {e.id: e for e in triggered.data.episodes}
        assert set(by_id) == {ids["edited"], ids["forgotten"], ids["current"]}
        assert by_id[ids["edited"]].status == "superseded"
        assert by_id[ids["edited"]].superseded_by == ids["current"]
        assert by_id[ids["forgotten"]].status == "forgotten"
        # The same answer as asking with include_past, but for the trigger.
        assert explicit.data.past_trigger is None
        assert "past_trigger" not in _json(explicit)
        assert _but_the_trigger(triggered) == _json(explicit)


async def test_a_question_without_a_trigger_is_answered_as_before(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path / "on", settings=_on()) as world:
        ids = await _seed(world)
        found = await world.memory.search(_search(PLAIN))
        assert {e.id for e in found.data.episodes} == {ids["current"]}
        assert "past_trigger" not in _json(found)
    async with GroupWorld.open(tmp_path / "off") as world:
        await _seed(world)
        off = await world.memory.search(_search(PLAIN))
    assert [(e.episode, e.score) for e in off.data.episodes] == [
        (e.episode, e.score) for e in found.data.episodes
    ]


async def test_a_date_trigger_is_read_against_asked_at(tmp_path: Path) -> None:
    question = "trip marker Lisbon hotel in 2025"
    async with GroupWorld.open(tmp_path, settings=_on()) as world:
        ids = await _seed(world)
        late = await world.memory.search(_search(question, asked_at=ASKED_2026))
        early = await world.memory.search(_search(question, asked_at=ASKED_2025))
        assert late.data.past_trigger == "in 2025"
        assert ids["edited"] in {e.id for e in late.data.episodes}
        # Asked in 2025, the year has not ended: no trigger, no past.
        assert early.data.past_trigger is None
        assert {e.id for e in early.data.episodes} == {ids["current"]}


# ── an explicit include_past wins ─────────────────────────────────────────────


async def test_an_explicit_include_past_false_beats_a_trigger(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path / "on", settings=_on()) as world:
        ids = await _seed(world)
        found = await world.memory.search(_search(PAST, include_past=False))
        assert {e.id for e in found.data.episodes} == {ids["current"]}
        dumped = _json(found)
        assert "past_trigger" not in dumped
        assert all("status" not in e for e in dumped["episodes"])
    async with GroupWorld.open(tmp_path / "off") as world:
        await _seed(world)
        off = await world.memory.search(_search(PAST))
    assert [(e.episode, e.score) for e in off.data.episodes] == [
        (e.episode, e.score) for e in found.data.episodes
    ]


async def test_an_explicit_include_past_true_names_no_trigger(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path, settings=_on()) as world:
        ids = await _seed(world)
        for query in (PLAIN, PAST):
            found = await world.memory.search(_search(query, include_past=True))
            assert ids["edited"] in {e.id for e in found.data.episodes}, query
            assert "past_trigger" not in _json(found), query


# ── only the question as asked ────────────────────────────────────────────────


class _SubQueryDecider:
    """Round 0 asks a sub-query full of past-time wording; round 1 keeps the
    first candidate and stops. Keeps every prompt it was shown."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, messages: Any, **_kwargs: Any) -> str:
        self.prompts.append("\n".join(str(getattr(m, "content", "")) for m in messages))
        if len(self.prompts) == 1:
            return json.dumps({"core": [], "next_queries": [SUB_QUERY]})
        return json.dumps({"core": [0], "next_queries": []})


async def test_a_sub_query_with_a_trigger_switches_nothing(tmp_path: Path) -> None:
    decider = _SubQueryDecider()
    async with GroupWorld.open(
        tmp_path, settings=_on(), decider=FakeLLMClient(handler=decider)
    ) as world:
        ids = await _seed(world)
        found = await world.memory.search(_search(PLAIN))
    assert len(decider.prompts) == 2
    assert f"[block: {SUB_QUERY}]" in decider.prompts[1]
    assert {e.id for e in found.data.episodes} == {ids["current"]}
    assert "past_trigger" not in _json(found)
    # Nothing the past holds reached the decider either.
    assert all("Alfama" not in p and "detour" not in p for p in decider.prompts)


# ── groups ────────────────────────────────────────────────────────────────────


async def _group(world: GroupWorld) -> dict[str, str]:
    """A group whose early and late plans were edited and one forgotten;
    ``u_evan`` joined between the early and the late plan."""
    await world.roster(
        "trip",
        [join("u_chen", T0, role="owner"), join("u_evan", T0 + 5_000)],
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
    ids = {"early": early, "late": late, "gone": gone}
    for name, record in (("early", early), ("late", late)):
        update = await world.memory.update_memory(
            app_id=APP, project_id=PROJECT, group_id="trip", asker_id="u_chen",
            record_id=record,
            changes=MemoryChanges(content=f"trip marker Lisbon edited {name}"),
        )  # fmt: skip
        ids[f"{name}_edit"] = update.id
    await world.memory.forget_in_group(
        app_id=APP, project_id=PROJECT, group_id="trip", asker_id="u_chen",
        target=ForgetTarget(kind="record", record_ids=(gone,)),
    )  # fmt: skip
    await world.project()
    return ids


def _group_search(asker: str, query: str, **fields: Any) -> SearchRequest:
    return SearchRequest(
        app_id=APP, project_id=PROJECT, group_id="trip", asker_id=asker,
        query=query, top_k=20, **fields,
    )  # fmt: skip


async def test_a_groups_windows_are_unchanged_under_a_trigger(tmp_path: Path) -> None:
    question = "what was the trip marker Lisbon plan at first"
    async with GroupWorld.open(tmp_path, settings=_on()) as world:
        ids = await _group(world)
        seen: dict[str, set[str]] = {}
        for asker in ("u_chen", "u_evan"):
            triggered = await world.memory.search(_group_search(asker, question))
            explicit = await world.memory.search(
                _group_search(asker, question, include_past=True)
            )
            assert triggered.data.past_trigger == "at first", asker
            assert _but_the_trigger(triggered) == _json(explicit), asker
            blob = json.dumps(_json(triggered))
            assert "dropped plan" not in blob, asker
            seen[asker] = {e.id for e in triggered.data.episodes}
        assert seen["u_chen"] == {
            ids["early"],
            ids["early_edit"],
            ids["late"],
            ids["late_edit"],
        }
        # The late joiner: the late versions only, never the early ones.
        assert seen["u_evan"] == {ids["late"], ids["late_edit"]}


async def test_include_groups_reads_a_triggered_past_by_the_same_rules(
    tmp_path: Path,
) -> None:
    question = "trip marker Lisbon plan we used to have"
    async with GroupWorld.open(tmp_path, settings=_on()) as world:
        ids = await _group(world)
        own = await world.personal_episode("u_evan", "trip marker Lisbon own", at=T0)
        await world.project()
        found = await world.memory.search(
            _search(question, user="u_evan", include_groups=True)
        )
        assert found.data.past_trigger == "used to"
        assert {e.id for e in found.data.episodes} == {
            own,
            ids["late"],
            ids["late_edit"],
        }
        blob = json.dumps(_json(found))
        assert "early plan" not in blob and "dropped plan" not in blob


async def test_the_rl_seam_reads_the_trigger_too(tmp_path: Path) -> None:
    class _Keep:
        async def __call__(self, *_args: Any, **_kwargs: Any) -> RoundDecision:
            return RoundDecision(stop=True, queries=[], core=list(range(10)))

    async with GroupWorld.open(tmp_path, settings=_on()) as world:
        ids = await _seed(world)
        manager = world.memory._search_manager
        triggered = await manager.search_episodes_with_decider(_search(PAST), _Keep())
        plain = await manager.search_episodes_with_decider(_search(PLAIN), _Keep())
        refused = await manager.search_episodes_with_decider(
            _search(PAST, include_past=False), _Keep()
        )
    assert {e.id for e in triggered} == set(ids.values())
    assert {e.status for e in triggered} == {"current", "superseded", "forgotten"}
    assert [e.id for e in plain] == [ids["current"]]
    assert [e.id for e in refused] == [ids["current"]]


# ── the wire ──────────────────────────────────────────────────────────────────


async def test_the_body_names_the_trigger_only_when_it_fired(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path, settings=_on()) as world:
        await _seed(world)
        app = create_app(application=make_application(memory=world.memory))
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t"
        ) as client:
            bodies = {}
            for name, extra in (
                ("past", {"query": PAST}),
                ("plain", {"query": PLAIN}),
                ("refused", {"query": PAST, "include_past": False}),
            ):
                response = await client.post(
                    "/api/v2/memory/search",
                    json={
                        "app_id": APP,
                        "project_id": PROJECT,
                        "user_id": "amy",
                        "top_k": 20,
                        **extra,
                    },
                )
                assert response.status_code == 200, response.text
                bodies[name] = response.json()["data"]
    assert bodies["past"]["past_trigger"] == "back then"
    assert "past_trigger" not in bodies["plain"]
    assert "past_trigger" not in bodies["refused"]
    assert all("status" not in e for e in bodies["plain"]["episodes"])
