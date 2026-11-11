"""The constraint block of a group search (task 14).

docs/group-memory.md, section 8.4. Real markdown, system.db and LanceDB;
segments stored as boundary detection stores them; group-profile items
seeded as the update strategy writes them. What these pin:

- every quote is a verbatim part of the message it cites; a constraint
  whose item quotes something its message does not say, whose segment is
  gone, or whose message was forgotten is absent;
- statuses on a conflict fixture: a decision with an open dissent about it
  is ``disputed`` and lists the version it replaced; without the dissent it
  is ``current``; an asker who cannot see the newer version gets the old one
  as ``superseded``, never the newer one;
- a core record no item covers gets a metadata constraint quoting its
  best-matching message;
- the model layer runs only when ``[search] constraints = "model"``; its
  bindings are checked and a failing one is dropped, leaving the rules
  layer's constraint as it was; a binding that does not apply may cite
  nothing, one that cites anything is checked in full;
- a binding call that yields nothing usable leaves the rules layer's block
  and is counted -- the search does not fail;
- quotes match with curly or straight quotation marks, and what is returned
  is the message's own words; the binding prompt is pinned and shows each
  speaker's member id.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage

from everos.component.llm import LLMCallError
from everos.config import Settings
from everos.core.observability.metrics import get_metrics_registry
from everos.memory.models import ProfileSource
from everos.memory.search.constraints import CONSTRAINT_BIND
from everos.service import SearchRequest
from tests.helpers.groups import APP, PROJECT, GroupWorld, join, leave

T0 = 1_772_323_200_000
QUERY = "offsite flights Lisbon venue"

SEGMENTS = {
    "mc_old": ("m-1", "u_chen", "Offsite flights: we fly out on May 1.", 1_000),
    "mc_new": (
        "m-6",
        "u_chen",
        "Offsite flights: May 1 is sold out. Let's leave on April 30 instead.",
        6_000,
    ),
    "mc_dissent": (
        "m-7",
        "u_betty",
        "Offsite flights on April 30 are too early.",
        7_000,
    ),
    "mc_venue": ("m-3", "u_chen", "I booked the Lisbon venue for the offsite.", 3_000),
}


def _cites(memcell: str, quote: str) -> list[ProfileSource]:
    """The evidence an item of ``memcell`` cites: its message, quoting ``quote``."""
    mid, who, _text, at = SEGMENTS[memcell]
    return [
        ProfileSource(
            memcell_id=memcell, message_id=mid, speaker_id=who, at=T0 + at, quote=quote
        )
    ]


class _Model:
    """Decider-free LLM: answers the binding call, records it.

    ``fail`` makes every call fail as the provider fails one whose replies
    never validate.
    """

    def __init__(
        self, bindings: list[dict[str, Any]] | None = None, *, fail: bool = False
    ) -> None:
        self.bindings = bindings or []
        self.fail = fail
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, messages: Any, **kwargs: Any) -> str:
        fmt = kwargs.get("response_format")
        assert fmt is not None and fmt.__name__ == "BindReply", fmt
        self.calls += 1
        self.prompts.append("\n".join(str(m.content) for m in messages))
        if self.fail:
            raise LLMCallError("no usable reply", reason="invalid_reply")
        return json.dumps({"bindings": self.bindings})


async def _world(
    tmp_path: Path, model: _Model, *, layer: str = "rules", dissent: bool = True
) -> AsyncIterator[GroupWorld]:
    base = Settings()
    settings = base.model_copy(
        update={"search": base.search.model_copy(update={"constraints": layer})}
    )
    async with GroupWorld.open(
        tmp_path, llm=FakeLLMClient(handler=model), settings=settings
    ) as world:
        await world.roster(
            "trip",
            [
                join("u_chen", T0, display_name="Chen", role="owner"),
                join("u_betty", T0, display_name="Betty"),
                join("u_dan", T0, display_name="Dan"),
                leave("u_dan", T0 + 2_500),
                join("u_dan", T0 + 7_500),
            ],
        )
        for memcell, (mid, who, text, at) in SEGMENTS.items():
            await world.segment(memcell, (mid, who, text, T0 + at))
            await world.episode(
                "trip",
                text,
                start=T0 + at,
                end=T0 + at,
                senders=(who,),
                memcell_id=memcell,
            )
        old = await world.profile_item(
            "trip",
            "we fly out on May 1",
            start=T0 + 1_000,
            end=T0 + 1_000,
            memcell_id="mc_old",
            statement="Fly out on May 1.",
            sources=_cites("mc_old", "we fly out on May 1"),
        )
        new = await world.profile_item(
            "trip",
            "Let's leave on April 30 instead",
            start=T0 + 6_000,
            end=T0 + 6_000,
            memcell_id="mc_new",
            statement="Depart on April 30 instead of May 1.",
            supersedes=old.entry_id,
            sources=_cites("mc_new", "Let's leave on April 30 instead"),
        )
        if dissent:
            await world.profile_item(
                "trip",
                "on April 30 are too early",
                kind="dissent",
                speaker="u_betty",
                start=T0 + 7_000,
                end=T0 + 7_000,
                memcell_id="mc_dissent",
                statement="Betty finds April 30 too early.",
                held_by=["u_betty"],
                about=new.entry_id,
                sources=_cites("mc_dissent", "on April 30 are too early"),
            )
        await world.project()
        yield world


async def _constraints(world: GroupWorld, asker: str) -> list[Any]:
    response = await world.memory.search(
        SearchRequest(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id=asker,
            query=QUERY,
            method="keyword",
            top_k=10,
        )
    )
    return response.data.constraints


async def _assert_verbatim(world: GroupWorld, constraints: list[Any]) -> None:
    cells = await world.stores.memcell.cells_by_id(
        [c.source.memcell_id for c in constraints]
    )
    for c in constraints:
        (message,) = [
            m
            for m in cells[c.source.memcell_id].items
            if isinstance(m, ChatMessage) and m.id == c.source.message_id
        ]
        assert c.source.quote and c.source.quote in " ".join(message.content.split())
        assert c.source.speaker_id == message.sender_id


@pytest.fixture
async def rules(tmp_path: Path) -> AsyncIterator[tuple[GroupWorld, _Model]]:
    model = _Model()
    async for world in _world(tmp_path, model):
        yield world, model


async def test_a_disputed_decision_and_its_history(
    rules: tuple[GroupWorld, _Model],
) -> None:
    world, model = rules
    constraints = await _constraints(world, "u_chen")

    await _assert_verbatim(world, constraints)
    by_record = {c.statement: c for c in constraints}
    decision = by_record["Depart on April 30 instead of May 1."]
    assert decision.status == "disputed" and decision.kind == "decision"
    assert [s.statement for s in decision.supersedes] == ["Fly out on May 1."]
    assert decision.source.quote == "Let's leave on April 30 instead"
    assert by_record["Betty finds April 30 too early."].status == "open"
    assert "Fly out on May 1." not in by_record  # it is history, not a constraint
    venue = next(c for c in constraints if c.kind == "episode")
    assert venue.source.memcell_id == "mc_venue"
    assert venue.source.quote == "I booked the Lisbon venue for the offsite."
    assert [c.id for c in constraints] == [
        f"c{n}" for n in range(1, len(constraints) + 1)
    ]
    assert model.calls == 0  # the rules layer makes no call


async def test_an_asker_who_cannot_see_the_newer_version(
    rules: tuple[GroupWorld, _Model],
) -> None:
    world, _ = rules
    constraints = await _constraints(world, "u_dan")  # away for both later items

    statements = {c.statement: c.status for c in constraints}
    assert statements.get("Fly out on May 1.") == "superseded"
    assert "Depart on April 30 instead of May 1." not in statements
    assert "Betty finds April 30 too early." not in statements
    blob = json.dumps([c.model_dump(mode="json") for c in constraints])
    assert "April 30" not in blob


async def test_without_dissent_the_decision_is_current(tmp_path: Path) -> None:
    async for world in _world(tmp_path, _Model(), dissent=False):
        constraints = await _constraints(world, "u_chen")
        decision = next(c for c in constraints if c.kind == "decision")
        assert decision.status == "current"


async def test_a_constraint_that_fails_its_check_is_absent(
    rules: tuple[GroupWorld, _Model],
) -> None:
    world, _ = rules
    # An item whose quote its message never said, and one whose segment is gone.
    await world.profile_item(
        "trip",
        "we fly out on June 9",
        start=T0 + 3_000,
        end=T0 + 3_000,
        memcell_id="mc_venue",
        statement="Fly out on June 9.",
        sources=_cites("mc_venue", "we fly out on June 9"),
    )
    await world.profile_item(
        "trip",
        "a lost segment",
        start=T0 + 3_000,
        end=T0 + 3_000,
        memcell_id="mc_gone",
        statement="From a lost segment.",
    )
    await world.project()
    # The dissent's message is forgotten: nothing may quote it any more.
    await world.stores.message_forget.record(
        world.session("s_group"), ["m-7"], forget_id="fg_test"
    )

    constraints = await _constraints(world, "u_chen")

    statements = {c.statement for c in constraints}
    assert "Fly out on June 9." not in statements
    assert "From a lost segment." not in statements
    assert "Betty finds April 30 too early." not in statements
    assert all(c.source.message_id != "m-7" for c in constraints)
    await _assert_verbatim(world, constraints)


async def test_the_model_layer_runs_only_when_configured(tmp_path: Path) -> None:
    bindings = [
        {
            "constraint": "c1",
            "applies": True,
            "applies_to": ["u_chen"],
            "segment": "mc_new",
            "message": "m-6",
            "quote": "May 1 is sold out.",
        },
        # A paraphrase: dropped, so c2 keeps its rules-layer form.
        {
            "constraint": "c2",
            "applies": False,
            "applies_to": "group",
            "segment": "mc_dissent",
            "message": "m-7",
            "quote": "April 30 is far too early",
        },
    ]
    model = _Model(bindings)
    async for world in _world(tmp_path, model, layer="model"):
        constraints = await _constraints(world, "u_chen")

        assert model.calls == 1
        first, second = constraints[0], constraints[1]
        assert first.applies is True and first.scope.applies_to == ["u_chen"]
        assert first.source.quote == "May 1 is sold out."
        assert second.applies is None and second.scope.applies_to == "group"
        await _assert_verbatim(world, constraints)


def test_the_binding_prompt_is_registered_and_pinned() -> None:
    assert CONSTRAINT_BIND.id == "search.constraint_bind"
    assert CONSTRAINT_BIND.version == 2
    assert CONSTRAINT_BIND.fingerprint == "58044af1423a7cdf"


async def test_a_binding_that_does_not_apply_needs_no_citation(tmp_path: Path) -> None:
    bindings = [
        # Beside the point, nothing cited, no scope: bound, its source and
        # scope kept.
        {"constraint": "c1", "applies": False, "applies_to": ""},
        # A quote without a message: a citation, checked in full, dropped.
        {
            "constraint": "c2",
            "applies": False,
            "applies_to": "group",
            "segment": "",
            "message": "",
            "quote": "Offsite flights on April 30 are too early.",
        },
        # One member id as a string, and a curly apostrophe where the message
        # has a straight one: bound to that member, with the message's words.
        {
            "constraint": "c3",
            "applies": True,
            "applies_to": "u_chen",
            "segment": "mc_new",
            "message": "m-6",
            "quote": "Let’s leave on April 30 instead.",
        },
        # It applies, but to no one: dropped.
        {
            "constraint": "c4",
            "applies": True,
            "applies_to": [],
            "segment": "mc_old",
            "message": "m-1",
            "quote": "we fly out on May 1",
        },
    ]
    model = _Model(bindings)
    async for world in _world(tmp_path, model, layer="model"):
        first, second, third, fourth = await _constraints(world, "u_chen")

        assert first.applies is False and first.scope.applies_to == "group"
        assert first.source.message_id == "m-6"
        assert first.source.quote == "Let's leave on April 30 instead"
        assert second.applies is None
        assert third.applies is True and third.scope.applies_to == ["u_chen"]
        assert third.source.message_id == "m-6"
        assert third.source.quote == "Let's leave on April 30 instead."
        assert fourth.applies is None
        assert "Chen (u_chen): Offsite flights" in model.prompts[0]
        await _assert_verbatim(world, [first, second, third, fourth])


def _bind_failures(reason: str) -> float:
    value = get_metrics_registry().get_sample_value(
        "everos_constraint_bind_failures_total", {"reason": reason}
    )
    return value or 0.0


async def test_a_failed_binding_call_leaves_the_rules_block(tmp_path: Path) -> None:
    before = _bind_failures("invalid_reply")
    async for world in _world(tmp_path, _Model()):
        expected = await _constraints(world, "u_chen")
    model = _Model(fail=True)
    async for world in _world(tmp_path / "model", model, layer="model"):
        constraints = await _constraints(world, "u_chen")

        assert model.calls == 1
        assert [c.model_dump(exclude={"audience"}) for c in constraints] == [
            c.model_dump(exclude={"audience"}) for c in expected
        ]
        assert all(c.applies is None for c in constraints)
    assert _bind_failures("invalid_reply") - before == 1


async def test_a_stored_quote_matches_with_curly_marks(
    rules: tuple[GroupWorld, _Model],
) -> None:
    world, _ = rules
    await world.profile_item(
        "trip",
        "Let’s leave on April 30 instead",
        start=T0 + 6_000,
        end=T0 + 6_000,
        memcell_id="mc_new",
        statement="Leave on April 30.",
        sources=_cites("mc_new", "Let’s leave on April 30 instead"),
    )
    await world.project()

    constraints = await _constraints(world, "u_chen")

    (item,) = [c for c in constraints if c.statement == "Leave on April 30."]
    assert item.source.quote == "Let's leave on April 30 instead"
    await _assert_verbatim(world, constraints)
