"""Relational expansion in a group's multi-round search (task 13).

docs/group-memory.md, section 8.3. Real markdown, system.db and LanceDB; a
scripted decider that cores the line carrying a marker and asks one
follow-up, so the second round shows what expansion added. What these pin:

- the query matches only an old decision's wording; the decider cores that
  episode; the newest version of the decision -- by its source segment's
  episode -- enters the next round, in the block of the sub-query that
  surfaced its anchor;
- expansion never adds a record the asker cannot see: a member who left
  before the new decision never gets it;
- at most ``[search] link_slots`` linked records join a round, and with
  ``[search] expansion = false`` none do;
- the decider's input format does not change: a linked record's line is the
  line the same record gets when recalled, and round 0 is identical with
  expansion on and off.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient

from everos.config import Settings, load_settings
from everos.service import SearchRequest
from tests.helpers.groups import APP, PROJECT, GroupWorld, join, leave

T0 = 1_772_323_200_000
OLD = "OLDMARK the group flies out on May 1 from Lisbon"
NEW = "NEWMARK departure moved to the thirtieth of April"
OTHER = "OTHERMARK Betty brings the projector for the slides"


FOLLOW_UP = "Betty projector slides"
"""Round 1's sub-query: it recalls ``OTHER``, never the new decision."""


class _Decider:
    """Round 0: core the line holding ``OLDMARK`` and ask one follow-up.

    Every later round: stop. Keeps every round's prompt.
    """

    def __init__(self, target: str = "OLDMARK") -> None:
        self.target = target
        self.prompts: list[str] = []

    def __call__(self, messages: Any, **_: Any) -> str:
        # The round's own part follows the last "QUESTION:" line; the worked
        # examples before it use the same block grammar and are left out.
        text = "\n".join(str(m.content) for m in messages)
        prompt = text[text.rindex("QUESTION:\n") :]
        self.prompts.append(prompt)
        if len(self.prompts) > 1:
            return json.dumps({"core": [], "next_queries": []})
        index = _index_of(prompt, self.target)
        return json.dumps(
            {"core": [] if index is None else [index], "next_queries": [FOLLOW_UP]}
        )


def _index_of(prompt: str, marker: str) -> int | None:
    for line in prompt.splitlines():
        found = re.match(r"^  (\d+): ", line)
        if found and marker in line:
            return int(found.group(1))
    return None


def _block(prompt: str, label: str) -> list[str]:
    """The candidate lines under ``[block: <label>]``."""
    lines = prompt.splitlines()
    out: list[str] = []
    inside = False
    for line in lines:
        if line.startswith("[block: "):
            inside = line == f"[block: {label}]"
            continue
        if inside and re.match(r"^  \d+: ", line):
            out.append(line)
        elif inside:
            inside = False
    return out


@pytest.fixture
def make_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A world per call; one block per sub-query, of one candidate each.

    ``seed_topk`` / ``subq_topk`` of 1 keep each block to its best match, so
    the only way a record no query matches reaches a round is expansion.
    """
    monkeypatch.setenv("EVEROS_DECIDER__SEED_TOPK", "1")
    monkeypatch.setenv("EVEROS_DECIDER__SUBQ_TOPK", "1")
    load_settings.cache_clear()
    built = 0

    async def build(
        decider: _Decider, *, link_slots: int = 5, expansion: bool = True
    ) -> AsyncIterator[GroupWorld]:
        nonlocal built
        built += 1
        base = Settings()
        settings = base.model_copy(
            update={
                "search": base.search.model_copy(
                    update={"link_slots": link_slots, "expansion": expansion}
                )
            }
        )
        async with GroupWorld.open(
            tmp_path / f"world_{built}",
            decider=FakeLLMClient(handler=decider),
            settings=settings,
        ) as world:
            await world.roster(
                "trip",
                [
                    join("u_chen", T0, display_name="Chen", role="owner"),
                    join("u_betty", T0, display_name="Betty"),
                    # Dan was away when the new decision was made.
                    join("u_dan", T0, display_name="Dan"),
                    leave("u_dan", T0 + 2_500),
                    join("u_dan", T0 + 7_500),
                ],
            )
            for text, start, memcell in (
                (OLD, 1_000, "mc_old"),
                (OTHER, 2_000, "mc_other"),
                (NEW, 6_000, "mc_new"),
            ):
                await world.segment(memcell, (f"m-{start}", "u_chen", text, T0 + start))
                await world.episode(
                    "trip",
                    text,
                    start=T0 + start,
                    end=T0 + start + 100,
                    memcell_id=memcell,
                )
            old = await world.profile_item(
                "trip",
                OLD,
                start=T0 + 1_000,
                end=T0 + 1_000,
                memcell_id="mc_old",
            )
            await world.profile_item(
                "trip",
                NEW,
                start=T0 + 6_000,
                end=T0 + 6_000,
                memcell_id="mc_new",
                supersedes=old.entry_id,
            )
            await world.project()
            yield world

    return build


def _search(asker: str) -> SearchRequest:
    return SearchRequest(
        app_id=APP,
        project_id=PROJECT,
        group_id="trip",
        asker_id=asker,
        query="When do we fly out on May 1?",
        method="llm_multiround",
        top_k=10,
    )


async def _run(make_world: Any, asker: str, **world: Any) -> _Decider:
    decider = _Decider()
    async for w in make_world(decider, **world):
        await w.memory.search(_search(asker))
    return decider


async def test_the_newest_version_enters_the_next_round(make_world: Any) -> None:
    decider = await _run(make_world, "u_chen")

    first, second = decider.prompts
    assert "NEWMARK" not in first  # the query alone never reached it
    linked = _block(second, "original question")
    assert any("NEWMARK" in line for line in linked)


async def test_expansion_never_adds_what_the_asker_cannot_see(
    make_world: Any,
) -> None:
    decider = await _run(make_world, "u_dan")  # away when it was decided

    first, second = decider.prompts
    assert "OLDMARK" in first
    assert "NEWMARK" not in second


async def test_the_slot_cap_holds_and_the_switch_turns_it_off(
    make_world: Any,
) -> None:
    capped = await _run(make_world, "u_chen", link_slots=1)
    off = await _run(make_world, "u_chen", expansion=False)

    assert len(_block(capped.prompts[1], "original question")) == 1
    assert _block(off.prompts[1], "original question") == []
    assert "NEWMARK" not in off.prompts[1]


async def test_the_decider_reads_a_linked_record_as_any_other(
    make_world: Any,
) -> None:
    on = await _run(make_world, "u_chen")
    off = await _run(make_world, "u_chen", expansion=False)
    direct = _Decider()
    async for w in make_world(direct):
        await w.memory.search(_search("u_chen").model_copy(update={"query": NEW}))

    assert on.prompts[0] == off.prompts[0]  # round 0 is the same with or without
    linked = _candidate_lines(on.prompts[1], "NEWMARK")
    recalled = _candidate_lines(direct.prompts[0], "NEWMARK")
    assert len(linked) == len(recalled) == 1
    strip = re.compile(r"^  \d+: ")
    assert strip.sub("", linked[0]) == strip.sub("", recalled[0])
    # Every evidence line keeps the block grammar: a header or a numbered line.
    lines = on.prompts[1].splitlines()
    first = next(n for n, line in enumerate(lines) if line.startswith("[block: "))
    for line in lines[first:]:
        if not line.strip():
            break
        assert line.startswith("[block: ") or re.match(r"^  \d+: .+ - ", line), line


def _candidate_lines(prompt: str, marker: str) -> list[str]:
    return [
        line
        for line in prompt.splitlines()
        if marker in line and re.match(r"^  \d+: ", line)
    ]
