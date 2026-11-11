"""update_member_profiles: the member-in-group profiles of a group (task 5).

docs/group-memory.md, section 7.4. In-memory stores and a scripted profile
model. What these pin:

- every member who spoke gets a profile in the group; every item cites the
  segments it was read from, each of which holds a message by that member;
- the profile operator reads only the member's own segments of this group,
  with roster names as speaker labels, never another member's segments;
- ``[groups] member_profile_every`` holds updates back until that many new
  segments, and a flush of the group runs every pending update;
- two concurrent updates of one member lose no item (the per-member lock);
  updates of two members run in parallel;
- a replayed event does not count its segment twice;
- the operator is told a language, never left to choose one: the event's
  (the group pipeline's resolved language), a flush's resolved from the
  segments it reads, a stored ``primary_language`` over either;
- through everalgo's own merge: an UPDATE op that rewrites an item by its
  index makes the new item name the old one (``supersedes``, ``since``) and
  the old one's history entry name it back; a delete with an add, and a
  REGROUP rewrite, link nothing; the version keys never reach the model.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import re
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, MemCell
from everalgo.user_memory import ProfileExtractor

from everos.config import Settings
from everos.memory.events import GroupEpisodeExtracted, GroupFlushed
from everos.memory.mappers import profile as profile_mapper
from everos.memory.models import ExplicitInfoItem, Group, Profile, RosterEvent
from everos.memory.ports import Scope
from everos.memory.strategies import update_member_profiles
from tests.fakes import make_deps

_SCOPE = Scope(app_id="default", project_id="p", owner_id="trip", owner_type="group")


class _ProfileModel:
    """INIT: one item naming the target; UPDATE: adds one more item.

    ``gate`` (optional) holds every call until ``release`` is set, recording
    which targets are waiting -- how a test sees two calls in flight.
    """

    def __init__(self, gate: asyncio.Event | None = None) -> None:
        self.prompts: list[str] = []
        self.waiting: list[str] = []
        self.gate = gate
        self.counter = 0

    async def __call__(self, messages: Any, *, response_format: type, **_: Any) -> str:
        text = "\n".join(str(m.content) for m in messages)
        self.prompts.append(text)
        target = re.search(r"TARGET USER: ([^*\n]+)\*\*", text)
        who = target.group(1).strip() if target else "?"
        self.counter += 1
        if self.gate is not None:
            self.waiting.append(who)
            await self.gate.wait()
        item = {
            "evidence": f"{who} said so",
            "description": f"Fact {self.counter} about {who}",
            "category": "plans",
        }
        if response_format.__name__ == "_ProfileItems":
            return json.dumps({"explicit_info": [item], "implicit_traits": []})
        if response_format.__name__ == "_MemCellUpdateOutput":
            op = {"action": "add", "type": "explicit_info", "data": item}
            return json.dumps({"operations": [op], "update_note": "added"})
        raise AssertionError(response_format.__name__)


def _deps(model: Any) -> Any:
    deps = make_deps(llm=FakeLLMClient(handler=model))
    deps.stores.groups.groups[("default", "p", "trip")] = Group(
        group_id="trip", created_at=0
    )
    deps.stores.groups.events[("default", "p", "trip")].extend(
        [
            RosterEvent(op="join", member_id="u_chen", at=0, display_name="Chen"),
            RosterEvent(op="join", member_id="u_betty", at=0, display_name="Betty"),
        ]
    )
    return deps


def _segment(deps: Any, memcell_id: str, at: int, *lines: tuple[str, str]) -> None:
    items = [
        ChatMessage(
            id=f"{memcell_id}_{i}",
            role="user",
            sender_id=sender,
            sender_name=sender,
            content=text,
            timestamp=at + i,
        )
        for i, (sender, text) in enumerate(lines)
    ]
    deps.stores.memcell.seed(
        memcell_id, MemCell(items=items, timestamp=at + len(items))
    )


def _event(
    memcell_id: str, *speakers: str, language: str | None = None
) -> GroupEpisodeExtracted:
    return GroupEpisodeExtracted(
        memcell_id=memcell_id,
        episode_entry_id=f"gep_{memcell_id}",
        episode_text="text",
        episode_timestamp_ms=1,
        group_id="trip",
        app_id="default",
        project_id="p",
        speaker_ids=list(speakers),
        language=language,
    )


_FLUSHED = GroupFlushed(
    group_id="trip", app_id="default", project_id="p", session_id="s"
)


@pytest.fixture
def every(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Set ``[groups] member_profile_every`` for the strategy under test."""

    def set_to(n: int) -> None:
        base = Settings()
        settings = base.model_copy(
            update={
                "groups": base.groups.model_copy(update={"member_profile_every": n})
            }
        )
        # The package re-exports the function under the module's name.
        module = importlib.import_module(
            "everos.memory.strategies.update_member_profiles"
        )
        monkeypatch.setattr(module, "load_settings", lambda: settings)

    return set_to


async def test_each_speaker_gets_a_profile_from_their_own_segments(every: Any) -> None:
    every(1)
    model = _ProfileModel()
    deps = _deps(model)
    _segment(deps, "mc_1", 1_000, ("u_chen", "I'll book flights"), ("u_betty", "ok"))
    _segment(deps, "mc_2", 2_000, ("u_chen", "Bring passports"))

    await update_member_profiles(_event("mc_1", "u_chen", "u_betty"), deps)
    await update_member_profiles(_event("mc_2", "u_chen"), deps)

    store = deps.stores.member_profile
    assert await store.members(_SCOPE) == ["u_betty", "u_chen"]
    chen = await store.read(_SCOPE, "u_chen")
    betty = await store.read(_SCOPE, "u_betty")
    assert chen is not None and betty is not None
    assert [i.description for i in chen.profile.explicit_info] == [
        "Fact 1 about Chen",
        "Fact 3 about Chen",
    ]
    spoke = {"u_chen": {"mc_1", "mc_2"}, "u_betty": {"mc_1"}}
    for member, profile in (("u_chen", chen), ("u_betty", betty)):
        for item in profile.profile.explicit_info:
            sources = set((item.model_extra or {})["source_memcell_ids"])
            assert sources and sources <= spoke[member]
    # The operator read roster names, and Betty's update never saw mc_2.
    betty_prompt = next(p for p in model.prompts if "TARGET USER: Betty" in p)
    assert "Bring passports" not in betty_prompt
    assert "Betty(user_id:u_betty)" in betty_prompt
    assert chen.pending_memcell_ids == []


async def test_updates_wait_for_member_profile_every_and_run_at_flush(
    every: Any,
) -> None:
    every(3)
    model = _ProfileModel()
    deps = _deps(model)
    for n in (1, 2):
        _segment(deps, f"mc_{n}", n * 1_000, ("u_chen", f"note {n}"))
        await update_member_profiles(_event(f"mc_{n}", "u_chen"), deps)
    # A replay of the same segment counts once.
    await update_member_profiles(_event("mc_2", "u_chen"), deps)

    chen = await deps.stores.member_profile.read(_SCOPE, "u_chen")
    assert chen is not None and chen.pending_memcell_ids == ["mc_1", "mc_2"]
    assert chen.profile.explicit_info == [] and model.prompts == []

    await update_member_profiles(_FLUSHED, deps)
    chen = await deps.stores.member_profile.read(_SCOPE, "u_chen")
    assert chen is not None and chen.pending_memcell_ids == []
    assert len(chen.profile.explicit_info) == 1
    assert len(model.prompts) == 1

    for n in (3, 4, 5):
        _segment(deps, f"mc_{n}", n * 1_000, ("u_chen", f"note {n}"))
        await update_member_profiles(_event(f"mc_{n}", "u_chen"), deps)
    assert len(model.prompts) == 2  # the third new segment ran the update


async def test_two_concurrent_updates_of_one_member_lose_no_item(every: Any) -> None:
    every(1)
    deps = _deps(_ProfileModel())
    _segment(deps, "mc_1", 1_000, ("u_chen", "first"))
    _segment(deps, "mc_2", 2_000, ("u_chen", "second"))

    await asyncio.gather(
        update_member_profiles(_event("mc_1", "u_chen"), deps),
        update_member_profiles(_event("mc_2", "u_chen"), deps),
    )
    chen = await deps.stores.member_profile.read(_SCOPE, "u_chen")
    assert chen is not None
    assert len(chen.profile.explicit_info) == 2  # the second read the first


async def test_updates_of_two_members_run_in_parallel(every: Any) -> None:
    every(1)
    gate = asyncio.Event()
    model = _ProfileModel(gate)
    deps = _deps(model)
    _segment(deps, "mc_1", 1_000, ("u_chen", "hi"), ("u_betty", "hello"))

    run = asyncio.create_task(
        update_member_profiles(_event("mc_1", "u_chen", "u_betty"), deps)
    )
    for _ in range(200):
        if len(model.waiting) == 2:
            break
        await asyncio.sleep(0.01)
    assert sorted(model.waiting) == ["Betty", "Chen"]  # both calls in flight
    gate.set()
    await asyncio.wait_for(run, timeout=5)


async def test_an_update_is_told_the_event_s_language(every: Any) -> None:
    every(1)
    model = _ProfileModel()
    deps = _deps(model)
    _segment(deps, "mc_1", 1_000, ("u_chen", "I'll book the flights for all of us"))

    await update_member_profiles(_event("mc_1", "u_chen", language="English"), deps)

    (prompt,) = model.prompts
    assert "Write every output field in English" in prompt
    chen = await deps.stores.member_profile.read(_SCOPE, "u_chen")
    assert chen is not None and chen.profile.primary_language == "English"


async def test_a_flush_update_resolves_the_language_from_its_segments(
    every: Any,
) -> None:
    """A flush brings no language; the segments it reads decide, in code."""
    every(5)
    model = _ProfileModel()
    deps = _deps(model)
    _segment(deps, "mc_1", 1_000, ("u_chen", "我来订机票，大家记得带好护照和充电器。"))
    await update_member_profiles(_event("mc_1", "u_chen"), deps)
    assert model.prompts == []  # queued, below member_profile_every

    await update_member_profiles(_FLUSHED, deps)

    (prompt,) = model.prompts
    assert "Write every output field in Chinese" in prompt
    chen = await deps.stores.member_profile.read(_SCOPE, "u_chen")
    assert chen is not None and chen.profile.primary_language == "Chinese"


async def test_a_stored_language_wins(every: Any) -> None:
    every(1)
    model = _ProfileModel()
    deps = _deps(model)
    _segment(deps, "mc_1", 1_000, ("u_chen", "I'll book the flights for all of us"))
    await update_member_profiles(_event("mc_1", "u_chen", language="German"), deps)
    _segment(deps, "mc_2", 2_000, ("u_chen", "Bring your passports and chargers"))

    await update_member_profiles(_event("mc_2", "u_chen", language="English"), deps)

    assert "Write every output field in German" in model.prompts[-1]


def _fact(description: str, category: str = "residence") -> dict[str, str]:
    return {"evidence": "said so", "description": description, "category": category}


class _ScriptedModel:
    """INIT returns ``init``; each UPDATE returns the next ``updates`` reply."""

    def __init__(
        self,
        init: list[dict[str, str]],
        *updates: list[dict[str, Any]],
        regroup: list[dict[str, str]] | None = None,
    ) -> None:
        self.init = init
        self.updates = list(updates)
        self.regroup = regroup
        self.calls: list[str] = []

    def __call__(self, messages: Any, *, response_format: type, **_: Any) -> str:
        name = response_format.__name__
        self.calls.append(name)
        if name == "_ProfileItems":
            return json.dumps({"explicit_info": self.init, "implicit_traits": []})
        if name == "_MemCellUpdateOutput":
            ops = self.updates.pop(0)
            return json.dumps({"operations": ops, "update_note": "n"})
        if name == "_MemCellRegroupOutput" and self.regroup is not None:
            return json.dumps({"items": self.regroup, "regroup_note": "n"})
        raise AssertionError(name)


async def _two_updates(model: _ScriptedModel) -> Any:
    deps = _deps(model)
    _segment(deps, "mc_1", 1_000, ("u_chen", "I live in Paris"))
    await update_member_profiles(_event("mc_1", "u_chen"), deps)
    _segment(deps, "mc_2", 2_000, ("u_chen", "We moved to Berlin last week"))
    await update_member_profiles(_event("mc_2", "u_chen"), deps)
    chen = await deps.stores.member_profile.read(_SCOPE, "u_chen")
    assert chen is not None
    return chen


async def test_an_update_op_rewriting_a_fact_links_the_new_value(every: Any) -> None:
    every(1)
    rewrite = {
        "action": "update",
        "type": "explicit_info",
        "index": 0,
        "data": _fact("Lives in Berlin"),
    }
    model = _ScriptedModel(
        [_fact("Lives in Paris"), _fact("Cycles to work", "habits")], [rewrite]
    )

    chen = await _two_updates(model)

    berlin, bike = chen.profile.explicit_info
    (paris,) = chen.history
    new = berlin.model_extra or {}
    assert berlin.description == "Lives in Berlin"
    # since = the end of the update's span: mc_2's one message, at 2_000 + 1.
    assert (new["supersedes"], new["since"]) == (paris["item_id"], 2_001)
    assert new["source_memcell_ids"] == ["mc_2"]
    assert paris["description"] == "Lives in Paris"
    assert (paris["superseded_by"], paris["replaced_at"]) == (new["item_id"], 2_001)
    assert not {"supersedes", "since"} & set(bike.model_extra or {})


async def test_a_delete_with_an_add_and_a_regroup_link_nothing(every: Any) -> None:
    every(1)
    replace = [
        {"action": "delete", "type": "explicit_info", "index": 0, "reason": "moved"},
        {"action": "add", "type": "explicit_info", "data": _fact("Lives in Berlin")},
    ]
    chen = await _two_updates(_ScriptedModel([_fact("Lives in Paris")], replace))
    assert [i.description for i in chen.profile.explicit_info] == ["Lives in Berlin"]
    assert "supersedes" not in (chen.profile.explicit_info[0].model_extra or {})
    assert "superseded_by" not in chen.history[0]

    # Nine plans: the UPDATE rewrites one by its index and adds a tenth, so the
    # group passes its cap and REGROUP rewrites it without any host key.
    plans = [_fact(f"Plan {n}", "plans") for n in range(9)]
    rewrite_and_add = [
        {"action": "update", "type": "explicit_info", "index": 0,
         "data": _fact("Plan 0, moved to May", "plans")},
        {"action": "add", "type": "explicit_info", "data": _fact("Plan 9", "plans")},
    ]  # fmt: skip
    regrouped = [_fact("Plan 0, moved to May", "plans"), _fact("Plans 1-9", "plans")]
    model = _ScriptedModel(plans, rewrite_and_add, regroup=regrouped)
    chen = await _two_updates(model)
    assert "_MemCellRegroupOutput" in model.calls
    assert [i.description for i in chen.profile.explicit_info] == [
        "Plan 0, moved to May",
        "Plans 1-9",
    ]
    for item in chen.profile.explicit_info:
        assert not {"supersedes", "since"} & set(item.model_extra or {})
    assert len(chen.history) == 9
    assert all("superseded_by" not in entry for entry in chen.history)


async def test_the_version_keys_never_reach_the_model() -> None:
    """The UPDATE prompt for a profile with versions is the one without them."""

    def stored(**version: object) -> Profile:
        return Profile(
            owner_id="u_chen",
            summary="",
            timestamp=1_000,
            explicit_info=[
                ExplicitInfoItem(
                    description="Lives in Berlin",
                    category="residence",
                    evidence="said so",
                    observed_at=1_000,
                    item_id="pi_berlin",
                    source_memcell_ids=["mc_1"],
                    source_start=900,
                    source_end=1_000,
                    **version,
                )
            ],
        )

    prompts: list[str] = []

    def model(messages: Any, *, response_format: type, **_: Any) -> str:
        prompts.append("\n".join(str(m.content) for m in messages))
        return json.dumps({"operations": [{"action": "none"}], "update_note": "n"})

    cell = MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                sender_id="u_chen",
                sender_name="Chen",
                content="Still in Berlin",
                timestamp=2_000,
            )
        ],
        timestamp=2_000,
    )
    extractor = ProfileExtractor(llm=FakeLLMClient(handler=model))
    for profile in (stored(), stored(supersedes="pi_paris", since=1_000)):
        await extractor.aextract(
            [cell],
            sender_id="u_chen",
            old_profile=profile_mapper.to_algo(profile),
            output_language="English",
        )
    plain, versioned = prompts
    assert versioned == plain
    assert "supersedes" not in versioned and "pi_paris" not in versioned
