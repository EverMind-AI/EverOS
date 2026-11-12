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
- a replayed event does not count its segment twice, nor queue one an
  update already read (an item rests on it);
- the operator is told a language, never left to choose one: the event's
  (the group pipeline's resolved language), a flush's resolved from the
  segments it reads, a stored ``primary_language`` over either;
- through everalgo's lineage: a ``modify`` that gives a fact a new value
  makes the new item name the old one (``supersedes``, ``since``) and the old
  one's history entry name it back; a deprecation with an add links nothing; a
  consolidation keeps the link of an item it leaves as it is and links none of
  the items it merges (their history entries name what they merged into); the
  version keys never reach the model;
- the member's pictures in the segments are read with them, nobody else's,
  and never one the group forgot.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import json
import re
from typing import Any

import pytest
from everalgo.llm import ImageUrlPart
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, MemCell
from everalgo.user_memory import ProfileExtractor

from everos.component.images import ImageLimits
from everos.config import Settings
from everos.memory.events import GroupEpisodeExtracted, GroupFlushed
from everos.memory.mappers import profile as profile_mapper
from everos.memory.models import ExplicitInfoItem, Group, Profile, RosterEvent
from everos.memory.ports import Scope
from everos.memory.strategies import update_member_profiles
from tests.fakes import make_deps

_SCOPE = Scope(app_id="default", project_id="p", owner_id="trip", owner_type="group")


class _ProfileModel:
    """Every update adds one item naming the target.

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
        target = re.search(r"The target user is (.+?), the speaker", text)
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
        if response_format.__name__ == "_ProfileUpdateReply":
            op = {"action": "add", "type": "explicit_info", "data": item}
            return json.dumps({"operations": [op]})
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
    betty_prompt = next(p for p in model.prompts if "target user is Betty" in p)
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


async def test_a_segment_already_read_is_not_queued_again(every: Any) -> None:
    """A replay after the update read the segment: no second read, no item."""
    every(1)
    model = _ProfileModel()
    deps = _deps(model)
    _segment(deps, "mc_1", 1_000, ("u_chen", "I'll book flights"))
    await update_member_profiles(_event("mc_1", "u_chen"), deps)

    await update_member_profiles(_event("mc_1", "u_chen"), deps)
    every(3)
    await update_member_profiles(_event("mc_1", "u_chen"), deps)

    chen = await deps.stores.member_profile.read(_SCOPE, "u_chen")
    assert chen is not None and chen.pending_memcell_ids == []
    assert [i.description for i in chen.profile.explicit_info] == ["Fact 1 about Chen"]
    assert len(model.prompts) == 1


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
    """The first update adds ``first``; each later one returns the next ``updates``."""

    def __init__(
        self,
        first: list[dict[str, str]],
        *updates: list[dict[str, Any]],
        consolidation: dict[str, Any] | None = None,
    ) -> None:
        self.replies = [[_added(item) for item in first], *updates]
        self.consolidation = consolidation
        self.calls: list[str] = []

    def __call__(self, messages: Any, *, response_format: type, **_: Any) -> str:
        name = response_format.__name__
        self.calls.append(name)
        if name == "_ProfileUpdateReply":
            return json.dumps({"operations": self.replies.pop(0)})
        if name == "_ConsolidateReply" and self.consolidation is not None:
            return json.dumps(self.consolidation)
        raise AssertionError(name)


def _added(item: dict[str, str]) -> dict[str, Any]:
    return {"action": "add", "type": "explicit_info", "data": item, "sources": ["s1"]}


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
    rewrite = {"action": "modify", "ref": "e1", "data": _fact("Lives in Berlin")}
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


async def test_a_deprecation_with_an_add_links_nothing(every: Any) -> None:
    every(1)
    replace = [
        {"action": "deprecate", "ref": "e1", "reason": "moved"},
        {"action": "add", "type": "explicit_info", "data": _fact("Lives in Berlin")},
    ]
    chen = await _two_updates(_ScriptedModel([_fact("Lives in Paris")], replace))
    assert [i.description for i in chen.profile.explicit_info] == ["Lives in Berlin"]
    assert "supersedes" not in (chen.profile.explicit_info[0].model_extra or {})
    (paris,) = chen.history
    assert "superseded_by" not in paris
    assert (paris["reason"], paris["deprecated_by"]) == ("moved", ["mc_2"])


async def test_a_consolidation_keeps_a_link_and_merges_without_one(every: Any) -> None:
    every(1)
    # Eight plans fill their name; the update gives one a new value and adds a
    # ninth, so the name passes its cap and is consolidated in the same call.
    plans = [_fact(f"Plan {n}", "plans") for n in range(8)]
    may = _fact("Plan 0, moved to May", "plans")
    rewrite_and_add = [
        {"action": "modify", "ref": "e1", "data": may},
        {"action": "add", "type": "explicit_info", "data": _fact("Plan 8", "plans")},
    ]
    consolidation = {
        "items": [
            {"from": ["e1"], **_fact("Plan 0, moved to May", "plans")},
            {"from": [f"e{n}" for n in range(2, 10)], **_fact("Plans 1-8", "plans")},
        ],
        "dropped": [],
    }
    model = _ScriptedModel(plans, rewrite_and_add, consolidation=consolidation)
    chen = await _two_updates(model)
    assert "_ConsolidateReply" in model.calls
    may, merged = chen.profile.explicit_info
    assert [may.description, merged.description] == [
        "Plan 0, moved to May",
        "Plans 1-8",
    ]
    # The new value the update linked keeps its link through a re-filing ...
    assert (may.model_extra or {})["supersedes"]
    # ... and an item merged from others is no new value of any of them.
    assert not {"supersedes", "since"} & set(merged.model_extra or {})
    linked = [e for e in chen.history if "superseded_by" in e]
    assert [e["description"] for e in linked] == ["Plan 0"]
    into = [e for e in chen.history if "merged_into" in e]
    assert len(into) == 7  # Plans 1-7 as stored; Plan 8 was added by this update
    assert {tuple(e["merged_into"]) for e in into} == {
        ((merged.model_extra or {})["item_id"],)
    }


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
        return json.dumps({"operations": [{"action": "none"}]})

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


async def test_a_picture_the_group_forgot_is_left_out_of_the_update(
    every: Any,
) -> None:
    every(1)
    cat, dog = "c" * 64, "d" * 64
    sent: list[Any] = []

    async def model(messages: Any, *, response_format: type, **_: Any) -> str:
        sent.append(messages)
        return json.dumps({"operations": [{"action": "none"}]})

    limits = ImageLimits(
        max_bytes=1 << 20,
        max_per_message=4,
        max_request_bytes=1 << 22,
        max_per_request=8,
        formats=("jpeg",),
    )
    deps = dataclasses.replace(_deps(model), picture_limits=limits)
    deps.stores.images.images[cat] = (b"cat", "jpeg")
    deps.stores.images.images[dog] = (b"dog", "jpeg")
    _segment(
        deps,
        "mc_1",
        100,
        ("u_chen", f"Our cat [IMAGE: {cat}]"),
        ("u_chen", f"Our dog [IMAGE: {dog}]"),
    )
    await deps.stores.forget_log.record(
        target="image",
        target_id=cat,
        app_id="default",
        project_id="p",
        owner_id="trip",
        owner_type="group",
    )

    await update_member_profiles(_event("mc_1", "u_chen"), deps)

    (messages,) = sent
    content = messages[1].content
    text = "\n".join(p.text for p in content if not isinstance(p, ImageUrlPart))
    assert cat not in text and "Our cat" in text
    assert "Our dog [IMAGE: img-1]" in text
    assert sum(isinstance(p, ImageUrlPart) for p in content) == 1
