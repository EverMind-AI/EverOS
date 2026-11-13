"""update_group_profile: the group profile's batches and versions (task 11).

docs/group-memory.md, section 7.5. In-memory stores and a scripted model.
What these pin:

- the prompt is registered (``chat.group_profile_update``, version and
  fingerprint pinned) and names the conversation's language, resolved in
  code; it shows the segments under their markers, one ``[id] time speaker
  (member id): text`` line per message, and the current items with their ids;
- a reply in loose form -- keys of unused fields left out, one id where a list
  is expected -- is read, and the verifier still judges what it says;
- a call that yields nothing usable is not raised: it is counted, the batch
  stays queued and no trigger tries it again before its back-off has passed
  (the idle tick fired on every scan used to call again each time), and the
  third failure in a row drops the batch;
- a batch runs after ``[groups] group_profile_batch`` segments, at a flush,
  and on the idle tick; a replayed segment counts once; nothing runs when the
  switch is off or nothing is queued;
- superseding a decision appends a new item and marks the old one
  superseded; a dissent moves from open to resolved; a topic update keeps
  every earlier version stored -- nothing is ever deleted;
- a decision stored from the reply that creates its topic links to it, and
  the next call shows it under that topic; a link that names nothing is
  dropped and counted, the item stored;
- a dropped operation writes nothing.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, MemCell

from everos.component.llm import LLMCallError
from everos.config import load_settings
from everos.core.observability.metrics import get_metrics_registry
from everos.core.utils.datetime import from_timestamp_ms, to_timestamp_ms
from everos.memory.events import EpisodeExtracted, GroupFlushed
from everos.memory.extract.group_profile import GROUP_PROFILE_UPDATE
from everos.memory.models import Group, RosterEvent
from everos.memory.ports import DeferredFailure, IdleTick, PendingBatch, Scope
from everos.memory.strategies import (
    UPDATE_GROUP_PROFILE_IDLE_SPEC,
    update_group_profile,
)
from tests.fakes import make_deps

_SCOPE = Scope(app_id="default", project_id="p", owner_id="trip", owner_type="group")

_BLANK: dict[str, Any] = {
    "target": "",
    "topic_id": "",
    "name": "",
    "summary": "",
    "status": "",
    "statement": "",
    "decided_by": [],
    "agreed_by": [],
    "assignees": [],
    "applies_to": [],
    "held_by": [],
    "about": "",
    "member_id": "",
    "role": "",
    "confidence": "",
}


def op(
    kind: str, evidence: Sequence[tuple[str, str, str]], /, **fields: Any
) -> dict[str, Any]:
    return {
        **_BLANK,
        **fields,
        "op": kind,
        "evidence": [{"segment": s, "message": m, "quote": q} for s, m, q in evidence],
    }


class _Model:
    """Replies with the queued operation lists, one per call; records prompts."""

    def __init__(self, *replies: list[dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def __call__(self, messages: Any, *, response_format: type, **_: Any) -> str:
        assert response_format.__name__ == "GroupProfileReply"
        self.prompts.append("\n".join(str(m.content) for m in messages))
        operations = self.replies.pop(0) if self.replies else []
        return json.dumps({"operations": operations})


@pytest.fixture
def groups(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Set ``[groups]`` values for the strategy under test."""

    def set_to(**values: Any) -> None:
        for key, value in values.items():
            monkeypatch.setenv(f"EVEROS_GROUPS__{key.upper()}", str(value).lower())
        load_settings.cache_clear()

    return set_to


def _deps(model: _Model) -> Any:
    deps = make_deps(llm=FakeLLMClient(handler=model))
    key = ("default", "p", "trip")
    deps.stores.groups.groups[key] = Group(group_id="trip", created_at=0)
    deps.stores.groups.events[key].extend(
        [
            RosterEvent(op="join", member_id="u_chen", at=0, display_name="Chen"),
            RosterEvent(op="join", member_id="u_betty", at=0, display_name="Betty"),
            RosterEvent(
                op="join",
                member_id="bot_ava",
                at=0,
                display_name="Ava",
                kind="assistant",
            ),
        ]
    )
    return deps


def _segment(deps: Any, memcell_id: str, at: int, *lines: tuple[str, str, str]) -> None:
    items = [
        ChatMessage(
            id=mid,
            role="assistant" if sender.startswith("bot") else "user",
            sender_id=sender,
            sender_name=sender,
            content=text,
            timestamp=at + i,
        )
        for i, (mid, sender, text) in enumerate(lines)
    ]
    deps.stores.memcell.seed(
        memcell_id, MemCell(items=items, timestamp=at + len(items))
    )


def _extracted(memcell_id: str) -> EpisodeExtracted:
    return EpisodeExtracted(
        memcell_id=memcell_id,
        episode_entry_id=f"gep_{memcell_id}",
        episode_text="text",
        episode_timestamp_ms=1,
        owner_id="trip",
        owner_type="group",
        app_id="default",
        project_id="p",
        speaker_ids=["u_chen"],
    )


_FLUSHED = GroupFlushed(
    group_id="trip", app_id="default", project_id="p", session_id="s"
)


async def _items(deps: Any) -> list[Any]:
    return [s.item for s in await deps.stores.group_profile.items(_SCOPE)]


def test_the_prompt_is_registered_and_pinned() -> None:
    assert GROUP_PROFILE_UPDATE.id == "chat.group_profile_update"
    assert GROUP_PROFILE_UPDATE.version == 3
    assert GROUP_PROFILE_UPDATE.fingerprint == "4e4f531149128e69"


async def test_a_batch_runs_at_the_batch_size_once_per_segment(groups: Any) -> None:
    groups(group_profile_batch=3)
    model = _Model()
    deps = _deps(model)
    for n in (1, 2, 3):
        _segment(deps, f"mc_{n}", n * 1_000, (f"m-{n}", "u_chen", f"note {n}"))
    await update_group_profile(_extracted("mc_1"), deps)
    await update_group_profile(_extracted("mc_2"), deps)
    await update_group_profile(_extracted("mc_2"), deps)  # a replay counts once
    assert model.prompts == []
    assert await deps.stores.group_profile.pending(_SCOPE) == ["mc_1", "mc_2"]

    await update_group_profile(_extracted("mc_3"), deps)

    (prompt,) = model.prompts
    for n in (1, 2, 3):
        assert f"=== SEGMENT mc_{n} ===" in prompt
    assert "[m-1] 1970-01-01 00:00 Chen (u_chen): note 1" in prompt
    assert await deps.stores.group_profile.pending(_SCOPE) == []


async def test_a_flush_and_the_idle_tick_run_what_is_queued(groups: Any) -> None:
    groups(group_profile_batch=5)
    model = _Model()
    deps = _deps(model)
    _segment(deps, "mc_1", 1_000, ("m-1", "u_chen", "note"))
    _segment(deps, "mc_2", 2_000, ("m-2", "u_chen", "note"))

    await update_group_profile(_FLUSHED, deps)  # nothing queued: no call
    await update_group_profile(_extracted("mc_1"), deps)
    await update_group_profile(_FLUSHED, deps)
    assert len(model.prompts) == 1

    await update_group_profile(_extracted("mc_2"), deps)
    tick = IdleTick(
        strategy_name=UPDATE_GROUP_PROFILE_IDLE_SPEC.name,
        bucket_key=_extracted("mc_2").group_key,
        idle_seconds=600,
    )
    await update_group_profile(tick, deps)
    assert len(model.prompts) == 2 and "SEGMENT mc_2" in model.prompts[1]
    await update_group_profile(tick, deps)  # still idle, nothing queued
    assert len(model.prompts) == 2


async def test_the_switch_turns_it_off(groups: Any) -> None:
    groups(group_profile="false", group_profile_batch=1)
    model = _Model()
    deps = _deps(model)
    _segment(deps, "mc_1", 1_000, ("m-1", "u_chen", "note"))

    await update_group_profile(_extracted("mc_1"), deps)
    await update_group_profile(_FLUSHED, deps)

    assert model.prompts == []
    assert await deps.stores.group_profile.pending(_SCOPE) == []


async def test_the_prompt_names_the_language_it_resolved(groups: Any) -> None:
    groups(group_profile_batch=1)
    model = _Model()
    deps = _deps(model)
    _segment(
        deps, "mc_1", 1_000, ("m-1", "u_chen", "我来订机票，大家记得带好护照和充电器。")
    )

    await update_group_profile(_extracted("mc_1"), deps)

    (prompt,) = model.prompts
    assert "Write every output field in Chinese" in prompt


async def test_versions_supersede_and_nothing_is_deleted(groups: Any) -> None:
    groups(group_profile_batch=1)
    first = [
        op(
            "topic.new",
            [("mc_1", "m-1", "Let's fly out on May 1")],
            name="Travel",
            summary="How the group gets there",
            status="exploring",
        ),
        op(
            "decision.new",
            [("mc_1", "m-1", "Let's fly out on May 1"), ("mc_1", "m-2", "OK")],
            statement="Fly out on May 1.",
            decided_by=["u_chen"],
            agreed_by=["u_betty"],
        ),
        op(
            "dissent.open",
            [("mc_1", "m-3", "too early for me")],
            statement="Betty finds May 1 too early.",
            held_by=["u_betty"],
        ),
    ]
    model = _Model(first)
    deps = _deps(model)
    _segment(
        deps,
        "mc_1",
        1_000,
        ("m-1", "u_chen", "Let's fly out on May 1."),
        ("m-2", "u_betty", "OK."),
        ("m-3", "u_betty", "Though it is too early for me."),
    )
    await update_group_profile(_extracted("mc_1"), deps)
    topic, decision, dissent = await _items(deps)
    assert (topic.status, decision.status, dissent.status) == (
        "exploring",
        "current",
        "open",
    )

    model.replies.append(
        [
            op(
                "topic.update",
                [("mc_2", "m-4", "sold out")],
                target=topic.topic_id,
                name="Travel",
                summary="Flights moved",
                status="consensus",
            ),
            op(
                "decision.supersede",
                [("mc_2", "m-4", "Let's leave on April 30 instead")],
                target=decision.item_id,
                statement="Depart on April 30 instead of May 1.",
                decided_by=["u_chen"],
            ),
            op(
                "dissent.resolve",
                [("mc_2", "m-5", "April 30 works")],
                target=dissent.item_id,
                statement="Betty agreed to April 30.",
            ),
        ]
    )
    _segment(
        deps,
        "mc_2",
        5_000,
        ("m-4", "u_chen", "May 1 is sold out. Let's leave on April 30 instead."),
        ("m-5", "u_betty", "April 30 works for me."),
    )
    await update_group_profile(_extracted("mc_2"), deps)

    prompt = model.prompts[1]
    assert f"[{decision.item_id}]" in prompt and f"[{topic.topic_id}] Travel" in prompt
    items = await _items(deps)
    assert len(items) == 6  # every version is kept
    statuses = [(i.kind, i.status) for i in items]
    assert statuses == [
        ("topic", "superseded"),
        ("decision", "superseded"),
        ("dissent", "superseded"),
        ("topic", "consensus"),
        ("decision", "current"),
        ("dissent", "resolved"),
    ]
    assert items[1].superseded_by == items[4].item_id
    assert items[3].topic_id == topic.topic_id


async def test_a_decision_links_to_the_topic_its_reply_creates(groups: Any) -> None:
    groups(group_profile_batch=1)
    links = (
        "everos_group_profile_links_dropped_total",
        {"reason": "unknown_reference"},
    )
    before = get_metrics_registry().get_sample_value(*links) or 0.0
    fly = [("mc_1", "m-1", "Let's fly out on May 1")]
    model = _Model(
        [
            # The decision names the topic before the topic.new that creates it.
            op(
                "decision.new",
                fly,
                topic_id="new_1",
                statement="Fly out on May 1.",
                decided_by=["u_chen"],
            ),
            op(
                "topic.new",
                fly,
                topic_id="new_1",
                name="Travel",
                summary="How the group gets there",
                status="consensus",
            ),
            # A topic id the model made up: the dissent is kept, unlinked.
            op(
                "dissent.open",
                [("mc_1", "m-2", "too early for me")],
                statement="Betty finds May 1 too early.",
                held_by=["u_betty"],
                about="tp_travel",
            ),
        ]
    )
    deps = _deps(model)
    _segment(
        deps,
        "mc_1",
        1_000,
        ("m-1", "u_chen", "Let's fly out on May 1."),
        ("m-2", "u_betty", "Though it is too early for me."),
    )

    await update_group_profile(_extracted("mc_1"), deps)

    decision, topic, dissent = await _items(deps)
    assert topic.topic_id and topic.topic_id.startswith("tp_")
    assert decision.topic_id == topic.topic_id and dissent.about is None
    assert (get_metrics_registry().get_sample_value(*links) or 0.0) - before == 1.0

    _segment(deps, "mc_2", 5_000, ("m-3", "u_chen", "Booking the flights now."))
    await update_group_profile(_extracted("mc_2"), deps)

    assert f"(topic {topic.topic_id}) Fly out on May 1." in model.prompts[1]


async def test_a_dropped_operation_writes_nothing(groups: Any) -> None:
    groups(group_profile_batch=1)
    model = _Model(
        [
            op(
                "decision.new",
                [("mc_1", "m-1", "two flights on April 30")],
                statement="Take an April 30 flight.",
            )
        ]
    )
    deps = _deps(model)
    _segment(
        deps, "mc_1", 1_000, ("m-1", "bot_ava", "There are two flights on April 30.")
    )

    await update_group_profile(_extracted("mc_1"), deps)

    assert len(model.prompts) == 1 and await _items(deps) == []
    assert (
        '"Ava (assistant)"' in model.prompts[0] or "Ava (assistant)" in model.prompts[0]
    )


async def test_a_reply_in_loose_form_is_read(groups: Any) -> None:
    groups(group_profile_batch=1)
    cite = [{"segment": "mc_1", "message": "m-1", "quote": "Let's fly out on May 1"}]
    loose = [
        # Only the fields a topic uses.
        {"op": "topic.new", "name": "Travel", "status": "exploring", "evidence": cite},
        # One id where a list is expected, and a null.
        {
            "op": "decision.new",
            "statement": "Fly out on May 1.",
            "decided_by": "u_chen",
            "agreed_by": None,
            "evidence": cite,
        },
        # Still judged: Betty's own message is not cited.
        {
            "op": "decision.new",
            "statement": "Betty decided May 1.",
            "decided_by": "u_betty",
            "evidence": cite,
        },
        # No evidence at all: dropped, not repaired.
        {"op": "role.add", "member_id": "u_chen", "role": "coordinator"},
    ]
    model = _Model(loose)
    deps = _deps(model)
    _segment(deps, "mc_1", 1_000, ("m-1", "u_chen", "Let's fly out on May 1."))

    await update_group_profile(_extracted("mc_1"), deps)

    topic, decision = await _items(deps)
    assert (topic.kind, topic.name, topic.summary) == ("topic", "Travel", "")
    assert decision.decided_by == ["u_chen"] and decision.agreed_by == []
    assert await deps.stores.group_profile.pending(_SCOPE) == []


class _Failing:
    """Fails every call as the provider fails one with no valid reply, then
    answers once ``ok_after`` calls have failed."""

    def __init__(self, ok_after: int | None = None) -> None:
        self.calls = 0
        self.ok_after = ok_after

    async def __call__(self, messages: Any, **_: Any) -> str:
        self.calls += 1
        if self.ok_after is not None and self.calls > self.ok_after:
            return json.dumps({"operations": []})
        raise LLMCallError("no usable reply", reason="invalid_reply")


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[datetime]:
    """The strategy's clock, moved by hand."""
    now = [from_timestamp_ms(1_000_000)]
    module = importlib.import_module("everos.memory.strategies.update_group_profile")
    monkeypatch.setattr(module, "get_utc_now", lambda: now[0])
    return now


def _failures(outcome: str, reason: str = "invalid_reply") -> float:
    value = get_metrics_registry().get_sample_value(
        "everos_group_profile_update_failures_total",
        {"reason": reason, "outcome": outcome},
    )
    return value or 0.0


def _tick() -> IdleTick:
    return IdleTick(
        strategy_name=UPDATE_GROUP_PROFILE_IDLE_SPEC.name,
        bucket_key=_extracted("mc_1").group_key,
        idle_seconds=600,
    )


async def test_an_unusable_reply_backs_off_then_drops_the_batch(
    groups: Any, clock: list[datetime]
) -> None:
    groups(group_profile_batch=1, group_profile_idle_seconds=600)
    model = _Failing()
    deps = make_deps(llm=FakeLLMClient(handler=model))
    deps.stores.groups.groups[("default", "p", "trip")] = Group(
        group_id="trip", created_at=0
    )
    _segment(deps, "mc_1", 1_000, ("m-1", "u_chen", "note"))
    _segment(deps, "mc_2", 2_000, ("m-2", "u_chen", "more"))
    backed_off, dropped = _failures("backed_off"), _failures("dropped")

    async def scans(n: int = 10) -> list[str]:
        """The scheduler's idle ticks, and flushes, while the group stays idle;
        the runs they failed."""
        failed: list[str] = []
        for _ in range(n):
            for event in (_tick(), _FLUSHED):
                try:
                    await update_group_profile(event, deps)
                except DeferredFailure as failure:
                    failed.append(str(failure))
        return failed

    # The run fails -- the engine records it, with no retry -- and the batch
    # is kept with a back-off.
    with pytest.raises(DeferredFailure, match=r"kept and read again in 600 s"):
        await update_group_profile(_extracted("mc_1"), deps)
    batch = await deps.stores.group_profile.pending_batch(_SCOPE)
    assert model.calls == 1 and batch.memcell_ids == ("mc_1",)
    assert (batch.failures, batch.retry_at) == (1, to_timestamp_ms(clock[0]) + 600_000)
    assert await scans() == []  # waiting: nothing read, nothing failed
    await update_group_profile(_extracted("mc_2"), deps)  # queued, not read
    assert model.calls == 1
    batch = await deps.stores.group_profile.pending_batch(_SCOPE)
    assert batch.memcell_ids == ("mc_1", "mc_2") and batch.failures == 1

    clock[0] += timedelta(seconds=600)
    (failed,) = await scans()
    assert "(invalid_reply)" in failed and "failure 2 of 3" in failed
    batch = await deps.stores.group_profile.pending_batch(_SCOPE)
    assert model.calls == 2 and batch.failures == 2
    assert batch.retry_at == to_timestamp_ms(clock[0]) + 1_200_000  # twice as long

    clock[0] += timedelta(seconds=1_200)
    (failed,) = await scans()
    assert "3 times in a row" in failed and "was dropped" in failed
    assert model.calls == 3
    assert await deps.stores.group_profile.pending_batch(_SCOPE) == PendingBatch()
    clock[0] += timedelta(days=1)
    assert await scans() == []
    assert model.calls == 3  # nothing left to try
    assert await _items(deps) == []
    assert _failures("backed_off") - backed_off == 2
    assert _failures("dropped") - dropped == 1


async def test_a_later_try_that_works_clears_the_record(
    groups: Any, clock: list[datetime]
) -> None:
    groups(group_profile_batch=1, group_profile_idle_seconds=600)
    model = _Failing(ok_after=1)
    deps = make_deps(llm=FakeLLMClient(handler=model))
    _segment(deps, "mc_1", 1_000, ("m-1", "u_chen", "note"))

    with pytest.raises(DeferredFailure):
        await update_group_profile(_extracted("mc_1"), deps)
    clock[0] += timedelta(seconds=601)
    await update_group_profile(_tick(), deps)

    assert model.calls == 2
    assert await deps.stores.group_profile.pending_batch(_SCOPE) == PendingBatch()


async def test_a_reply_that_never_parses_is_a_failure_too(
    groups: Any, clock: list[datetime]
) -> None:
    groups(group_profile_batch=1)
    calls: list[int] = []

    def garbage(messages: Any, **_: Any) -> str:
        calls.append(1)
        return "not json"

    deps = make_deps(llm=FakeLLMClient(handler=garbage))
    _segment(deps, "mc_1", 1_000, ("m-1", "u_chen", "note"))

    with pytest.raises(DeferredFailure):
        await update_group_profile(_extracted("mc_1"), deps)

    assert len(calls) == 3  # the operator's own attempts, then the back-off
    batch = await deps.stores.group_profile.pending_batch(_SCOPE)
    assert batch.memcell_ids == ("mc_1",) and batch.failures == 1
