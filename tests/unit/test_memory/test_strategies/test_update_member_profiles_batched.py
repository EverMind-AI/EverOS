"""update_member_profiles in its default ``batched`` mode (docs/group-memory.md, 7.4).

The group is read like one more person: its segments are queued on the group,
and one model call (everalgo ``ProfileExtractor.aupdate_many``) updates every
member who spoke in a batch. In-memory stores and a scripted model. What
these pin:

- one call per batch: nothing until ``[groups] group_profile_batch`` segments
  are queued, then one call showing each segment once, for every speaker;
- each member's operations land on their own document only, with sources
  among their own segments; a member the reply leaves out is not written;
- the batch holds every speaker's profile lock, taken in sorted order, and a
  writer of one member waits for it;
- the attribution checks: an operation quoting only another member's line is
  left out, one quoting nothing found anywhere is kept, both counted;
- the cadence: the idle tick reads a short batch, a flush reads nothing
  unless ``profile_on_flush``, a catch-up reads even a batch in back-off;
  a failed call keeps the batch with a back-off and fails its run;
- a forget's queue on a member's document survives the batch and is read
  one member per call, as in ``per_member`` mode;
- members whose stored languages differ are read in one call per language;
- a replayed segment every speaker's profile rests on is not queued again;
- every member's pictures go in once, each item read off one reaches its poster;
  a member whose only message was a forgotten picture is not read;
- an idle tick with nothing queued reads no member's document; input everalgo
  refuses backs off like a failed call.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import json
import re
from datetime import datetime
from typing import Any

import pytest
from everalgo.llm import ImageUrlPart
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, MemCell
from everalgo.user_memory import ProfileExtractor
from everalgo.utils import InvalidInputError

from everos.component.images import ImageLimits
from everos.component.llm import LLMCallError
from everos.config import load_settings
from everos.core.observability.metrics import get_metrics_registry
from everos.core.utils.datetime import from_timestamp_ms
from everos.memory import _partition_locks
from everos.memory.events import EpisodeExtracted, GroupFlushed, GroupProfilesDue
from everos.memory.models import ExplicitInfoItem, Group, Profile, RosterEvent
from everos.memory.ports import DeferredFailure, IdleTick, Scope
from everos.memory.strategies import (
    UPDATE_MEMBER_PROFILES_IDLE_SPEC,
    update_member_profiles,
)
from tests.fakes import make_deps

_SCOPE = Scope(app_id="default", project_id="p", owner_id="trip", owner_type="group")
_FLUSHED = GroupFlushed(
    group_id="trip", app_id="default", project_id="p", session_id="s"
)
_DUE = GroupProfilesDue(group_id="trip", app_id="default", project_id="p")


class _Model:
    """Replies to a batched update with ``reply(targets)``; to a one-person
    update with one added item; records every prompt by reply kind."""

    def __init__(self, reply: Any = None, gate: asyncio.Event | None = None) -> None:
        self.many: list[str] = []
        self.single: list[str] = []
        self.reply = reply
        self.gate = gate
        self.entered = asyncio.Event()

    async def __call__(self, messages: Any, *, response_format: type, **_: Any) -> str:
        text = "\n".join(
            part if isinstance(part, str) else str(getattr(part, "text", ""))
            for m in messages
            for part in ([m.content] if isinstance(m.content, str) else m.content)
        )
        if response_format.__name__ == "_ProfileUpdateManyReply":
            self.many.append(text)
            self.entered.set()
            if self.gate is not None:
                await self.gate.wait()
            targets = re.findall(
                r"^- (p\d+): .+?, lines tagged user_id:(\S+),", text, re.M
            )
            people = (
                self.reply(dict(targets))
                if self.reply
                else [
                    {"user_id": uid, "operations": [_add(f"Fact about {uid}")]}
                    for _, uid in targets
                ]
            )
            return json.dumps({"people": people})
        if response_format.__name__ == "_ProfileUpdateReply":
            self.single.append(text)
            return json.dumps({"operations": [_add("Read again after a forget")]})
        raise AssertionError(response_format.__name__)


def _add(
    description: str, evidence: str = "x", sources: list[str] | None = None
) -> dict[str, Any]:
    item = {"evidence": evidence, "description": description, "category": "plans"}
    return {
        "action": "add",
        "type": "explicit_info",
        "data": item,
        "sources": sources or [],
        "pictures": [],
    }


def _deps(model: Any) -> Any:
    deps = make_deps(llm=FakeLLMClient(handler=model))
    deps.stores.groups.groups[("default", "p", "trip")] = Group(
        group_id="trip", created_at=0
    )
    deps.stores.groups.events[("default", "p", "trip")].extend(
        [
            RosterEvent(op="join", member_id=m, at=0, display_name=name)
            for m, name in (("u_chen", "Chen"), ("u_betty", "Betty"), ("u_ann", "Ann"))
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


def _event(memcell_id: str, *speakers: str) -> EpisodeExtracted:
    return EpisodeExtracted(
        memcell_id=memcell_id,
        episode_entry_id=f"gep_{memcell_id}",
        episode_text="text",
        episode_timestamp_ms=1,
        owner_id="trip",
        owner_type="group",
        app_id="default",
        project_id="p",
        speaker_ids=list(speakers),
    )


def _tick() -> IdleTick:
    return IdleTick(
        strategy_name=UPDATE_MEMBER_PROFILES_IDLE_SPEC.name,
        bucket_key=_event("mc_x").group_key or "",
        idle_seconds=600,
    )


async def _profile(deps: Any, member: str) -> Profile | None:
    return await deps.stores.profile.read(_SCOPE, member)


async def _pending(deps: Any) -> list[str]:
    return list(
        (await deps.stores.group_profile.member_pending_batch(_SCOPE)).memcell_ids
    )


@pytest.fixture
def groups(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Set ``[groups]`` values for the strategy under test (batched by default)."""

    def set_to(**values: Any) -> None:
        for key, value in values.items():
            monkeypatch.setenv(f"EVEROS_GROUPS__{key.upper()}", str(value).lower())
        load_settings.cache_clear()

    return set_to


async def _queue_five(deps: Any) -> None:
    speakers = [
        ("u_chen", "u_betty"),
        ("u_chen",),
        ("u_ann",),
        ("u_betty",),
        ("u_chen", "u_ann"),
    ]
    for n, who in enumerate(speakers, start=1):
        _segment(deps, f"mc_{n}", n * 1_000, *((m, f"{m} line {n}") for m in who))
        await update_member_profiles(_event(f"mc_{n}", *who), deps)


# ----------------------------------------------------------------------------
# One call per batch, applied per member
# ----------------------------------------------------------------------------


async def test_one_call_reads_a_full_batch_for_every_speaker(groups: Any) -> None:
    groups(group_profile_batch=5)
    model = _Model()
    deps = _deps(model)

    speakers = [("u_chen", "u_betty"), ("u_chen",), ("u_ann",), ("u_betty",)]
    for n, who in enumerate(speakers, start=1):
        _segment(deps, f"mc_{n}", n * 1_000, *((m, f"{m} line {n}") for m in who))
        await update_member_profiles(_event(f"mc_{n}", *who), deps)
    assert model.many == [] and await _pending(deps) == ["mc_1", "mc_2", "mc_3", "mc_4"]

    _segment(
        deps, "mc_5", 5_000, ("u_chen", "u_chen line 5"), ("u_ann", "u_ann line 5")
    )
    await update_member_profiles(_event("mc_5", "u_chen", "u_ann"), deps)

    (prompt,) = model.many
    assert model.single == []
    for n in range(1, 6):
        assert prompt.count(f"line {n}") == len(speakers[n - 1]) if n < 5 else 2
    for member in ("u_ann", "u_betty", "u_chen"):
        assert f"user_id:{member}, pictures posted" in prompt
        profile = await _profile(deps, member)
        assert profile is not None
        (item,) = profile.explicit_info
        sources = (item.model_extra or {})["source_memcell_ids"]
        assert all(f"{member} line {m[3:]}" in prompt for m in sources)
    assert await _pending(deps) == []


async def test_each_member_gets_only_their_own_operations(groups: Any) -> None:
    groups(group_profile_batch=5)

    def reply(targets: dict[str, str]) -> list[dict[str, Any]]:
        return [
            {
                "user_id": "u_betty",
                "operations": [_add("Books the hotel", sources=["s1"])],
            }
        ]

    model = _Model(reply)
    deps = _deps(model)
    await _queue_five(deps)

    betty = await _profile(deps, "u_betty")
    assert betty is not None and [i.description for i in betty.explicit_info] == [
        "Books the hotel"
    ]
    assert (betty.explicit_info[0].model_extra or {})["source_memcell_ids"] == ["mc_1"]
    assert await _profile(deps, "u_chen") is None  # left out: nothing written
    assert await _profile(deps, "u_ann") is None


async def test_the_batch_holds_every_speaker_s_lock_in_sorted_order(
    groups: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    groups(group_profile_batch=5)
    gate = asyncio.Event()
    model = _Model(gate=gate)
    deps = _deps(model)
    taken: list[str] = []
    real = _partition_locks.profile_lock

    def recording(scope: Scope, person: str | None = None) -> Any:
        taken.append(str(person))
        return real(scope, person)

    monkeypatch.setattr(_partition_locks, "profile_lock", recording)
    speakers = [("u_chen", "u_betty"), ("u_chen",), ("u_ann",), ("u_betty",)]
    for n, who in enumerate(speakers, start=1):
        _segment(deps, f"mc_{n}", n * 1_000, *((m, f"{m} line {n}") for m in who))
        await update_member_profiles(_event(f"mc_{n}", *who), deps)
    _segment(deps, "mc_5", 5_000, ("u_chen", "u_chen line 5"))
    event = _event("mc_5", "u_chen")
    batch = asyncio.create_task(update_member_profiles(event, deps))
    await model.entered.wait()
    assert taken == ["u_ann", "u_betty", "u_chen"]

    edited = asyncio.Event()

    async def edit() -> None:
        async with real(_SCOPE, "u_betty"):
            edited.set()

    writer = asyncio.create_task(edit())
    await asyncio.sleep(0.01)
    assert not edited.is_set()  # Betty's writer waits for the batch
    gate.set()
    await asyncio.gather(batch, writer)
    assert edited.is_set()


# ----------------------------------------------------------------------------
# Attribution
# ----------------------------------------------------------------------------


def _attribution(outcome: str) -> float:
    value = get_metrics_registry().get_sample_value(
        "everos_member_profile_attribution_total", {"outcome": outcome}
    )
    return value or 0.0


async def test_a_quote_of_another_member_only_is_left_out_and_counted(
    groups: Any,
) -> None:
    groups(group_profile_batch=1)

    def reply(targets: dict[str, str]) -> list[dict[str, Any]]:
        return [
            {
                "user_id": "u_chen",
                "operations": [
                    _add(
                        "Is vegetarian",
                        evidence="2026-03-02: 'I have been vegetarian for ten years'",
                    ),
                    _add(
                        "Prefers trains",
                        evidence="2026-03-02: 'trains are always better than planes'",
                    ),
                ],
            }
        ]

    model = _Model(reply)
    deps = _deps(model)
    before = {o: _attribution(o) for o in ("other_speaker", "unmatched")}
    _segment(
        deps,
        "mc_1",
        1_000,
        ("u_chen", "Let's pick the restaurant."),
        ("u_betty", "I have been vegetarian for ten years, remember."),
    )
    await update_member_profiles(_event("mc_1", "u_chen", "u_betty"), deps)

    chen = await _profile(deps, "u_chen")
    assert chen is not None and [i.description for i in chen.explicit_info] == [
        "Prefers trains"
    ]
    assert _attribution("other_speaker") - before["other_speaker"] == 1
    assert _attribution("unmatched") - before["unmatched"] == 1


# ----------------------------------------------------------------------------
# Cadence
# ----------------------------------------------------------------------------


async def test_the_idle_tick_reads_a_short_batch(groups: Any) -> None:
    groups(group_profile_batch=5)
    model = _Model()
    deps = _deps(model)
    _segment(deps, "mc_1", 1_000, ("u_chen", "a"))
    await update_member_profiles(_event("mc_1", "u_chen"), deps)
    assert model.many == []

    await update_member_profiles(_tick(), deps)
    assert len(model.many) == 1 and await _pending(deps) == []
    await update_member_profiles(_tick(), deps)  # nothing queued: no call
    assert len(model.many) == 1


async def test_a_flush_reads_nothing_unless_profile_on_flush(groups: Any) -> None:
    groups(group_profile_batch=5)
    model = _Model()
    deps = _deps(model)
    _segment(deps, "mc_1", 1_000, ("u_chen", "a"))
    await update_member_profiles(_event("mc_1", "u_chen"), deps)

    await update_member_profiles(_FLUSHED, deps)
    assert model.many == [] and await _pending(deps) == ["mc_1"]

    groups(group_profile_batch=5, profile_on_flush=True)
    await update_member_profiles(_FLUSHED, deps)
    assert len(model.many) == 1 and await _pending(deps) == []


async def test_a_catch_up_reads_the_batch_at_once(groups: Any) -> None:
    groups(group_profile_batch=5)
    model = _Model()
    deps = _deps(model)
    _segment(deps, "mc_1", 1_000, ("u_chen", "a"))
    await update_member_profiles(_event("mc_1", "u_chen"), deps)

    await update_member_profiles(_DUE, deps)
    assert len(model.many) == 1 and await _pending(deps) == []


class _Failing:
    """Fails every call, as the provider fails one with no valid reply, until
    ``recover`` is set; then answers as :class:`_Model`."""

    def __init__(self) -> None:
        self.calls = 0
        self.recover = False
        self.model = _Model()

    async def __call__(self, messages: Any, **kwargs: Any) -> str:
        self.calls += 1
        if self.recover:
            return await self.model(messages, **kwargs)
        raise LLMCallError("no usable reply", reason="invalid_reply")


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[datetime]:
    now = [from_timestamp_ms(1_000_000)]
    module = importlib.import_module("everos.memory.strategies._member_batch")
    monkeypatch.setattr(module, "get_utc_now", lambda: now[0])
    return now


async def test_a_failed_call_keeps_the_batch_with_a_back_off(
    groups: Any, clock: list[datetime]
) -> None:
    groups(group_profile_batch=1, group_profile_idle_seconds=600)
    failing = _Failing()
    deps = _deps(failing)
    _segment(deps, "mc_1", 1_000, ("u_chen", "a"))

    with pytest.raises(DeferredFailure, match="kept and read again in 600 s"):
        await update_member_profiles(_event("mc_1", "u_chen"), deps)
    batch = await deps.stores.group_profile.member_pending_batch(_SCOPE)
    assert batch.memcell_ids == ("mc_1",) and batch.failures == 1
    calls = failing.calls

    await update_member_profiles(_tick(), deps)  # waiting out the back-off
    assert failing.calls == calls

    failing.recover = True
    await update_member_profiles(_DUE, deps)  # a catch-up does not wait
    assert len(failing.model.many) == 1 and await _pending(deps) == []


# ----------------------------------------------------------------------------
# Forget queues, languages, replays
# ----------------------------------------------------------------------------


async def test_a_forget_queue_survives_the_batch_and_is_read_per_member(
    groups: Any,
) -> None:
    groups(group_profile_batch=1)
    model = _Model()
    deps = _deps(model)
    _segment(deps, "mc_old", 500, ("u_chen", "an older line"))
    queued = Profile(
        owner_id="u_chen",
        summary="",
        timestamp=0,
        explicit_info=[
            ExplicitInfoItem(evidence="e", description="Old fact", category="plans")
        ],
        pending_memcell_ids=["mc_old"],
    )
    await deps.stores.profile.write(_SCOPE, queued)
    _segment(deps, "mc_1", 1_000, ("u_chen", "a"), ("u_betty", "b"))

    await update_member_profiles(_event("mc_1", "u_chen", "u_betty"), deps)

    assert len(model.many) == 1
    (single,) = model.single  # Chen's forget queue, one member per call
    assert "an older line" in single and "The target user is Chen" in single
    chen = await _profile(deps, "u_chen")
    assert chen is not None and chen.pending_memcell_ids == []
    assert {i.description for i in chen.explicit_info} >= {
        "Fact about u_chen",
        "Read again after a forget",
    }


async def test_members_of_different_languages_are_read_in_one_call_each(
    groups: Any,
) -> None:
    groups(group_profile_batch=1)
    model = _Model()
    deps = _deps(model)
    german = Profile(
        owner_id="u_betty",
        summary="",
        timestamp=0,
        explicit_info=[
            ExplicitInfoItem(
                evidence="e", description="Wohnt in Berlin", category="Heim"
            )
        ],
        primary_language="German",
    )
    await deps.stores.profile.write(_SCOPE, german)
    _segment(
        deps,
        "mc_1",
        1_000,
        ("u_chen", "I will book the flights for everyone"),
        ("u_betty", "I will book the hotel for us"),
    )

    await update_member_profiles(_event("mc_1", "u_chen", "u_betty"), deps)

    assert len(model.many) == 2
    assert sum("German" in prompt for prompt in model.many) == 1


async def test_a_replayed_segment_every_speaker_read_is_not_queued(groups: Any) -> None:
    groups(group_profile_batch=1)
    model = _Model()
    deps = _deps(model)
    _segment(deps, "mc_1", 1_000, ("u_chen", "a"))
    await update_member_profiles(_event("mc_1", "u_chen"), deps)
    assert len(model.many) == 1

    await update_member_profiles(_event("mc_1", "u_chen"), deps)  # the replay
    assert len(model.many) == 1 and await _pending(deps) == []


async def test_each_picture_is_attached_once_and_reaches_its_poster_only(
    groups: Any,
) -> None:
    """D17: one call attaches every member's pictures, each listed beside its
    poster; an item read off a picture lands on the poster, and an operation
    resting only on someone else's picture is left out."""
    groups(group_profile_batch=1)
    cat, dog = "c" * 64, "d" * 64
    sent: list[Any] = []

    def reply(targets: dict[str, str]) -> list[dict[str, Any]]:
        pic = "picture sent 2026-03-02 shows a pet"
        chen = _add("Has a cat", evidence=pic) | {"pictures": ["img-1"]}
        betty = _add("Has a cat too", evidence=pic) | {"pictures": ["img-1"]}
        return [
            {"user_id": "u_chen", "operations": [chen]},
            {"user_id": "u_betty", "operations": [betty]},
        ]

    model = _Model(reply)

    async def recording(messages: Any, **kwargs: Any) -> str:
        sent.append(messages)
        return await model(messages, **kwargs)

    limits = ImageLimits(
        max_bytes=1 << 20,
        max_per_message=4,
        max_request_bytes=1 << 22,
        max_per_request=8,
        formats=("jpeg",),
    )
    deps = dataclasses.replace(_deps(recording), picture_limits=limits)
    deps.stores.images.images[cat] = (b"cat", "jpeg")
    deps.stores.images.images[dog] = (b"dog", "jpeg")
    _segment(
        deps,
        "mc_1",
        100,
        ("u_chen", f"Our cat [IMAGE: {cat}]"),
        ("u_betty", f"And our dog [IMAGE: {dog}]"),
    )

    await update_member_profiles(_event("mc_1", "u_chen", "u_betty"), deps)

    (messages,) = sent
    content = messages[1].content
    assert sum(isinstance(p, ImageUrlPart) for p in content) == 2
    system = messages[0].content
    assert "user_id:u_chen, pictures posted: img-1" in system
    assert "user_id:u_betty, pictures posted: img-2" in system
    chen = await _profile(deps, "u_chen")
    assert chen is not None
    (item,) = chen.explicit_info
    assert (item.model_extra or {})["source_image_ids"] == [cat]
    betty = await _profile(deps, "u_betty")
    assert betty is not None and betty.explicit_info == []


async def test_an_idle_tick_with_nothing_queued_reads_no_member_document(
    groups: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scheduler fires an idle tick on every scan while a group stays idle:
    with nothing queued it must not list or read the members' documents."""
    groups(group_profile_batch=5)
    model = _Model()
    deps = _deps(model)
    listed: list[Scope] = []
    real_people = deps.stores.profile.people

    async def people(scope: Scope) -> list[str]:
        listed.append(scope)
        return await real_people(scope)

    monkeypatch.setattr(deps.stores.profile, "people", people)

    await update_member_profiles(_tick(), deps)
    assert listed == [] and model.many == []

    await update_member_profiles(_DUE, deps)  # a catch-up reads the queues
    assert listed == [_SCOPE]


async def test_a_member_whose_only_message_was_a_forgotten_picture_is_not_read(
    groups: Any,
) -> None:
    """A forgotten picture leaves the text with its message when the message
    held nothing else: that member is no target, the rest of the batch is read."""
    groups(group_profile_batch=1)
    cat = "c" * 64
    model = _Model()
    limits = ImageLimits(
        max_bytes=1 << 20,
        max_per_message=4,
        max_request_bytes=1 << 22,
        max_per_request=8,
        formats=("jpeg",),
    )
    deps = dataclasses.replace(_deps(model), picture_limits=limits)
    deps.stores.images.images[cat] = (b"cat", "jpeg")
    _segment(
        deps,
        "mc_1",
        100,
        ("u_chen", "Let's book the flights today."),
        ("u_betty", f"[IMAGE: {cat}]"),
    )
    await deps.stores.forget_log.record(
        target="image",
        target_id=cat,
        app_id="default",
        project_id="p",
        owner_id="trip",
        owner_type="group",
    )

    await update_member_profiles(_event("mc_1", "u_chen", "u_betty"), deps)

    (prompt,) = model.many
    assert "user_id:u_chen, pictures posted" in prompt
    assert "user_id:u_betty, pictures posted" not in prompt
    assert await _profile(deps, "u_chen") is not None
    assert await _profile(deps, "u_betty") is None
    assert await _pending(deps) == []


async def test_input_everalgo_refuses_backs_off_instead_of_blocking(
    groups: Any, clock: list[datetime], monkeypatch: pytest.MonkeyPatch
) -> None:
    groups(group_profile_batch=1, group_profile_idle_seconds=600)
    model = _Model()
    deps = _deps(model)

    async def refuse(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise InvalidInputError("refused for the test")

    monkeypatch.setattr(ProfileExtractor, "aupdate_many", refuse)
    _segment(deps, "mc_1", 1_000, ("u_chen", "a"))

    with pytest.raises(DeferredFailure, match=r"invalid_input.*kept and read again"):
        await update_member_profiles(_event("mc_1", "u_chen"), deps)
    batch = await deps.stores.group_profile.member_pending_batch(_SCOPE)
    assert batch.memcell_ids == ("mc_1",) and batch.failures == 1
