"""Tests for :func:`extract_user_profile`: one new segment per update.

The strategy reads the profile and the event's segment from their stores,
attaches the pictures the owner posted in it, runs everalgo's
``ProfileExtractor.aupdate`` against a scripted model and writes the stamped
profile. In-memory doubles from :mod:`tests.fakes` stand in for every port,
so the assertions are on what the model was sent and what was stored.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from typing import Any

import pytest
from everalgo.llm import ImageUrlPart, LLMMessage
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, MemCell

from everos.component.images import ImageLimits
from everos.config import Settings
from everos.memory._partition_locks import _reset_for_tests
from everos.memory.deps import MemoryDeps
from everos.memory.events import EpisodeExtracted, SessionFlushed
from everos.memory.models import FORGOTTEN_MARK, Episode, ExplicitInfoItem, Profile
from everos.memory.ports import NO_SOURCE_CHECK, Scope, SessionKey
from everos.memory.strategies import EXTRACT_USER_PROFILE_SPEC, extract_user_profile
from tests.fakes import make_deps

_TS = 1_700_000_000_000
_LIMITS = ImageLimits(
    max_bytes=1 << 20,
    max_per_message=4,
    max_request_bytes=1 << 22,
    max_per_request=8,
    formats=("jpeg",),
)
_CAT = "c" * 64
_DOG = "d" * 64


@pytest.fixture(autouse=True)
def _isolate_partition_locks() -> None:
    _reset_for_tests()


def _every(monkeypatch: pytest.MonkeyPatch, n: int) -> None:
    """Set ``[profile] update_every`` for the strategy under test."""
    base = Settings()
    settings = base.model_copy(
        update={"profile": base.profile.model_copy(update={"update_every": n})}
    )
    module = importlib.import_module("everos.memory.strategies.extract_user_profile")
    monkeypatch.setattr(module, "load_settings", lambda: settings)


@pytest.fixture(autouse=True)
def _every_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The update itself is under test here: every segment is read at once
    (``update_every = 1``); the cadence has its own tests below."""
    _every(monkeypatch, 1)


def _scope(owner_id: str = "u_alice") -> Scope:
    return Scope(
        app_id="default", project_id="default", owner_id=owner_id, owner_type="user"
    )


def _event(
    memcell_id: str = "mc_1",
    *,
    owner_id: str = "u_alice",
    language: str | None = "English",
) -> EpisodeExtracted:
    return EpisodeExtracted(
        memcell_id=memcell_id,
        episode_entry_id=f"ep_{memcell_id}",
        episode_text="...",
        episode_timestamp_ms=_TS,
        owner_id=owner_id,
        language=language,
    )


def _memcell(*lines: tuple[str, str], ts: int = _TS) -> MemCell:
    return MemCell(
        items=[
            ChatMessage(
                id=f"m{i}_{ts}",
                role="user",
                content=text,
                timestamp=ts + i,
                sender_id=sender,
                sender_name=sender.removeprefix("u_").title(),
            )
            for i, (sender, text) in enumerate(lines)
        ],
        timestamp=ts + len(lines),
    )


def _add(description: str, *, pictures: list[str] | None = None) -> dict[str, Any]:
    return {
        "action": "add",
        "type": "explicit_info",
        "data": {"evidence": "e", "description": description, "category": "c"},
        "sources": ["s1"],
        "pictures": pictures or [],
    }


class _Model:
    """A scripted model that records what each call was sent."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.sent: list[list[LLMMessage]] = []
        self.llm = FakeLLMClient(handler=self._answer)

    def _answer(self, messages: list[LLMMessage], **_kwargs: Any) -> str:
        self.sent.append(messages)
        return self.replies.pop(0)

    def user_text(self, call: int = 0) -> str:
        content = self.sent[call][1].content
        if isinstance(content, str):
            return content
        return "\n".join(p.text for p in content if not isinstance(p, ImageUrlPart))

    def images(self, call: int = 0) -> int:
        content = self.sent[call][1].content
        return (
            0
            if isinstance(content, str)
            else sum(isinstance(p, ImageUrlPart) for p in content)
        )


def _deps(model: _Model, **kwargs: Any) -> MemoryDeps:
    return _with_episodes(make_deps(llm=model.llm, **kwargs))


def _with_episodes(deps: MemoryDeps) -> MemoryDeps:
    """``deps`` whose seeded segments come with their episodes, as the pipeline
    writes each one (a person's, for every user sender) before its event."""
    seed = deps.stores.memcell.seed

    def seed_with_episodes(memcell_id: str, cell: MemCell, *args: Any) -> None:
        seed(memcell_id, cell, *args)
        for sender in dict.fromkeys(
            item.sender_id for item in cell.items if item.role == "user"
        ):
            deps.stores.episode.seed(
                _scope(sender),
                # The id _seed_episode gives the segment's alice episode.
                f"ep_{memcell_id}"
                if sender == "u_alice"
                else f"ep_{memcell_id}_{sender}",
                Episode(
                    owner_id=sender,
                    episode="...",
                    summary="...",
                    timestamp=cell.timestamp,
                    session_id="sess_1",
                    parent_id=memcell_id,
                ),
            )

    deps.stores.memcell.seed = seed_with_episodes  # type: ignore[method-assign]
    return deps


def test_spec_listens_to_a_users_own_episodes_only() -> None:
    spec = EXTRACT_USER_PROFILE_SPEC
    assert spec.name == "extract_user_profile"
    assert list(spec.trigger.on) == [  # type: ignore[union-attr]
        EpisodeExtracted,
        SessionFlushed,
    ]
    assert spec.applies_to is not None
    assert spec.applies_to(_event(), make_deps())
    assert spec.applies_to(_event(), make_deps(embedding=None))
    group = _event(owner_id="trip").model_copy(update={"owner_type": "group"})
    assert not spec.applies_to(group, make_deps())
    assert spec.max_retries == 2


async def test_a_first_update_writes_items_resting_on_the_segment() -> None:
    model = _Model(json.dumps({"operations": [_add("Lives in Paris.")]}))
    deps = _deps(model)
    deps.stores.memcell.seed("mc_1", _memcell(("u_alice", "I live in Paris.")))

    await extract_user_profile(_event(), deps)

    stored = await deps.stores.profile.read(_scope())
    assert stored is not None
    (item,) = stored.explicit_info
    extra = item.model_extra or {}
    assert item.description == "Lives in Paris."
    assert extra["item_id"].startswith("pi_")
    assert extra["source_memcell_ids"] == ["mc_1"]
    assert (extra["source_start"], extra["source_end"]) == (_TS, _TS + 1)
    assert stored.primary_language == "English"
    assert '<segment id="s1">' in model.user_text()


async def test_an_update_reads_only_the_new_segment_against_the_stored_profile() -> (
    None
):
    model = _Model(json.dumps({"operations": [{"action": "none"}]}))
    deps = _deps(model)
    deps.stores.memcell.seed("mc_1", _memcell(("u_alice", "Old talk about Paris.")))
    deps.stores.memcell.seed(
        "mc_2", _memcell(("u_alice", "New talk about Lyon."), ts=_TS + 10_000)
    )
    await deps.stores.profile.write(
        _scope(),
        Profile(
            owner_id="u_alice",
            summary="s",
            timestamp=_TS,
            explicit_info=[
                ExplicitInfoItem(
                    description="Lives in Paris.",
                    category="Home",
                    item_id="pi_1",
                    source_memcell_ids=["mc_1"],
                )
            ],
        ),
    )

    await extract_user_profile(_event("mc_2"), deps)

    user = model.user_text()
    assert "New talk about Lyon." in user
    assert "Old talk about Paris." not in user
    assert '<item ref="e1">' in user and "Lives in Paris." in user
    assert "pi_1" not in user and "mc_1" not in user


async def test_the_next_update_writes_a_summary_a_forget_dropped() -> None:
    """A forget leaves the summary empty; the next update writes it again,
    from the items -- even when its reply changes nothing."""
    model = _Model(json.dumps({"operations": [{"action": "none"}]}))
    deps = _deps(model)
    deps.stores.memcell.seed("mc_2", _memcell(("u_alice", "Nice weather.")))
    await deps.stores.profile.write(
        _scope(),
        Profile(
            owner_id="u_alice",
            summary="",
            timestamp=_TS,
            explicit_info=[
                ExplicitInfoItem(
                    description="Works in Python.",
                    category="Work",
                    item_id="pi_1",
                    source_memcell_ids=["mc_1"],
                )
            ],
        ),
    )

    await extract_user_profile(_event("mc_2"), deps)

    stored = await deps.stores.profile.read(_scope())
    assert stored is not None and stored.summary == "Works in Python."


async def test_a_segment_already_read_is_not_read_again() -> None:
    model = _Model()
    deps = _deps(model)
    deps.stores.memcell.seed("mc_1", _memcell(("u_alice", "I live in Paris.")))
    await deps.stores.profile.write(
        _scope(),
        Profile(
            owner_id="u_alice",
            summary="s",
            timestamp=_TS,
            explicit_info=[
                ExplicitInfoItem(description="x", source_memcell_ids=["mc_1"])
            ],
        ),
    )

    await extract_user_profile(_event(), deps)

    assert model.sent == []


async def test_a_deleted_segment_is_skipped() -> None:
    model = _Model()

    await extract_user_profile(_event("mc_gone"), _deps(model))

    assert model.sent == []


async def test_the_owners_pictures_are_attached_and_another_senders_are_not() -> None:
    model = _Model(json.dumps({"operations": [_add("Has a cat.", pictures=["img-1"])]}))
    deps = _deps(model, picture_limits=_LIMITS)
    deps.stores.images.images[_CAT] = (b"cat", "jpeg")
    deps.stores.images.images[_DOG] = (b"dog", "jpeg")
    deps.stores.memcell.seed(
        "mc_1",
        _memcell(
            ("u_alice", f"My cat [IMAGE: {_CAT}]"),
            ("u_bob", f"My dog [IMAGE: {_DOG}]"),
        ),
    )

    await extract_user_profile(_event(), deps)

    user = model.user_text()
    assert "My cat [IMAGE: img-1]" in user
    assert "My dog [IMAGE]" in user and _DOG not in user
    assert model.images() == 1
    stored = await deps.stores.profile.read(_scope())
    assert stored is not None
    assert (stored.explicit_info[0].model_extra or {})["source_image_ids"] == [_CAT]


def _seed_episode(deps: MemoryDeps, memcell_id: str = "mc_1") -> str:
    """The segment's episode, as the pipeline wrote it before the event."""
    entry_id = f"ep_{memcell_id}"
    if any(r.ref.entry_id == entry_id for r in deps.stores.episode.records):
        return entry_id  # seeded with its segment (:func:`_with_episodes`)
    deps.stores.episode.seed(
        _scope(),
        entry_id,
        Episode(
            owner_id="u_alice",
            episode="...",
            summary="...",
            timestamp=_TS,
            session_id="sess_1",
            parent_id=memcell_id,
        ),
    )
    return entry_id


async def test_a_segment_whose_episode_is_forgotten_is_not_read() -> None:
    model = _Model(json.dumps({"operations": [{"action": "none"}]}))
    deps = _deps(model)
    deps.stores.memcell.seed("mc_1", _memcell(("u_alice", "I live in Paris.")))
    entry_id = _seed_episode(deps)
    (stored,) = await deps.stores.episode.read_many(_scope(), [entry_id])
    await deps.stores.episode.deprecate(
        _scope(),
        entries=[(stored.ref.md_path, entry_id)],
        replaced_by=FORGOTTEN_MARK,
        only_if=NO_SOURCE_CHECK,
    )
    event = _event().model_copy(update={"session_id": "sess_1"})

    await extract_user_profile(event, deps)
    assert model.sent == []

    await deps.stores.episode.clear_deprecations(_scope(), [entry_id])
    await extract_user_profile(event, deps)
    assert len(model.sent) == 1  # remembered again


async def test_a_conversation_sent_again_after_its_forget_is_read() -> None:
    """A live forget of the conversation covers the segments it marked, not
    new ones the conversation got afterwards (sent again, or continued)."""
    model = _Model(json.dumps({"operations": [{"action": "none"}]}))
    deps = _deps(model)
    deps.stores.memcell.seed("mc_1", _memcell(("u_alice", "I live in Paris.")))
    _seed_episode(deps)
    await deps.stores.forget_log.record(
        target="session",
        target_id="sess_1",
        app_id="default",
        project_id="default",
        owner_id="u_alice",
    )

    await extract_user_profile(
        _event().model_copy(update={"session_id": "sess_1"}), deps
    )
    assert len(model.sent) == 1


async def test_an_update_keeps_a_held_item_and_the_new_value_beside_it() -> None:
    reply = {
        "operations": [
            {
                "action": "modify",
                "ref": "e1",
                "data": {
                    "description": "Lives in Berlin.",
                    "category": "Home",
                    "evidence": "I moved to Berlin.",
                },
                "sources": ["s1"],
                "pictures": [],
            }
        ]
    }
    model = _Model(json.dumps(reply))
    deps = _deps(model)
    deps.stores.memcell.seed("mc_2", _memcell(("u_alice", "I moved to Berlin.")))
    await deps.stores.profile.write(
        _scope(),
        Profile(
            owner_id="u_alice",
            summary="s",
            timestamp=_TS,
            explicit_info=[
                ExplicitInfoItem(
                    description="Lives in Paris.",
                    category="Home",
                    item_id="pi_paris",
                    source_memcell_ids=["mc_1"],
                    held=True,
                    edited_by="u_alice",
                )
            ],
        ),
    )

    await extract_user_profile(_event("mc_2"), deps)

    stored = await deps.stores.profile.read(_scope())
    assert stored is not None
    paris, berlin = stored.explicit_info
    assert paris.description == "Lives in Paris."
    assert (paris.model_extra or {})["held"] is True
    assert berlin.description == "Lives in Berlin."
    assert "supersedes" not in (berlin.model_extra or {})
    assert stored.history == []


async def test_a_forgotten_picture_is_taken_out_of_the_update() -> None:
    model = _Model(json.dumps({"operations": [{"action": "none"}]}))
    deps = _deps(model, picture_limits=_LIMITS)
    deps.stores.images.images[_CAT] = (b"cat", "jpeg")
    deps.stores.images.images[_DOG] = (b"dog", "jpeg")
    deps.stores.memcell.seed(
        "mc_1",
        _memcell(
            ("u_alice", f"My cat [IMAGE: {_CAT}]"),
            ("u_alice", f"My dog [IMAGE: {_DOG}]"),
        ),
    )
    await deps.stores.forget_log.record(
        target="image",
        target_id=_CAT,
        app_id="default",
        project_id="default",
        owner_id="u_alice",
    )

    await extract_user_profile(_event(), deps)

    user = model.user_text()
    assert _CAT not in user and "My cat" in user
    assert "My dog [IMAGE: img-1]" in user
    assert model.images() == 1


async def test_a_same_named_groups_forgotten_picture_stays_in_the_update() -> None:
    """A group whose id is the owner's forgot the picture: not the owner's."""
    model = _Model(json.dumps({"operations": [{"action": "none"}]}))
    deps = _deps(model, picture_limits=_LIMITS)
    deps.stores.images.images[_CAT] = (b"cat", "jpeg")
    deps.stores.memcell.seed("mc_1", _memcell(("u_alice", f"My cat [IMAGE: {_CAT}]")))
    await deps.stores.forget_log.record(
        target="image",
        target_id=_CAT,
        app_id="default",
        project_id="default",
        owner_id="u_alice",
        owner_type="group",
    )

    await extract_user_profile(_event(), deps)

    assert "My cat [IMAGE: img-1]" in model.user_text()
    assert model.images() == 1


async def test_without_vision_no_picture_is_attached() -> None:
    model = _Model(json.dumps({"operations": [{"action": "none"}]}))
    deps = _deps(model)
    deps.stores.images.images[_CAT] = (b"cat", "jpeg")
    deps.stores.memcell.seed("mc_1", _memcell(("u_alice", f"My cat [IMAGE: {_CAT}]")))

    await extract_user_profile(_event(), deps)

    assert model.images() == 0


async def test_the_profile_language_is_frozen_once_known() -> None:
    model = _Model(
        json.dumps({"operations": [_add("Lives in Paris.")]}),
        json.dumps({"operations": [{"action": "none"}]}),
    )
    deps = _deps(model)
    deps.stores.memcell.seed("mc_1", _memcell(("u_alice", "I live in Paris.")))
    deps.stores.memcell.seed("mc_2", _memcell(("u_alice", "Hola."), ts=_TS + 10))

    await extract_user_profile(_event(language="English"), deps)
    await extract_user_profile(_event("mc_2", language="Spanish"), deps)

    stored = await deps.stores.profile.read(_scope())
    assert stored is not None and stored.primary_language == "English"
    assert "Write every output field in English" in str(model.sent[1][0].content)


def _modify(ref: str, description: str) -> dict[str, Any]:
    return {
        "action": "modify",
        "ref": ref,
        "data": {"evidence": "e", "description": description, "category": "c"},
        "sources": ["s1"],
        "pictures": [],
    }


async def test_a_new_value_links_the_old_one_and_keeps_it_in_the_history() -> None:
    """Nothing is lost in place: the old value moves to ``user.md``'s history."""
    model = _Model(
        json.dumps({"operations": [_add("Lives in Paris.")]}),
        json.dumps({"operations": [_modify("e1", "Lives in Lyon.")]}),
    )
    deps = _deps(model)
    deps.stores.memcell.seed("mc_1", _memcell(("u_alice", "I live in Paris.")))
    deps.stores.memcell.seed(
        "mc_2", _memcell(("u_alice", "We moved to Lyon."), ts=_TS + 10_000)
    )

    await extract_user_profile(_event(), deps)
    first = await deps.stores.profile.read(_scope())
    assert first is not None and first.history == []
    old_id = (first.explicit_info[0].model_extra or {})["item_id"]
    await extract_user_profile(_event("mc_2"), deps)

    stored = await deps.stores.profile.read(_scope())
    assert stored is not None
    (item,) = stored.explicit_info
    extra = item.model_extra or {}
    (old,) = stored.history
    assert item.description == "Lives in Lyon."
    assert extra["item_id"] != old_id
    assert (extra["supersedes"], extra["since"]) == (old_id, _TS + 10_001)
    assert extra["source_memcell_ids"] == ["mc_2"]
    assert old["description"] == "Lives in Paris."
    assert (old["item_id"], old["superseded_by"]) == (old_id, extra["item_id"])
    assert old["source_memcell_ids"] == ["mc_1"]
    assert old["replaced_at"] == _TS + 10_001


async def test_a_deprecated_item_moves_to_the_history_with_its_reason() -> None:
    model = _Model(
        json.dumps({"operations": [_add("Has a car.")]}),
        json.dumps(
            {
                "operations": [
                    {
                        "action": "deprecate",
                        "ref": "e1",
                        "reason": "sold it",
                        "sources": ["s1"],
                    }
                ]
            }
        ),
    )
    deps = _deps(model)
    deps.stores.memcell.seed("mc_1", _memcell(("u_alice", "I have a car.")))
    deps.stores.memcell.seed(
        "mc_2", _memcell(("u_alice", "I sold my car."), ts=_TS + 10_000)
    )

    await extract_user_profile(_event(), deps)
    await extract_user_profile(_event("mc_2"), deps)

    stored = await deps.stores.profile.read(_scope())
    assert stored is not None and stored.explicit_info == []
    (old,) = stored.history
    assert old["description"] == "Has a car."
    assert (old["deprecated_by"], old["reason"]) == (["mc_2"], "sold it")


async def _probe(owner_a: str, owner_b: str) -> list[str]:
    log: list[str] = []

    async def slow(messages: list[LLMMessage], **_kwargs: Any) -> str:
        owner = "u_alice" if "u_alice" in str(messages[0].content) else "u_bob"
        log.append(f"enter:{owner}")
        await asyncio.sleep(0.02)
        log.append(f"leave:{owner}")
        return json.dumps({"operations": [{"action": "none"}]})

    deps = _with_episodes(make_deps(llm=FakeLLMClient(handler=slow)))
    for n, owner in enumerate((owner_a, owner_b)):
        deps.stores.memcell.seed(f"mc_{n}", _memcell((owner, "hello"), ts=_TS + n))
    await asyncio.gather(
        extract_user_profile(_event("mc_0", owner_id=owner_a), deps),
        extract_user_profile(_event("mc_1", owner_id=owner_b), deps),
    )
    return log


async def test_partition_lock_serialises_runs_on_same_owner() -> None:
    log = await _probe("u_alice", "u_alice")
    assert log == ["enter:u_alice", "leave:u_alice", "enter:u_alice", "leave:u_alice"]


async def test_partition_lock_lets_different_owners_run_in_parallel() -> None:
    log = await _probe("u_alice", "u_bob")
    assert log.index("enter:u_alice") < log.index("leave:u_bob")
    assert log.index("enter:u_bob") < log.index("leave:u_alice")


# ── Cadence: [profile] update_every ─────────────────────────────────────


def _none() -> str:
    return json.dumps({"operations": [{"action": "none"}]})


def _segments_sent(model: _Model, call: int) -> int:
    return model.user_text(call).count("<segment id=")


async def test_a_first_segment_is_read_at_once_then_every_n_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _every(monkeypatch, 3)
    model = _Model(json.dumps({"operations": [_add("Lives in Paris.")]}), _none())
    deps = _deps(model)
    for n in range(1, 5):
        deps.stores.memcell.seed(
            f"mc_{n}", _memcell(("u_alice", f"note {n}"), ts=_TS + n * 10)
        )

    await extract_user_profile(_event("mc_1"), deps)  # no profile yet: at once
    assert len(model.sent) == 1
    for n in (2, 3):
        await extract_user_profile(_event(f"mc_{n}"), deps)
        await extract_user_profile(_event(f"mc_{n}"), deps)  # a replay: once
    assert len(model.sent) == 1
    stored = await deps.stores.profile.read(_scope())
    assert stored is not None and stored.pending_memcell_ids == ["mc_2", "mc_3"]

    await extract_user_profile(_event("mc_4"), deps)
    assert len(model.sent) == 2
    assert _segments_sent(model, 1) == 3
    stored = await deps.stores.profile.read(_scope())
    assert stored is not None and stored.pending_memcell_ids == []
    assert [i.description for i in stored.explicit_info] == ["Lives in Paris."]


async def test_a_flush_reads_what_its_conversation_queued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _every(monkeypatch, 5)
    model = _Model(_none(), _none())
    deps = _deps(model)
    session = SessionKey("default", "default", "sess_1")
    for n in (1, 2):
        cell = _memcell(("u_alice", f"note {n}"), ("u_bob", "ok"), ts=_TS + n * 10)
        deps.stores.memcell.seed(f"mc_{n}", cell, session)
    for n in (1, 2):
        await extract_user_profile(_event(f"mc_{n}"), deps)
    assert len(model.sent) == 1  # the first at once, the second queued

    flushed = SessionFlushed(session_id="sess_1")
    assert EXTRACT_USER_PROFILE_SPEC.applies_to(flushed, deps)  # type: ignore[misc]
    await extract_user_profile(flushed, deps)
    assert len(model.sent) == 2  # alice's queue; bob has no profile, nothing
    stored = await deps.stores.profile.read(_scope())
    assert stored is not None and stored.pending_memcell_ids == []
    await extract_user_profile(flushed, deps)
    assert len(model.sent) == 2  # nothing left to read


async def test_a_queued_segment_forgotten_before_its_update_is_not_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _every(monkeypatch, 2)
    model = _Model(_none(), _none())
    deps = _deps(model)
    for n in (1, 2, 3):
        deps.stores.memcell.seed(
            f"mc_{n}", _memcell(("u_alice", f"note {n}"), ts=_TS + n * 10)
        )
    await extract_user_profile(_event("mc_1"), deps)
    await extract_user_profile(_event("mc_2"), deps)
    (stored,) = await deps.stores.episode.read_many(_scope(), ["ep_mc_2"])
    await deps.stores.episode.deprecate(
        _scope(),
        entries=[(stored.ref.md_path, "ep_mc_2")],
        replaced_by=FORGOTTEN_MARK,
        only_if=NO_SOURCE_CHECK,
    )
    await extract_user_profile(_event("mc_3"), deps)
    assert len(model.sent) == 2
    assert "note 3" in model.user_text(1)
    assert "note 2" not in model.user_text(1)


class _DownError(Exception):
    """The profile model's endpoint failed after its retries."""


async def test_a_failed_update_keeps_its_segments_for_the_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead-lettered update loses nothing: its segments stay queued, and
    the next update reads them with its own."""
    _every(monkeypatch, 5)
    replies: list[str | Exception] = [
        _DownError("down"),
        json.dumps({"operations": [_add("Lives in Paris."), _add("Has a dog.")]}),
    ]
    sent: list[list[LLMMessage]] = []

    def answer(messages: list[LLMMessage], **_kwargs: Any) -> str:
        sent.append(messages)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    deps = _with_episodes(make_deps(llm=FakeLLMClient(handler=answer)))
    deps.stores.memcell.seed("mc_1", _memcell(("u_alice", "I live in Paris.")))
    deps.stores.memcell.seed(
        "mc_2", _memcell(("u_alice", "I have a dog."), ts=_TS + 10)
    )

    with pytest.raises(Exception, match="down"):
        await extract_user_profile(_event("mc_1"), deps)
    stored = await deps.stores.profile.read(_scope())
    assert stored is not None and stored.pending_memcell_ids == ["mc_1"]

    await extract_user_profile(_event("mc_2"), deps)
    assert len(sent) == 2
    text = "\n".join(str(m.content) for m in sent[1])
    assert "I live in Paris." in text and "I have a dog." in text
    stored = await deps.stores.profile.read(_scope())
    assert stored is not None and stored.pending_memcell_ids == []
    assert {i.description for i in stored.explicit_info} == {
        "Lives in Paris.",
        "Has a dog.",
    }
