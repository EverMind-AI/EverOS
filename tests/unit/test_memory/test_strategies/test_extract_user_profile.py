"""Tests for :func:`extract_user_profile`: one new segment per update.

The strategy reads the profile and the event's segment from their stores,
attaches the pictures the owner posted in it, runs everalgo's
``ProfileExtractor.aupdate`` against a scripted model and writes the stamped
profile. In-memory doubles from :mod:`tests.fakes` stand in for every port,
so the assertions are on what the model was sent and what was stored.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from everalgo.llm import ImageUrlPart, LLMMessage
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, MemCell

from everos.component.images import ImageLimits
from everos.memory._partition_locks import _reset_for_tests
from everos.memory.deps import MemoryDeps
from everos.memory.events import EpisodeExtracted
from everos.memory.models import FORGOTTEN_MARK, Episode, ExplicitInfoItem, Profile
from everos.memory.ports import NO_SOURCE_CHECK, Scope
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


def _scope(owner_id: str = "u_alice") -> Scope:
    return Scope(
        app_id="default", project_id="default", owner_id=owner_id, owner_type="user"
    )


def _event(
    memcell_id: str = "mc_1",
    *,
    owner_id: str = "u_alice",
    source: str = "pipeline",
    language: str | None = "English",
) -> EpisodeExtracted:
    return EpisodeExtracted(
        memcell_id=memcell_id,
        episode_entry_id=f"ep_{memcell_id}",
        episode_text="...",
        episode_timestamp_ms=_TS,
        owner_id=owner_id,
        source=source,
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
    return make_deps(llm=model.llm, **kwargs)


def test_spec_listens_to_pipeline_episodes_only() -> None:
    spec = EXTRACT_USER_PROFILE_SPEC
    assert spec.name == "extract_user_profile"
    assert list(spec.trigger.on) == [EpisodeExtracted]  # type: ignore[union-attr]
    assert spec.applies_to is not None
    assert spec.applies_to(_event(), make_deps())
    assert spec.applies_to(_event(), make_deps(embedding=None))
    assert not spec.applies_to(_event(source="reflection"), make_deps())
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

    deps = make_deps(llm=FakeLLMClient(handler=slow))
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
