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
from everos.memory.models import ExplicitInfoItem, Profile
from everos.memory.ports import Scope
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
