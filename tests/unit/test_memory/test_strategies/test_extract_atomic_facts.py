from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import structlog.testing
from everalgo.types import AtomicFact

from everos.memory.events import (
    AtomicFactsExtracted,
    EpisodeExtracted,
    ReflectedEpisode,
    ReflectionContext,
)
from everos.memory.strategies import EXTRACT_ATOMIC_FACTS_SPEC, extract_atomic_facts
from tests.fakes import make_deps

_EXTRACTOR = "everos.memory.strategies.extract_atomic_facts.AtomicFactExtractor"


def _fact(text: str) -> AtomicFact:
    return AtomicFact(owner_id=None, content=text, timestamp=1_700_000_000_000)


def _event(
    *,
    owner_id: str = "u_alice",
    memcell_id: str = "mc_a",
    session_id: str = "s1",
    episode_text: str = "alice likes hiking and lives in tokyo",
    episode_timestamp_ms: int = 1_700_000_000_000,
    **extra: object,
) -> EpisodeExtracted:
    return EpisodeExtracted(
        memcell_id=memcell_id,
        episode_entry_id="ep_20260517_0001",
        episode_text=episode_text,
        episode_timestamp_ms=episode_timestamp_ms,
        owner_id=owner_id,
        session_id=session_id,
        **extra,
    )


def test_spec() -> None:
    spec = EXTRACT_ATOMIC_FACTS_SPEC
    assert spec.name == "extract_atomic_facts"
    assert EpisodeExtracted in spec.trigger.on
    assert spec.emits == frozenset({AtomicFactsExtracted})
    assert spec.max_retries == 2


async def test_extracts_from_episode_text_and_writes_under_event_owner() -> None:
    """Single LLM call on episode_text; all facts written under event.owner_id."""
    deps = make_deps()
    facts = [
        _fact("alice mentioned a weekend trip to tokyo"),
        _fact("alice said she needs hiking gear"),
    ]
    with (
        patch(_EXTRACTOR) as extractor_cls,
        structlog.testing.capture_logs() as captured,
    ):
        extractor_cls.return_value.aextract_from_text = AsyncMock(return_value=facts)
        events = await extract_atomic_facts(_event(), deps)

    # Exactly one LLM call with the episode text.
    extractor_cls.assert_called_once_with(llm=deps.llm)
    call = extractor_cls.return_value.aextract_from_text.call_args
    assert call.args[0] == "alice likes hiking and lives in tokyo"
    assert call.kwargs["timestamp"] == 1_700_000_000_000

    # Single owner → one batch with 2 facts, under the event's scope.
    stored = deps.stores.atomic_fact.records
    assert len(stored) == 2
    assert {s.scope.owner_id for s in stored} == {"u_alice"}
    assert {(s.scope.app_id, s.scope.project_id) for s in stored} == {
        ("default", "default")
    }
    assert all(s.item.session_id == "s1" for s in stored)
    assert all(s.item.parent_id == "ep_20260517_0001" for s in stored)
    assert sorted(s.item.content for s in stored) == [
        "alice mentioned a weekend trip to tokyo",
        "alice said she needs hiking gear",
    ]

    (done,) = events
    assert isinstance(done, AtomicFactsExtracted)
    assert done.episode_entry_id == "ep_20260517_0001"
    assert done.fact_entry_ids == [s.ref.entry_id for s in stored]
    assert done.source == "pipeline"

    record = next(e for e in captured if e.get("event") == "atomic_facts_extracted")
    assert record["count"] == 2
    assert record["owner_id"] == "u_alice"


async def test_no_facts_writes_nothing_but_still_reports() -> None:
    deps = make_deps()
    with (
        patch(_EXTRACTOR) as extractor_cls,
        structlog.testing.capture_logs() as captured,
    ):
        extractor_cls.return_value.aextract_from_text = AsyncMock(return_value=[])
        events = await extract_atomic_facts(_event(), deps)

    record = next(e for e in captured if e.get("event") == "atomic_facts_extracted")
    assert record["count"] == 0
    assert deps.stores.atomic_fact.records == []
    (done,) = events
    assert done.fact_entry_ids == []


async def test_reflection_context_is_carried_through() -> None:
    context = ReflectionContext(
        cluster_id="cl_1",
        mode="init",
        source_members=[("ep_1", "episode")],
        episodes=[ReflectedEpisode(entry_id="ep_1", md_path="a.md", parent_id="mc_1")],
        merged_episode="merged",
        last_ts=1,
    )
    with patch(_EXTRACTOR) as extractor_cls:
        extractor_cls.return_value.aextract_from_text = AsyncMock(return_value=[])
        (done,) = await extract_atomic_facts(
            _event(source="reflection", reflection=context), make_deps()
        )

    assert done.source == "reflection"
    assert done.reflection == context


@pytest.mark.parametrize("language", ["Chinese", None])
async def test_facts_are_written_in_the_episode_language(
    language: str | None,
) -> None:
    """The event's language is the extractor's output language; ``None`` lets
    the extractor infer its own."""
    deps = make_deps()
    with patch(_EXTRACTOR) as extractor_cls:
        extractor_cls.return_value.aextract_from_text = AsyncMock(return_value=[])
        await extract_atomic_facts(_event(language=language), deps)

    call = extractor_cls.return_value.aextract_from_text.call_args
    assert call.kwargs["output_language"] == language
