from __future__ import annotations

import importlib
from collections.abc import Mapping
from unittest.mock import AsyncMock, patch

import pytest
import structlog.testing
from everalgo.types import AtomicFact

from everos.infra.ome.testing import FakeStrategyContext
from everos.memory.events import EpisodeExtracted
from everos.memory.strategies.extract_atomic_facts import extract_atomic_facts

mod = importlib.import_module("everos.memory.strategies.extract_atomic_facts")


def _fact(text: str) -> AtomicFact:
    return AtomicFact(owner_id=None, content=text, timestamp=1_700_000_000_000)


def _event(
    *,
    owner_id: str = "u_alice",
    memcell_id: str = "mc_a",
    session_id: str = "s1",
    episode_text: str = "alice likes hiking and lives in tokyo",
    episode_timestamp_ms: int = 1_700_000_000_000,
) -> EpisodeExtracted:
    return EpisodeExtracted(
        memcell_id=memcell_id,
        episode_entry_id="ep_20260517_0001",
        episode_text=episode_text,
        episode_timestamp_ms=episode_timestamp_ms,
        owner_id=owner_id,
        session_id=session_id,
    )


async def test_strategy_meta_is_attached() -> None:
    meta = extract_atomic_facts.meta
    assert meta.name == "extract_atomic_facts"
    assert EpisodeExtracted in meta.trigger.on
    assert meta.emits == frozenset()
    assert meta.max_retries == 2


async def test_extracts_from_episode_text_and_writes_under_event_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single LLM call on episode_text; all facts written under event.owner_id."""
    monkeypatch.setattr(mod, "_writer", None, raising=False)
    generic_facts = [
        _fact("alice mentioned a weekend trip to tokyo"),
        _fact("alice said she needs hiking gear"),
    ]

    with (
        patch(
            "everos.memory.strategies.extract_atomic_facts.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_atomic_facts.AtomicFactExtractor"
        ) as mock_cls,
        patch(
            "everos.memory.strategies.extract_atomic_facts.AtomicFactWriter"
        ) as mock_wcls,
        structlog.testing.capture_logs() as captured,
    ):
        mock_cls.return_value.aextract_from_text = AsyncMock(return_value=generic_facts)
        mock_wcls.return_value.append_entries = AsyncMock(return_value=[])

        await extract_atomic_facts(_event(), FakeStrategyContext())

    # Exactly one LLM call with the episode text.
    assert mock_cls.return_value.aextract_from_text.await_count == 1
    call = mock_cls.return_value.aextract_from_text.call_args
    assert call.args[0] == "alice likes hiking and lives in tokyo"
    assert call.kwargs["timestamp"] == 1_700_000_000_000

    # Single owner → one batch call with 2 facts.
    assert mock_wcls.return_value.append_entries.call_count == 1
    batch_call = mock_wcls.return_value.append_entries.call_args
    assert batch_call.args[0] == "u_alice"
    items: list[tuple[Mapping, Mapping]] = batch_call.args[1]
    assert len(items) == 2

    for inline, _sections in items:
        assert inline["owner_id"] == "u_alice"
        assert inline["session_id"] == "s1"
        assert inline["parent_type"] == "episode"
        assert inline["parent_id"] == "ep_20260517_0001"

    fact_texts = sorted(sections["Fact"] for _, sections in items)
    assert fact_texts == [
        "alice mentioned a weekend trip to tokyo",
        "alice said she needs hiking gear",
    ]

    matching = [e for e in captured if e.get("event") == "atomic_facts_extracted"]
    assert matching, "expected atomic_facts_extracted log line"
    record = matching[0]
    assert record["count"] == 2
    assert record["owner_id"] == "u_alice"


async def test_skips_when_extractor_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "_writer", None, raising=False)
    with (
        patch(
            "everos.memory.strategies.extract_atomic_facts.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_atomic_facts.AtomicFactExtractor"
        ) as mock_cls,
        patch(
            "everos.memory.strategies.extract_atomic_facts.AtomicFactWriter"
        ) as mock_wcls,
        structlog.testing.capture_logs() as captured,
    ):
        mock_cls.return_value.aextract_from_text = AsyncMock(return_value=[])
        mock_wcls.return_value.append_entries = AsyncMock(return_value=[])
        await extract_atomic_facts(_event(), FakeStrategyContext())

    matching = [e for e in captured if e.get("event") == "atomic_facts_extracted"]
    assert matching, "log line should still fire (count=0)"
    assert matching[0]["count"] == 0
    mock_wcls.return_value.append_entries.assert_not_called()


async def test_passes_app_id_and_project_id_to_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "_writer", None, raising=False)
    facts = [_fact("some fact")]

    with (
        patch(
            "everos.memory.strategies.extract_atomic_facts.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_atomic_facts.AtomicFactExtractor"
        ) as mock_cls,
        patch(
            "everos.memory.strategies.extract_atomic_facts.AtomicFactWriter"
        ) as mock_wcls,
    ):
        mock_cls.return_value.aextract_from_text = AsyncMock(return_value=facts)
        mock_wcls.return_value.append_entries = AsyncMock(return_value=[])

        event = _event()
        await extract_atomic_facts(event, FakeStrategyContext())

    batch_call = mock_wcls.return_value.append_entries.call_args
    assert batch_call.kwargs["app_id"] == "default"
    assert batch_call.kwargs["project_id"] == "default"
