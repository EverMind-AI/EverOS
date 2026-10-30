"""Dynamic legacy overrides remain distinct from cached base settings."""

from __future__ import annotations

import pytest

from everos.config.search_runtime import SearchConfigSource
from everos.config.settings import load_settings


def test_same_source_rereads_legacy_override_and_preserves_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVEROS_DECIDER__MAX_ROUNDS", "4")
    load_settings.cache_clear()
    source = SearchConfigSource()
    monkeypatch.setenv("EVEROS_LLMMR_MAX_ROUNDS", "2")
    assert source.tuning().max_rounds == 2
    monkeypatch.setenv("EVEROS_LLMMR_MAX_ROUNDS", "5")
    assert source.tuning().max_rounds == 5
    assert load_settings().decider.max_rounds == 4
    monkeypatch.setenv("EVEROS_LLMMR_MAX_ROUNDS", "invalid")
    assert source.tuning().max_rounds == 4
    monkeypatch.setenv("EVEROS_LLMMR_MAX_ROUNDS", "   ")
    assert source.tuning().max_rounds == 4


def test_base_settings_change_only_after_cache_invalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EVEROS_LLMMR_MAX_ROUNDS", raising=False)
    monkeypatch.setenv("EVEROS_DECIDER__MAX_ROUNDS", "4")
    load_settings.cache_clear()
    source = SearchConfigSource()
    assert source.tuning().max_rounds == 4
    monkeypatch.setenv("EVEROS_DECIDER__MAX_ROUNDS", "6")
    assert source.tuning().max_rounds == 4
    load_settings.cache_clear()
    assert source.tuning().max_rounds == 6


def test_same_source_rereads_trace_path(monkeypatch: pytest.MonkeyPatch) -> None:
    source = SearchConfigSource()
    monkeypatch.setenv("EVEROS_LLMMR_TRACE_DUMP", " first.jsonl ")
    assert source.trace_path() == "first.jsonl"
    monkeypatch.setenv("EVEROS_LLMMR_TRACE_DUMP", "second.jsonl")
    assert source.trace_path() == "second.jsonl"
    monkeypatch.setenv("EVEROS_LLMMR_TRACE_DUMP", "   ")
    assert source.trace_path() is None
