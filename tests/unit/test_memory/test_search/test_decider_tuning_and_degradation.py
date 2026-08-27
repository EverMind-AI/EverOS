"""The decider's loop settings are configurable, and a degraded result says so.

Two defects, one root cause: a run's behaviour was not readable off its configuration.

* Twelve loop parameters existed only as ``EVEROS_LLMMR_*`` environment variables read
  once at import time. Nothing in the config named them, so an operator could not
  discover them, and an import-time read cannot answer to a config reload.
* When every decider attempt fails, the loop falls back to a fixed top-N core and stops
  after one round. It logged a *warning* and returned HTTP 200 with a full episode list.
  That produced a batch of runs reporting plausible accuracies while every decider
  call 404'd -- the same degraded path twelve times, indistinguishable from healthy
  results, and the numbers were written up before anyone read a trace.

So: the parameters live in ``[decider]``, resolved per search; and a fallback marks the
response and logs at error.
"""

from __future__ import annotations

import pytest

from everos.config.settings import DeciderSettings
from everos.core.context import (
    get_degradations,
    mark_degraded,
    reset_degradations,
    restore_degradations,
)
from everos.memory.search import llm_multiround as lmr
from everos.memory.search.dto import SearchData

_TUNING = (
    "max_rounds",
    "seed_topk",
    "subq_topk",
    "max_subqueries",
    "rrf_k",
    "no_new_core_patience",
    "per_subquery_guarantee",
    "retries",
    "retry_backoff_seconds",
    "core_overflow",
    "full_text",
    "fallback_core",
)


@pytest.mark.parametrize("field", _TUNING)
def test_every_loop_parameter_is_a_config_field(field: str) -> None:
    """Declared, so `everos config` shows it and a TOML can set it.

    Parameterised by name rather than asserted as a set: a rename should fail on the
    field that moved, not on an opaque set difference.
    """
    assert field in DeciderSettings.model_fields


def test_defaults_match_what_the_published_runs_used() -> None:
    """Promoting these must not silently change any of them.

    Every reference number in the repository was produced with these values. A default
    that shifted in the move would make the published results unreproducible while every
    test still passed.
    """
    d = DeciderSettings()
    assert (d.max_rounds, d.seed_topk, d.subq_topk, d.max_subqueries) == (3, 50, 20, 3)
    assert (d.rrf_k, d.no_new_core_patience, d.per_subquery_guarantee) == (60, 1, 1)
    assert (d.retries, d.retry_backoff_seconds, d.fallback_core) == (3, 0.5, 3)
    assert d.core_overflow is False and d.full_text is False


def test_the_legacy_env_names_still_win(monkeypatch: pytest.MonkeyPatch) -> None:
    """Launch scripts export these; an in-flight comparison must not shift.

    Precedence is deliberate: env over config. The reverse would change the behaviour of
    every script that already sets one the moment this landed.
    """
    monkeypatch.setenv("EVEROS_LLMMR_MAX_ROUNDS", "7")
    monkeypatch.setenv("EVEROS_LLMMR_DECIDER_FULL_TEXT", "1")
    monkeypatch.setenv("EVEROS_LLMMR_DECIDER_BACKOFF_S", "0.25")
    t = lmr._tuning()
    assert (t.max_rounds, t.full_text, t.retry_backoff_seconds) == (7, True, 0.25)


def test_a_malformed_override_is_ignored_not_guessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Falling back to the configured value beats inventing one.

    The alternative -- crashing -- would take down a search over a typo in an
    environment variable, and the run would already have been paid for.
    """
    monkeypatch.setenv("EVEROS_LLMMR_MAX_ROUNDS", "not-a-number")
    assert lmr._tuning().max_rounds == DeciderSettings().max_rounds


def test_resolution_is_per_call_not_frozen_at_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The property that made the old shape untestable.

    The root conftest resets the settings cache per test, so an import-time read would
    serve a value from whichever test imported the module first.
    """
    monkeypatch.setenv("EVEROS_LLMMR_MAX_ROUNDS", "2")
    assert lmr._tuning().max_rounds == 2
    monkeypatch.setenv("EVEROS_LLMMR_MAX_ROUNDS", "5")
    assert lmr._tuning().max_rounds == 5


def test_the_fallback_marks_the_response() -> None:
    """`decider_fallback` is the identifier a client switches on."""
    token = reset_degradations()
    try:
        mark_degraded("decider_fallback")
        assert get_degradations() == ("decider_fallback",)
        assert SearchData(degraded=list(get_degradations())).degraded == [
            "decider_fallback"
        ]
    finally:
        restore_degradations(token)


def test_repeats_collapse() -> None:
    """A per-question loop hits the same fallback every round.

    The response should say what degraded, not how many times -- a list that grows with
    the round count reads like several distinct faults.
    """
    token = reset_degradations()
    try:
        for _ in range(5):
            mark_degraded("decider_fallback")
        mark_degraded("something_else")
        assert get_degradations() == ("decider_fallback", "something_else")
    finally:
        restore_degradations(token)


def test_a_healthy_result_carries_nothing() -> None:
    """Empty on the normal path, so a client that ignores the field sees no change."""
    token = reset_degradations()
    try:
        assert get_degradations() == ()
        assert SearchData().degraded == []
    finally:
        restore_degradations(token)


def test_reset_stops_one_degraded_search_marking_later_ones() -> None:
    """Worker contexts are reused, and a sticky flag is worse than no flag.

    A field that eventually marks everything trains the reader to ignore it, which
    leaves the original defect in place with extra machinery on top.
    """
    token = reset_degradations()
    try:
        mark_degraded("decider_fallback")
        inner = reset_degradations()
        assert get_degradations() == ()
        restore_degradations(inner)
        assert get_degradations() == ("decider_fallback",)
    finally:
        restore_degradations(token)


def test_blank_reasons_are_not_recorded() -> None:
    """An empty string would render as a degradation with no name."""
    token = reset_degradations()
    try:
        mark_degraded("")
        mark_degraded("   ")
        assert get_degradations() == ()
    finally:
        restore_degradations(token)


def test_the_fallback_logs_at_error_not_warning() -> None:
    """The level is the signal that got missed.

    A warning said "noted" for the mechanism under test not running at all, and twelve
    arms of results were published on top of it.
    """
    from pathlib import Path

    src = Path(lmr.__file__).read_text(encoding="utf-8")
    block = src[src.index("fallback = list(range(min(tune.fallback_core") :][:900]
    assert 'logger.error(\n                "llm_multiround_decider_fallback"' in block
    assert 'mark_degraded("decider_fallback")' in block
