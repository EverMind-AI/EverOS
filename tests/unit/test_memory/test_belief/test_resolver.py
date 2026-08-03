"""Pins the resolver contract: supersession, audit trail, and abstention.

What a caller is entitled to rely on:

- the memory adopts a correction from a trusted channel;
- a flood on an untrusted channel changes nothing it asserts;
- every observation leaves a replayable record, including the ones that
  were rejected — "why does memory say X" must be answerable;
- when the memory holds no usable position it says so instead of
  asserting the marginally-leading candidate.
"""

from __future__ import annotations

import datetime as dt

import pytest

from everos.memory.belief import (
    BeliefResolver,
    BeliefState,
    FactObservation,
    ProvenanceTier,
)

_T0 = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
_KEY = "user_42:coffee_ratio"


def _obs(
    fact: str,
    *,
    day: int = 0,
    tier: ProvenanceTier = ProvenanceTier.USER_DIRECT,
    confidence: float = 0.9,
    source_id: str = "mc_1",
) -> FactObservation:
    return FactObservation(
        belief_key=_KEY,
        fact=fact,
        observed_at=_T0 + dt.timedelta(days=day),
        tier=tier,
        content_confidence=confidence,
        source_id=source_id,
    )


def test_trusted_correction_supersedes() -> None:
    resolver = BeliefResolver()
    resolver.observe(_obs("6 ounces of water per tablespoon", day=0))
    resolver.observe(_obs("5 ounces of water per tablespoon", day=7))

    verdict = resolver.verdict(_KEY)

    assert verdict.fact == "5 ounces of water per tablespoon"
    assert verdict.superseded == ["6 ounces of water per tablespoon"]
    assert verdict.observation_count == 2


def test_untrusted_flood_changes_nothing() -> None:
    resolver = BeliefResolver()
    resolver.observe(_obs("5 ounces of water per tablespoon", day=0))
    for day in range(1, 51):
        resolver.observe(
            _obs(
                "12 ounces of water per tablespoon",
                day=day,
                tier=ProvenanceTier.WEB_FETCH,
                confidence=0.99,
                source_id="scraped",
            )
        )

    verdict = resolver.verdict(_KEY)

    assert verdict.fact == "5 ounces of water per tablespoon"


def test_every_observation_is_auditable() -> None:
    """Rejected observations are recorded too — that is the point of an audit."""
    resolver = BeliefResolver()
    resolver.observe(_obs("5 ounces", day=0))
    revision = resolver.observe(
        _obs("12 ounces", day=1, tier=ProvenanceTier.UNTRUSTED, confidence=0.99)
    )

    assert revision.belief_key == _KEY
    assert revision.fact == "12 ounces"
    assert revision.reliability <= 0.10
    assert revision.accepted is False
    assert revision.admitted is True
    assert revision.shift < 0.05
    assert revision.prior != {}
    assert revision.posterior["5 ounces"] > revision.posterior["12 ounces"]
    assert revision.source_id == "mc_1"


def test_first_observation_is_accepted_and_moves_the_belief() -> None:
    resolver = BeliefResolver()
    revision = resolver.observe(_obs("5 ounces", day=0))

    assert revision.accepted is True
    assert revision.admitted is True
    assert 0.0 < revision.shift <= 1.0
    assert revision.prior == {}


def test_unknown_key_yields_an_empty_verdict() -> None:
    verdict = BeliefResolver().verdict("nothing:here")

    assert verdict.fact is None
    assert verdict.probability == 0.0
    assert verdict.observation_count == 0


def test_a_split_belief_reports_itself_as_uncertain() -> None:
    """Two comparable candidates should be surfaced, not silently picked."""
    resolver = BeliefResolver()
    resolver.observe(_obs("option a", day=0, confidence=0.6))
    resolver.observe(_obs("option b", day=1, confidence=0.6))

    verdict = resolver.verdict(_KEY)

    assert verdict.is_uncertain
    assert verdict.probability < 0.75


def test_decay_makes_an_unconfirmed_belief_uncertain() -> None:
    resolver = BeliefResolver(half_life_days=30.0)
    resolver.observe(_obs("5 ounces", day=0))
    resolver.observe(_obs("6 ounces", day=1, confidence=0.6))

    fresh = resolver.verdict(_KEY)
    stale = resolver.verdict(_KEY, at=_T0 + dt.timedelta(days=400))

    assert stale.entropy_bits > fresh.entropy_bits
    assert stale.fact is not None


def test_persisted_state_round_trips() -> None:
    resolver = BeliefResolver()
    resolver.observe(_obs("5 ounces", day=0))
    saved = list(resolver.states.values())

    restored = BeliefResolver()
    restored.load(BeliefState(**state.model_dump()) for state in saved)

    assert restored.verdict(_KEY).fact == "5 ounces"
    assert restored.verdict(_KEY).probability == pytest.approx(
        resolver.verdict(_KEY).probability
    )
