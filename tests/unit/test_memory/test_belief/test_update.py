"""Pins the update rule's guarantees.

The three properties the belief layer is allowed to claim:

1. a channel at or below the pivot cannot change the asserted fact, at
   any volume and for any number of candidates — including by admitting
   a brand-new candidate, which is the path that bypasses the likelihood
   ratio entirely;
2. a channel above the pivot can, in one observation;
3. repeated confirmation saturates, so a movement-gated audit log stays
   finite.

If any of these break, the layer is either useless or unsafe, so they are
pinned rather than spot-checked.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from everos.memory.belief import (
    PIVOT,
    UNKNOWN,
    ProvenanceTier,
    compose_reliability,
    entropy_bits,
    entry_mass,
    likelihood_ratio,
    posterior,
)
from everos.memory.belief.update import decay, total_variation


def _asserted(distribution: dict[str, float]) -> str:
    return max(distribution, key=lambda k: distribution[k])


@pytest.mark.parametrize("candidates", [2, 3, 5, 12])
@pytest.mark.parametrize("reliability", [0.10, 0.30, 0.40, 0.50])
def test_sub_pivot_channel_cannot_flip_a_known_candidate(
    candidates: int, reliability: float
) -> None:
    """1000 sub-pivot observations do not move the mode, for any |V|."""
    dist = {"true": 0.9}
    for i in range(candidates - 1):
        dist[f"other_{i}"] = 0.1 / (candidates - 1)

    for _ in range(1000):
        dist = posterior(dist, "other_0", reliability)

    assert _asserted(dist) == "true"


@pytest.mark.parametrize("reliability", [0.10, 0.30, 0.50])
def test_sub_pivot_channel_cannot_admit_a_new_candidate(reliability: float) -> None:
    """The admission path is gated by the same pivot as the likelihood.

    Without this the guarantee above is vacuous: an attacker simply
    asserts a value the belief has never seen, which is admitted before
    any reliability test runs.
    """
    dist = {"true": 0.8, "unknown": 0.2}
    for _ in range(1000):
        dist = posterior(dist, "injected", reliability)

    assert _asserted(dist) == "true"
    assert dist["injected"] < 0.01


def test_trusted_channel_supersedes_in_one_observation() -> None:
    """A single trusted correction overturns a single trusted claim.

    The property naive implementations lose: with a fixed epsilon entry
    mass no reliability is ever enough, and the memory keeps asserting
    the stale value forever.
    """
    established = posterior({}, "old_value", 0.9)
    corrected = posterior(established, "new_value", 0.9)

    assert _asserted(corrected) == "new_value"


@pytest.mark.parametrize("reliability", [0.55, 0.7, 0.9, 0.98])
def test_one_sighting_leaves_the_belief_at_the_channel_reliability(
    reliability: float,
) -> None:
    """A fact heard once is believed to the degree the channel is trusted.

    The number is only worth storing if it means something. Normalising a
    lone observation to 1.0 — which is what happens when the residual
    mass has nowhere to sit — makes every belief in the store read as
    certain and the whole layer decorative.
    """
    dist = posterior({}, "heard_once", reliability)

    assert dist["heard_once"] == pytest.approx(reliability)
    assert dist[UNKNOWN] == pytest.approx(1.0 - reliability)


def test_entry_mass_is_pivot_conditional() -> None:
    assert entry_mass(PIVOT) < 1e-3
    assert entry_mass(PIVOT - 0.01) < 1e-3
    assert entry_mass(0.9) == pytest.approx(0.9)
    assert entry_mass(0.9, scale=0.5) == pytest.approx(0.45)


def test_likelihood_ratio_pivots_at_one_half() -> None:
    assert likelihood_ratio(0.3) == pytest.approx(1.0)
    assert likelihood_ratio(0.5) == pytest.approx(1.0)
    assert likelihood_ratio(0.9) == pytest.approx(9.0)


def test_content_confidence_cannot_exceed_the_channel_ceiling() -> None:
    """Confident phrasing on a scraped page stays at the page's ceiling."""
    assert compose_reliability(ProvenanceTier.WEB_FETCH, 0.99) == pytest.approx(0.30)
    assert compose_reliability(ProvenanceTier.USER_DIRECT, 0.25) == pytest.approx(0.25)


def test_taint_applies_the_stricter_ceiling() -> None:
    """Laundering untrusted content through a trusted tool does not launder trust."""
    laundered = compose_reliability(
        ProvenanceTier.TOOL_OUTPUT, 0.9, taint=ProvenanceTier.UNTRUSTED
    )
    assert laundered <= 0.10


def test_repeated_confirmation_saturates() -> None:
    """Each identical confirmation moves the belief less than the last."""
    dist = {"x": 0.5, "unknown": 0.5}
    movements = []
    for _ in range(8):
        updated = posterior(dist, "x", 0.9)
        movements.append(total_variation(updated, dist))
        dist = updated

    assert all(a > b for a, b in pairwise(movements))
    assert movements[-1] < 0.05


def test_decay_loses_precision_without_losing_the_fact() -> None:
    dist = {"a": 0.95, "b": 0.05}
    entropies = [entropy_bits(decay(dist, days, 30.0)) for days in (0, 30, 90, 365)]

    assert all(a < b for a, b in pairwise(entropies))
    assert entropies[-1] > 0.98
    assert decay(dist, 365, 30.0)["a"] > 0.0


def test_total_variation_is_bounded_even_when_a_candidate_is_new() -> None:
    """The audit measure stays in [0, 1] where KL would diverge."""
    admitted = posterior({}, "first_ever", 0.9)

    assert 0.0 <= total_variation(admitted, {}) <= 1.0
    assert total_variation({"a": 1.0}, {"a": 1.0}) == pytest.approx(0.0)
    assert total_variation({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)
