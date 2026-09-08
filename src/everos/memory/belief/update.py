"""The update rule: closed-form categorical Bayes with a trust pivot.

Three properties this rule is built to have, in the order they matter:

1. **A channel at or below the pivot cannot change what is asserted.**
   Not by phrasing, not by volume. This is what makes memory poisoning a
   non-event rather than an arms race against prompt wording.
2. **A single trusted correction can supersede a single trusted claim.**
   Sounds trivial; it is the property naive implementations lose. If a
   new candidate enters the distribution at a fixed epsilon, no
   reliability is ever enough to promote it in one step, and the memory
   silently keeps asserting the stale value forever.
3. **Repetition saturates.** Confirming what is already believed moves
   the distribution by a shrinking number of bits, so an audit log gated
   on movement stays readable instead of recording every restatement.

(1) and (2) are the same knob and that is the whole subtlety. Admission of
a never-before-seen candidate happens *before* any likelihood test, so an
entry mass large enough to satisfy (2) hands an untrusted channel a free
seat at the table. :func:`entry_mass` resolves this by conditioning
admission on the same pivot the likelihood ratio uses.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from .models import DEFAULT_TIER, PIVOT, TIER_CEILING, UNKNOWN, ProvenanceTier

_EPS = 1e-9
_INERT_MASS = 1e-6
"""Admission mass for a candidate seen only on a sub-pivot channel.

Not zero: the candidate is still recorded and still retrievable, which
matters for audit. It simply cannot win.
"""


def compose_reliability(
    tier: ProvenanceTier,
    content_confidence: float,
    taint: ProvenanceTier | None = None,
) -> float:
    """Reliability of one observation: ``min(ceiling, content)``.

    Content confidence may only lower trust within the channel's ceiling.
    Reading trust off the text instead would let an attacker who controls
    the text control the trust: "Confirmed, verified, this is final"
    scores near 1.0 on any hedging rubric.

    Args:
        tier: Channel the observation arrived on.
        content_confidence: Speaker commitment, in ``[0, 1]``.
        taint: Origin tier if the content was laundered through this
            channel from a less trusted one.

    Returns:
        Reliability in ``[0.01, 0.99]``.
    """
    ceiling = TIER_CEILING.get(tier, TIER_CEILING[DEFAULT_TIER])
    if taint is not None:
        ceiling = min(ceiling, TIER_CEILING.get(taint, TIER_CEILING[DEFAULT_TIER]))
    return max(0.01, min(ceiling, content_confidence))


def likelihood_ratio(reliability: float) -> float:
    """Evidential weight of one observation, clamped at the pivot.

    ``LR = max(1, r / (1 - r))``. The clamp is deliberate: a source you
    do not trust asserting X is not evidence *against* X, it is simply
    not evidence. Clamping rather than inverting also fixes the pivot at
    ``r = 0.5`` independently of how many candidates the belief holds —
    the textbook form ``r if match else (1-r)/(|V|-1)`` has a threshold
    at ``1/|V|``, which drifts as candidates accumulate, so no guarantee
    can be stated about it.

    Args:
        reliability: Composed reliability of the observation.

    Returns:
        Multiplier ``>= 1.0`` applied to the observed candidate.
    """
    r = min(max(reliability, 0.0), 0.999)
    return max(1.0, r / max(1.0 - r, _EPS))


def entry_mass(reliability: float, scale: float = 1.0) -> float:
    """Prior mass a never-before-seen candidate is admitted with.

    Conditioned on the same pivot as :func:`likelihood_ratio`, for the
    reason given in the module docstring: a fixed entry mass cannot serve
    both supersession and poisoning resistance. Above the pivot a
    trusted first sighting enters at a mass proportional to how much the
    channel is trusted; below it, the candidate is recorded but inert.

    Args:
        reliability: Composed reliability of the observation.
        scale: Tuning factor on the admitted mass, in ``(0, 1]``. Lower
            values make the memory more conservative about adopting new
            candidates from trusted channels.

    Returns:
        Prior mass for the new candidate.
    """
    if reliability <= PIVOT:
        return _INERT_MASS
    return min(0.99, reliability * scale)


def entropy_bits(distribution: Mapping[str, float]) -> float:
    """Shannon entropy in bits — how unsure the belief is."""
    return -sum(p * math.log2(max(p, _EPS)) for p in distribution.values() if p > 0)


def kl_bits(
    posterior_dist: Mapping[str, float], prior_dist: Mapping[str, float]
) -> float:
    """``KL(posterior || prior)`` in bits — how far the belief moved."""
    total = 0.0
    for key in set(posterior_dist) | set(prior_dist):
        p = posterior_dist.get(key, _EPS)
        q = prior_dist.get(key, _EPS)
        if p > _EPS:
            total += p * math.log2(p / max(q, _EPS))
    return max(0.0, total)


def total_variation(
    posterior_dist: Mapping[str, float], prior_dist: Mapping[str, float]
) -> float:
    """Share of the belief's mass an observation relocated, in ``[0, 1]``.

    The audit measure. :func:`kl_bits` is the natural one while the set of
    candidates is fixed, but it diverges the moment an observation
    introduces a candidate the prior had never heard of — precisely the
    supersession case worth logging. The reported figure would then be a
    function of the implementation's zero-floor rather than of anything
    that happened, so this is used for the write gate instead.
    """
    keys = set(posterior_dist) | set(prior_dist)
    return 0.5 * sum(
        abs(posterior_dist.get(key, 0.0) - prior_dist.get(key, 0.0)) for key in keys
    )


def decay(
    distribution: Mapping[str, float], elapsed_days: float, half_life_days: float
) -> dict[str, float]:
    """Mix toward uniform as a belief goes unconfirmed.

    Forgetting as precision loss rather than deletion: a fact last
    confirmed two years ago should come back with low confidence, not
    come back wrong and not vanish.

    Args:
        distribution: Current distribution.
        elapsed_days: Days since the last observation.
        half_life_days: Days for half the concentration to be lost.

    Returns:
        The decayed distribution.
    """
    if not distribution or elapsed_days <= 0 or half_life_days <= 0:
        return dict(distribution)
    retention = (0.5 ** (1.0 / half_life_days)) ** elapsed_days
    uniform = 1.0 / len(distribution)
    return {
        key: retention * p + (1.0 - retention) * uniform
        for key, p in distribution.items()
    }


def posterior(
    prior: Mapping[str, float],
    fact: str,
    reliability: float,
    entry_scale: float = 1.0,
) -> dict[str, float]:
    """Apply one observation to a distribution.

    ``p(fact) ∝ LR(r) · p_prior(fact)``; every other candidate keeps its
    mass and is renormalised. ``O(|candidates|)``, no gradients.

    A **first** sighting is admitted at :func:`entry_mass` and returns
    there without a likelihood boost, so one observation on a channel of
    reliability ``r`` leaves the belief at about ``r`` — the calibrated
    reading of "a 0.9-reliable channel said this once". Applying the
    ratio on the admission step too would land it at 0.99 instead, which
    is a memory certain of something it has been told once. Subsequent
    sightings take the ratio path and saturate.

    Args:
        prior: Distribution before the observation. Empty means a belief
            that has never been observed; it starts wholly on
            :data:`~everos.memory.belief.models.UNKNOWN`.
        fact: The observed candidate.
        reliability: Composed reliability of the observation.
        entry_scale: Passed to :func:`entry_mass` for a new candidate.

    Returns:
        The posterior distribution, normalised.
    """
    dist = dict(prior) or {UNKNOWN: 1.0}
    if fact not in dist:
        admitted = entry_mass(reliability, entry_scale)
        total = sum(dist.values()) or 1.0
        scaled = {key: p / total * (1.0 - admitted) for key, p in dist.items()}
        scaled[fact] = admitted
        return scaled
    ratio = likelihood_ratio(reliability)
    weighted = {key: (ratio if key == fact else 1.0) * p for key, p in dist.items()}
    normaliser = sum(weighted.values()) or 1.0
    return {key: p / normaliser for key, p in weighted.items()}
