"""BeliefResolver — folds fact observations into beliefs and verdicts.

Stateless with respect to storage: the caller owns persistence. Load the
states you need, fold observations in, hand the revisions to whatever
audit sink you keep. That keeps the domain rule testable without a
database and lets the state live in SQLite, where derived state belongs,
rather than in the LanceDB fact index.

Usage::

    resolver = BeliefResolver()
    for obs in observations:
        revision = resolver.observe(obs)
    verdict = resolver.verdict("user_42:coffee_ratio")
    if verdict.is_uncertain:
        ...  # ask rather than assert
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping

from everos.component.utils.datetime import ensure_utc
from everos.core.observability.logging import get_logger

from .models import (
    UNKNOWN,
    BeliefRevision,
    BeliefState,
    BeliefVerdict,
    FactObservation,
)
from .update import (
    compose_reliability,
    decay,
    entropy_bits,
    posterior,
    total_variation,
)

logger = get_logger(__name__)

_SECONDS_PER_DAY = 86400.0


class BeliefResolver:
    """Maintains one categorical belief per ``belief_key``.

    Args:
        gate_shift: Total variation below which a revision is recorded
            as not accepted. The belief still updates and the observation
            still counts; the audit log just stays free of restatements.
            0.05 is roughly "the fourth identical confirmation".
        entry_scale: Tuning factor on how much prior mass a trusted new
            candidate is admitted with. See ``update.entry_mass``.
        half_life_days: Days for an unconfirmed belief to lose half its
            concentration. ``None`` disables decay.
    """

    def __init__(
        self,
        *,
        gate_shift: float = 0.05,
        entry_scale: float = 1.0,
        half_life_days: float | None = None,
    ) -> None:
        self._gate_shift = gate_shift
        self._entry_scale = entry_scale
        self._half_life_days = half_life_days
        self._states: dict[str, BeliefState] = {}

    @property
    def states(self) -> Mapping[str, BeliefState]:
        """Current belief states, keyed by ``belief_key``."""
        return self._states

    def load(self, states: Iterable[BeliefState]) -> None:
        """Seed the resolver with persisted states.

        Args:
            states: Previously stored belief states.
        """
        for state in states:
            self._states[state.key] = state

    def observe(self, observation: FactObservation) -> BeliefRevision:
        """Apply one observation and return its audit record.

        Args:
            observation: The sighting to fold in.

        Returns:
            The revision record, whether or not it cleared the gate.
        """
        observed_at = ensure_utc(observation.observed_at)
        state = self._states.get(observation.belief_key)
        prior = self._prior(state, observed_at)

        reliability = compose_reliability(
            observation.tier, observation.content_confidence, observation.taint
        )
        post = posterior(
            prior, observation.fact, reliability, entry_scale=self._entry_scale
        )
        moved = total_variation(post, prior)

        self._states[observation.belief_key] = BeliefState(
            key=observation.belief_key,
            distribution=post,
            updated_at=observed_at,
            observation_count=(state.observation_count if state else 0) + 1,
        )

        revision = BeliefRevision(
            belief_key=observation.belief_key,
            fact=observation.fact,
            reliability=reliability,
            prior=dict(prior),
            posterior=post,
            shift=moved,
            admitted=observation.fact not in prior,
            accepted=moved > self._gate_shift,
            observed_at=observed_at,
            source_id=observation.source_id,
        )
        if revision.accepted:
            logger.debug(
                "belief_revised",
                belief_key=observation.belief_key,
                reliability=round(reliability, 3),
                shift=round(moved, 3),
                source_id=observation.source_id,
            )
        return revision

    def verdict(
        self, belief_key: str, *, at: dt.datetime | None = None
    ) -> BeliefVerdict:
        """What the memory asserts for one belief.

        Args:
            belief_key: The belief to read.
            at: Read the belief as of this moment, applying decay. Defaults
                to the last observation time (no decay applied).

        Returns:
            The verdict, with the losing candidates listed as superseded.
        """
        state = self._states.get(belief_key)
        if state is None:
            return BeliefVerdict(
                belief_key=belief_key, fact=None, probability=0.0, entropy_bits=0.0
            )

        dist = (
            self._prior(state, ensure_utc(at)) if at is not None else state.distribution
        )
        leader = max(dist, key=lambda k: dist[k]) if dist else None
        # UNKNOWN winning means no observation has earned a position; that
        # is a real answer and must not be dressed up as a fact.
        asserted = None if leader == UNKNOWN else leader
        return BeliefVerdict(
            belief_key=belief_key,
            fact=asserted,
            probability=dist.get(leader, 0.0) if leader else 0.0,
            entropy_bits=entropy_bits(dist),
            candidate_count=len(dist),
            superseded=sorted(key for key in dist if key != leader and key != UNKNOWN),
            observation_count=state.observation_count,
        )

    def _prior(self, state: BeliefState | None, at: dt.datetime) -> dict[str, float]:
        """Distribution to update against, decayed to ``at`` if configured."""
        if state is None:
            return {}
        if self._half_life_days is None:
            return dict(state.distribution)
        elapsed = (at - ensure_utc(state.updated_at)).total_seconds() / _SECONDS_PER_DAY
        return decay(state.distribution, elapsed, self._half_life_days)
