"""Domain models for the belief layer.

An atomic fact today is a sentence with a timestamp. Two facts that
contradict each other are both stored, both retrieved, and the answering
model is left to pick. These models add the missing dimension: a set of
mutually exclusive candidate facts (a *belief*) carrying a probability
distribution, and a per-observation audit record of how that distribution
moved.

The trust model is the load-bearing part. Reliability is a property of the
**channel** a fact arrived on, never of the sentence. A confidently phrased
claim from a scraped page must not outrank a hedged claim from the user,
and no amount of repetition on a low tier may change that — otherwise
volumetric memory poisoning works by construction.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

PIVOT = 0.5
"""Reliability below which an observation carries no evidential weight.

The likelihood ratio clamps to 1 here, so a channel at or under the pivot
cannot shift probability mass between candidates that are already known,
and cannot admit a candidate that is not (see ``update.entry_mass``).
"""

UNKNOWN = "__unknown__"
"""Reserved candidate holding the mass no observation has claimed.

Every belief starts entirely on it. Without an explicit residual the
first observation of a fact would normalise to probability 1.0 — a
memory certain of something it has been told exactly once, which is the
opposite of the point. When ``UNKNOWN`` is the most probable candidate
the memory holds no position and reports ``fact = None``.
"""


class ProvenanceTier(StrEnum):
    """Channel a fact arrived on. Determines its reliability ceiling."""

    USER_DIRECT = "user_direct"
    USER_EDIT = "user_edit"
    AGENT_OWN = "agent_own"
    TOOL_OUTPUT = "tool_output"
    DOCUMENT = "document"
    AGENT_THIRD_PARTY = "agent_third_party"
    WEB_FETCH = "web_fetch"
    UNTRUSTED = "untrusted"


TIER_CEILING: Mapping[ProvenanceTier, float] = {
    ProvenanceTier.USER_EDIT: 0.99,
    ProvenanceTier.USER_DIRECT: 0.98,
    ProvenanceTier.AGENT_OWN: 0.80,
    ProvenanceTier.TOOL_OUTPUT: 0.65,
    ProvenanceTier.DOCUMENT: 0.55,
    ProvenanceTier.AGENT_THIRD_PARTY: 0.40,
    ProvenanceTier.WEB_FETCH: 0.30,
    ProvenanceTier.UNTRUSTED: 0.10,
}
"""Maximum reliability a channel can ever earn.

Everything at or below :data:`PIVOT` is inert: it may be recorded and
retrieved, but it cannot change what the memory asserts.
"""

DEFAULT_TIER = ProvenanceTier.UNTRUSTED


class FactObservation(BaseModel):
    """One sighting of a candidate fact on one channel.

    Args:
        belief_key: Identifies the set of mutually exclusive candidates
            this fact competes within — one belief per key. Supplied by
            the caller; see ``docs/belief-layer.md`` for how the
            extraction layer is expected to derive it.
        fact: The atomic fact sentence, canonical surface form.
        observed_at: When the claim was made (not when it was indexed).
        tier: Channel provenance. Caps reliability.
        content_confidence: How firmly the speaker committed to the
            claim, read off hedging or definiteness. May only *lower*
            reliability within the tier ceiling, never raise it.
        taint: Origin tier when the content reached this channel through
            an untrusted intermediary. The stricter ceiling applies.
        source_id: Provenance pointer for audit — memcell / session id.
    """

    model_config = ConfigDict(frozen=True)

    belief_key: str
    fact: str
    observed_at: dt.datetime
    tier: ProvenanceTier = DEFAULT_TIER
    content_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    taint: ProvenanceTier | None = None
    source_id: str = ""


class BeliefState(BaseModel):
    """Current distribution over the candidate facts of one belief."""

    key: str
    distribution: dict[str, float] = Field(default_factory=dict)
    updated_at: dt.datetime
    observation_count: int = 0

    @property
    def asserted(self) -> str | None:
        """The most probable candidate, or ``None`` while empty."""
        if not self.distribution:
            return None
        return max(self.distribution, key=lambda k: self.distribution[k])

    @property
    def probability(self) -> float:
        """Probability mass on :attr:`asserted`."""
        fact = self.asserted
        return self.distribution[fact] if fact is not None else 0.0


class BeliefRevision(BaseModel):
    """Audit record of one observation applied to one belief.

    This is the decision chain. It answers "why does the memory currently
    say X" with a replayable sequence rather than a single overwrite:
    what arrived, on which channel, how far it moved the distribution, and
    whether it counted.

    Args:
        shift: Total variation distance between prior and posterior — how
            much of the belief's mass this observation relocated, in
            ``[0, 1]``. Deliberately not KL: an observation that
            introduces a candidate the prior had never heard of has
            unbounded KL, dominated by whatever floor the implementation
            picked, which makes the number unusable for a threshold and
            misleading in an audit log.
        admitted: Whether this observation introduced a new candidate
            rather than reweighting known ones.
    """

    model_config = ConfigDict(frozen=True)

    belief_key: str
    fact: str
    reliability: float
    prior: dict[str, float]
    posterior: dict[str, float]
    shift: float
    admitted: bool
    accepted: bool
    observed_at: dt.datetime
    source_id: str = ""


class BeliefVerdict(BaseModel):
    """What the memory asserts for one belief, and how sure it is."""

    model_config = ConfigDict(frozen=True)

    belief_key: str
    fact: str | None
    probability: float
    entropy_bits: float
    candidate_count: int = 0
    superseded: list[str] = Field(default_factory=list)
    observation_count: int = 0

    @property
    def normalised_entropy(self) -> float:
        """Entropy as a fraction of the maximum for this many candidates.

        Raw bits are not comparable across beliefs: two candidates cap at
        one bit, twelve cap at 3.58, so any absolute threshold silently
        means something different per belief. This is the comparable one.
        """
        if self.candidate_count < 2:
            return 0.0
        return self.entropy_bits / math.log2(self.candidate_count)

    @property
    def is_uncertain(self) -> bool:
        """True when the belief should be surfaced as a question.

        Above half the available entropy the memory holds no usable
        position, and saying so is more useful to a caller than asserting
        the marginally-leading candidate as if it were settled.
        """
        return self.fact is None or self.normalised_entropy >= 0.5
