"""Belief layer — probabilistic conflict resolution over atomic facts.

Atomic facts are stored and retrieved independently today, so two facts
that contradict each other both survive and both reach the answering
model. This package adds the missing arbitration: mutually exclusive
candidates share a ``belief_key`` and hold a probability distribution,
updated by a reliability-weighted Bayesian rule whose evidential weight
is capped by the provenance of the channel each fact arrived on.

    from everos.memory.belief import BeliefResolver, FactObservation

    resolver = BeliefResolver()
    resolver.observe(FactObservation(...))
    verdict = resolver.verdict("user_42:coffee_ratio")

Two properties it is built to hold, both covered by
``tests/unit/test_memory/test_belief``: a channel at or below the trust
pivot cannot change what the memory asserts at any volume, and a single
trusted correction can still supersede a single trusted claim.
"""

from .models import PIVOT as PIVOT
from .models import TIER_CEILING as TIER_CEILING
from .models import UNKNOWN as UNKNOWN
from .models import BeliefRevision as BeliefRevision
from .models import BeliefState as BeliefState
from .models import BeliefVerdict as BeliefVerdict
from .models import FactObservation as FactObservation
from .models import ProvenanceTier as ProvenanceTier
from .resolver import BeliefResolver as BeliefResolver
from .update import compose_reliability as compose_reliability
from .update import entropy_bits as entropy_bits
from .update import entry_mass as entry_mass
from .update import kl_bits as kl_bits
from .update import likelihood_ratio as likelihood_ratio
from .update import posterior as posterior
from .update import total_variation as total_variation

__all__ = [
    "PIVOT",
    "TIER_CEILING",
    "UNKNOWN",
    "BeliefResolver",
    "BeliefRevision",
    "BeliefState",
    "BeliefVerdict",
    "FactObservation",
    "ProvenanceTier",
    "compose_reliability",
    "entropy_bits",
    "entry_mass",
    "kl_bits",
    "likelihood_ratio",
    "posterior",
    "total_variation",
]
