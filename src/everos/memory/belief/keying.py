"""Deriving ``belief_key`` — which facts are competing for the same slot.

The resolver arbitrates between facts that share a key. Nothing produces
that key: an atomic fact is an undecomposed sentence, with no
``(subject, attribute)`` to group on.

The rule here is that two facts compete when they are **about the same
thing while differing in the value**. So the signature drops value-bearing
tokens — numbers, quantities, amounts — and keeps the topic:

    "I got pre-approved for $350,000 from Wells Fargo"
    "I got pre-approved for $400,000 from Wells Fargo"
                    ^ signature identical, values differ

Terms are IDF-weighted against the facts seen so far in the scope. Without
that, "really", "looking", "forward" count as much as "pre-approved" and
"Wells Fargo", and unrelated chat about anything at all starts scoring as
a contradiction. Weighting is worth roughly a factor of ten in precision on
the benchmark (`benchmarks/belief_key.py`).

Errors here are not symmetric, and the threshold is set accordingly
-----------------------------------------------------------------
A **missed** link leaves two contradicting facts unarbitrated — exactly
where EverOS is today, so no ground is lost. A **false** link declares two
unrelated facts mutually exclusive, and one then suppresses the other: a
true fact stops being asserted because something irrelevant outranked it.

Partial arbitration is worth having; wrong arbitration is not. The default
threshold is therefore set where false links are rare rather than where
the F1 is best, and the losing candidate always stays retrievable — a
belief suppresses a fact from being *asserted*, never from being *found*.

This is a lexical stand-in. EverOS already embeds every fact, and matching
on those embeddings is the better implementation; :class:`BeliefKeyer` is
written against a similarity it takes as a parameter so that swap is a
constructor argument rather than a rewrite.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9']+")

_STOPWORD_TEXT = (
    "a an the i you we my your our me it is are was were be been am do does "
    "did have has had of to for in on at with and or but so that this these "
    "those there here about just really some any mine can could would should "
    "will i'm i've it's don't very much more most also as by from up out if "
    "then than what when how why who"
)

_STOPWORDS = frozenset(_STOPWORD_TEXT.split())
"""Terms too common to say anything about what a fact is about.

Kept short on purpose. Aggressive stopword lists start deleting the words
that distinguish one belief from another, and IDF weighting already
demotes whatever is common in a given store.
"""

_VALUE = re.compile(
    r"^(\d[\w:./%,-]*|\$.*|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|first|second|third|fourth|fifth|sixth|seventh|eighth|"
    r"ninth|tenth)$"
)
"""Tokens that carry the *value* rather than the topic.

A belief's candidates differ precisely here, so these must not enter the
signature — otherwise the two readings of a changed quantity look like
different topics and never compete.
"""

_DEFAULT_THRESHOLD = 0.25
"""Weighted-overlap score above which two facts share a belief.

Chosen from the precision side of the curve, not the F1 peak: on
``benchmarks/belief_key.py`` it links 81.4% of true pairs at a 0.45%
false-link rate. Dropping to 0.20 buys 4 points of linking and costs 3.5x
the false links; raising it to 0.30 cuts false links by a further 4x for
13 points of linking, which is the right trade for a scope holding many
near-duplicate topics.
"""


def signature(fact: str) -> frozenset[str]:
    """Topic terms of a fact, with value-bearing tokens removed."""
    return frozenset(
        token
        for token in _TOKEN.findall(fact.lower())
        if len(token) > 1 and token not in _STOPWORDS and not _VALUE.match(token)
    )


class BeliefKeyer:
    """Assigns facts to beliefs by topic signature within a scope.

    Stateful by design: term frequencies and known signatures accumulate
    as facts arrive, so the weighting adapts to what a given owner
    actually talks about. Feed it the scope's existing facts on startup
    to restore that.

    Args:
        threshold: Weighted overlap above which a fact joins an existing
            belief. See :data:`_DEFAULT_THRESHOLD` for how it was picked.
    """

    def __init__(self, *, threshold: float = _DEFAULT_THRESHOLD) -> None:
        self._threshold = threshold
        self._document_frequency: Counter[str] = Counter()
        self._documents = 0
        self._signatures: dict[str, frozenset[str]] = {}
        self._scopes: dict[str, str] = {}

    def key_for(self, fact: str, *, scope: str = "") -> str:
        """Return the belief this fact belongs to, minting one if new.

        Args:
            fact: The atomic fact sentence.
            scope: Owner / app partition. Facts never compete across
                scopes, whatever they say.

        Returns:
            A stable ``belief_key``.
        """
        terms = signature(fact)
        self._document_frequency.update(terms)
        self._documents += 1

        if terms:
            match = self._best_match(terms, scope)
            if match is not None:
                return match

        key = self._mint(terms, scope, fact)
        self._signatures[key] = terms
        self._scopes[key] = scope
        return key

    def _best_match(self, terms: frozenset[str], scope: str) -> str | None:
        """Highest-scoring known belief in ``scope``, if it clears the bar."""
        best_key: str | None = None
        best_score = self._threshold
        for key, known in self._signatures.items():
            if self._scopes.get(key) != scope:
                continue
            score = self._overlap(terms, known)
            if score >= best_score:
                best_key, best_score = key, score
        return best_key

    def _overlap(self, left: frozenset[str], right: frozenset[str]) -> float:
        """IDF-weighted overlap coefficient of two signatures, in ``[0, 1]``.

        Overlap rather than Jaccard: one fact restating a topic at greater
        length is the same belief, and Jaccard punishes it for the extra
        words.
        """
        if not left or not right:
            return 0.0
        shared = sum(self._idf(term) for term in left & right)
        smaller = min(
            sum(self._idf(term) for term in left),
            sum(self._idf(term) for term in right),
        )
        return shared / smaller if smaller else 0.0

    def _idf(self, term: str) -> float:
        """Inverse document frequency, smoothed, always ``>= 1``."""
        return (
            math.log((self._documents + 1) / (self._document_frequency[term] + 1)) + 1.0
        )

    def _mint(self, terms: frozenset[str], scope: str, fact: str) -> str:
        """A stable key for a belief this scope has not held before.

        Derived from the founding signature, and never revised afterwards:
        widening a belief's signature as members join lets one belief drift
        into the territory of another and swallow it, and that failure is
        both silent and unrecoverable.

        A fact with no topic terms at all ("it is 5") falls back to the
        sentence itself, so two of them get two beliefs. There is no
        evidence they compete, and with no evidence the rule is not to
        link — see the module docstring on which error costs more.
        """
        parts = sorted(terms) if terms else [fact]
        digest = hashlib.sha256(
            "\x00".join([scope, *parts]).encode("utf-8")
        ).hexdigest()[:16]
        return f"{scope}:{digest}" if scope else digest
