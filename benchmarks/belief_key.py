"""Belief-key derivation benchmark — does the grouping find real conflicts?

The resolver arbitrates between facts that share a ``belief_key``. This
measures whether :class:`~everos.memory.belief.BeliefKeyer` puts the right
facts together, and — the number that actually governs the threshold —
how often it puts the wrong ones together.

Labels come free from LongMemEval ``knowledge-update``: the two evidence
spans of one instance are *by construction* two readings of the same
attribute, and spans from different instances are not. That gives 70
positive pairs and every cross-instance pair as a negative, with no
annotation of my own.

Two things are reported, and they are not symmetric:

``linked``
    Share of true pairs the keyer groups. A miss leaves two contradicting
    facts unarbitrated — which is where EverOS is today, so nothing is
    lost that was not already lost.
``false links``
    Share of unrelated pairs it groups. Each one lets an irrelevant fact
    suppress a true one. This is the error that costs something, so the
    default threshold is set from this column.

The last section runs the full KU supersession benchmark with *derived*
keys instead of the oracle key, which is the only number that says what
the layer would do on real data.

Usage::

    uv run python benchmarks/belief_key.py --data data/longmemeval_oracle.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure the repo root is on sys.path when run as a script (see run.py)
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from benchmarks.belief_ku import Instance, build_stream, load_instances  # noqa: E402
from everos.core.observability.logging import configure_logging  # noqa: E402
from everos.memory.belief import BeliefKeyer, BeliefResolver, ProvenanceTier  # noqa: E402
from everos.memory.belief.keying import _DEFAULT_THRESHOLD  # noqa: E402

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class Pair:
    """Two statements of one attribute, from the same KU instance."""

    question: str
    stale: str
    fresh: str


def _salient_sentence(turn: str, question: str) -> str:
    """The sentence in a turn that the question is asking about.

    Stands in for extraction: the algo layer would emit one atomic fact per
    claim, and this picks the claim rather than the whole turn of chat
    around it. Selection is by question overlap, which is symmetric between
    the two sessions and so does not favour either reading.
    """
    wanted = set(_WORD.findall(question.lower()))
    best, best_score = turn, -1
    for sentence in _SENTENCE.split(turn):
        candidate = sentence.strip()
        if len(candidate) < 10:
            continue
        score = len(wanted & set(_WORD.findall(candidate.lower())))
        if score > best_score:
            best, best_score = candidate, score
    return best


def load_pairs(path: Path) -> list[Pair]:
    """KU instances reduced to the two competing claims, one sentence each."""
    pairs: list[Pair] = []
    for item in json.loads(path.read_text()):
        if item["question_type"] != "knowledge-update":
            continue
        sessions = item["haystack_sessions"]
        if len(sessions) != 2:
            continue
        spans = [
            [turn["content"] for turn in session if turn.get("has_answer")]
            for session in sessions
        ]
        if not spans[0] or not spans[1]:
            continue
        pairs.append(
            Pair(
                question=item["question"],
                stale=_salient_sentence(spans[0][-1], item["question"]),
                fresh=_salient_sentence(spans[1][-1], item["question"]),
            )
        )
    return pairs


def _fitted_keyer(pairs: list[Pair], threshold: float) -> BeliefKeyer:
    """A keyer whose term weights have seen the corpus, holding no beliefs.

    IDF needs a corpus. Feeding the facts in per scope means each belief is
    minted in isolation while the weighting still knows which terms are
    common, which is what a warm store looks like.
    """
    keyer = BeliefKeyer(threshold=threshold)
    for index, pair in enumerate(pairs):
        keyer.key_for(pair.stale, scope=f"warm_{index}")
        keyer.key_for(pair.fresh, scope=f"warm_{index}")
    return keyer


def measure_linking(
    pairs: list[Pair], threshold: float, *, samples: int = 4000
) -> tuple[float, float]:
    """Share of true pairs linked, and of unrelated pairs falsely linked."""
    keyer = _fitted_keyer(pairs, threshold)

    linked = 0
    for index, pair in enumerate(pairs):
        scope = f"eval_{index}"
        if keyer.key_for(pair.stale, scope=scope) == keyer.key_for(
            pair.fresh, scope=scope
        ):
            linked += 1

    rng = random.Random(0)
    false_links = 0
    for draw in range(samples):
        left, right = rng.sample(range(len(pairs)), 2)
        scope = f"neg_{draw}"
        if keyer.key_for(pairs[left].fresh, scope=scope) == keyer.key_for(
            pairs[right].fresh, scope=scope
        ):
            false_links += 1

    return 100.0 * linked / len(pairs), 100.0 * false_links / samples


def measure_end_to_end(path: Path, threshold: float) -> tuple[float, float]:
    """Supersession accuracy with derived keys, clean and under attack.

    Same protocol and the same claim sentences as the linking measurement
    above — the only difference from ``belief_ku.py`` is that nothing tells
    the resolver which facts compete.

    An instance counts as correct when the memory asserts the updated
    claim **and no longer asserts the stale one**. Both halves are needed:
    a keyer that links nothing scores a free pass on the first half while
    leaving the contradiction exactly where it found it.
    """
    pairs = load_pairs(path)
    instances = [
        _instance(pair, source, index)
        for index, (pair, source) in enumerate(
            zip(pairs, load_instances(path), strict=True)
        )
    ]
    results = []
    for arm in ("clean", "poison-5"):
        hits = 0
        for index, instance in enumerate(instances):
            keyer = _fitted_keyer(pairs, threshold)
            resolver = BeliefResolver()
            stream = build_stream(
                instance,
                arm,
                repetitions=5,
                poison_tier=ProvenanceTier.WEB_FETCH,
                novel=instances[(index + 1) % len(instances)].stale,
            )
            keys = set()
            for observation in stream:
                key = keyer.key_for(observation.fact, scope=instance.key)
                keys.add(key)
                resolver.observe(observation.model_copy(update={"belief_key": key}))
            asserted = {resolver.verdict(key).fact for key in keys}
            hits += instance.fresh in asserted and instance.stale not in asserted
        results.append(100.0 * hits / len(instances))
    return results[0], results[1]


def _instance(pair: Pair, source: Instance, index: int) -> Instance:
    """A ``belief_ku`` instance carrying the claim sentences, not whole turns."""
    return Instance(
        key=f"e2e_{index}",
        stale=pair.stale,
        fresh=pair.fresh,
        stale_at=source.stale_at,
        fresh_at=source.fresh_at,
    )


def main() -> int:
    """Run both measurements and print the threshold curve."""
    configure_logging("WARNING")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/longmemeval_oracle.json")
    args = parser.parse_args()

    path = Path(args.data)
    if not path.exists():
        print(f"missing {path} — see the module docstring for the download command")
        return 2

    pairs = load_pairs(path)
    print(f"LongMemEval knowledge-update — {len(pairs)} labelled pairs\n")
    print(f"{'threshold':>10}{'linked':>10}{'false links':>14}")
    print("-" * 34)
    for threshold in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
        linked, false_links = measure_linking(pairs, threshold)
        marker = "  <- default" if threshold == _DEFAULT_THRESHOLD else ""
        print(f"{threshold:>10.2f}{linked:>9.1f}%{false_links:>13.2f}%{marker}")

    print(
        "\nA miss leaves two facts unarbitrated, which is today's behaviour.\n"
        "A false link lets an unrelated fact suppress a true one. The default\n"
        "is set from the right-hand column."
    )

    clean, poisoned = measure_end_to_end(path, _DEFAULT_THRESHOLD)
    print("\nSupersession with derived keys (nothing tells it what competes):")
    print(f"  clean      {clean:>6.1f}%")
    print(f"  poison-5   {poisoned:>6.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
