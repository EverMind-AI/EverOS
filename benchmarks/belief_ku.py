"""Belief-layer benchmark — LongMemEval knowledge-update, with overlays.

Complements ``benchmarks/run.py``. That one measures the whole pipeline on
LoCoMo and needs a server, providers, and an LLM judge. This one measures
exactly one thing — what the memory asserts when a fact changes — and needs
none of them: no server, no API key, no network after the dataset download,
about two seconds.

Why this slice
--------------
``knowledge-update`` is the only part of a public memory benchmark whose
instances are supersessions: a fact stated in session 0, revised in
session 1, gold answer being the revision. All 78 are exactly two sessions
with turn-level ``has_answer`` annotations, so the gold evidence spans can
be fed straight into :class:`~everos.memory.belief.BeliefResolver` with no
extractor and no retriever in the way. A number produced this way is
attributable to the update rule, which is the only claim being made.

    supersession accuracy = P(asserted fact == the session-1 claim)

Arms
----
``clean``
    Both claims on ``user_direct``. Pure recency, so last-write-wins is
    optimal by construction and this arm only shows no regression.
``poison-k``
    Then the **stale** claim replayed k times on a low-trust channel,
    dated after the update, phrased with maximum confidence.
``novel-k``
    Same, but asserting a claim the belief has never seen — the path that
    bypasses the likelihood ratio and goes through admission instead.
``lowtrust-fix``
    The **true** update arrives on the low-trust channel. Measures what
    the provenance ceiling costs, not only what it buys.

The three overlays are constructed on top of the benchmark; they are not
part of LongMemEval and are labelled as such in the output. What they
reuse is its real dialogue text and its real gold supersession.

Usage::

    mkdir -p data && cd data
    wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
    cd ..
    uv run python benchmarks/belief_ku.py --data data/longmemeval_oracle.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path

from everos.core.observability.logging import configure_logging
from everos.memory.belief import (
    BeliefResolver,
    FactObservation,
    ProvenanceTier,
    compose_reliability,
)

_DATE = re.compile(r"(\d{4})/(\d{2})/(\d{2})\s*\([^)]*\)\s*(\d{2}):(\d{2})")
_POISON_PREFIX = "Confirmed, and verified: to be clear, this is final. "

# Content confidence is the extraction layer's job in production — the algo
# library emits it alongside the fact. This rubric stands in for it so the
# benchmark stays offline, and it is deliberately crude: the misses it
# causes are reported rather than tuned away.
_HEDGE = re.compile(
    r"\b(maybe|might|possibly|perhaps|i think|i believe|probably|apparently|"
    r"not sure|unsure|if i recall|iirc|roughly|could be|seems?|i'd say|"
    r"leaning towards?)\b",
    re.IGNORECASE,
)
_FIRM = re.compile(
    r"\b(definitely|confirmed|actually|in fact|to be clear|we decided|final|"
    r"settled|verified|i measured|i tested|correction[:,])\b",
    re.IGNORECASE,
)


def content_confidence(text: str) -> float:
    """Speaker commitment read off epistemic markers, in ``[0.05, 0.98]``."""
    score = 0.70
    if _HEDGE.search(text):
        score -= 0.42
    if _FIRM.search(text):
        score += 0.25
    return max(0.05, min(0.98, score))


@dataclass(frozen=True)
class Instance:
    """One KU instance reduced to its two competing claims."""

    key: str
    stale: str
    fresh: str
    stale_at: dt.datetime
    fresh_at: dt.datetime


def _parse_date(raw: str) -> dt.datetime:
    match = _DATE.match(raw)
    if match is None:
        return dt.datetime(1970, 1, 1, tzinfo=dt.UTC)
    year, month, day, hour, minute = (int(part) for part in match.groups())
    return dt.datetime(year, month, day, hour, minute, tzinfo=dt.UTC)


def load_instances(path: Path) -> list[Instance]:
    """Read the KU instances that carry evidence spans in both sessions."""
    out: list[Instance] = []
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
        out.append(
            Instance(
                key=item["question_id"],
                stale=" ".join(spans[0])[:240],
                fresh=" ".join(spans[1])[:240],
                stale_at=_parse_date(item["haystack_dates"][0]),
                fresh_at=_parse_date(item["haystack_dates"][1]),
            )
        )
    return out


def build_stream(
    instance: Instance,
    arm: str,
    *,
    repetitions: int,
    poison_tier: ProvenanceTier,
    novel: str,
) -> list[FactObservation]:
    """Observations for one instance under one arm, in arrival order."""

    def observation(
        fact: str, at: dt.datetime, tier: ProvenanceTier, text: str | None = None
    ) -> FactObservation:
        return FactObservation(
            belief_key=instance.key,
            fact=fact,
            observed_at=at,
            tier=tier,
            content_confidence=content_confidence(text or fact),
            source_id=f"{instance.key}:{tier.value}",
        )

    fresh_tier = (
        poison_tier if arm == "lowtrust-fix" else ProvenanceTier.USER_DIRECT
    )
    stream = [
        observation(instance.stale, instance.stale_at, ProvenanceTier.USER_DIRECT),
        observation(instance.fresh, instance.fresh_at, fresh_tier),
    ]

    payload = None
    if arm.startswith("poison"):
        payload = instance.stale
    elif arm.startswith("novel"):
        payload = novel
    if payload is not None:
        for step in range(1, repetitions + 1):
            stream.append(
                observation(
                    payload,
                    instance.fresh_at + dt.timedelta(days=step),
                    poison_tier,
                    _POISON_PREFIX + payload,
                )
            )
    return stream


def score(
    instances: list[Instance],
    arm: str,
    policy: str,
    *,
    repetitions: int,
    poison_tier: ProvenanceTier,
) -> tuple[float, float]:
    """Supersession accuracy and mean asserted probability for one cell."""
    hits = 0
    confidence_total = 0.0
    for index, instance in enumerate(instances):
        novel = instances[(index + 1) % len(instances)].stale
        stream = build_stream(
            instance,
            arm,
            repetitions=repetitions,
            poison_tier=poison_tier,
            novel=novel,
        )
        if policy == "lww":
            asserted, probability = stream[-1].fact, 1.0
        else:
            resolver = BeliefResolver()
            for item in stream:
                resolver.observe(item)
            verdict = resolver.verdict(instance.key)
            asserted, probability = verdict.fact, verdict.probability
        hits += asserted == instance.fresh
        confidence_total += probability
    total = len(instances)
    return 100.0 * hits / total, confidence_total / total


def main() -> int:
    """Run the benchmark and print the arm-by-policy table."""
    # One line per revision is the right default for a memory runtime and
    # the wrong one for a benchmark that performs ~1,700 of them.
    configure_logging("WARNING")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/longmemeval_oracle.json")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--poison-tier",
        default=ProvenanceTier.WEB_FETCH.value,
        choices=[tier.value for tier in ProvenanceTier],
    )
    args = parser.parse_args()

    path = Path(args.data)
    if not path.exists():
        print(f"missing {path} — see the module docstring for the download command")
        return 2

    instances = load_instances(path)
    poison_tier = ProvenanceTier(args.poison_tier)
    ceiling = compose_reliability(poison_tier, 1.0)
    arms = [
        "clean",
        f"poison-{args.repetitions}",
        f"novel-{args.repetitions}",
        "lowtrust-fix",
    ]
    policies = ["lww", "belief"]

    print(
        f"LongMemEval knowledge-update — {len(instances)} instances with gold "
        f"evidence spans in both sessions"
    )
    print(
        f"poison tier `{poison_tier.value}` (ceiling {ceiling:.2f}), "
        f"k={args.repetitions}\n"
    )
    header = f"{'arm':<16}" + "".join(f"{policy:>14}" for policy in policies)
    print(header)
    print("-" * len(header))
    for arm in arms:
        row = f"{arm:<16}"
        confidences = []
        for policy in policies:
            accuracy, confidence = score(
                instances,
                arm,
                policy,
                repetitions=args.repetitions,
                poison_tier=poison_tier,
            )
            row += f"{accuracy:>13.1f}%"
            confidences.append(confidence)
        print(row)

    print(
        "\nclean is a recency test — last-write-wins is optimal there by "
        "construction.\nThe overlays are the discriminating arms, and they are "
        "constructed: they are\nnot part of LongMemEval."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
