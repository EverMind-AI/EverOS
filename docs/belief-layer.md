# Belief layer — probabilistic conflict resolution over atomic facts

Status: proposal, domain layer implemented and benchmarked, persistence and
wiring not yet built.

## The gap

EverOS stores an atomic fact as a sentence with a timestamp, an owner, and a
pointer to its source MemCell. There is no mechanism anywhere in the write
path that notices two facts disagree.

`deprecated_by` looks like one and is not: Reflection sets it when it merges
*fragmented* cluster members into a consolidated episode (Select → Merge →
Re-extract → Deprecate). It is a consolidation marker, not a contradiction
verdict. A grep for dedup / contradiction / conflict handling over `src/`
returns SQLite `ON CONFLICT` clauses and nothing else.

So when a user says "I use 6 ounces of water per tablespoon" in March and
"I've switched to 5 ounces" in June, both facts are live, both are indexed,
both match the same query, and both are handed to the model. Which one comes
back first is a function of BM25 scoring and embedding distance — neither of
which knows which is true.

Two things are missing, and they are the same thing:

1. **arbitration** — which of two mutually exclusive facts does the memory
   assert?
2. **calibration** — how sure is it, and can a caller act on that number?

`confidence` exists on `AgentSkill` today (LLM-emitted, alongside
`maturity_score`) but not on facts, and an LLM-emitted scalar is not a
probability: nothing normalises it across candidates and nothing updates it
when new evidence arrives.

## What this adds

`everos.memory.belief` — a pure domain module, no I/O, no LLM.

Mutually exclusive facts share a `belief_key` and hold a categorical
distribution. Each observation carries the channel it arrived on; the update
is Bayesian with the evidential weight of an observation capped by that
channel's reliability ceiling.

```python
from everos.memory.belief import BeliefResolver, FactObservation, ProvenanceTier

resolver = BeliefResolver()
resolver.observe(FactObservation(
    belief_key="user_42:coffee_ratio",
    fact="6 ounces of water per tablespoon",
    observed_at=march,
    tier=ProvenanceTier.USER_DIRECT,
    content_confidence=0.9,
))
resolver.observe(FactObservation(..., fact="5 ounces...", observed_at=june))

verdict = resolver.verdict("user_42:coffee_ratio")
# fact="5 ounces...", probability=0.7, superseded=["6 ounces..."]
if verdict.is_uncertain:
    ...  # ask rather than assert
```

Three design commitments, each with a test that fails if it breaks.

**Trust is a property of the channel, never of the text.** Reliability is
`min(channel ceiling, content confidence)`. Content confidence may lower
trust within the ceiling, never raise it. Reading trust off the sentence
instead hands it to whoever writes the sentence: "Confirmed, verified, this
is final" scores near 1.0 on any hedging rubric.

**A channel at or below r = 0.5 cannot change what the memory asserts,
at any volume.** The likelihood ratio clamps to 1 at that pivot — a source
you distrust asserting X is not evidence against X, it is simply not
evidence. Clamping rather than inverting also fixes the pivot independently
of how many candidates the belief holds.

**A single trusted correction can still supersede a single trusted claim.**
This is the property that is easy to lose, and losing it is silent. See
below.

## The one subtle part

Admission and evidence are different operations and they need separate
gates.

A candidate the belief has never seen cannot be reweighted, only admitted,
and admission happens *before* any likelihood test runs. So the entry mass
is a second, hidden trust knob — and it controls both properties at once:

| entry mass | supersession accuracy | untrusted novel claim wins |
|---|---|---|
| 0.02 | 0.0% | 0% |
| 0.20 | 28.6% | 28.6% |
| 0.30 | 72.9% | 72.9% |
| 0.40 | 94.3% | 90.0% |
| 0.50 | 100.0% | 100.0% |

No setting satisfies both. At the small end the memory is immune because it
never learns anything; at the large end an untrusted channel can install any
belief it likes. A layer tuned at either end is worse than no layer at all,
and it fails quietly.

`entry_mass()` resolves this by conditioning admission on the same pivot the
likelihood ratio uses: above it a trusted first sighting enters at a mass
proportional to the channel's reliability, below it the candidate is
recorded but inert. It is still retrievable and still auditable — it simply
cannot win.

A related detail: a first observation admitted at reliability `r` leaves the
belief at `p ≈ r`, not at 1.0. The residual sits on a reserved `UNKNOWN`
candidate. Without it, a fact heard exactly once normalises to certainty and
every number in the store reads as 1.0.

## Results

`benchmarks/belief_ku.py`, LongMemEval `knowledge-update`: 70 instances that
are two-session supersessions with turn-level gold evidence spans. The spans
feed the resolver directly — no extractor, no retriever, no LLM — so the
number is attributable to the update rule. Runs offline in about two
seconds.

| arm | last-write-wins | belief |
|---|---|---|
| clean | **100.0%** | 78.6% |
| poison-5 (stale claim replayed on `web_fetch`) | 0.0% | **78.6%** |
| novel-5 (unseen claim asserted on `web_fetch`) | 0.0% | **78.6%** |
| lowtrust-fix (true update on `web_fetch`) | 100.0% | 0.0% |

Read the rows, not the cells.

- `clean` is a pure recency test — last-write-wins is optimal there by
  construction, and this arm exists to show the belief layer does not
  regress on the ordinary case.
- The identical 78.6% across the three arms is the point: the attacks have
  **zero** effect. Not "reduced" — the asserted fact is the same one, with
  the same probability, whether or not the attacker is there.
- The 21.4% gap on `clean` is fully accounted for: 15 of 70 updates are
  phrased with hedges ("I'd say the marketing campaign is the priority"),
  the stand-in content-confidence rubric drops them below the pivot, and
  admission is blocked. `15/70 = 21.4%` exactly. This is an extraction
  quality number, not an arbitration number — the algo layer emits a real
  confidence in production and the rubric in the benchmark is a placeholder
  for it.
- `lowtrust-fix` = 0.0% is the honest cost. A ceiling that stops a bad
  correction on a low-trust channel stops a good one identically. There is
  no setting that gets both; this row is the price of the other three.

## Deriving `belief_key`

The resolver arbitrates between facts sharing a key, and nothing in EverOS
produces one — an atomic fact is an undecomposed sentence with no
`(subject, attribute)` to group on.

`keying.BeliefKeyer` groups on the observation that competing facts are
*about the same thing while differing in the value*. The signature drops
value-bearing tokens and keeps the topic, so "pre-approved for $350,000
from Wells Fargo" and "pre-approved for $400,000 from Wells Fargo" collapse
to the same signature. Terms are IDF-weighted against what the scope has
already said; without that, "really" and "looking" count as much as
"pre-approved", and the false-link rate is roughly ten times worse.

**The errors are not symmetric, and that sets the threshold.** A missed
link leaves two contradicting facts unarbitrated — exactly today's
behaviour, so nothing is lost. A false link declares two unrelated facts
mutually exclusive and lets one suppress the other. Partial arbitration is
worth having; wrong arbitration is not.

`benchmarks/belief_key.py` takes its labels from KU pair membership: the
two evidence spans of an instance are two readings of one attribute by
construction, and spans from different instances are not.

| threshold | true pairs linked | unrelated pairs linked |
|---|---|---|
| 0.15 | 90.0% | 6.03% |
| 0.20 | 85.7% | 1.60% |
| **0.25** (default) | **81.4%** | **0.45%** |
| 0.30 | 68.6% | 0.12% |
| 0.40 | 48.6% | 0.05% |

Running the supersession benchmark again with *derived* keys — nothing
telling the resolver what competes — gives **78.6% clean and 78.6% under
`poison-5`**. An instance counts as correct only when the memory asserts
the update *and* has stopped asserting the stale claim; both halves are
needed, since a keyer that links nothing would otherwise score full marks
while leaving every contradiction exactly where it found it.

That figure coincides with the oracle-key run without being the same
result. There the entire loss was hedged updates falling below the pivot;
here the claim sentences are shorter and only 5.7% hedge, while 18.6% fail
to link. The dominant error moved from extraction to keying.

This is a lexical stand-in and should not survive contact with production.
EverOS already embeds every fact, and matching on those embeddings is the
better implementation — `BeliefKeyer` is written so that swap is a
constructor argument. The lexical version exists to establish that the
grouping problem is tractable at all before anyone spends embedding calls
on it.

## What is not built

**Persistence.** `BeliefState` and `BeliefRevision` are derived state and
belong in SQLite (`~/.everos/.index/sqlite/system.db`), not in the LanceDB
fact table — no index migration, and the states rebuild from the revision
log. Needs a repo + an alembic revision.

**Search integration.** `search/filters.py` already excludes
`deprecated_by IS NOT NULL`; the analogous move is to rank or filter by
posterior and to surface `probability` on the recall DTO so an answering
model can see how sure the memory is.

**Tier assignment.** `ProvenanceTier` is an enum with a ceiling table.
Mapping EverOS's existing scoping (`owner_type`, `app_id`, `session_id`,
`sender_ids`) onto tiers should be config, in `everos.toml`, in the operator's
hands — not inferred, and never from anything an agent can write.

**Calibration.** The layer reports probabilities. Whether they are *true*
probabilities is an empirical question that needs outcomes to score against
(Brier / ECE over resolved beliefs). Until that is measured, `entropy` is
the honest thing to show a caller and `probability` should be read as a
ranking, not a frequency.
