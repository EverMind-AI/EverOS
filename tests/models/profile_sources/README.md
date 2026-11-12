# Cluster-triggered profile source model

Run with Java 11+ and the official TLA+ `tla2tools.jar` (checked with TLC 2.19).
From this directory, use a separate output directory for each configuration:

```sh
java -cp /path/to/tla2tools.jar tlc2.TLC -metadir /tmp/profile-member-states -config Member.cfg ProfileSources.tla
```

Repeat with `Index.cfg`, `Fallback.cfg`, and `SkipCleanup.cfg`, each with a distinct
`-metadir`. `Member` must pass. The other three must fail respectively on
`CurrentSourceIncluded`, `NoCompletedDeletedSource`, and
`NoCompletedDeletedSource`.

With TLC 2.19, the member configuration explored 895 distinct states (2,620
generated), reaching depth 17 without an invariant or liveness violation.
The three counterexamples were also observed; rerun all four configurations
when changing the model or its guarantees.

The model bounds are one owner, one cluster, two episodes, and one queued profile
task triggered by episode 2. `Persist`, `Link`, and `Emit` represent the user
pipeline and `trigger_profile_clustering`. `Index` and `Erase` represent the
independent cascade. `Remove`, `Detach`, and `Return` represent the separate
awaited stages of `Deleter`; arbitrary stuttering between them includes failure
and retry delays. `Snapshot` and `Select` represent `extract_user_profile` after
it acquires the profile partition lock.

A deletion completed before the cluster snapshot must not contribute a source.
The current event's member must be selected regardless of index visibility.
Historical supplementation remains best effort through the index. Weak fairness
of selection actions ensures an emitted task eventually finishes selection.

The model does not cover multi-owner discovery, tenant isolation, permanent event
loss, process crash recovery, multiple writers, manual file edits, or cancellation
of an LLM request already using a source. The real-store regressions in
`tests/integration/test_profile_source_selection.py` cover deletion retry including
lost owner discovery. The state machine in
`tests/unit/test_properties/test_profile_source_laws.py` covers source selection
with actual markdown and SQLite, stale indexes, duplicate parents and projects.
This bounded model is not a proof of the whole implementation.
