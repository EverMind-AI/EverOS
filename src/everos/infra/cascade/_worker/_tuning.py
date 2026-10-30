"""Tuning constants for the cascade worker, each with the measurement behind it.

The ``DEFAULT_*`` values are constructor defaults (four of them are exposed as
``[cascade]`` settings); the rest are fixed thresholds and deadlines that are
deliberately not configurable — each docstring says why it has the value it
has, so a change starts from the evidence rather than from a guess.
"""

from __future__ import annotations

DEFAULT_BATCH_SIZE = 50
DEFAULT_HANDLER_CONCURRENCY = 4
"""Maximum handler executions in flight on one worker.

The bound is per worker, not per :meth:`CascadeWorker.drain_once` call: a
server drains from two places at once -- the background loop and whatever
calls ``drain_fully`` (``POST /cascade/quiesce`` does, before ``stop``) --
and a semaphore built per call would let those two overlap at twice this
number, in exactly the backlog-plus-quiesce case the bound exists for.

A daily-log handler first resolves existing rows by ``md_path``. On fragmented
Lance tables that lookup is a native flat scan: 50 simultaneous handlers grew
the production-data replay past 8.29 GiB and did not finish in 80 seconds,
while 1/2/4/8-way peaks were about 0.26/0.61/0.98/1.73 GiB. Keeping the claim
batch at 50 preserves queue semantics; bounding only execution prevents native
queries that outlive Python cancellation from multiplying memory and file
handles after a timeout.

A slot is held across :meth:`CascadeWorker._process_one`, inline retry
backoff included, so a row failing against a down provider parks one slot for
``retry_backoff * max_retry * (max_retry + 1) / 2`` seconds (12s at the
defaults) with no native query in flight. That is deliberate -- backing off
harder against a provider that is already failing is the behaviour we want --
but it is why a fully-failing batch drains slower than it did unbounded.
"""
DEFAULT_MAX_RETRY = 3
DEFAULT_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0
DEFAULT_OPTIMIZE_MIN_INTERVAL_SECONDS = 60.0
"""Throttle between ``optimize()`` runs on one kind.

Not a visibility delay. A row is searchable the moment its upsert commits —
LanceDB flat-scans the unindexed tail, and that covers BM25 as well as vector
and scalar (verified: a row with ``num_unindexed_rows=1`` is returned by
``nearest_to_text``). What ``optimize`` buys is folding that row out of the tail
and into the index, i.e. speed. Sparse writes do not even wait: the scheduler
uses ``max(0, interval - elapsed)``, so when the last run is already older than
the interval the next one starts immediately.

It is also the **ceiling on index-directory growth**, which is the reason to
think twice before lowering it. Every beat leaves new ``_indices/<uuid>/``
dirs behind, and lance never removes the empty ones, so the accrual rate is
capped by this interval rather than by write volume: past roughly one write per
table per interval the beats coalesce and writing harder adds nothing. Measured
at a 10s ceiling: ~127k dirs/day across three active tables, ~1.8M at the
~14-day reclaim horizon (~7GB of empty dirs — the horizon is the file wait
plus the 7-day age gate, see ``_HUSK_MIN_AGE_SECONDS`` in the lancedb
repository module). The rate scales with the interval, so the 60s default
costs a sixth of that: ~21k dirs/day, ~300k at the horizon (~1.2GB).

Why 60s rather than 10s: a 10h soak at sustained load showed the index file
count falling the whole way (631 -> 489 across quarters) with search p50 flat
or improving, so the shorter cooldown was buying speed the workload never
needed while paying six times the directory accrual. The cost of raising it is
a longer flat-scanned tail between beats, which that same soak bounds: keyword
p50 stayed at 0.02s and vector at 0.6s with the tail one beat deep.
"""
DEFAULT_OPTIMIZE_HEARTBEAT_SECONDS = 60.0
OPTIMIZE_FAILURE_ALERT_THRESHOLD = 5
"""Consecutive **non-benign** ``optimize()`` failures (per kind) before
the log escalates ``warning``→``error``, a fallback rebuild is triggered
(:meth:`IndexMaintainer._run_rebuild_once`), and
:meth:`CascadeWorker.health` reports the kind degraded. A benign commit
conflict (lost concurrency race, either beat) is expected under churn, logged
at ``debug``, and does **not** count — otherwise the streak pins high on a
busy table and drowns the real signal (the disk-bloat failure mode behind
lance-format/lance#7653).

Prune that stops succeeding is **not** detected here: a lost race is
excluded by design, and an intervening light-beat success resets the
streak anyway. That failure mode belongs to the per-kind prune-staleness
signal (:data:`PRUNE_STALE_FACTOR`), which fires on the symptom (nothing
reclaimed for 3 cadences) rather than on a particular exception."""
DRAIN_FAILURE_ALERT_THRESHOLD = 3
"""Consecutive :meth:`CascadeWorker.drain_once` exceptions at or above which
:meth:`CascadeWorker.health` reports degraded — the md → LanceDB
projection is not making progress, so accepted writes are not indexed."""
PRUNE_STALE_FACTOR = 3.0
"""Multiple of the prune interval without any successful prune before
:meth:`CascadeWorker.health` reports degraded. The primary disk-bloat
signal — fires whether the cause is lost commit-conflict races, a stuck
heavy beat, or a wedged worker, without depending on a particular
exception showing up."""
MAX_TOTAL_RETRIES = 12
"""Total retry budget across scanner re-enqueue cycles.

Each cycle: worker retries up to ``MAX_RETRY`` (default 3) inline;
scanner re-enqueues on the next 30s sweep; retry_count accumulates.
12 ≈ 4 scanner cycles × 3 inline retries ≈ 2 minutes of retrying.

Once exhausted, ``mark_failed(retryable=False)`` so the reconciler
stops re-enqueuing. Recover via ``cascade fix --apply`` (resets
retry_count) or editing the md (mtime change resets retry_count)."""

MAINTENANCE_TASK_TIMEOUT_SECONDS = 180.0
"""Last-resort deadline on one whole maintenance call (compact or prune).

The repo bounds its own critical sections, but this scheduler is the thing that
breaks if a call never returns at all: it runs one task per kind and skips a
kind whose task is still in flight, so a single non-returning await parks that
kind's maintenance forever — silently, since nothing failed. A soak run hit
exactly that (one table stopped pruning for 13 minutes with zero failure logs
while its siblings pruned normally), through an await that sat outside the
repo's deadline. Generous enough never to fire on a healthy beat (prune's own
budget is 60s), tight enough that a hang costs one cadence, not forever.
"""

REBUILD_CONFLICT_BACKOFFS_SECONDS = (600.0, 1800.0, 10800.0)
"""Delay before each re-attempt at a kind's index rebuild: 10min, 30min, 3h.

Only a lost commit race is retried — lance marks it ``Retryable`` and the table
is fine; a concurrent writer in another process simply won the manifest. Any
other failure defers to the next sweep, because retrying a real error just
burns the write lock.

Scaled to the 12h rebuild cadence, not to the conflict. A seconds-long backoff
would spend the whole budget inside one contention window and then leave the
kind unindexed for a full cadence; spreading the attempts over hours means the
retries land in genuinely different load conditions. The schedule is a
*deadline recorded on the kind*, never a sleep:
:meth:`IndexMaintainer.rebuild_loop` walks kinds sequentially, so sleeping
here would park every later kind behind this one (7 kinds x 3h would outlast
the cadence itself)."""

REBUILD_LOOP_TICK_SECONDS = 60.0
"""How often the rebuild loop wakes to pick up due retries between sweeps.
Cheap: a dict scan per tick, and only kinds with a due deadline do work."""

LOOP_RESTART_BACKOFF_SECONDS = (5.0, 15.0, 45.0)
"""Backoff before each restart of a background loop that raised.

The three long-lived loops (drain / heartbeat / rebuild) are plain
``create_task`` coroutines. Without supervision, one uncaught exception ends
that loop **permanently and silently**: nothing restarts it, and because
the worker's task attribute keeps a strong reference the interpreter never prints the
"Task exception was never retrieved" warning either (that fires on GC). The
loop's job simply stops happening. So each loop runs under
:meth:`LoopSupervisor.supervise`, which logs, waits, and restarts.

Escalating rather than fixed: a transient cause (a closing event loop, a
momentarily unavailable table) clears within seconds, while a deterministic
one would otherwise spin. After the last entry is used the worker asks the
process to exit (see :func:`request_process_exit`) — a server
whose projection pipeline is permanently dead should not keep serving as if
healthy.

The budget is **per incident, not per process lifetime**: a body that ran at
least :data:`LOOP_STABLE_RUN_SECONDS` before crashing gets a full budget
again. Without that reset, rare *independent* transients — one every few
days, each recovered by a single restart — would still spend the budget one
by one and SIGTERM the server weeks later on the 4th, which punishes exactly
the case supervision exists to absorb. This mirrors how process supervisors
count restarts within a window (systemd ``StartLimitIntervalSec``, Erlang
``max_restarts`` per ``max_seconds``) rather than forever.
"""

LOOP_STABLE_RUN_SECONDS = 60.0
"""A supervised loop body that ran at least this long before raising is
treated as a fresh incident (restart budget resets). Sized well above the
escalation ladder's total (5+15+45 = 65s of *backoff*, but each attempt's
run time counts from body start): a deterministic crash-on-startup fails in
milliseconds and cannot reach it, while a loop that did an hour of honest
work before hitting a transient obviously should not inherit stale strikes."""

DEFAULT_OPTIMIZE_REBUILD_INTERVAL_SECONDS = 12 * 60 * 60.0
"""How often (per kind) to do a full ``drop_index + create_index`` rebuild.

This is the **only** application-level mechanism we have to bound the
active index UUID / segment count growth — see
:meth:`LanceRepoBase.rebuild_indexes` for the full provenance: Rust
``OptimizeOptions.num_indices_to_merge`` is the right knob but
``lancedb.AsyncTable.optimize()`` does not expose it (verified on
lancedb main 2026-05-28), and on the embedded ``lance crate 4.0`` the
merge behaviour itself is broken so even calling Lance directly
wouldn't help.

12 hours is a conservative pick: rebuild cost is ~0.3s per 50k rows
× indexed columns (measured locally), so even a small EverOS
deployment can absorb it without scheduling around peaks. Smaller
intervals work fine functionally; we just don't need them — under
realistic single-user / small-team write rates 12h keeps active UUIDs
bounded well below any FD ceiling. Tune via the constructor argument.

**Remove this scheduler** once lancedb exposes ``num_indices_to_merge``
on the async Python API and the embedded lance crate ships the
working merge implementation; ``optimize(num_indices_to_merge=1)``
in the regular hot path will do the same job for ~free.
"""

DEFAULT_OPTIMIZE_PRUNE_INTERVAL_SECONDS = 300.0
"""**Cadence** — how often (per kind) the heavy write-locked
:meth:`LanceRepoBase.prune` beat runs, vs the light lock-free
:meth:`~LanceRepoBase.optimize` compaction on every other tick.

prune holds the per-table write lock (brief same-table write stall,
~seconds on a churned table), so it runs on this slow beat rather than
every throttle tick. This is the *frequency* of reclamation; how much
history each prune keeps is a separate knob — see
:data:`DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS`.

Does **not** shrink active index internals (FTS ``part_N`` count or
vector index UUID count): those only collapse via ``drop_index +
create_index``, which is intentionally out of scope here.
"""

DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS = 60.0
"""**Retention window** — ``cleanup_older_than`` passed to prune: files
belonging to dataset versions replaced more than this long ago become
eligible for physical deletion.

Decoupled from (and much shorter than) the prune *cadence*: prune runs
under the per-table write lock, so no concurrent writer can be
referencing an old version, and the only thing the window must outlive
is an **in-flight read** (a ``/search`` holding a version reference) —
sub-second to a few seconds. 60s is comfortably safe.

Why short matters: under sustained churn each compaction leaves a
full-table-sized *superseded* fragment behind; the retention window is
how long those pile up before reclamation. The storage soak showed a
300s window × the churn rate retaining ~24 full-table copies (a
transient ~15G/table peak that reclaimed to ~625MB live once churn
eased). Shrinking the window to 60s cuts that transient footprint ~5×
without changing the reclaimed floor. Tune via the constructor."""

PRUNE_STALE_SECONDS_ALERT = PRUNE_STALE_FACTOR * DEFAULT_OPTIMIZE_PRUNE_INTERVAL_SECONDS
"""Absolute prune-staleness alert threshold (seconds) = 900s under the
shipped 300s prune cadence (three missed heavy beats)."""
