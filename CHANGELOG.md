# Changelog

All notable changes to **EverOS** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.3] - 2026-08-07

**Background maintenance that fails loudly instead of quietly.** A soak run on
1.2.2 found a table that had stopped reclaiming disk for 100 minutes while
`/health` stayed green — nothing had *failed*, a call had simply never returned,
and every signal was built from failure counters. Auditing for that shape turned
up six more places it could happen: reads with no deadline (which stop the whole
md to LanceDB projection, not just one table), background loops that die
permanently on one exception with no log at all, an alert counter reset by the
remediation it triggers. All of them are now bounded, and a stall that does
happen names the table it happened to. Alongside that, agent-skill extraction is
rescued from a retry-then-dead-letter loop, keyword search no longer returns 500
during an index rebuild, and the maintenance cadences moved into settings.

### Fixed

- **Agent skill extraction is no longer stuck in a retry-then-dead-letter
  loop.** Target case data now travels on `SkillClusterUpdated` and existing
  skills for the cluster are read from markdown (strong-consistency), so the
  strategy never races cascade indexing. Prior to this fix, running a fresh
  agent trajectory produced zero `SKILL.md` files — `.skills/` did not exist.
  **The related stale-index clobber is fully closed only for clusters at or
  below `MAX_SKILLS_IN_PROMPT` (10).** Above it, markdown still supplies the
  candidate set but LanceDB orders it, and the skill a lagging index omits is
  by definition the one written most recently — the one most likely to need
  `update` — so it can be ranked out of the prompt and re-added instead. The
  window is narrow (it needs a cluster over 10 skills *and* an index that has
  not caught up) and the consequence is the pre-existing full-replace, not a
  new failure mode.
- **`POST /api/v2/ome/trigger` no longer masks strategy state.** The `status`
  field now distinguishes `not_dispatched` (all dispatch gates rejected the
  strategy — usually a missing `"force": true`) from `ok` (dispatched and
  settled). The new `runs` field surfaces dead-lettered strategy runs that
  were previously invisible to the caller. **If your client matches
  `status` exhaustively (Python `Literal`, TypeScript union), add a
  `not_dispatched` branch.**
- **Agentic search on agent memory now uses the skill-shaped rerank
  passage.** The cross-encoder previously saw only the raw `description`
  field instead of the `name + description + skill instruction` triple that
  the HYBRID lane uses. A skill with empty `description` (a legal everalgo
  output — see `everalgo/agent_memory/skill_ops.py:294`) no longer causes
  HTTP 500 during the LLM sufficiency check.
- **OME strategy retries now back off between attempts.** A retry-class error
  (e.g. waiting on eventually-consistent state) previously exhausted its
  `max_retries` budget in milliseconds; the loop now sleeps
  `min(base * 2**(attempt-1), cap)` plus up to `jitter` seconds
  (defaults: `1s` base / `10s` cap / `0.5s` jitter — code-only defaults,
  not currently exposed via `everos.toml` or `ome.toml`). **`engine_sem` is
  now held per attempt rather than across the whole retry chain**, so the
  backoff sleep does not occupy a concurrency slot. The cap bounds
  concurrent strategy *work* — LLM calls, embeddings, storage IO — and a
  coroutine waiting to retry consumes none of it; holding the slot would
  have turned a partial outage into a total stall, since enough
  simultaneously-failing runs park every one of the
  `max_concurrent_runs` slots in `asyncio.sleep` and starve strategies that
  would have succeeded. Backpressure on failing work is intended;
  backpressure on everything else is not.
- **Path-traversal hardening for LLM-generated agent-skill names (CWE-22).**
  `AgentSkillFrontmatter.name` comes straight from LLM output
  (`extract_agent_skill`) and was concatenated unsanitized into the
  `skills/skill_<name>/` directory segment on both the write and read
  paths; given a sufficiently long `../` prefix, the write target could
  escape the memory root. This is the same class of defect previously
  fixed for knowledge-upload titles/categories (see `knowledge_writer.py`
  in an earlier 1.2.x). The sanitizer is now a single shared helper
  (`everos.core.persistence.markdown.sanitize_dirname`) used by both
  `KnowledgeWriter` and the new `SkillPathMixin.skill_dir_name()` /
  `sanitize_skill_name()`, instead of two independently maintained copies.
  `extract_agent_skill` now sanitizes the LLM-emitted name *before*
  constructing `AgentSkillFrontmatter`, so **`AgentSkillFrontmatter.name`
  and the LanceDB `agent_skill` primary key now hold the sanitized name**
  (spaces become `_`, characters outside `[\w\-.]` are dropped, capped at
  50 chars), not the raw LLM output — a user-visible change for anything
  that reads a skill's `name` field expecting the verbatim LLM string.
  `AgentSkillFrontmatter.name` also gained a validator rejecting a name
  containing a path separator, or being exactly `..`, so a hand-edited
  `SKILL.md` that bypasses the writer's sanitization is caught on read
  rather than silently relocated (the substring form, e.g. a name that
  merely *contains* `..`, is deliberately allowed — sanitized output can
  legitimately contain runs of literal dots). `sanitize_dirname` itself
  falls back (not just on an empty result, but also on `.` or `..`) so a
  short input that is itself a sanitizer fixpoint — e.g. `"../"` sanitizes
  to `".."` verbatim without this fallback — cannot resolve to the same
  directory or its parent; this closes both the agent-skill case and an
  equivalent one-level escape on the knowledge-upload path, which has no
  `skill_`-style prefix protecting its sanitized segment. **No data
  migration is needed for agent skills**: extraction has never
  successfully produced a `SKILL.md` before this release (see the
  cascade-lag fix above), so there is no legacy skill corpus whose
  directory names would change. **Knowledge documents do have a
  pre-existing corpus**, and two inputs resolve to a different directory
  than before: a decomposed (NFD) topic or category now keeps its
  combining marks (`"Résumé"` no longer degrades to `"Resume"`) because
  the shared helper NFC-normalizes first, and a topic or category of
  exactly `.` or `..` now falls back instead of resolving onto the
  parent directory. Precomposed input — including CJK — is unaffected;
  the character class is unchanged from the previous private copy.
  Sanitizing is lossy: skills whose raw names
  differ only in characters the sanitizer drops or replaces (e.g.
  `"fix django"` vs. `"fix_django"`) now share one `SKILL.md`, and so do
  names differing only in a combining mark regardless of script (e.g.
  Devanagari `"किताब"` vs. `"कताब"` — a combining mark alone is not `\w`
  and is stripped either way; same for Thai tone marks, Hebrew niqqud,
  Arabic harakat). The later write wins — the earlier skill's
  `source_case_ids`, `maturity_score`, and body are silently lost, not
  merged. Case is *not* folded, so `"Fix Django"` and `"fix django"` stay
  two distinct sanitized names — two LanceDB rows, but one directory on a
  case-insensitive filesystem (macOS APFS and Windows NTFS defaults),
  where the index then advertises a name whose content was overwritten.
  This is accepted for now rather than mitigated: detecting a collision
  and raising would reintroduce the dead-letter DoS the sanitizer was
  built to avoid, and a disambiguating suffix — the workable option —
  needs a collision probe plus a case-folding rule, so it is deferred to
  a deliberate pass rather than added here.
- **A renamed skill no longer leaves an orphan directory that pollutes the
  next extraction.** everalgo treats a name change as a first-class update
  (`skill_ops._apply_update` preserves `prior.id` while swapping the name),
  so the emitted skill was written to a new `skill_<new_name>/` while the
  old directory survived carrying the same `cluster_id`. Because existing
  skills are now read from markdown rather than LanceDB, that orphan did
  not merely sit on disk — it came back in the next run's
  `existing_relevant_skills` as a duplicate of a skill the LLM had already
  renamed, feeding exactly the `add`-instead-of-`update` full-replace
  clobber this release set out to close, once more per rename. The old
  directory is now reaped after the new one is written, keyed on the
  skill's `id` (the only thing that survives a rename; a fresh `add` mints
  a uuid and can never match). A prior name that another skill in the same
  batch just claimed is never deleted.
- **`extract_agent_skill` retire ops are documented as unimplemented rather
  than silently mispersisted.** `AgentSkillExtractor.aextract` returns a
  flat list with no op discriminator, so a retirement arrives as an
  ordinary skill with `confidence < retire_confidence` and was written back
  like any other — staying in markdown, in the next prompt, and in search.
  The behaviour is unchanged; the module docstring no longer claims retire
  is handled. Honouring it is a design decision (delete the directory, or
  add a `retired` flag that the enumeration, cascade, and search all
  filter on) deferred to its own change.
- **`reference_name` and `script_filename` are sanitized.** Both are
  appended *after* the `skill_<name>` segment, so `skill_dir_name` never
  covered them; they now go through the same `sanitize_dirname` primitive
  on both the reader and the writer. No caller in `src/` reaches them
  today, so nothing was exploitable — this closes the gap before
  progressive disclosure wires them up.
- **A single unparseable `SKILL.md` no longer disables skill extraction
  for its whole cluster.** `AgentSkillReader.list_by_cluster` propagated
  any frontmatter `ValidationError`, which aborted the enumeration that
  feeds `extract_agent_skill` its existing skills — so one hand-edited
  file (or, after a future schema revision adds a required field, every
  existing file at once) dead-lettered that cluster's extraction on every
  run. Offending files are now logged and skipped. `read_main` still
  raises, since a caller naming one specific skill needs to hear about
  corruption rather than receive the `None` that already means "not
  created yet".
  merged. This is accepted, not mitigated, on two grounds: a
  disambiguating suffix would break the `name` ≡ directory-suffix
  identity the reader/writer relies on, and detecting a collision and
  raising would reintroduce the dead-letter DoS the sanitizer was built
  to avoid.
  `"fix django"` vs. `"fix_django"`) now share one `SKILL.md`, and the
  later write wins — the earlier skill's `source_case_ids`,
  `maturity_score`, and body are silently lost, not merged. This is
  accepted, not mitigated: the LLM's add/update decision is keyed on the
  name it sees, so a collision usually reads as an intended update
  anyway.
  under the new sanitizer.
  `KnowledgeWriter` and the new `SkillPathMixin.skill_dir_name()`, instead
  of two independently maintained copies. `AgentSkillFrontmatter.name`
  also gained a validator rejecting path separators / `..` so a
  hand-edited `SKILL.md` is caught on read rather than silently
  relocated. No data migration: agent-skill extraction has never
  successfully produced a `SKILL.md` before this release (see the
  cascade-lag fix above), so there is no legacy skill corpus whose
  directory names would change under the new sanitizer.
- **Reads now carry a deadline** (`count` / `get_by_id` / `find_where` /
  `find_where_paginated` / `search`). The write-side deadline work skipped them
  on the reasoning that a read takes no lock and so blocks no writer — true, but
  the cascade drain loop reads on every batch and advances strictly one batch at
  a time, so a read that never returns stops the **whole md → LanceDB
  projection**: claimed rows stay `processing` forever, nothing new is indexed,
  and `/health` still reports healthy because a hang raises nothing. Budget 60s
  (~1000x the measured 62ms flat scan over 117k rows); expiry raises the
  retryable `VectorStoreBusyError`.
- **Background loops are supervised.** The drain / heartbeat / rebuild loops were
  plain `create_task` coroutines: one uncaught exception ended that loop
  permanently, and because the worker holds a strong reference to the task the
  interpreter never printed "Task exception was never retrieved" either — the
  loop's job simply stopped happening with zero output. Each now runs under a
  supervisor that logs and restarts with escalating backoff (5s / 15s / 45s),
  then asks the process to exit via `SIGTERM` so a restarting supervisor
  (systemd, Docker, k8s) can recover it. The restart budget counts consecutive
  *quick* crashes, not crashes over the process lifetime — a body that ran 60s+
  before raising starts a fresh incident, so independent transients days apart
  cannot pool into a process exit (same windowed counting as systemd's
  `StartLimitIntervalSec`). A done-callback covers the case the supervisor
  itself ends unexpectedly.
- **The optimize-failure alert is reachable again.** The fallback rebuild reset
  the same counter the health verdict reads, so a table failing 100% of the time
  cycled `1..5 → 0 → 1..` and the threshold value existed only during the
  sub-second rebuild — roughly 1% observable against a 30s scrape, so
  `cascade.healthy` stayed green while the table never reclaimed a version. The
  rate limiter now lives in its own counter (`failures_since_fallback`); only a
  successful optimize clears the alert streak. Same shape as the cross-kind
  `max()` masking bug: a remediation path refreshing the signal meant to report
  it.
- **A rebuild no longer leaves the column without an FTS index.** `rebuild_indexes`
  dropped every index and recreated it, on the assumption that LanceDB falls
  back to a brute-force scan meanwhile. That holds for vector search and **not**
  for FTS: with no inverted index a BM25 query raises `Cannot perform full text
  search unless an INVERTED index has been created`, and since the recall legs
  are gathered without `return_exceptions`, one failing leg 500s the whole
  search request. Now uses `create_index(replace=True)`, which swaps atomically
  — measured 0 failures across 49 queries spanning 3 replaces, versus 55
  failures for the same test against drop-then-create — and collapses the live
  index fragment set exactly as before (7 index files back to 4).
- **The empty-index-dir sweep is bounded by lance's own threshold.** lance's
  `cleanup.rs` unlinks a superseded index's files but never its directory — it
  contains no `rmdir` at all, which is structural rather than an oversight: it
  targets object stores, where paths are flat keys and an empty directory does
  not exist. Only a local filesystem materialises them, and a soak run reached
  13061 dirs, 98% empty. everos sweeps them, now with three independent
  guarantees instead of a self-chosen age: `rmdir` cannot delete a non-empty
  directory (the kernel refuses it, so no file can be lost and there is no
  check-then-act window), live index UUIDs are excluded via `list_indices()`,
  and anything else must outlive `UNVERIFIED_THRESHOLD_DAYS = 7` — lance's own
  bound for deciding an unreferenced index UUID is dead rather than mid-build.
  The previous 300s was our invention, which is what made it indefensible.
  Two consequences worth knowing: the age gate reads the dir's mtime, which
  POSIX bumps when lance's cleanup empties it, so the effective reclaim horizon
  is up to ~14 days (file wait + age gate) and the ceiling-load steady state is
  ~1.8M dirs / ~7GB; and a sweep that blows its 60s deadline is swallowed
  inside `prune()` — the cleanup commit already succeeded, so escaping would
  bill a prune "failure" (feeding the fallback-rebuild threshold) and stall the
  prune-staleness clock for a cleanup stall that did not happen.
- **The optimize runner's wait on an in-flight rebuild is bounded too.** The two
  maintenance jobs park on each other — whichever arrives second waits — so an
  unbounded wait on this side is the same hazard as the one already fixed on
  the rebuild side, just seen from the other end: the kind's task slot stays
  occupied, `_schedule_optimize` keeps short-circuiting on it, and that table
  quietly stops being pruned. It was left open on the argument that
  `rebuild_indexes` carries its own 300s deadline, which covers its critical
  section but not the task's dispatch and teardown, so the transitive bound was
  never real. Now bounded at 180s, logging
  `cascade_lancedb_optimize_skipped_rebuild_unfinished` and skipping the beat
  rather than compacting under a live rebuild — the two commit on the same
  manifest, which is what the wait exists to prevent.
- **A rebuild that loses a commit race is retried** instead of waiting out the
  full 12h cadence. Lance labels the conflict `Retryable` and it is: a
  concurrent writer in another process won the manifest, nothing is wrong with
  the table. Retries are scheduled on the kind (10min / 30min / 3h) rather than
  slept through, so the other kinds in the sweep are not parked behind the
  backoff. A soak run at a 600s cadence hit 3 conflicts in 119 attempts, all
  while a concurrent CLI storm was running.

- **The index-rebuild sweep can no longer park forever** waiting on the
  optimize runner. That wait had no deadline, and the runner's loop condition is
  "keep going while there is unindexed data" — which under sustained writes is
  never, since the drain loop re-raises the flag every second against a 10s
  cooldown. Now bounded at 180s, logging
  `cascade_lancedb_rebuild_skipped_optimize_unfinished` and skipping the sweep
  rather than dropping indices under a live optimize. **This makes the stall
  visible, not absent**: under sustained ingest every sweep still times out, so
  active index-UUID / FTS `part_N` growth stays unbounded there. The functional
  fix requires the optimize runner to yield when a rebuild is pending, which
  changes the optimize/rebuild mutual-exclusion contract and needs its own
  validation — the rebuild cadence is 12h, longer than any soak run so far, so
  the periodic sweep has never been exercised under load.
- **The memory-root lock wait is bounded and visible.** Acquisition polls with
  `LOCK_NB` instead of blocking inside a worker thread: a blocking `flock` could
  not be bounded or cancelled — cancelling the awaiting coroutine left the
  thread to acquire the lock later with nobody to release it. The wait itself is
  by design (the second process is supposed to wait, then find the migration
  already done), but it now logs `memory_root_lock_waiting` and gives up after
  `timeout_seconds` (default 30min) instead of leaving a server startup looking
  like a hang whose last message is `lifespan_provider_startup name=lancedb`.
  The default sits an order of magnitude above the worst legitimate hold (a
  large migration is minutes) on purpose: the wait is already visible from the
  first poll, and against the one case the bound exists for — a holder that is
  alive but wedged — giving up at 5 minutes buys nothing over 30, while a bound
  near the legitimate hold turns a slow migration into startup crashes for
  every waiting process.

### Added

- `OfflineEngine.trigger_manual` now returns
  `tuple[BaseEvent, list[tuple[StrategyMeta, str]]]` instead of `None`,
  enabling the `dispatched`/`runs` fields below.
- `TriggerResponse` gains `dispatched: int` and `runs: list[RunSummary]`.
- `OMEConfig` gains `retry_backoff_base_seconds`, `retry_backoff_cap_seconds`,
  and `retry_jitter_seconds` for the retry-loop sleep.
- `AgentSkillReader.list_by_cluster()` enumerates the cluster's SKILL.md
  files from markdown (strong-consistency existence check).
- **`[cascade]` settings section** — the four maintenance cadences
  (`optimize_heartbeat_seconds`, `optimize_prune_interval_seconds`,
  `optimize_prune_retention_seconds`, `optimize_rebuild_interval_seconds`) are
  now configurable. They were already constructor arguments on `CascadeWorker`,
  but `CascadeConfig` did not carry them and no production path passed one, so
  the defaults were unreachable — which is why the 12h rebuild sweep could not
  be exercised by any soak run shorter than half a day. The deadlines that
  bound a hung call are deliberately **not** exposed: they are hang-catchers
  sized from measured durations, where too low manufactures failures on a
  healthy table and too high leaves a wedged one invisible for longer. Note
  `optimize_prune_retention_seconds` has a second effect worth reading before
  tuning — it also decides how long index files keep a manifest naming them,
  and below LanceDB's 7-day unverified window they then wait out the full 7
  days.

### Changed

- **`extract_foresight` now ships disabled** (`enabled=False`). Not because
  it is broken — the crash below is fixed — but because it is one LLM call
  per sender per memcell whose output nothing in EverOS reads today: no
  search route surfaces foresights and no prompt slot consumes them. Until
  something does, running it by default spends tokens on write-only data.
  **Re-enable per install** in `ome.toml` (hot-reloaded, no restart):

  ```toml
  [strategies.extract_foresight]
  enabled = true
  ```

  Editing `default_ome.toml` alone would not have reached existing installs
  — `everos init` does not overwrite an existing `~/.everos/ome.toml` — so
  the code default is what changed.
- **`extract_foresight` no longer crashes on a memcell containing tool
  calls.** The sender scan read `m.role` off every item, but only
  `ChatMessage` carries it (`ToolCallRequest` has `sender_id` without it,
  `ToolCallResult` has neither), so the first tool call raised
  `AttributeError` — before any sender was resolved. The strategy was
  correct on plain user chat and dead-lettered every time on agent
  trajectories. everalgo explicitly contracts for the mixed case
  (`user_memory/_render.chat_messages`: the caller need not pre-filter),
  and every other user-memory extractor gets that for free by delegating;
  this was the one place the filter was hand-rolled. The scan now tests
  `isinstance(m, ChatMessage)`, so a pure agent trajectory yields no senders
  and returns without an LLM call. Matters even with the strategy off by
  default: it is what makes the opt-in above actually usable.
- **`SkillClusterUpdated` carries the case's 1024-dim embedding, growing the
  OME `run_record` table.** The event payload is persisted verbatim in
  `run_record.event_payload` (and in the APScheduler jobstore while a job is
  queued), so a `skill_cluster_updated` record goes from roughly 0.8 KB to
  14 KB. At the default `max_records_per_strategy = 1000` ring buffer that is
  ~14 MB for this one strategy instead of ~0.8 MB. **Operators sizing
  `~/.everos/.index/sqlite/ome.db` should expect this.** The vector is only
  read when a cluster holds more skills than `MAX_SKILLS_IN_PROMPT`, so it
  usually rides along unused; trimming it from the persisted copy is not a
  local change, because crash recovery replays `event_payload` to rebuild the
  event and a trimmed payload would silently take the recovered run down a
  different branch than the original. Tracked as a follow-up.
- **`cascade_lancedb_optimize_conflict` now records `pruned`** — which
  maintenance beat lost the commit race. Lance labels both beats' commit the
  same way (`This Rewrite transaction was preempted by concurrent transaction
  …`), so the message alone cannot separate a free loss from a costly one: a
  lost **light** beat retries ~10s later, while a lost **heavy** beat means
  that table skipped a whole prune cadence and its superseded files stay on
  disk. Attributing index-dir growth previously meant back-inferring which
  beats were heavy from the 300s cadence. Log level (`debug`) and the
  benign-conflict semantics are unchanged — this adds one field.

### Upgrade

Two behaviour changes to know about before upgrading.

`extract_foresight` now ships disabled — a deployment relying on foresight
entries must set `enabled = true` for it in `ome.toml`.

`SkillClusterUpdated` now carries the case's 1024-dim embedding, so a
`skill_cluster_updated` row in the OME `run_record` table grows from ~0.8 KB to
~14 KB — about 14 MB for that strategy's default 1000-record ring buffer,
against ~0.8 MB before. Anyone sizing `~/.everos/.index/sqlite/ome.db` should
account for it.

Nothing else needs action: the new `[cascade]` section is optional and a config
written by an earlier version falls back to the same defaults (verified on a
clean install).

One deployment note. When a supervised background loop crashes repeatedly and
exhausts its restart budget, the worker now sends itself `SIGTERM` rather than
serving on with a dead projection pipeline. That assumes something restarts the
process — systemd `Restart=always`, Docker `restart: unless-stopped`, a k8s
Deployment. Without one the process simply stops, which is still preferable to a
server answering searches from a silently frozen index, but it is worth knowing
before the first time it happens.

## [1.2.2] - 2026-08-04

### Added

- **`GET /health` now carries a `cascade` readiness block** — `healthy`,
  human-readable `reasons`, and the counters behind them (`pending`,
  `failed_permanent`, `failed_retryable`, `drain_consecutive_failures`,
  `unrecoverable_total`, `optimize_failure_streak`, `prune_stale_seconds`).
  `null` when the app runs without the cascade lifespan. **Alert on
  `cascade.healthy`**: it flips false only on operational faults — drain loop
  failing (≥3 in a row), index maintenance wedged (≥5), or version cleanup
  stalled on some table (≥3 missed 300s beats, and `reasons` names the table).
  `failed_permanent` is a data-quality backlog awaiting `cascade fix` and
  deliberately does **not** flip `healthy`, otherwise the signal sits red until
  a human edits markdown. The HTTP status stays 200 even when the block says
  unhealthy — it is a liveness signal, and a degraded projection must not
  trigger a container restart. If the probe itself fails (locked / full
  SQLite), the block returns `healthy=false` with a `cascade health probe
  failed: …` reason and zeroed counters — read zeros next to that reason as
  "unknown", not "clean".
- **`everos cascade rebuild` CLI command** — drops every business LanceDB table,
  clears the cascade queue, and re-indexes all markdown from scratch. The
  supported recovery from a drifted or corrupt index: unlike deleting the index
  directory, it re-enqueues every file (a bare `rm -rf` leaves the queue marked
  `done`, so nothing re-indexes and the index comes back empty), and unlike
  deleting `.index/` it preserves SQLite state that markdown cannot rebuild —
  notably `unprocessed_buffer`. **Requires the server to be stopped**: it
  refuses to start (exit code `3`) while a server holds the OME lock, because a
  live daemon keeps writing through cached table handles to the dropped
  dataset. `--yes/-y` for non-interactive use; `Ctrl-C` exits `130` and the
  re-index resumes on the next run or server start.
- **Startup schema verification now detects column *type* drift**, not just
  missing / extra columns. Catches the class of corruption behind #337 — an
  `episode.subject_vector` left as `string` by an older build while the schema
  declares a 1024-d `fixed_size_list` — which a name-only check waved through
  and which then failed deep inside `merge_insert` with an opaque
  `LanceError(IO)`. The error now points at `everos cascade rebuild`.

### Changed

- **LanceDB maintenance is split into compaction and reclamation.**
  `optimize()` is lock-free compaction; the new `prune()` runs
  `cleanup_older_than` under the per-table write lock. Fixes unbounded index
  growth: the previous bundled call issued a Rewrite that concurrent writes
  kept preempting, so version cleanup lost the race indefinitely (a soak run
  measured 16 successes against 547 conflicts over 21h, with the index
  directory growing to the disk guardrail). Reclamation now completes on every
  beat, at the cost of a brief same-table write stall (measured ~40ms).
  Retention is decoupled from cadence: files older than 60s are eligible,
  reclaimed on a 300s beat.
- **Every write-lock critical section is now bounded.** All seven operations
  (`add` / `upsert` / `update` / `delete` / `delete_by_md_path` / `prune` /
  `rebuild_indexes`) run under a deadline that covers **lock acquisition as
  well as the body**, so no code path can wait for the lock — or hold it —
  indefinitely. Budgets are sized from measured durations (row writes are
  2–25ms, worst observed 63ms → 15s; index rebuild → 300s; prune → 60s).
  Expiry raises the retryable `VectorStoreBusyError`, so the cascade worker
  retries the row instead of marking it permanently failed. Without this, one
  operation stuck outside the old narrow timeout wedged a table permanently:
  every writer blocked on acquire, and the maintenance scheduler skipped a kind
  whose task never finished, so that table stopped reclaiming versions
  altogether (observed: 150 versions retained, disk 11x live size, with nothing
  logged because nothing failed).
- **Benign LanceDB commit conflicts no longer count as failures.** A lost
  optimistic-concurrency race logs at `debug` on either maintenance beat. The
  heavy beat needs this too: its write lock is in-process only, so a second
  process (`cascade sync`, `cascade backfill`) can preempt its commit. Counting
  those triggered spurious fallback index rebuilds, which drop every index
  before recreating them — and if the rebuild also lost the race, the table sat
  without an FTS index and every search on that kind returned 500 until the
  next 12h sweep.
- **A query vector whose width disagrees with the embedding provider's declared
  `dim` now fails immediately** with `CONFIGURATION_ERROR` instead of reaching
  LanceDB. It previously surfaced as an opaque `ValueError` after the query was
  built — 13–14s per request, as an unhandled 500.
- **Exception logging no longer renders frame locals.** structlog's default
  traceback formatter (`show_locals=True`, up to 100 frames) rendered one
  unhandled exception on an async stack into 6423 log lines — 85MB of logs
  across 11 exceptions in one soak run — at ~290ms of synchronous CPU each, and
  risked printing request payloads into logs. Now locals-off and capped at 15
  frames: the same traceback is 103 lines.
- **`cascade backfill` reclaims through the daemon's retention window** rather
  than at zero age, so it cannot delete files out from under an in-flight
  `/search` in the server process.
- **`lancedb` pinned to `>=0.34.0,<0.35.0`.** 0.35 embeds lance-rust v9; 0.34.0
  is the version validated under sustained churn. Environments installing from
  `uv.lock` are unaffected (already 0.34.0). Never widen the floor below 0.34 —
  older lance cannot read v8 data.

### Fixed

- **Maintenance deadlines now cover the whole call, not just the critical
  section.** Resolving a table handle sat outside the timeout, so a hang there
  never returned — and because the scheduler runs one maintenance task per kind
  and skips a kind whose task is in flight, that table stopped being maintained
  permanently and silently (a soak run caught one table 13 minutes without a
  reclaim, retained versions climbing, while its siblings reclaimed normally and
  nothing was logged because nothing failed). Handle resolution moved inside the
  deadline for all seven locked operations, the lock-free compaction beat got
  its own deadline, and the scheduler adds a last-resort 180s bound on the whole
  call. The per-kind staleness alert added in this release is what surfaced it.
- **AGENTIC search crashed on agent memory** (`agent_case` / `agent_skill`) —
  candidate metadata now satisfies the everalgo `_format_docs` contract,
  removing a `TypeError` in the sufficiency / multi-query steps.
- **Per-kind version-cleanup staleness is no longer masked.** The health signal
  reported time since the newest successful prune across all kinds, so on a
  multi-kind deployment one kind whose cleanup died was hidden by the others
  pruning on schedule. It now reports the worst kind and names it.
- **`cascade backfill` silently skipped compaction and reclamation** — it still
  called the removed `optimize(cleanup_older_than=…)` signature, and the
  resulting `TypeError` was swallowed by a best-effort `except`, so the disk
  growth this release fixes came back after every backfill.
- **Empty `_indices/<uuid>/` husks are removed after cleanup** (a soak run
  accumulated 13061 directories, 98% of them empty), which bloated inode usage
  and slowed directory scans.

### Docs

- Rewrote the cascade runbook's recovery paths: the `/health` cascade block and
  its alert thresholds, `cascade rebuild` (including the stop-the-server
  requirement), why `rm -rf .index/lancedb` yields an empty index, and why
  `rm -rf .index` loses un-extracted buffered messages.

## [1.2.1] - 2026-07-29

### Added

- **`[embedding]` and `[rerank]` are now soft runtime dependencies** — EverOS
  boots and serves requests with only `[llm]` configured. Missing or
  misconfigured embedding / rerank / multimodal providers no longer abort
  startup; the corresponding accessor logs `<provider>_capability_build_failed`
  and reports `available=False`. Features degrade gracefully into three tiers:
  Tier 1 (`[llm]` only) → KEYWORD search + add/flush + md writes + cascade sync;
  Tier 2 (`+ [embedding]`) → adds VECTOR/HYBRID search + reflection + skill
  extraction + backfill; Tier 3 (`+ [rerank]`) → adds AGENTIC search + knowledge
  write/search. Tier upgrades require a server restart. Downgrades are
  read-safe: knowledge documents stay readable/renamable/deletable after
  a Tier-3 → Tier-2 downgrade; only write / search endpoints return 422.
- **`everos cascade backfill` CLI command** — three-phase interactive
  backfill (`vectors` → `clusters` → `skills`, or `--phase all`) for
  upgrading Tier-1 rows to Tier-2 after `[embedding]` is configured. Each
  phase prints row/token estimates and blocks on `y/N`; `--yes`/`-y` for
  CI. Exit codes: `0` success, `1` user declined, `2` phase preconditions
  unmet, `3` server running, `4` completed-with-failures, `130` SIGINT.
- **LanceDB schema v2** — the six business tables (`episode`, `atomic_fact`,
  `foresight`, `agent_case`, `agent_skill`, `knowledge_topic`) now allow
  `vector NULL`. Cascade writes rows without vectors when `[embedding]` is
  unavailable; a later backfill fills them in. Migration runs once on first
  startup under a cross-process `memory_root_lock` (`fcntl.flock`),
  followed by `optimize(cleanup_older_than=timedelta(0))` per table to
  physically prune older manifest versions.
- **Startup unbackfilled-rows banner** — after LanceDB lifespan, an
  unconditional sweep emits `unbackfilled_memory_rows` when rows with
  `vector IS NULL` exist, pointing at `everos cascade backfill` in the
  hint text.
- **PyPI Trusted Publishing workflow** — tag-triggered `.github/workflows/release.yml`
  builds, smoke-tests, and uploads via OIDC (no stored token) behind the
  `release` environment's manual-approval gate. Version-tag mismatch
  aborts publish. Companion `/release` skill lives under
  `.claude/skills/release/`.

### Changed

- **`ProviderNotConfiguredError` → HTTP 422 `CAPABILITY_UNAVAILABLE`** —
  write / search endpoints that need embed or rerank now return 422 with
  a section-aware hint (points at `everos.toml` section, never at
  `EVEROS_*` env vars) instead of erroring at startup or 500-ing at
  request time.
- **`GET /health` returns a Pydantic `HealthResponse` model** — with typed
  `capabilities` and `disabled_features` fields, so OpenAPI codegen
  produces real shapes instead of `additionalProperties: true`.
- **`MemoryRoot.default()` → `MemoryRoot.resolve()`** — the classmethod
  that resolves the memory root from `--root` / `EVEROS_ROOT` / default was
  renamed to make its behavior explicit (`resolve` walks the precedence
  chain; `default` was ambiguous with "default location"). `MemoryRoot`
  is publicly exported from `everos.core.persistence`; callers outside
  the repo may have used the old name. **A `default()` alias is kept**
  as a backward-compatibility shim that forwards to `resolve()` and
  emits a `DeprecationWarning`. The alias will be removed in a future
  major release — update call sites when convenient.
- **Uncalibrated recall scores moved to their own name** — `KEYWORD` and
  single-route `VECTOR` searches now report their top score as
  `recall_top_score_raw`; `recall_top_score` is reserved for the calibrated
  methods (`HYBRID` LR sigmoid, `AGENTIC` cross-encoder), whose values share a
  comparable `[0, 1]` scale. Langfuse aggregates scores by name, so the previous
  single name meant a chart could average an unbounded BM25 score together with
  a probability. Every recall score also carries
  `metadata = {"method": ..., "calibrated": ...}` now, a structured field that
  can be split on, alongside the existing human-readable comment. Dashboards
  built on `recall_top_score` for keyword search need to switch to the new name.
- **Docs and examples now use `/api/v2`** — README, QUICKSTART, the `docs/`
  reference set, the Langfuse example, and `everos demo --live` all call the
  canonical `/api/v2` prefix instead of `/api/v1`. `/api/v1` keeps resolving
  to the same handlers, so nothing breaks; it is now described as a **legacy
  compatibility alias that may be removed in a future major release** rather
  than a permanent one. New integrations should target `/api/v2`.
- **`cluster_repo.find_cluster_id_for_member` now requires
  `(app_id, project_id, owner_id)`** — reverse-index lookups JOIN `Cluster`
  for scope filtering. `entry_id` is per-owner unique by design; the
  reverse index alone could collide across owners writing on the same day.
- **All six cascade handlers register unconditionally** — Tier-3 → Tier-2/1
  downgrade no longer strands DELETE / PATCH events. Embed-requiring
  branches inside each handler body-guard on capability availability at
  execution time.
- **Interactive TTY log level defaults to WARNING** — avoids INFO log lines
  drowning out backfill y/N prompts; non-interactive / CI stays at INFO.
  `--verbose` / `-v` forces INFO.
- **`click>=8.1` promoted to first-class dependency** — was previously
  transitive via typer. `typer.Abort` and `click.exceptions.Abort` are
  distinct classes under typer 0.15+ (typer vendored click); the interrupt
  catch in `cascade backfill` covers both.
- **Test harness pins `EVEROS_ROOT` to a temp path** — `conftest.py` scrubs
  every `EVEROS_*` env var so a developer's `~/.everos/everos.toml` cannot
  make tests accidentally green against a real provider.

### Fixed

- **Fixes present in `1.1.4` but missing from `1.2.0`** — cascade retry
  classification + total-retry budget, the delete/modify race, the
  embedding empty-data guard, and episode extraction retries (each
  detailed under 1.1.4) were absent from the branch `1.2.0` was built
  from, so `1.2.0` regressed to pre-1.1.4 behaviour on all of them. See
  Security below for the path traversal.
- **Filename validation on knowledge upload** — NUL byte and > 255-byte
  UTF-8 filenames now fail fast with `InvalidInputError → HTTP 400`
  instead of surfacing OS errors as 500 with a half-written md file.
- **HTML upload no longer takes the UTF-8 fast-path** — knowledge
  upload's plaintext short-circuit uses an explicit allowlist
  (`text/plain`, `text/markdown`, `text/x-rst`, `text/x-markdown`) plus
  known extensions; `text/html` is deliberately excluded so HTML still
  goes through everalgo's `clean_html_for_llm`. Prevents 503 when a
  Tier-3 user without `[multimodal]` uploads a markdown doc.
- **Broken table-of-contents links in `docs/api.md`** — the endpoint anchors
  still pointed at the pre-1.2.0 `#post-apiv1…` slugs after the headings moved
  to `/api/v2`, so all five endpoint links in the TOC were dead.

### Removed

- **README "Star us" section** — cleanup.

### Security

- **Knowledge upload path traversal (CWE-22)** — see the 1.1.4 entry for
  the fix description. **Affected:** every release before `1.1.4`, and
  `1.2.0`. **Not affected:** `1.1.4`. **Fixed in:** `1.2.1`. The fix
  shipped in `1.1.4` but was not present on the branch `1.2.0` was built
  from, so upgrading `1.1.4` → `1.2.0` reintroduced it.

## [1.2.0] - 2026-07-24

> This release is missing fixes that shipped in `1.1.4`, including a
> knowledge-upload path traversal (CWE-22). See the Security section
> under 1.2.1 for the affected-version range.

### Added

- **`/api/v2` API prefix** — every business endpoint (`memory/*`, `ome/*`,
  `knowledge/*`) is now served under `/api/v2`, aligning the open-source API
  with the EverOS Cloud contract. `/api/v1` is retained as a legacy
  compatibility alias: both prefixes resolve to the same handlers with
  identical request/response contracts, so existing integrations keep working
  unchanged. Infrastructure endpoints (`/health`, `/metrics`) stay unversioned.
- **Native OpenTelemetry tracing** — memory operations (add / flush, memcell
  boundary, episode extraction, search, and OME reflection) export to any
  OTLP backend (e.g. Langfuse) as nested traces carrying LLM/embedding token
  usage, per-request correlation, and recall-quality scores. Off by default;
  enabled via the `[observability]` config with the optional `otel` extra.
  Content capture (query / extracted memory) is opt-in and redaction-aware.

## [1.1.4] - 2026-07-20

> The entries below reflect the code shipped as `everos==1.1.4` on PyPI.
> The 1.1.4 sdist was built from the internal release lane and contains
> fixes that were not represented in this file when 1.1.4 was tagged;
> this section restores them so the changelog matches the wheel.

### Fixed

- **Knowledge upload path traversal (CWE-22)** — the original-file write
  path is now contained to the document directory; adversarial filenames
  (`..`, absolute paths, symlink games) are rejected on
  `POST /api/v1/knowledge/documents` and `POST /api/v2/knowledge/documents`.
- **Cascade reliability — retry classification, budget, and races** — the
  worker catches `ExternalServiceError` (embedding / LLM / rerank transient
  failures) and retries inline up to 3 times before marking
  `retryable=True`; a total retry budget of 12 attempts across scanner
  cycles bounds retries on prolonged outages. The reconciler no longer
  re-enqueues `pending` / `processing` rows on stable mtime (previously
  overwrote the worker's `mark_done`); `failed` rows with
  `retryable=False` on stable mtime skip auto-retry so users can edit and
  re-save. SQLite `REAL` float precision loss in mtime comparisons is
  now absorbed via a 10 ms tolerance. LanceDB `optimize()` failures
  escalate to a `drop_index + create_index` rebuild after 5 consecutive
  misses (workaround for `lance-format/lance#7653` panic path).
- **Cascade delete/modify race** — when a file disappears after its
  modified event is queued, the worker processes it as a deletion
  instead of leaving a stale indexed row and permanently failed queue
  item.
- **Embedding provider raises on empty API data** — the provider now
  raises `EmbeddingServiceError` when the API returns HTTP 200 with an
  empty `data` array (previously silently returned zero-length vectors,
  corrupting search).
- **Episode extraction retries on malformed LLM output** — the `/flush`
  synchronous path retries everalgo `ValueError` (typically OpenRouter
  truncated responses) twice with 1 s / 2 s backoff before surfacing a
  500.
- **Langfuse live-server traces use only real telemetry** — synthetic
  child spans are now limited to responses that provide stage details,
  while real servers emit accurate top-level latency, output, and
  recall-quality scores.

### Added

- **Optional `dimensions` parameter for MRL-capable embedding models** —
  opt-in via `[embedding] dimensions = N` in `everos.toml`; forwarded to
  the API for server-side truncation with re-normalization (OpenAI
  text-embedding-3-\*, Qwen3-Embedding).
- **LLM `finish_reason` diagnostic warnings** — logs `content_len` /
  `content_tail` / `model` when the provider returns a non-`stop`
  `finish_reason`, aiding OpenRouter truncation triage.
- **Langfuse integration example** — added an OpenTelemetry-based
  wrapper for tracing EverOS add, flush/extract, search, and reflection
  operations, with a built-in mock and support for connecting to a real
  EverOS server.

### Changed

- **`everalgo-user-memory` bumped 0.3.1 → 0.3.2**.

## [1.1.3] - 2026-07-10

### Fixed

- **LanceDB FTS optimize crash and disk growth** — disabled unused positional
  data in OR-mode BM25 indexes, automatically rebuilds affected indexes, and
  escalates repeated optimize failures so cleanup cannot fail silently.

## [1.1.2] - 2026-07-07

### Fixed

- **Agent-track search broken by `deprecated_by IS NULL` filter** —
  `compile_filters()` unconditionally appended a `deprecated_by IS NULL`
  clause to every LanceDB query, but only `episode` and `atomic_fact`
  tables have this column. Agent-track search (`agent_case`,
  `agent_skill`) failed on any method. The clause is now conditional on
  `owner_type == "user"`.

## [1.1.1] - 2026-07-06

### Added

- **DashScope rerank provider** — Aliyun Bailian `gte-rerank-v2` adapter;
  configure with `rerank.provider = "dashscope"` in `everos.toml`.
- **`everos demo` TUI command** — Textual-based interactive CLI demo for
  showcasing EverOS core features.
- **Benchmark runner** — full LoCoMo benchmark suite: `benchmarks/run.py`
  with TOML configuration, automated ingestion, search evaluation, and
  scoring.
- **Hybrid search: heap-expand algorithm** — rewrote `hierarchy.py` to
  heap-driven lazy expansion with global top-N competition, replacing the
  serial four-layer pipeline.

### Fixed

- **Knowledge: atomic upsert prevents StaleDataError** — cascade handler
  switched from get→update to `INSERT ... ON CONFLICT DO UPDATE`, fixing
  concurrent cascade race conditions.
- **API: OpenAPI version read from `__version__`** — no longer hardcoded to
  `0.1.0`; version now stays in sync with `pyproject.toml`.
- **Profile middleware no longer swallows exceptions** — inner handler
  errors now re-raise correctly instead of silently returning HTTP 200.

### Performance

- **Cascade optimize throttle 1s → 10s** — reduced unnecessary LanceDB
  `optimize()` I/O by raising the minimum interval between calls.

### CI / Build

- **CI Python version matrix** — test and integration jobs now run on both
  Python 3.12 and 3.13.
- **pyproject.toml improvements** — added `project.urls`, `Typing :: Typed`
  classifier, relaxed `jieba` version constraint, removed unused
  `python-dotenv` dependency, cleaned up sdist include list, added `RUF`
  lint rules and coverage configuration.
- **`make ci` includes coverage** — `ci` target now runs
  `lint + test + integration + cov`.

### Documentation

- Fixed stale references across 13 files (v1.1.0 freshness sweep).
- Added GitHub sync guide (`docs/github-sync.md`).
- Added v1.1.0 release notes and v1.0.0 migration guide as standalone docs.
- Added `README.zh-CN.md` (Chinese README).
- Expanded `QUICKSTART.md` with source install instructions and `uv run`
  usage notes.
- Clarified cascade `optimize()` semantics in docstrings and runbook.

## [1.1.0] - 2026-06-24

### Added

- **Knowledge base subsystem** — full-stack document management exposed via
  `/api/v1/knowledge/*`. Upload documents (PDF / HTML / DOCX via multimodal
  parser), CRUD operations, and hybrid search (BM25 + vector + rerank +
  category boost). Ships with a 20-category default taxonomy
  (`.taxonomy.md`, auto-generated on first use). Original uploaded files are
  preserved alongside extracted Markdown. New settings group:
  `knowledge.*` (search tuning, `max_upload_bytes`, etc.).
- **Reflection V1** — offline memory self-improvement engine.
  Select → Merge → Re-extract → Deprecate: clusters related episodes within
  existing 7-day windows, merges them via LLM, re-extracts consolidated
  episodes, and deprecates the originals. Runs as an OME strategy
  (`reflect_episodes`); configure via `ome.toml`
  (`[strategies.reflect_episodes]`, cron `0 2 * * 1`), changes are
  hot-reloaded within ~2 s, no restart needed; **disabled by default**.
  Requires `everalgo-user-memory>=0.3.1`.
- **Standardized error response contract.** All API errors now return a
  canonical envelope with a semantic `ErrorCode` (10 codes: `NOT_FOUND`,
  `CONFLICT`, `INVALID_INPUT`, `EXTRACTION_EMPTY`, `UNSUPPORTED_FORMAT`,
  `EXTERNAL_SERVICE_UNAVAILABLE`, `CAPABILITY_UNAVAILABLE`,
  `CONFIGURATION_ERROR`, `INTERNAL_ERROR`, `BAD_REQUEST`), per-type
  exception handlers with MRO dispatch, and an `ErrorResponse` Pydantic
  model visible in OpenAPI docs. Replaces the v1.0 two-code scheme
  (`HTTP_ERROR` / `SYSTEM_ERROR`).
- **Search: hierarchical fact eviction** (Layer-4) with `min_score` floor —
  low-confidence atomic facts are evicted before fusion, improving
  precision.
- **Knowledge search degradation guidance** — when the embedding or rerank
  provider fails at call time, the knowledge search route enriches the
  error message with actionable guidance (e.g. retry with `method=keyword`,
  which needs no embedding) before returning `503`.
- **Knowledge topic recaller** — dual-column BM25 recall for knowledge
  topics, integrated into the search manager alongside existing recall
  types.

### Changed

- **`everos init` now generates `gpt-4.1-mini`** as the default LLM model
  (was `gpt-4o-mini`). Existing user configurations are not affected.
- **API error `code` values have changed.** v1.0 returned only `HTTP_ERROR`
  (all 4xx) and `SYSTEM_ERROR` (all 5xx). v1.1 returns fine-grained
  semantic codes (see Added above). Clients that match on `error.code`
  string values need to update. The envelope structure
  (`request_id` + `error.{code, message, timestamp, path}`) is unchanged.
- **DDD-aligned exception hierarchy** — domain errors reorganized:
  `ValidationError` → `InvalidInputError`;
  `DocumentAlreadyExistsError` → `DuplicateDocumentError`;
  `EmbeddingError` → `EmbeddingServiceError`;
  `RerankError` → `RerankServiceError`;
  `LLMError` → `LLMServiceError` (at the boundary);
  `MultimodalError` split into `UnsupportedModalityError` (domain) +
  `MultimodalNotEnabledError` (infrastructure).
  New base classes: `CapabilityError`, `ConfigurationError`.
- **`infra/` restructured** — storage adapters moved under
  `infra/persistence/{markdown,sqlite,lancedb}`; each sub-package's
  `__init__.py` is the sole public API (enforced by import-linter).
- **Parser capability extracted** to `component/parser` (shared by memorize
  and knowledge upload paths).

### Fixed

- **Knowledge search no longer returns a bare `500 INTERNAL_ERROR` when the
  embedding or rerank provider is unconfigured.** `_require_search_providers`
  now raises `ConfigurationError` → `500 CONFIGURATION_ERROR`. A provider
  that is configured but fails at call time still surfaces as
  `503 EXTERNAL_SERVICE_UNAVAILABLE`.
- **Knowledge document uploads are capped** at `knowledge.max_upload_bytes`
  (default 50 MiB); oversized uploads are rejected with `422` before parsing.
- **Knowledge search `query` is bounded** to 2000 chars.
- **`GET /knowledge/documents?sort_by=updated_at`** is now accepted.
- **`POST /knowledge/documents` returns `original_file_path`** so callers no
  longer need a follow-up `GET` to locate the preserved upload.
- **Rerank providers no longer echo the upstream HTTP response body** into the
  client-facing `503` message (vLLM / DeepInfra); the body is logged instead.
- **Knowledge FK cascade race** — removed the foreign key on
  `knowledge_topics.doc_id` that caused delete-order race conditions;
  cascade cleanup handled at application level.
- **Knowledge `replace_document`** — atomic PUT: backup old Markdown before
  re-extraction; removed explicit SQLite delete for atomicity.
- **Knowledge duplicate `doc_id`** rejected on create; title collision
  resolved by appending `doc_id` to directory name.
- **Knowledge `md_path` resolution** fixed in `delete_document` (was not
  resolved against `memory_root`).
- **OME file-handle leak** — portalocker file handle is now closed on lock
  contention instead of being left open.
- **jieba / Python 3.12 compatibility** — deferred jieba import to avoid
  `SyntaxError` from invalid escape sequences; suppressed
  `DeprecationWarning` in tests.
- **Test isolation** — tests no longer leak `.env` state or depend on module
  import ordering.

### Documentation

- Added knowledge base technical documentation.
- Corrected the onboarding flow: `everos init` writes `everos.toml` +
  `ome.toml` (TOML), not a `.env` file; removed the nonexistent
  `--xdg` / `--env-file` options and the false `0600`-permissions claim
  from `README.md` / `QUICKSTART.md`; fixed the stable-version line
  (`v1.0.1`) and completed the `docs/cli.md` command tree.
- Updated error handling docs to match the new DDD exception hierarchy.

## [1.0.1] - 2026-06-16

### Security

- **Path-traversal hardening for caller-supplied identifiers.** `sender_id`
  (which flows through to `owner_id` and becomes a directory segment on the
  episode write path) now carries the same path-safety guard as `app_id` /
  `project_id`: a character whitelist plus rejection of the `.` / `..` tokens.
  The whitelist admits `@` and `+` so real-world ids (email-style,
  plus-addressing) still pass.
- **Defense-in-depth write containment.** `MarkdownWriter` now rejects any
  write target that resolves outside the configured memory root, before any
  filesystem touch (both the write `mkdir` and the append read-modify-write
  read). This backstop holds even if an identifier reaches the writer
  unsanitised (e.g. an `owner_id` set in the extract pipeline rather than from
  the DTO). The API layer maps the resulting error to HTTP 400.

### Documentation

- Add a multimodal usage guide and correct the multimodal error semantics
  after end-to-end verification.
- Rename the algorithm library to `everalgo` across docs and
  code comments (no code identifiers changed).
- Fix accuracy drift found in an adversarial doc audit; reflect the
  `everalgo` packages being published and the v1.0.0 stable status.

## [1.0.0] - 2026-06-03

First public release of EverOS — a Markdown-first memory extraction framework
for AI agents.

### Added

- **Markdown as source of truth** — all memory persists as plain `.md` files you
  can open, edit, grep, and version with Git.
- **Lightweight three-piece storage** — Markdown (truth) + SQLite (state / queue
  / audit) + LanceDB (vector + BM25 + scalar index). No external services
  required.
- **Hybrid retrieval** — BM25, vector, and scalar filtering in a single LanceDB
  query.
- **Cascade index sync** — editing a `.md` file triggers a file watcher →
  entry-level diff → sub-second LanceDB sync.
- **Dual-track memory** — user-track (Episodes / Profiles) and agent-track
  (Cases / Skills).
- **Multi-source extraction** — conversations, workflows, agent traces, and file
  knowledge.
- **CLI + HTTP API** — the `everos` command-line tool and a FastAPI server,
  async-first throughout.
- **Pluggable providers** — LLM / embedding / rerank via the OpenAI-compatible
  protocol (works with OpenAI, OpenRouter, vLLM, Ollama, …).
- **Decoupled algorithms** — memory extraction algorithms live in the standalone
  `everalgo-*` libraries published on PyPI.

[Unreleased]: https://github.com/EverMind-AI/everos/compare/v1.1.4...HEAD
[1.1.4]: https://github.com/EverMind-AI/everos/compare/v1.1.3...v1.1.4
[1.1.3]: https://github.com/EverMind-AI/everos/compare/v1.1.2...v1.1.3
[1.1.2]: https://github.com/EverMind-AI/everos/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/EverMind-AI/everos/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/EverMind-AI/everos/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/EverMind-AI/everos/releases/tag/v1.0.1
[1.0.0]: https://github.com/EverMind-AI/everos/releases/tag/v1.0.0
