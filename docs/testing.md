# Testing Reference

> Companion docs: the CI gates and commit conventions live in
> [engineering.md](engineering.md); this document covers **what the test
> layers cover, which ones CI does not run, and what a release has to
> establish beyond a green suite**.

## Layers

| Layer | Files | Runs in CI | Covers |
|---|---|---|---|
| `tests/unit` | 279 | yes (`make test`) | pure logic, adapters against fakes, script guards |
| `tests/integration` | 27 | yes (`make integration`) | real SQLite + LanceDB on a temp root, in-process app |
| `tests/e2e` | 9 | **no** | a real `everos server start` process driven over HTTP |

`make ci` is `lint + test + integration + package`. It never reaches
`tests/e2e`, and that is deliberate: those cases need real model credentials
and take minutes each.

## The two markers

```toml
addopts = "-v --tb=short -m 'not slow and not live_llm'"
```

Both markers are deselected **by default**, so a plain `pytest` run — local or
CI — silently skips them. They are not optional extras:

| Marker | Meaning | Run with |
|---|---|---|
| `slow` | a case that takes ≥10s regardless of dependencies | `pytest -m slow` |
| `live_llm` | needs real LLM / embedder / reranker credentials from `.env`; burns tokens | `pytest -m live_llm` |

**What only they cover.** Unit and integration do exercise every strategy,
but always against a fake model — `unittest.mock` over `get_llm_client` in
unit, `FakeLLMClient` returning canned JSON in integration. That pins the
wiring and nothing else: a prompt regression, a drift in the real model's
output shape, or an extractor that yields nothing usable on real prose all
pass a fake and fail a user. `slow` + `live_llm` are the only place a real
model drives those chains end to end, so a release that skips them has
verified the plumbing of `agent_case`, `agent_skill`, `foresight` and
`user_profile` but not that they produce anything. Run both before tagging:

```bash
uv run pytest -m slow
uv run pytest -m live_llm          # requires .env
```

Report what actually ran. A green line that deselected 2937 cases is not the
same claim as a green suite, and skipped counts belong in the release notes,
not rounded away.

## What a release has to establish

Three stages, in order. Each answers a question the previous one cannot.

| Stage | Question | Passes when |
|---|---|---|
| Suites | does the code agree with itself? | unit + integration + `slow` + `live_llm` all green, skip counts stated |
| Smoke | does the product work end to end? | all 8 memory kinds materialise, all 5 search routes answer |
| Soak | does it survive running? | four metrics bounded under load, flat when idle |

### Smoke

Drive a real `everos server start` process over HTTP — not an in-process ASGI
client, which cannot catch a startup or packaging fault. Feed a real corpus,
trigger every OME strategy, wait for the strategies to converge, then assert
two things:

- **All eight kinds exist**: `episode`, `atomic_fact`, `foresight`,
  `agent_case`, `agent_skill`, `user_profile`, `knowledge_document`,
  `knowledge_topic`.
- **All five search routes answer**: `keyword`, `vector`, `hybrid`,
  `agentic`, `llm_multiround`.

`extract_foresight` ships disabled — it costs one LLM call per sender for data
nothing reads yet — so a smoke harness must enable it in `ome.toml` or the
eighth kind is missing by design and the run proves nothing about it.

### Soak

Four metrics, two regimes. Dropping either column hides a class of defect.

| | Load (sustained writes + reads) | Idle (server up, zero requests) |
|---|---|---|
| Disk | grows with data, must show reclamation | reclaimed, then flat |
| Memory | growth rate must decay | flat |
| Descriptors | bounded, must fall as well as rise | flat |
| CPU | stable | near zero |

The idle regime is the one that separates a cache from a leak: **a cache stops
growing when traffic stops, a leak does not.** Run it on a root a load phase
already filled — an empty root holds no indexes open and proves nothing.

**Reading the result.** Do not compare the first sample to the last. Use
quarter medians plus the number of samples that fell below the previous one:

- `[49, 51, 50, 50]` with frequent falls — converged, eviction working.
- `[50, 80, 110, 140]` with zero falls — a leak.

For memory, fit a slope over nested windows (whole run, last 3h, last 2h,
last 1h) rather than reading single segments. A one-hour window on a real
workload swings by several MB/min in both directions and will support
whichever conclusion you already hold.

## Measurement pitfalls

Every one of these produces a plausible, stable, completely wrong number.
A metric that has never once reported a problem is a metric to distrust.

| Pitfall | Symptom | What to do |
|---|---|---|
| Measuring the launcher, not the server | On Windows `everos.exe` spawns the real server as a grandchild; sampling that pid reports ~5 MB and ~59 handles no matter the load | Sum the whole tree (`children(recursive=True)`) |
| Per-object CPU sampling | `psutil.cpu_percent()` measures against the previous call *on the same object*; re-fetching children each loop returns `0.0` forever | Cache `Process` objects across samples |
| Unlinked files invisible | `psutil.open_files()` omits deleted-but-open files on macOS — exactly the state a descriptor leak leaves behind | `/proc/<pid>/fd` on Linux, `lsof` on macOS |
| `lsof` line count as a descriptor count | Includes `mem`/`txt` mappings; inflates by a hundred or more | Count numeric FD rows only |
| Comparing Windows handles to POSIX fds | Handles cover threads, events, mutexes, registry keys — they track thread count (measured r = +0.99, ~3 handles per thread plus a ~400 baseline) | Compare a platform to itself, and watch the index-file subset |

## Reference figures

Measured for 1.4.2 across macOS (arm64), Ubuntu (x86_64) and Windows 11.
Useful as an order-of-magnitude baseline, not as a pass threshold.

**Storage, per 100k memories** — the index dominates; markdown is ~3% of total.

| Component | Size |
|---|---|
| Markdown (source of truth) | 50–60 MB |
| SQLite | 5–50 MB |
| LanceDB index, compacted | ~1.4 GB |
| LanceDB index, write peak | ~3.0 GB |

Plan capacity at ~1.5 GB per 100k memories steady-state and leave 2–3× headroom
for the write peak; a measured run compacted from 2789 MB down to 1232 MB.

**Behaviour under a 10h soak** — descriptors converged on all three platforms,
memory growth decayed to flat, disk reclaimed 24–68% on compaction, and search
p50 did not degrade (vector 0.643s → 0.562s first quarter to last).
