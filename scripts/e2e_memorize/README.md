# End-to-end memorize test

In-process driver that pushes a realistic fixture through `MemoryService.add`,
batching by 6 messages per `/add` call and then `/flush` at the end.

## What's here

| File | Purpose |
|---|---|
| `fixtures/chat_session.json` | 22 messages · 3 topic shifts · multi-user (Alice → Bob) |
| `run.py` | In-process runner (no HTTP) |

## Prereqs

1. **LLM client configured** in `.env`:
   - `EVEROS_LLM__API_KEY=...`
   - `EVEROS_LLM__BASE_URL=...` (OpenAI-compatible)
   - `EVEROS_LLM__MODEL=...` (defaults to `gpt-4.1-mini`)
   - Without these, the boundary stage logs `memorize_no_llm_client` and skips the run.
2. **Memory root**: defaults to `~/.everos`; override with `EVEROS_ROOT=...`.

## Run

```bash
# Boundary uses everalgo.boundary.adetect_boundaries
uv run python scripts/e2e_memorize/run.py \
    --fixture scripts/e2e_memorize/fixtures/chat_session.json

# Dry run (print batch plan, no LLM calls)
uv run python scripts/e2e_memorize/run.py \
    --fixture scripts/e2e_memorize/fixtures/chat_session.json --dry-run
```

## What to verify after a run

### 1. Console output

Each batch prints `status=` (`accumulated` while buffering, `extracted` when
cells got cut). Final `flush` should be `extracted` if any cell remained
in the tail. The trailing file walker lists md / sqlite files modified
in the last 10 minutes.

### 2. Episode md (sync — 4A)

```
~/.everos/users/<owner_id>/episodes/episode-YYYY-MM-DD.md
```

- Chat fixture: 2 owners (`u_alice`, `u_bob`) — expect Episodes split into
  ~3-4 cells aligned with topic shifts (Python bug → weekend ramen → Q3
  review → SRE handoff/ramen wrap).

### 3. SQLite memcell rows

```bash
sqlite3 ~/.everos/.index/sqlite/system.db \
    "select memcell_id, track, owner_id, owner_type, json_array_length(sender_ids_json) as senders
     from memcell order by timestamp"
```

- One row per cut cell (`raw_type` `Conversation`).

### 4. Unprocessed buffer

```bash
sqlite3 ~/.everos/.index/sqlite/system.db \
    "select session_id, count(*) from unprocessed_buffer
     where track='memorize' group by session_id"
```

After `flush` the buffer should be empty for the test session.

### 5. OME async output

- `users/<owner>/user.md` — the profile (`extract_user_profile`, run on each
  new episode)

### 6. Reset between runs

The fixture's session_id is randomised per invocation, so previous runs
don't pollute the new one. To wipe everything:

```bash
rm -rf ~/.everos/users ~/.everos/.index/sqlite/system.db
```

## Boundary expectations cheat sheet

### Chat fixture topic shifts (timestamps ms)

| Range | Topic |
|---|---|
| msgs 1-6  (`1747396800–1747397010`) | Python KeyError debugging |
| msgs 7-12 (`1747400400–1747400610`) | Weekend ramen plans |
| msgs 13-16 (`1747407600–1747407720`) | Q3 revenue review meeting prep |
| msgs 17-22 (`1747411200–1747411410`) | Bob joins, SRE handoff + ramen + Q3 deck deadline |

Boundary detector should cut on topic gaps; 3 cuts → 4 cells is the most likely outcome.
