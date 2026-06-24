# End-to-end memorize test

In-process driver that pushes a realistic fixture through `service.memorize`,
batching by 6 messages per `/add` call and then `/flush` at the end.

## What's here

| File | Purpose |
|---|---|
| `fixtures/chat_session.json` | 22 messages · 3 topic shifts · multi-user (Alice → Bob) — chat-mode fixture |
| `fixtures/agent_session.json` | 21 items · 2 task threads · interleaved `tool_calls` / `tool` results — agent-mode fixture |
| `run.py` | In-process runner (no HTTP) |

## Prereqs

1. **LLM client configured** in `.env`:
   - `EVEROS_LLM__API_KEY=...`
   - `EVEROS_LLM__BASE_URL=...` (OpenAI-compatible)
   - `EVEROS_LLM__MODEL=...` (defaults to `gpt-4.1-mini`)
   - Without these, the boundary stage logs `memorize_no_llm_client` and skips the run.
2. **Memory root**: defaults to `~/.everos`; override with `EVEROS_ROOT=...`.
3. **Mode** is read from `settings.memorize.mode` (toml/env) before the first `memorize()` call.

## Run

```bash
# Chat mode — boundary uses everalgo.boundary.detect_boundaries
EVEROS_MEMORIZE__MODE=chat uv run python scripts/e2e_memorize/run.py \
    --fixture scripts/e2e_memorize/fixtures/chat_session.json \
    --expected-mode chat

# Agent mode — boundary uses everalgo.agent_memory.AgentBoundaryDetector
# (filter→detect→remap; tool items preserved in cells)
EVEROS_MEMORIZE__MODE=agent uv run python scripts/e2e_memorize/run.py \
    --fixture scripts/e2e_memorize/fixtures/agent_session.json \
    --expected-mode agent

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
- Agent fixture: 1 user (`u_alice`) — expect ~2 Episodes aligned with the
  two task threads (latency rollback → DB index fix).

### 3. SQLite memcell rows

```bash
sqlite3 ~/.everos/.index/sqlite/system.db \
    "select memcell_id, track, owner_id, owner_type, json_array_length(sender_ids_json) as senders
     from memcell order by timestamp"
```

- Chat run: rows with `track=user_memory`, `owner_type=user`.
- Agent run: parallel rows for both tracks (`user_memory` **and**
  `agent_memory`) since agent mode dispatches both pipelines.

### 4. Unprocessed buffer

```bash
sqlite3 ~/.everos/.index/sqlite/system.db \
    "select session_id, count(*) from unprocessed_buffer
     where track='memorize' group by session_id"
```

After `flush` the buffer should be empty for the test session.

### 5. OME async output (only if subscribers exist)

- `users/<owner>/atomic_facts/atomic_fact-YYYY-MM-DD.md` (always; `extract_atomic_facts` is registered)
- `users/<owner>/foresights/foresight-YYYY-MM-DD.md` (always; `extract_foresight` is registered)
- `agents/<agent>/agent_cases/agent_case-YYYY-MM-DD.md` (**only after `extract_agent_cases` strategy is written + registered** — currently absent, the emit is a no-op)

### 6. Reset between runs

The fixture's session_id is randomised per invocation, so previous runs
don't pollute the new one. To wipe everything:

```bash
rm -rf ~/.everos/users ~/.everos/agents ~/.everos/.index/sqlite/system.db
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

### Agent fixture task threads

| Range | Task |
|---|---|
| items 1-13 (`1747396800–1747397140`) | API latency spike → identify keepalive pool regression → rollback |
| items 14-21 (`1747400400–1747400720`) | DB connection pool exhaustion → find unindexed query → CREATE INDEX CONCURRENTLY |

Boundary detector should cut between item 13 and item 14 (timestamp jump
~55 minutes, topic flip). Tool items inside each cell stay attached to
their initiating chat turn.
