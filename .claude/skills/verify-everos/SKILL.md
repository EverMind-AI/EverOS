---
name: verify-everos
description: Verify an EverOS change on a real server - start your own instance, drive it over HTTP or through Claude Code with everos-mcp, read the outcome from the layer that owns it (markdown, SQLite, LanceDB, metrics), and tear it down. Use before saying a memory feature works end to end.
---

# /verify-everos - check a change on a real server

Unit and integration tests run the code with stand-ins; this is how to see
the real `everos server` do it. Grown one task at a time: each task adds the
features it actually walked (see the feature map at the end), nothing more.

## 1. Start your own instance

Never use `~/.everos` or a server someone else started. Give every check its
own root and port.

```bash
ROOT=$(mktemp -d)/root; PORT=18611
lsof -nP -iTCP:$PORT -sTCP:LISTEN          # must print nothing
uv run everos init --root "$ROOT"
EVEROS_LLM__API_KEY=$OPENROUTER_API_KEY \
EVEROS_EMBEDDING__API_KEY=$DEEPINFRA_API_KEY \
EVEROS_RERANK__API_KEY= \
  uv run everos server start --root "$ROOT" --port $PORT > "$ROOT.log" 2>&1 &
echo $! > "$ROOT.pid"
```

The shipped defaults point the LLM at OpenRouter and the embedding at
DeepInfra, so those two keys are all it needs. Reflection is off by default
(`docs/reflection.md`); to turn it on, replace the `$ROOT/ome.toml` that
`init` scaffolds with `[strategies.reflect_episodes]\nenabled = true` before
starting, as the acceptance tests do.

## 2. Is it ready?

`curl -s http://127.0.0.1:$PORT/health` answers 200 within a few seconds.
Its `cascade.pending` is the number of markdown changes not yet in the index.
Before reading the index, poll it once a second until it reads 0 on two
consecutive samples (`docs/api.md`, and `wait_converged` in
`tests/acceptance/memory_delete/observe.py`): a single 0 can come between a
markdown write and its projection being queued. A server that exits instead: the last
lines of `$ROOT.log` (usually a missing key).

## 3. Drive it

- **HTTP**: `POST /api/v2/memory/add`, `/flush`, `/delete`
  (`user_id` / `agent_id` / `session_id`), `/delete_by_ids`
  (`{"memory_type": "episode", "memory_ids": ["<owner>_<entry_id>"]}`),
  `/memory/search`, `/memory/get`. Run Reflection now with
  `POST /api/v2/ome/trigger {"name": "reflect_episodes", "force": true}`.
- **Claude Code, the way users do it**: run everos-mcp against your port and
  start Claude Code with only that server, in a tmux socket of your own
  (`tmux -L <name>`), sending keys and reading the pane:

  ```json
  {"mcpServers": {"everos": {"command": "uv",
    "args": ["run", "--no-project", "--with", "everos-mcp==0.1.1",
             "python", "-m", "everos_mcp.server"],
    "env": {"EVEROS_BASE_URL": "http://127.0.0.1:18611",
            "EVEROS_USER_ID": "me", "EVEROS_SESSION_ID": "walk-1"}}}}
  ```

  `claude --mcp-config <that file> --strict-mcp-config`. A turn is finished
  when the pane shows its completion line (`... for Ns · done`), not when
  some `for Ns` appears (the thinking line has one too).
- **Racing a background task**: `make acceptance` (needs the two keys; the
  Milvus cases need docker) starts servers behind a local proxy that can hold
  one model call (`tests/acceptance/memory_delete/proxy.py`), so a deletion
  can land while an extraction or a Reflection merge is in flight. Check that
  the held call is still the only one of its kind when the deletion answers:
  a model call that times out retries past the hold.

## 4. Read the outcome where it lives

| Fact | Read it from |
|---|---|
| A memory exists | the markdown: `$ROOT/<app>/<project>/users/<id>/episodes/episode-<day>.md`, `.atomic_facts/`, `.foresights/`, `user.md`; agents under `agents/<id>/` (`tests/acceptance/memory_delete/observe.py` `md_entries` parses them) |
| Session, cluster, queue state | `$ROOT/.index/sqlite/system.db` (`memcell`, `cluster`, `cluster_member`, `memorize_inbox`, ...); OME runs in `ome.db` beside it |
| Index rows | LanceDB under `$ROOT/.index/lancedb/` (`observe.py` `lance_ids`); on Milvus read with `consistency_level="Strong"`, or you may see rows from before a deletion |
| Lock waits and holds | `GET /metrics`: `everos_lock_wait_seconds`, `everos_lock_hold_seconds` (label `kind`: `session`, `owner`, `partition:<strategy>`); compare before/after deltas |
| What the server complained about | `$ROOT.log`; it is coloured, so strip ANSI codes before grepping (`perl -pe 's/\e\[[0-9;]*m//g'`), or `[error` matches nothing |

A search that finds nothing is not proof of a deletion: read the markdown
and the index directly.

## 5. Tear down

Stop the pid you started, after checking it is yours, then wait for the port:

```bash
ps -o args= -p $(cat "$ROOT.pid")             # must show `everos server start`
kill $(cat "$ROOT.pid")
for i in $(seq 1 50); do lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null || break; sleep 0.2; done
tmux -L <name> kill-server                     # if you drove Claude Code
```

Never `pkill` by pattern: other sessions on the machine run their own
servers.

## Feature map

One row per feature a task has walked on a real server.

| Feature | Reach it | Drive it | End state that proves it works |
|---|---|---|---|
| Forget a conversation from Claude Code | everos-mcp `forget_session` (Section 3) | ask the model to remember a sentence (`add_memory`, wait true), then "forget this conversation" | no entry with that `session_id` in the markdown, no `.md` file contains the sentence, `search_memory` finds nothing |
| Delete a session (`session_id` only) | `POST /api/v2/memory/delete {"session_id": ...}` | while its episode extraction is held at the proxy | 200; the session's `memcell` rows gone; no entry of the session appears in the markdown after the hold is released |
| Delete a whole user / agent | `POST /api/v2/memory/delete {"user_id": ...}` | after seeding several kinds | 200 with `count`; only today's empty day files remain (they keep the id counters), `user.md` gone, the owner's `cluster` rows and index rows gone; `..` and `a/../b` answer 422 |
| Delete episodes by id with Reflection on | `POST /api/v2/memory/delete_by_ids` | while the Reflection merge call is held | no merged episode (`parent_type: cluster`) appears; the deleted source and its atomic facts are gone, the other source stays |
| Deletion latency | `tests/acceptance/memory_delete/test_t18_1_tb11_1_perf.py`, alone on the machine | a year of day files per user | each deletion <= 1 s; owner-lock holds of all session deletions p99 <= 20 ms, none over 100 ms |
| Purge one project | `POST /api/v2/memory/purge {"scope": "project", ...}`, then `/purge/confirm` with the nonce, poll `GET /purge/{job_id}`; `tests/acceptance/purge/` runs it with no credentials (fake LLM / embedding / rerank upstream, `make acceptance-purge`) | after seeding four scopes (`seed_purge_world`) | job `succeeded`; nothing of the project left by `s_leftovers` (markdown entries, `system.db` rows, index rows, ledger rows other than today's counter files, `ome.db` rows carrying its events' content), the other scopes' rows unchanged (`scope_dump`), a second request's summary all 0 |
| Nothing blocked during a purge | every route, v1 and v2, while a purge is held with `EVEROS_TEST_FAILPOINTS=purge.before_step8=pause` (release: create `$EVEROS_TEST_FAILPOINT_DIR/purge.before_step8.release`) | every route the running server lists in `/openapi.json` (`ENV=DEV`), except `/cascade/quiesce` (it stops the cascade) | each route answers the status code it answered before the purge; no 409; an `/add` to the purged project and to another one both 200 (`test_no_blocking.py`) |
| Purge resumed after a kill | `EVEROS_TEST_FAILPOINTS=purge.before_step3` (or `purge.after_step2b`) kills the server with 137 mid-purge; start it again without the point | a run of the scope held at the proxy before the kill | job `succeeded` with `attempts` 2; the held run's event has a `crashed` row and a replay dropped at the gate; the scope stays empty; no `.tmp/purge-*` left; new entries never reuse a purged id |
| Background backlog dropped by a purge | `EVEROS_OME__MAX_CONCURRENT_RUNS=1`; hold another project's extraction at the proxy (it takes the only slot), add agent sessions to the scope behind it, purge the scope (it does not wait for the other project), release the hold after the job ends (`test_ome.py::test_t9_1_backlog_never_writes_back`) | request + confirm + poll as above | job `succeeded` while the backlog is still queued; after it runs, `s_leftovers` is empty (no skill, cluster or fact written back, no event content in `ome.db`) |
| Old work in flight waited for | hold one of the scope's background runs at the proxy (e.g. `_SkillExtractReply`), confirm a purge, wait past a minute (`test_drain.py::test_t31_1_*`) | request + confirm + poll | job `running` with `heartbeat_at` moving until the hold is released, then `succeeded` and `s_leftovers` empty |
