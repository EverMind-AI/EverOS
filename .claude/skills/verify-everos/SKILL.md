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
  uv run everos server start --root "$ROOT" --port $PORT > "$ROOT.log" 2>&1 &
echo $! > "$ROOT.pid"
```

The shipped defaults point the LLM at OpenRouter and the embedding at
DeepInfra, so those two keys are all it needs. Scenes are off by default
(`docs/scenes.md`); to walk them, start with `EVEROS_SCENES__USERS=true`
(or `EVEROS_SCENES__GROUPS=true`).

## 2. Is it ready?

`curl -s http://127.0.0.1:$PORT/health` answers 200 within a few seconds.
Its `cascade.pending` is the number of markdown changes not yet in the index.
Before reading the index, poll it once a second until it reads 0 on two
consecutive samples (`docs/api.md`, eventual consistency): a single 0 can come
between a markdown write and its projection being queued. A server that exits instead: the last
lines of `$ROOT.log` (usually a missing key).

## 3. Drive it

- **HTTP**: `POST /api/v2/memory/add`, `/flush`, `/memory/forget` (a dry run
  unless `"dry_run": false`; a real one answers its `forget_id`) and
  `/memory/forget/undo`, the aliases `/delete` (`user_id` / `session_id`) and
  `/delete_by_ids` (`{"memory_type": "episode", "memory_ids":
  ["<owner>_<entry_id>"]}`), which forget too, `/memory/search` (multi-round
  retrieval, `method: "llm_multiround"`, the only method; it calls the
  `[decider]` model), `/memory/get`, `GET /memory/history`.
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

## 4. Read the outcome where it lives

| Fact | Read it from |
|---|---|
| A memory exists | the markdown: `$ROOT/<app>/<project>/users/<id>/episodes/episode-<day>.md`, `images/`, `scenes/`, `user.md`; a group's under `groups/<id>/`. A forget marks rather than removes: an entry is live only while its file's `deprecated_entries` does not name it (`live_markdown` in `tests/helpers/groups.py` reads a file that way) |
| Session, cluster, queue state | `$ROOT/.index/sqlite/system.db` (`memcell`, `cluster`, `cluster_member`, `memorize_inbox`, ...); OME runs in `ome.db` beside it |
| Index rows | LanceDB under `$ROOT/.index/lancedb/`; a forgotten or replaced entry keeps its row with `deprecated_by` set, and no read reaches it. On Milvus read with `consistency_level="Strong"`, or you may see rows from before a forget |
| Lock waits and holds | `GET /metrics`: `everos_lock_wait_seconds`, `everos_lock_hold_seconds` (label `kind`: `session`, `owner`, `partition:<strategy>`); compare before/after deltas |
| What the server complained about | `$ROOT.log`; it is coloured, so strip ANSI codes before grepping (`perl -pe 's/\e\[[0-9;]*m//g'`), or `[error` matches nothing |

A search that finds nothing is not proof of a forget: read the markdown and
the index directly.

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
| Forget a conversation from Claude Code | everos-mcp `forget_session` (Section 3) | ask the model to remember a sentence (`add_memory`, wait true), then "forget this conversation" | every entry with that `session_id` is marked `forgotten` in its file's `deprecated_entries`, no live entry holds the sentence, `search_memory` finds nothing; `/memory/forget/undo` with the answer's `forget_id` brings it back |
| Delete a session (`session_id` only) | `POST /api/v2/memory/delete {"session_id": ...}` | after seeding the session | 200 with the forgets' `forget_ids`; the session's entries marked `forgotten`, its index rows `deprecated_by` set, nothing of it in a search; the undo clears the marks |
| Delete a whole user | `POST /api/v2/memory/delete {"user_id": ...}` | after seeding several kinds | 200 with `count` and a `forget_id`; every current entry marked `forgotten`, `user.md` emptied (its values kept in the forget log); `..` and `a/../b` answer 422 |
