# Local Agent Smoke Runbook

This runbook is for agent builders who want to prove that a local EverOS
service can be used safely by more than one runtime, such as a coding
assistant, a CLI agent, a desktop app, or a local orchestrator.

The goal is deliberately narrow:

1. write one scoped memory into a temporary local root;
2. force the markdown -> LanceDB index to catch up;
3. read the same memory from a second process;
4. delete the source markdown in the temporary root;
5. prove `/get` and `/search` no longer return it.

Use this before wiring EverOS into any automatic post-session hook. A
manual smoke test is cheaper than debugging an agent that silently wrote
to the wrong scope.

## Assumptions

- You have already installed and configured EverOS. See
  [QUICKSTART.md](../QUICKSTART.md).
- The server runs on the default loopback address,
  `http://127.0.0.1:8000`.
- The smoke uses a throwaway `EVEROS_MEMORY__ROOT`, so deleting the test
  markdown is safe.
- EverOS has no built-in authentication. Keep the server on loopback
  unless you place your own gateway in front. See
  [SECURITY.md](../SECURITY.md).

## 0. Keep loopback traffic off proxies

Many agent runtimes inherit `HTTP_PROXY` / `HTTPS_PROXY` from the
developer shell. Make sure local EverOS traffic stays local:

```bash
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost,::1"
curl --noproxy '*' http://127.0.0.1:8000/health
```

If the health check hangs only inside one agent runtime, check its proxy
environment first.

## 1. Start an isolated local root

Terminal A:

```bash
export EVEROS_MEMORY__ROOT="$(
  python3 -c 'import tempfile; print(tempfile.mkdtemp(prefix="everos-agent-smoke-"))'
)"
echo "$EVEROS_MEMORY__ROOT"
export EVEROS_MEMORY__TIMEZONE=UTC
everos server start --host 127.0.0.1 --port 8000
```

Using Python's `tempfile` avoids the `mktemp` flag/template differences
between GNU and BSD/macOS environments.

Terminal B:

```bash
export EVEROS_MEMORY__ROOT="<same value printed in Terminal A>"
export BASE_URL="http://127.0.0.1:8000"
export APP_ID="agent_smoke_app"
export PROJECT_ID="agent_smoke_project"
export USER_ID="agent_smoke_user"
export SESSION_ID="agent_smoke_session_$(date +%s)"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
curl --noproxy '*' "$BASE_URL/health"
```

Expected:

```json
{"status":"ok"}
```

## 2. Write and flush one memory

```bash
TS=$(($(date +%s)*1000))
curl --noproxy '*' -sS -X POST "$BASE_URL/api/v1/memory/add" \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\": \"$SESSION_ID\",
    \"app_id\": \"$APP_ID\",
    \"project_id\": \"$PROJECT_ID\",
    \"messages\": [
      {
        \"sender_id\": \"$USER_ID\",
        \"role\": \"user\",
        \"timestamp\": $TS,
        \"content\": \"Agent smoke marker: EverOS should remember that this runtime prefers local-first memory.\"
      },
      {
        \"sender_id\": \"$USER_ID\",
        \"role\": \"assistant\",
        \"timestamp\": $((TS+1000)),
        \"content\": \"Acknowledged. This is a scoped local smoke test, not production memory.\"
      }
    ]
  }"

curl --noproxy '*' -sS -X POST "$BASE_URL/api/v1/memory/flush" \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\": \"$SESSION_ID\",
    \"app_id\": \"$APP_ID\",
    \"project_id\": \"$PROJECT_ID\"
  }"
```

`/flush` is strong with respect to markdown persistence: when it returns
`status: "extracted"`, the episode markdown is on disk. `/search` and
`/get` read LanceDB, which is eventually consistent, so force the queue
before asserting read-your-write:

```bash
everos cascade sync
```

## 3. Read from another runtime

Open a different terminal, coding assistant, or local process. Reuse only
the target identifiers, not the previous process state:

```bash
export BASE_URL="http://127.0.0.1:8000"
export APP_ID="agent_smoke_app"
export PROJECT_ID="agent_smoke_project"
export USER_ID="agent_smoke_user"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"

curl --noproxy '*' -sS -X POST "$BASE_URL/api/v1/memory/search" \
  -H 'Content-Type: application/json' \
  -d "{
    \"user_id\": \"$USER_ID\",
    \"app_id\": \"$APP_ID\",
    \"project_id\": \"$PROJECT_ID\",
    \"query\": \"local-first memory preference\",
    \"top_k\": 5
  }"
```

Pass criteria:

- the response includes at least one `episodes[]` item;
- the episode belongs to `agent_smoke_user`;
- the summary or episode text contains the smoke marker;
- the result stays inside `agent_smoke_app` / `agent_smoke_project`.

For browsing instead of ranked recall:

```bash
curl --noproxy '*' -sS -X POST "$BASE_URL/api/v1/memory/get" \
  -H 'Content-Type: application/json' \
  -d "{
    \"user_id\": \"$USER_ID\",
    \"app_id\": \"$APP_ID\",
    \"project_id\": \"$PROJECT_ID\",
    \"memory_type\": \"episode\"
  }"
```

## 4. Delete source markdown and prove absence

Markdown is the source of truth; SQLite and LanceDB are derived indexes.
For this isolated smoke root, remove the generated episode markdown and
then force cascade to reconcile the deletion:

```bash
EPISODE_DIR="$EVEROS_MEMORY__ROOT/$APP_ID/$PROJECT_ID/users/$USER_ID/episodes"
find "$EPISODE_DIR" -type f -name 'episode-*.md' -print -delete
everos cascade sync
```

If `find` prints no episode file, stop and debug the write/flush step before
asserting deletion behavior.

Now prove both browse and search paths no longer return the deleted
episode:

```bash
curl --noproxy '*' -sS -X POST "$BASE_URL/api/v1/memory/get" \
  -H 'Content-Type: application/json' \
  -d "{
    \"user_id\": \"$USER_ID\",
    \"app_id\": \"$APP_ID\",
    \"project_id\": \"$PROJECT_ID\",
    \"memory_type\": \"episode\"
  }"

curl --noproxy '*' -sS -X POST "$BASE_URL/api/v1/memory/search" \
  -H 'Content-Type: application/json' \
  -d "{
    \"user_id\": \"$USER_ID\",
    \"app_id\": \"$APP_ID\",
    \"project_id\": \"$PROJECT_ID\",
    \"query\": \"local-first memory preference\",
    \"top_k\": 5
  }"
```

Pass criteria:

- `/get` returns `count: 0` or an empty `episodes` array for the smoke
  scope;
- `/search` returns no `episodes[]` item containing the smoke marker;
- `everos cascade status` shows no stuck pending or failed row for the
  deleted markdown path.

## What not to infer

- A passing smoke does not mean automatic session hooks are safe. It only
  proves the local service, scope ids, index catch-up, and absence path.
- Do not inspect `system.db` for memory content. SQLite stores state,
  buffers, audit, and the cascade queue; the durable user-visible memory
  is markdown, and search state is derived into LanceDB.
- Do not treat old cloud plugin examples as the current OSS integration
  contract. The supported local OSS API is documented in
  [api.md](api.md) and [migration-to-1.0.0.md](migration-to-1.0.0.md).
