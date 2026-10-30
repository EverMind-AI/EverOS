# EverOS HTTP API (v2)

Human-readable reference for the EverOS HTTP API. Schema names, types
and validation constraints mirror the OpenAPI spec served at
`GET /openapi.json` (when the server runs with `ENV=DEV`; see
[OpenAPI spec source](#openapi-spec-source)). This document adds the
business semantics the raw spec does not carry.

## Table of contents

- [Overview](#overview)
  - [Base URL and versioning](#base-url-and-versioning)
  - [Content type](#content-type)
  - [Authentication](#authentication)
  - [Response envelope](#response-envelope)
  - [Eventual consistency](#eventual-consistency)
  - [Conventions](#conventions)
  - [ScopeId: `app_id` and `project_id`](#scopeid-app_id-and-project_id)
- [Errors](#errors)
- [Common types](#common-types)
  - [MessageItem](#messageitem)
  - [ContentItem](#contentitem)
  - [ToolCall](#toolcall)
  - [ToolFunction](#toolfunction)
  - [FilterNode (filter DSL)](#filternode-filter-dsl)
  - [SearchMethod](#searchmethod)
  - [GetMemoryType](#getmemorytype)
- [Endpoints](#endpoints)
  - [POST /api/v2/memory/add](#post-apiv2memoryadd)
  - [POST /api/v2/memory/flush](#post-apiv2memoryflush)
  - [Asynchronous mode](#asynchronous-mode)
  - [GET /api/v2/tasks/{task_id}](#get-apiv2taskstask_id)
  - [GET /api/v2/tasks](#get-apiv2tasks)
  - [POST /api/v2/memory/search](#post-apiv2memorysearch)
  - [POST /api/v2/memory/get](#post-apiv2memoryget)
  - [POST /api/v2/ome/trigger](#post-apiv2ometrigger)
  - [Knowledge endpoints](#knowledge-endpoints)
- [OpenAPI spec source](#openapi-spec-source)

## Overview

### Base URL and versioning

| Setting | Default | Override |
|---|---|---|
| Host | `127.0.0.1` (loopback only) | `EVEROS_API__HOST` env var or `--host` flag |
| Port | `8000` | `EVEROS_API__PORT` env var or `--port` flag |
| Version prefix | `/api/v2` | — |

Business endpoints live under `/api/v2/memory/`, `/api/v2/ome/`, and
`/api/v2/knowledge/`. Knowledge endpoints have their own dedicated
reference at [docs/knowledge.md](knowledge.md) and are cross-referenced
below. The operational endpoints `GET /health`, `GET /metrics` and
`POST /api/v2/cascade/quiesce` exist but are intentionally outside this
reference — they are runtime probes and shutdown hooks for deployment, not
part of the application contract. `quiesce` drains the md → index queue and
stops the watcher; calling it twice is safe, and it returns a bare status body
(`quiesced`, `drained`, `pending_before`, `pending_after`,
`failed_permanent`), see `docs/openapi.json`. One note for anyone keying off
`/health`: `capabilities.embed` reports whether the embedding service
answered a dimension probe **when the process started**, not whether it is
reachable now -- a service that recovers later keeps reading `false` until
the next restart. There is no `capabilities.llm`: the server does not start
without an extraction LLM, so a running server always has one.

`/api/v2` is the canonical prefix, aligned with the EverOS Cloud API. Every
business endpoint is **also** served under `/api/v1`, kept as a legacy
compatibility alias: the two prefixes resolve to the same handlers with
identical request/response contracts, so existing `/api/v1` integrations keep
working today. The alias carries no long-term guarantee — it may be removed in
a future major release, and it will be announced in the changelog with a
deprecation window before that happens. Write new integrations against
`/api/v2`, and migrate existing ones when convenient.

### Content type

All `POST` endpoints require `Content-Type: application/json`. Request
and response bodies are UTF-8 JSON.

### Authentication

EverOS ships **no built-in authentication**. The server binds to
`127.0.0.1` by default; place your own gateway or auth layer in front
before exposing the API on any other interface. See
[../SECURITY.md](../SECURITY.md) for the threat model.

### Response envelope

Successful responses from the business endpoints (`/memory/*`,
`/knowledge/*`) wrap the payload in the envelope below (`200 OK`, or
`201 Created` for a knowledge document upload). `POST /api/v2/ome/trigger`
and the operational endpoints return their body bare — see each section.

```json
{
  "request_id": "<32-char hex>",
  "data": { /* endpoint-specific payload */ }
}
```

`request_id` is generated server-side (32 lowercase hex chars) and is
echoed by structured logs / tracing / metrics, so a single request can
be correlated end-to-end. Error responses carry their own envelope —
`request_id` at the top level alongside a nested `error` object (not a
bare FastAPI `detail`); see [Errors](#errors).

### Eventual consistency

`/add` and `/flush` write the markdown file (the source of truth)
**synchronously** — when the call returns with `status: "extracted"`,
the new entry exists on disk. The configured vector / BM25 / scalar
index backend is rebuilt by the in-process **cascade coroutine
asynchronously**.

That means `/search` and `/get` may not see a record immediately after
the `/flush` that produced it. Typical sync latency is sub-second, but
under load it can reach ~10–15 seconds. If you need read-your-write
semantics, retry with backoff. The markdown file is durable
regardless; index lag never loses data.

The same applies one level up for everything an OME strategy produces
**after** the extraction call returns: agent cases and skills (agent track),
profiles, atomic facts. `/flush` returns once the episode is on disk; the
`extract_agent_case` strategy then runs its own LLM call, and skill
clustering runs after that, so `/get` with `memory_type: "agent_case"` can
legitimately report `total_count: 0` for tens of seconds after a flush that
returned `"extracted"` (14 s for the case and 32 s for the skill on a
laptop with a hosted LLM). `GET /health`'s `cascade.pending` does **not**
cover this window — it only tracks the md → index queue, which is empty
while the strategy is still thinking. Poll `/get` with a budget instead.
The agent track also has extraction gates of its own, applied before any
LLM call: a trajectory with fewer than three tool-call rounds is rejected as
too simple (`AgentCaseExtractor(min_tool_call_rounds=3)`, the EverOS
default — so a session with no tool use never yields a case), and so is one
that does not end on an assistant text message. Trajectories that pass
those gates can still be dropped by the extractor's own "worth extracting"
LLM filter. EverOS logs the skip as `agent_case_skipped_by_algo` with the
memcell and session ids; the reason itself is on the adjacent everalgo log
line, `skipping memcell (n_items=N): <reason> <detail>`. The same
"seconds later" rule for every OME-produced kind is summarised in
[how-memory-works.md](how-memory-works.md).

### Conventions

#### Timestamps

EverOS runs a **two-zone discipline**: every stored byte is UTC; every
rendered value carries the configured **display timezone**. The
display timezone is set by `EVEROS_MEMORY__TIMEZONE` (env var) or
`[memory] timezone` (TOML); default `"UTC"`.

| Direction | Format | Notes |
|---|---|---|
| **Request** — `messages[].timestamp` (`/add`) | Integer, **Unix epoch milliseconds (ms)** | Server auto-detects `>= 10^12` as ms for backward compat, but the contract is ms |
| **Request** — `filters.timestamp.{gte,lt,…}` (`/search` / `/get`) | Integer, **Unix epoch ms** *or* ISO-8601 string | Same auto-detect; pick whichever you have |
| **Response** — every `timestamp` field | ISO-8601 string with **explicit timezone offset** (`+HH:MM` or `Z`) | Always rendered in the configured display tz; never naive |

With the default `EVEROS_MEMORY__TIMEZONE=UTC` an episode produced
at 11:30 UTC renders as `"2026-05-28T11:30:36Z"` (Pydantic
canonicalises `timezone.utc` to `Z`). Switch to
`EVEROS_MEMORY__TIMEZONE=Asia/Shanghai` and the same UTC instant
renders as `"2026-05-28T19:30:36+08:00"` — bytes on disk are
unchanged.

**Fallback for naive input.** If you submit a `filters.timestamp` ISO
string without an offset (e.g. `"2026-05-28T11:30:00"`), the server
treats it as already display-tz-local before comparing against
storage. This is the same rule users see when reading rendered output:
"if you didn't say a zone, we assume your zone."

**For internal architecture.** The storage / display split lives in
`everos.core.utils.datetime`
(`get_utc_now` / `ensure_utc` / `UtcDatetime` for storage;
`get_now_with_timezone` / `to_display_tz` for display). See
[datetime.md](datetime.md) for the design rationale.

#### Other conventions

- **Server-generated IDs** follow `<owner>_<kind>_<YYYYMMDD>_<NNN>`,
  e.g. `alice_ep_20260528_00000001` for an episode, `alice_af_...`
  for an atomic fact. See
  [storage_layout.md §4](storage_layout.md) for the encoding.
- **All endpoints are POST** for `/api/v2/memory/*` even when the
  semantics look like a read (`/search`, `/get`) — the request bodies
  are too rich (filters, methods, paging) to encode in a query string.

### ScopeId: `app_id` and `project_id`

`app_id` and `project_id` partition memory at the disk layer. Every
write lands under
`~/.everos/<app>/<project>/users/<user_id>/...` (or `.../agents/...`
for the agent track). The default scope materialises on disk as
`default_app` / `default_project` (the `_app` / `_project` suffix is
added only for the literal id `"default"` so the default space stays
visually distinct from user-named scopes).

A `/search` or `/get` query never crosses scopes — different
`(app_id, project_id)` pairs are isolated.

Both fields share the same validation:

| Setting | Value |
|---|---|
| Type | `string` |
| Default | `"default"` |
| Length | 1–128 chars |
| Charset | `^[a-zA-Z0-9_.-]+$` |
| Rejected literals | `"."` and `".."` (path-traversal guard) |

## Errors

Every non-2xx response uses a uniform error envelope — `request_id` at
the top level (mirroring the success envelope) alongside a nested
`error` object. It is **not** a bare FastAPI `detail`:

```json
{
  "request_id": "<32-char hex>",
  "error": {
    "code": "NOT_FOUND",
    "message": "Document 'abc123' not found",
    "timestamp": "2026-06-01T12:24:46+00:00",
    "path": "/api/v2/knowledge/documents/abc123"
  }
}
```

### error.code values

`error.code` is a machine-readable `ErrorCode` enum. Clients can switch
on this value to decide retry / display / routing behaviour without
parsing the human-readable `message` field.

| `error.code` | HTTP | Retryable? | When |
|---|---|---|---|
| `NOT_FOUND` | `404` | No | Requested resource does not exist |
| `CONFLICT` | `409` | No | Operation conflicts with existing state (e.g. duplicate document) |
| `INVALID_INPUT` | `422` | No | Request-body validation failure. Also covers `/search` / `/get` filter-DSL compile errors — the compile reason rides in `message` — and, on `/add` and `/flush`, a session whose buffered messages the model could not segment on three consecutive calls: the buffer is moved to the `quarantine` track of `unprocessed_buffer` so the session can continue, and `message` names the session and the number of messages set aside |
| `EXTRACTION_EMPTY` | `422` | No | Knowledge document parsed but the extractor produced no topics. An empty or whitespace-only upload is rejected earlier as `INVALID_INPUT` (`"Uploaded file has no valid content."`) |
| `BAD_REQUEST` | `400` | No | Path traversal attempt or other malformed input |
| `UNSUPPORTED_FORMAT` | `415` | No | File format or modality not supported (e.g. unsupported `ContentItem` type, missing `ext` for `base64`) |
| `PAYLOAD_TOO_LARGE` | `413` | No | An `async_mode` request body exceeds `memorize.async_queue.max_payload_bytes`; the batch was not accepted |
| `QUEUE_SATURATED` | `429` | Yes | The memorize queue is at its per-session or global cap; the batch was not accepted — retry later or keep it client-side |
| `EXTERNAL_SERVICE_UNAVAILABLE` | `503` | **Yes** | An external service (LLM, embedding, rerank) returned an error or timed out |
| `CAPABILITY_UNAVAILABLE` | `503` | No | A required server-side capability is missing (e.g. `everos[multimodal]` extra not installed, LibreOffice absent) — requires admin action, not retry |
| `CONFIGURATION_ERROR` | `500` | No | A required configuration is missing or invalid (e.g. embedding model not set) |
| `INTERNAL_ERROR` | `500` | No | Unhandled exception (internal details are logged, never leaked) |

### error object

| Field | Type | Description |
|---|---|---|
| `code` | `string` | One of the `ErrorCode` values listed above |
| `message` | `string` | Human-readable reason. For `INVALID_INPUT` from request validation, **only the first** validation error is surfaced, formatted `"<msg>: <dotted-loc>"` with the leading `body` segment stripped (e.g. `"Field required: messages"`); a model-level validator with no field location surfaces just `"<msg>"` (e.g. `"Value error, exactly one of user_id / agent_id must be provided"`) |
| `timestamp` | `string` | ISO-8601 with timezone offset (display tz) |
| `path` | `string` | Request path, e.g. `/api/v2/memory/add` |

> Unlike FastAPI's default, the full per-field validation array is **not**
> returned — only the first error's message. A client that needs the
> offending field can read the `<loc>` suffix in `message`.

## Common types

### MessageItem

One turn in a `/add` batch. Shape mirrors the OpenAI Chat Completions
message structure plus a stable `sender_id` for indexing.

| Field | Type | Required | Default | Constraints |
|---|---|---|---|---|
| `sender_id` | `string` | yes | — | `minLength=1` |
| `sender_name` | `string \| null` | no | `null` | — |
| `role` | `"user" \| "assistant" \| "tool"` | yes | — | — |
| `timestamp` | `integer` | yes | — | `> 0` — **Unix epoch milliseconds (ms)** per v1 contract |
| `content` | `string \| array<ContentItem>` | yes | — | — |
| `tool_calls` | `array<ToolCall> \| null` | no | `null` | — |
| `tool_call_id` | `string \| null` | no | `null` | — |

**`sender_id`** — Stable identifier of the entity producing this turn.
For `role: "user"`, this is the `user_id` the server will index the
extracted memory under: markdown lands at
`users/<sender_id>/episodes/...`, and `/search` / `/get` queries with
`user_id: "<sender_id>"` reach it. For `role: "assistant"` or
`"tool"`, the field is informational (the LLM sees it during
extraction but it is not used as an indexing key).

**`sender_name`** — Optional human-readable display name. Lets the LLM
use a real name during extraction without changing the indexing key.
For example, set `sender_id: "u_42"` (stable internal id) and
`sender_name: "Alice"` (what the LLM sees).

**`role`** — Speaker role; one of:
- `"user"` — content originates from a human (or human-acting client).
- `"assistant"` — content from the AI assistant.
- `"tool"` — output of a tool call; pair with `tool_call_id`.

**`timestamp`** — Wall-clock anchor in **Unix epoch milliseconds (ms)**
per the v1 API contract. Used by extraction to anchor the resulting
episode and by the daily-log writer to bucket the entry into the right
file (`episode-<YYYY-MM-DD>.md` etc.).

> Implementation note: the algo layer auto-detects seconds vs ms (values
> `>= 10^12` are treated as ms, smaller as seconds) for backward compat,
> but **clients should send ms** to honour the contract.

**`content`** — The message body.
- A bare **string** is shorthand for a single text content item.
- An **array of `ContentItem`** is for mixed-modality input (text +
  image / pdf / audio / ...); non-text items are parsed by the
  multimodal LLM configured via `EVEROS_MULTIMODAL__*` env vars. See
  [ContentItem](#contentitem).

**`tool_calls`** — When `role: "assistant"`, the tool calls the
assistant emitted in this turn (OpenAI Chat Completions shape).

**`tool_call_id`** — When `role: "tool"`, the `id` of the call this
message is the response to. Required for `role: "tool"`; a tool row
without it is rejected with `422`.

### ContentItem

Mixed-modality message-body element. Carry the payload in exactly one
of `text` / `uri` / `base64`; the others must be `null`. For
`type: "text"` use `text`; for every **non-text** type use `uri`
(`http(s)://`) or `base64` (with `ext`). Non-text items are routed
through the multimodal parser, which needs a fetchable or decodable
payload — a non-text item carrying only `text` returns `415`.

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `type` | `"text" \| "image" \| "audio" \| "doc" \| "pdf" \| "html" \| "email"` | yes | — | — |
| `text` | `string \| null` | no | `null` | Required when `type: "text"` |
| `uri` | `string \| null` | no | `null` | `http(s)://` (fetched server-side) or `file://` (read from the server's local fs, guardrailed) pointer |
| `base64` | `string \| null` | no | `null` | Inline payload, plain base64 (no `data:` prefix) |
| `ext` | `string \| null` | no | `null` | File-extension hint when `uri` lacks one |
| `name` | `string \| null` | no | `null` | Display filename, used in logs |
| `extras` | `object \| null` | no | `null` | Provider-specific metadata, opaque to EverOS |

**`type`** — The content kind. Each non-text type is dispatched to the
multimodal LLM. If the multimodal endpoint cannot handle the supplied
payload, `/add` returns `415 Unsupported Media Type`.

**`text`** — The literal text payload; valid **only** for
`type: "text"`. A non-text type (including `"html"`) is always routed
to the parser and must carry `uri` or `base64`; passing only `text` on
a non-text item returns `415`. To inline HTML as plain text, send it
as `type: "text"`.

**`uri`** — `http(s)://` or `file://` pointer to the asset. An
`http(s)` uri is fetched by the server and dispatched by the response
Content-Type (use it for assets hosted elsewhere — S3 / OSS presigned
URL, http server). A `file://` uri is read from the **server's** local
filesystem (the path must be reachable by the server process), subject
to these guardrails:

- the resolved path (symlinks followed) must be an existing regular file;
- its size must be ≤ `EVEROS_MULTIMODAL__FILE_URI_MAX_BYTES` (default 50 MiB);
- when `EVEROS_MULTIMODAL__FILE_URI_ALLOW_DIRS` is set, the path must lie
  within one of the listed roots (unset = any readable file, the
  local-first default).

A guardrail violation surfaces as `415`. Mutually exclusive with `base64`.

**`base64`** — Inline binary payload, base64-encoded (no `data:`
prefix). Mutually exclusive with `uri`.

> **Size caution.** base64 inflates the payload ~4/3× on the wire, and
> the encoded blob is held **verbatim in the server's staging buffer
> (SQLite) from `/add` until the session is flushed** — a multi-MB PDF
> becomes multi-MB of SQLite text for the buffer's lifetime. It is *not*
> persisted past extraction (the memory cell and episode store only the
> parsed text, never the raw bytes), but the transient footprint is
> real, and a large inline blob also slows request parsing. **Prefer
> `uri` (`http(s)://`) for large assets**: the buffer then holds only
> the URL plus the parsed text, and the bytes are fetched transiently at
> parse time rather than stored. Reserve `base64` for small assets or
> when no reachable URL exists.

**`ext`** — File-extension hint (`"pdf"`, `"png"`, `"html"`, ...). For
`base64` payloads this drives modality dispatch and is effectively
**required** (without it the server falls back to `mime`, then `415`s
if neither resolves). For `uri` payloads it is optional — the fetched
Content-Type usually suffices.

**`name`** — Filename / human label for logging and traceability.
Does not affect parsing.

**`extras`** — Free-form bag of provider-specific metadata (e.g.
caption, page hints). Opaque to EverOS; passed through to the
multimodal LLM context.

### ToolCall

OpenAI-shaped tool invocation attached to an assistant turn.

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `id` | `string` | yes | — | Stable id; echoed by the paired `role: "tool"` turn via `tool_call_id` |
| `type` | `string` | no | `"function"` | Only `"function"` is meaningful today |
| `function` | [ToolFunction](#toolfunction) | yes | — | The function being called |

### ToolFunction

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | `string` | yes | Function name as registered with the agent |
| `arguments` | `string` | yes | **JSON-encoded string** (OpenAI convention — not an object) |

### FilterNode (filter DSL)

A recursive boolean tree of predicates. Used by `/search.filters` and
`/get.filters`. The Pydantic envelope only checks the recursive
combinator shape; field-level validity (which scalar fields are
filterable, which operators apply, value coercion) runs when the
node is compiled to a backend-specific filter clause server-side. Compile
errors surface as `422` with the offending field / operator in
`error.message`.

#### Shape

A node is a JSON object whose keys are one of:

| Key | Value | Notes |
|---|---|---|
| `AND` | `array<FilterNode>` | All child nodes must match. Omit if not needed |
| `OR` | `array<FilterNode>` | At least one child node must match. Omit if not needed |
| *<allowed field>* | scalar or operator map | Predicate on that field — see [Allowed fields](#allowed-fields) and [Operators](#operators) |

`AND`, `OR`, and scalar predicates **mix freely at the same level**;
they are implicitly joined with `AND`. A node with only scalar keys is
fine — `AND` / `OR` arrays are not required.

Example — `session_id == "demo-001"` AND (`sender_id` contains `"alice"` OR `"bob"`):

```json
{
  "session_id": "demo-001",
  "OR": [
    { "sender_id": "alice" },
    { "sender_id": "bob" }
  ]
}
```

<a id="filter-allowed-fields"></a>

#### Allowed fields

These are the only field names that may appear as a predicate key.
Anything else returns `422`.

| Field | Column kind | Value type | Notes |
|---|---|---|---|
| `session_id` | string | `string` | Session the row was extracted from |
| `parent_type` | string | `string` | Parent kind, e.g. `"memcell"` |
| `parent_id` | string | `string` | Parent id |
| `timestamp` | timestamp | `integer` (Unix epoch **ms** per v1 contract) or `string` (ISO-8601) | Implementation auto-detects `>= 10^12` as ms / smaller as sec for backward compat, but the contract is ms |
| `sender_id` | array of strings | `string` | Matches via `array_has(sender_ids, <value>)` — checks whether the row's `sender_ids` list contains the given id |

The following names are **reserved** and rejected inside `filters`.
They must be set at the top of the request (`/search` or `/get`),
not inside the DSL:

```
owner_id   owner_type   app_id   project_id
```

<a id="filter-operators"></a>

#### Operators

A predicate value may be a **scalar shorthand** (equality) or an
**operator map** (multiple operators in one dict are AND-joined):

| Operator | SQL | Applies to |
|---|---|---|
| `eq` | `=` | string, timestamp; on `sender_id` becomes `array_has(...)` |
| `ne` | `!=` | string, timestamp |
| `gt` | `>` | timestamp |
| `gte` | `>=` | timestamp |
| `lt` | `<` | timestamp |
| `lte` | `<=` | timestamp |
| `in` | `IN (...)` | string (list of values); on `sender_id` becomes `OR` of `array_has` |

Examples:

```json
// Equality shorthand
{ "session_id": "demo-001" }

// Operator map (operators inside one dict are AND-joined)
{ "timestamp": { "gte": 1748390400000, "lt": 1748400000000 } }

// IN list
{ "session_id": { "in": ["demo-001", "demo-002"] } }

// Array-membership on sender_ids
{ "sender_id": "alice" }                       // array_has(sender_ids, 'alice')
{ "sender_id": { "in": ["alice", "bob"] } }    // OR of two array_has
```

### SearchMethod

| Value | Behaviour |
|---|---|
| `"keyword"` | BM25 only — pure lexical match, no embedding cost |
| `"vector"` | Dense vector ANN only — semantic recall, no lexical |
| `"hybrid"` *(default)* | Reciprocal-rank fuse of BM25 + vector + optional scalar filter against the configured derived index backend |
| `"agentic"` | Iterative cluster-path retrieval driven by a cross-encoder rerank loop; higher quality at higher latency / cost |

`"hybrid"` is the default because it balances recall and precision
without requiring the agentic loop. `"agentic"` calls the LLM in a loop
and should be reserved for offline or background workflows.

### GetMemoryType

| Value | Track | Returned in `data.<plural>` |
|---|---|---|
| `"episode"` | user | `data.episodes` — [GetEpisodeItem](#getepisodeitem) |
| `"profile"` | user | `data.profiles` — [GetProfileItem](#getprofileitem) |
| `"agent_case"` | agent | `data.agent_cases` — [GetAgentCaseItem](#getagentcaseitem) |
| `"agent_skill"` | agent | `data.agent_skills` — [GetAgentSkillItem](#getagentskillitem) |

`memory_type` must match the requested owner kind: `"episode"` /
`"profile"` require `user_id`; `"agent_case"` / `"agent_skill"`
require `agent_id`. The mismatching combinations are rejected with
`422`.

## Endpoints

### POST /api/v2/memory/add

Append a batch of messages to a session buffer. The server
accumulates messages until the boundary detector decides the session
is complete (or you force it with `/flush`); on boundary the LLM
extracts a memory cell and the writers persist markdown + index rows.

The response distinguishes "buffered" from "extracted this call" via
the `status` field — see [Response body](#response-body) below.

#### Request body

| Field | Type | Required | Default | Constraints |
|---|---|---|---|---|
| `session_id` | `string` | yes | — | length 1–128 |
| `app_id` | `string` *(ScopeId)* | no | `"default"` | see [ScopeId](#scopeid-app_id-and-project_id) |
| `project_id` | `string` *(ScopeId)* | no | `"default"` | see [ScopeId](#scopeid-app_id-and-project_id) |
| `messages` | `array<MessageItem>` | yes | — | 1–500 items |
| `defer_extraction` | `boolean` | no | `false` | buffer durably without boundary or extraction LLM calls |
| `async_mode` | `boolean` | no | `false` | queue the batch and return `202` immediately; see [Asynchronous mode](#asynchronous-mode) |

**`session_id`** — Identifies the conversation buffer on the server.
Messages POSTed with the same `(session_id, app_id, project_id)`
accumulate into one extraction batch. Different `session_id`s never
share a buffer, even within the same scope.

**`app_id` / `project_id`** — Scope identifiers; see
[ScopeId](#scopeid-app_id-and-project_id).

**`messages`** — Ordered list of [MessageItem](#messageitem) to
append. Order matters (it is preserved when the buffer is later
extracted). The 1–500 cap is a per-request safety bound, not a
session lifetime cap — you can call `/add` many times for the same
`session_id`.

**`defer_extraction`** — When `true`, `/add` only merges the messages
into the durable SQLite `unprocessed_buffer`. It skips boundary
detection, memcell creation, and all extraction pipelines, so batching
clients can capture every turn cheaply and call `/flush` after an idle
window, a size threshold, or a session switch. A successful deferred
add always reports `status: "accumulated"`.

**`async_mode`** — When `true`, `/add` persists the batch to the durable
memorize queue and returns `202 Accepted` without running boundary
detection or extraction; a background worker delivers queued batches in
arrival order per session, several sessions at a time. See
[Asynchronous mode](#asynchronous-mode) for the `202` body, the ordering
rule and how to poll progress.

#### Response body

`200 OK` returns a SuccessEnvelope wrapping:

| Field | Type | Notes |
|---|---|---|
| `message_count` | `integer` | Number of messages accepted in this call |
| `status` | `"accumulated" \| "extracted"` | What the server did with the batch |

**`message_count`** — Always equal to `len(messages)` on a `200` —
there is no partial accept. The field is present mainly for log
correlation.

**`status`** — Two-state outcome:
- `"accumulated"` — Messages were added to the buffer; the boundary
  detector did **not** trigger this call. The buffer remains open.
- `"extracted"` — During this call the boundary detector tripped; the
  LLM ran extraction and the writers persisted at least one memory
  cell. The buffer is empty after the call returns.

`202 Accepted` — only with `async_mode`, or when the session already has
queued batches (see [Asynchronous mode](#asynchronous-mode)) — returns a
SuccessEnvelope wrapping:

| Field | Type | Notes |
|---|---|---|
| `status` | `"queued"` | The batch is persisted; delivery happens in the background |
| `message_count` | `integer` | Number of messages accepted in this call |
| `pending` | `integer` | Unfinished queued batches of this session, this one included |

The envelope's `request_id` is the handle for
[`GET /tasks/{task_id}`](#get-apiv2taskstask_id).

#### cURL example

```bash
TS=$(( $(date +%s) * 1000 ))
curl -X POST http://127.0.0.1:8000/api/v2/memory/add \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\": \"demo-002\",
    \"app_id\": \"default\",
    \"project_id\": \"default\",
    \"messages\": [
      {\"sender_id\": \"alice\", \"role\": \"user\", \"timestamp\": $TS, \"content\": \"I love climbing in Yosemite every spring.\"},
      {\"sender_id\": \"alice\", \"role\": \"user\", \"timestamp\": $((TS+10)), \"content\": \"My favorite coffee shop is Blue Bottle in SOMA.\"},
      {\"sender_id\": \"alice\", \"role\": \"user\", \"timestamp\": $((TS+20)), \"content\": \"I bike to work most days.\"}
    ]
  }"
```

Response (real capture):

```json
{
    "request_id": "ae78d3f689c941eea135893e702fd171",
    "data": {
        "message_count": 3,
        "status": "accumulated"
    }
}
```

### POST /api/v2/memory/flush

Force the boundary detector to decide **now** for the given session
buffer. The LLM runs extraction (one call) regardless of whether the
heuristic would have tripped on its own. Useful at the end of a chat
or agent run to make sure pending context becomes durable memory.

#### Request body

| Field | Type | Required | Default | Constraints |
|---|---|---|---|---|
| `session_id` | `string` | yes | — | length 1–128 |
| `app_id` | `string` *(ScopeId)* | no | `"default"` | — |
| `project_id` | `string` *(ScopeId)* | no | `"default"` | — |

**`session_id`** — Identifies which buffer to flush. Must match the
`session_id` of prior `/add` calls in the same `(app_id, project_id)`
scope.

**`app_id` / `project_id`** — Scope identifiers; see
[ScopeId](#scopeid-app_id-and-project_id).

#### Response body

| Field | Type | Notes |
|---|---|---|
| `status` | `"extracted" \| "no_extraction"` | What happened |

**`status`**:
- `"extracted"` — The buffer had at least one message; extraction ran
  and produced at least one memory cell.
- `"no_extraction"` — The buffer was empty (no prior `/add` calls for
  this `(session_id, app_id, project_id)`, or it was already flushed).

`202 Accepted` with the same body as `/add` (`message_count` is `0`) only
when the session already has queued batches, so the flush runs after them;
`/flush` has no `async_mode` of its own. See
[Asynchronous mode](#asynchronous-mode).

A `200` from `/flush` is synchronous with respect to markdown persistence: by the
time the response returns, the new entry is on disk. Derived index
sync is still asynchronous — see
[Eventual consistency](#eventual-consistency).

#### cURL example

```bash
curl -X POST http://127.0.0.1:8000/api/v2/memory/flush \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo-002","app_id":"default","project_id":"default"}'
```

Response (real capture; this call took several seconds because of the
extraction LLM call):

```json
{
    "request_id": "e65bcf4f56c042e39cdf50866810672c",
    "data": {
        "status": "extracted"
    }
}
```

### Asynchronous mode

`async_mode: true` on `/add` turns the call into "accept and return": the
request body is written to the SQLite `memorize_inbox` table in one
transaction and the response is `202 Accepted`. `/flush` has no
`async_mode` (the Cloud API has no asynchronous flush either): it runs
inline, except that a flush on a session with queued batches joins the
queue behind them and answers `202` (see Ordering). An in-process
worker then delivers queued batches to the same code path a synchronous
call uses: each session's batches in the order they were accepted, several
sessions at a time. The field name, the `202` body
(`status: "queued"` plus the envelope's `request_id`) and the status
vocabulary (`queued` / `processing` / `success` / `failed`) match the
EverOS Cloud gateway, so one client can talk to both.

**Ordering.** Batches of one session are delivered strictly in arrival
order, and a `/flush` runs after every `/add` accepted before it. To keep
that true, a synchronous call (`async_mode` omitted or `false`) on a
session that still has queued batches is queued too and answered with
`202` — otherwise it would overtake work the client was already told is
accepted. Clients that never use `async_mode` never see this: their
sessions never have queued batches. Sessions are independent of each
other: up to `max_concurrency` batches (default 4) run at once, never two
from one session, so one slow session does not hold up the others, and
there is no ordering between sessions. Like the Cloud consumer, which runs
groups in parallel and each group in order.

**Backpressure.** The queue is bounded. A batch is rejected before it is
stored when:

| Condition | HTTP | `error.code` | Default limit |
|---|---|---|---|
| Serialised body too large | `413` | `PAYLOAD_TOO_LARGE` | 4 MiB |
| Session has too many unfinished batches | `429` | `QUEUE_SATURATED` | 50 |
| Process has too many unfinished batches | `429` | `QUEUE_SATURATED` | 200 |

The caps are checked before the insert without a lock, so a burst of
concurrent requests can overshoot them by a few rows. A rejected batch was
**not** accepted: retry later or keep it client-side.
`pending` in every `202` body lets a client slow down before hitting `429`.
Limits are configured under `[memorize.async_queue]`, see
[configuration.md](configuration.md#memorizeasync_queue).

**Failure.** A batch whose delivery raises is marked `failed` and skipped
so its session keeps moving; `GET /tasks/{request_id}` shows `error`.
There is no automatic retry by default (`max_attempts = 1`), because the
memorize pipeline is not atomic: boundary detection writes the memcell and
consumes the session buffer before extraction runs. A failure before that
point (the common one: the boundary LLM call failed) leaves nothing behind,
and re-sending the batch is safe. A failure after it leaves the memcell
without its episode; re-sending then cuts a second memcell, and a `/flush`
finds an empty buffer. `last_error` does not say which step failed. The
server never re-drives a failed batch and deletes its row with the others
after seven days. Synchronous calls have the same property; they just hand
the decision to the client as a 5xx.

**Delivery is at-least-once across a crash.** A row that was in flight
when the process died is delivered again on the next start. If the first
delivery had already written its memcell, an `/add` replay duplicates that
memory; a `/flush` replay finds an empty buffer and does nothing, so a crash
that fell between the memcell write and the episode write leaves that
memcell without its episode. A
normal shutdown waits up to 30 seconds for the rows in flight (up to
`max_concurrency` of them, one per session); a row that takes longer, or a
process manager that kills earlier (`docker stop` sends SIGKILL after 10
seconds by default), is replayed the same way, so one restart can replay
up to `max_concurrency` batches. So is a
batch whose delivery had just failed when the process died before its
`failed` mark was written, and a batch whose status write was lost (see
Stuck rows) if the server restarts before the row is marked. Replays are
bounded: a batch that has been replayed `max_replays` times (default 3)
without ever finishing is marked `failed` with `error_code`
`replays_exhausted` instead of being requeued again, so a batch that kills
the process cannot crash-loop it.

**Operations.** `GET /health` carries a `memorize_queue` block: `healthy`
(the delivery loop is alive), `pending`, `oldest_pending_age_s`, `failed`,
`loop_restarts` and `last_loop_error_at`. The probe only observes: a loop
that dies relaunches itself (with backoff) and is relaunched by the next
`/add`, and `loop_restarts` counts those. Alert on `healthy` and on
`loop_restarts` changing; a growing `oldest_pending_age_s` with a live loop
means delivery is slower than arrival.

**Retention.** Every queue row, whatever its status, is deleted seven days
(`row_retention_hours`) after it was created. `GET /tasks/{task_id}` returns `404`
after that. A delivered row drops its request body at once. A `failed` row
keeps its body (in `memorize_inbox.payload_json`, found by `request_id`)
until purged, and failed rows are not capped: an outage in which every batch
fails keeps every body accepted during it for the retention period. SQLite
reuses the freed pages but does not shrink `system.db`. A row that ages out
while still `queued` was never delivered; the purge logs a warning when that
happens.

**Stuck rows.** A row left `processing` because its status write was lost
(SQLite busy past its timeout, disk full) is marked `failed` before the
worker claims its next batch, or at its next scheduling pass if that write
fails too, so a session's `pending` does not stay non-zero because of it;
batches the worker is still running are never swept. Its `last_error`
reads `status write lost; outcome unknown`:
the batch may well have been delivered in full (the lost write is usually
the `success` one), so check the session's memories before re-sending it.
If the server restarts before that mark lands, the row is replayed like a
crash (see above).

Setting `[memorize.async_queue] enabled = false` closes the door:
`async_mode` is accepted but ignored and new calls run synchronously unless
their session still has queued rows. Rows accepted while it was on are still
delivered by the worker, and a session that still has such rows still queues
its synchronous calls behind them, so no drain is needed before switching.
Downgrading to a release without `async_mode` is different: queued rows are
neither delivered nor deleted there. Set `enabled = false` and restart first,
so `async_mode` requests stop adding rows (a synchronous call still queues
behind its session's backlog until it drains); wait until
`select count(*) from memorize_inbox where status in ('queued', 'processing')`
on `system.db` is 0; then stop and downgrade.

### GET /api/v2/tasks/{task_id}

The outcome of one accepted request. The path and the body are the EverOS
Cloud gateway's (`GET /api/v2/tasks/{task_id}`), so a client polls both the
same way; `task_id` is the `request_id` a `202` carried.

`200 OK`, or `404 NOT_FOUND` when the row never existed or has aged out of
`row_retention_hours`.

| Field | Type | Notes |
|---|---|---|
| `id` | `string` | The request id |
| `status` | `"queued" \| "processing" \| "success" \| "failed"` | Where the batch is. Cloud can also report `pending`; this build never does |
| `task_type` | `"memory_add" \| "memory_flush"` | — |
| `error` | `string \| null` | Summary of the failure, if any |
| `error_code` | `string \| null` | `status_write_lost` for a row the sweep failed, `replays_exhausted` for one replayed `max_replays` times without finishing, otherwise the exception type that failed the batch |
| `created_at` | `string` (ISO 8601) | Accepted at |
| `finished_at` | `string \| null` | Set once `status` is `success` or `failed` |

### GET /api/v2/tasks

Accepted requests, newest first, one page at a time. Mirrors the Cloud
gateway's listing.

#### Query parameters

| Parameter | Type | Required | Default | Notes |
|---|---|---|---|---|
| `session_id` | `string` | no | — | Only this session's requests |
| `app_id` / `project_id` | `string` | no | `"default"` | Scope for `session_id` |
| `status` | `string` | no | — | One of the four statuses; anything else is `422 INVALID_INPUT` |
| `page` | `integer` | no | `1` | `≥ 1` |
| `page_size` | `integer` | no | `20` | `1..100`, larger values are capped |

#### Response body

| Field | Type | Notes |
|---|---|---|
| `total` | `integer` | Rows matching the filters, all pages |
| `page` / `page_size` | `integer` | As requested (`page_size` after capping) |
| `items` | `array<TaskItem>` | The page, newest first; fields as in `GET /tasks/{task_id}` |

Drain a session before exiting by polling `?session_id=X&status=queued` and
`?session_id=X&status=processing` until both `total` are `0`. That means
every batch reached the memorize pipeline (unless rows aged out before
delivery, see Retention); it does not mean every message is an episode yet:
messages still in the session buffer wait for the boundary detector or the
next `/flush`.

### POST /api/v2/memory/search

Hybrid retrieval over the memory store. Combines BM25, dense vector
ANN, optional scalar filtering, optional cross-encoder rerank, and
optional final LLM rerank. Returns ranked items grouped by kind.

#### Request body

| Field | Type | Required | Default | Constraints |
|---|---|---|---|---|
| `user_id` | `string \| null` | XOR with `agent_id` | `null` | `minLength=1` if set |
| `agent_id` | `string \| null` | XOR with `user_id` | `null` | `minLength=1` if set |
| `app_id` | `string` *(ScopeId)* | no | `"default"` | — |
| `project_id` | `string` *(ScopeId)* | no | `"default"` | — |
| `query` | `string` | yes | — | `minLength=1` |
| `method` | [SearchMethod](#searchmethod) | no | `"hybrid"` | — |
| `top_k` | `integer` | no | `-1` | `-1` or `1..100` |
| `radius` | `number \| null` | no | `null` | `0.0 ≤ x ≤ 1.0` if set |
| `min_score` | `number \| null` | no | `null` | `0.0 ≤ x ≤ 1.0` if set |
| `include_profile` | `boolean` | no | `false` | — |
| `enable_llm_rerank` | `boolean` | no | `false` | — |
| `filters` | [FilterNode](#filternode-filter-dsl) `\| null` | no | `null` | — |

**`user_id` / `agent_id`** — **Exactly one** must be set. Determines
which track is searched: `user_id` → user-memory (episodes /
profiles); `agent_id` → agent-memory (cases / skills).

**`app_id` / `project_id`** — Scope identifiers; results never cross
scopes.

**`query`** — The retrieval query. The same string is fed to BM25
tokenization and to the embedding model, then the two recall lists
are fused.

**`method`** — Retrieval method; see [SearchMethod](#searchmethod).
Default `"hybrid"` is the recommended starting point.

**`top_k`** — Maximum number of items to return per kind. Two modes:
- `-1` (default) — "Use server defaults." Recall falls back to a
  fixed internal cap, and a server-side `radius` default kicks in if
  the caller did not pass one (see `radius` below). Use for "give me
  whatever you find worth returning."
- `1..100` — Explicit cap. Recall is sized as `top_k × multiplier`
  and **no** default `radius` is applied unless you set it yourself.

Values of `0` or outside `-1` / `1..100` are rejected with `422`.

**`radius`** — Optional **cosine-similarity threshold** in `[0.0, 1.0]`,
applied to the **dense (vector) recall before fusion or rerank**:
candidates whose raw cosine is below it never enter the pool. Every
method's vector route honours it (`vector`, `hybrid`, `agentic`);
`keyword` has no cosine and ignores it. Use it to cut a long tail of
weak matches. Three-level fallback for the effective radius:

1. Caller-supplied `radius` (including the literal `0.0`) always wins.
2. With `top_k=-1` and no caller-supplied `radius`, a server-side
   default radius kicks in.
3. With `top_k>0` and no caller-supplied `radius`, no threshold is
   applied (`null`).

**`min_score`** — Optional **post-fusion relevance floor** in
`[0.0, 1.0]`. Results below this score are evicted after fusion,
independent of `radius` (which is a per-recall cosine threshold).

**`include_profile`** — When `user_id` is set, also fetch the user's
profile and include it in `data.profiles`. The profile is not
ranked; `score` is `null`. Ignored when `agent_id` is set.

**`enable_llm_rerank`** — Opt-in LLM rerank pass for
`method: "hybrid"`. Applies to `agent_case` and `agent_skill` fusion
only; the episode hybrid path has built-in fact eviction and
ignores this flag. Adds one LLM call per request. Ignored by
`keyword` / `vector` (no fusion to rerank) and `agentic` (uses its
own cross-encoder loop).

**`filters`** — Optional filter-DSL node; see
[FilterNode](#filternode-filter-dsl). Applied **before** ranking, so
it does not perturb the ranker.

#### Response body

`200 OK` returns a SuccessEnvelope wrapping `SearchData`. All five
arrays are always present so client code can iterate without
branching on owner type; arrays that do not apply to the requested
owner kind stay as `[]`.

| Field | Type | Notes |
|---|---|---|
| `episodes` | `array<SearchEpisodeItem>` | Populated when `user_id` is set |
| `profiles` | `array<SearchProfileItem>` | Populated when `user_id` is set **and** `include_profile=true` |
| `agent_cases` | `array<SearchAgentCaseItem>` | Populated when `agent_id` is set |
| `agent_skills` | `array<SearchAgentSkillItem>` | Populated when `agent_id` is set |
| `unprocessed_messages` | `array<UnprocessedMessageDTO>` | Populated **only** when `filters.session_id` is a top-level eq scalar; otherwise stays `[]`. Independent of `user_id` / `agent_id` (buffer rows have no owner attribution — boundary detection runs before owner inference) |

#### SearchEpisodeItem

User-track conversation episode hit. `score` is the fused retrieval
score; `atomic_facts` lists single-sentence facts that matched the
query within this episode (already nested, no separate call needed).

| Field | Type | Notes |
|---|---|---|
| `id` | `string` | `<user_id>_ep_<YYYYMMDD>_<NNN>` |
| `user_id` | `string \| null` | Owner of this episode |
| `app_id` | `string` | Scope where the episode lives |
| `project_id` | `string` | Scope where the episode lives |
| `session_id` | `string` | The `session_id` whose messages produced this episode |
| `timestamp` | `string` | ISO-8601 with timezone offset; default `UTC` renders as `Z`, non-UTC defaults render as `±HH:MM` — see [Conventions](#conventions) |
| `sender_ids` | `array<string>` | Distinct `sender_id`s in the underlying messages |
| `summary` | `string` | Short summary (~200 chars), suitable for list-view rendering |
| `subject` | `string` | One-line subject / title |
| `episode` | `string` | Full extracted narrative |
| `type` | `"Conversation"` | Reserved; today only conversation-derived episodes ship |
| `score` | `number` | Fused retrieval score for this episode |
| `atomic_facts` | `array<SearchAtomicFactItem>` | Sub-facts extracted from the same episode that matched the query |

> **`id` is unique within `(app_id, project_id)`.** The same `id` can appear
> under two projects (each project keeps its own daily files and sequence);
> address a row globally by `(app_id, project_id, id)`. Within one response
> ids never repeat, because a request is always scoped to one project.

#### SearchAtomicFactItem

A single-sentence fact pulled out of an episode during extraction.

| Field | Type | Notes |
|---|---|---|
| `id` | `string` | `<user_id>_af_<YYYYMMDD>_<NNN>` |
| `content` | `string` | The fact as a single sentence |
| `score` | `number` | Same score scale as the parent episode |

#### SearchProfileItem

User profile. Only populated when `include_profile=true` and
`user_id` is set.

| Field | Type | Notes |
|---|---|---|
| `id` | `string` | Profile id |
| `user_id` | `string \| null` | Owner |
| `app_id` | `string` | Scope |
| `project_id` | `string` | Scope |
| `profile_data` | `object` | Free-form structured profile fields produced by extraction (schema is profile-specific) |
| `score` | `number \| null` | `null` for direct fetches (no query-aware profile ranking yet) |

#### SearchAgentCaseItem

Agent-track case hit. Returned only when the request uses `agent_id`.

| Field | Type | Notes |
|---|---|---|
| `id` | `string` | `<agent_id>_ac_<YYYYMMDD>_<NNN>` |
| `entry_id` | `string` | `ac_<YYYYMMDD>_<NNN>`; the id a skill's `source_case_ids` refers to |
| `agent_id` | `string` | Owner |
| `app_id` | `string` | Scope |
| `project_id` | `string` | Scope |
| `session_id` | `string` | The agent run this case came from |
| `task_intent` | `string` | What the agent was trying to do (one-sentence framing) |
| `approach` | `string` | How the agent went about it |
| `quality_score` | `number` | Quality assessed at extraction time, in `[0, 1]` |
| `key_insight` | `string \| null` | One-line takeaway; `null` if extraction did not produce one |
| `timestamp` | `string` | ISO-8601 UTC of when the run happened |
| `score` | `number` | Fused retrieval score for this case |

#### SearchAgentSkillItem

Agent-track skill hit — a named procedural memory produced by
clustering related cases.

| Field | Type | Notes |
|---|---|---|
| `id` | `string` | `<agent_id>_sk_<YYYYMMDD>_<NNN>` |
| `agent_id` | `string` | Owner |
| `app_id` | `string` | Scope |
| `project_id` | `string` | Scope |
| `name` | `string` | Skill name (a stable, human-meaningful slug) |
| `description` | `string` | One-line description |
| `content` | `string` | Full skill body (procedure / playbook text) |
| `confidence` | `number` | Extraction confidence in `[0, 1]` |
| `maturity_score` | `number` | Maturity assessed at clustering time, in `[0, 1]` |
| `source_case_ids` | `array<string>` | `entry_id` values of the `agent_case` items that contributed to this skill |
| `score` | `number` | Fused retrieval score for this skill |

#### UnprocessedMessageDTO

A raw message still sitting in the boundary-detection buffer — sent to
`/add` but not yet carved into an episode / case. Returned **only**
when the request's `filters` contains `session_id` as a top-level eq
scalar (`{"session_id": "<sid>"}`); compound shapes (`AND` / `OR`
combinators, operator maps such as `{"eq": ...}` / `{"in": ...}`) do
not trigger the lookup because there is no defensible buffer-scope
mapping for them. Buffer rows have no `user_id` / `agent_id`
attribution, so `session_id` is the only meaningful query dimension.

| Field | Type | Notes |
|---|---|---|
| `id` | `string` | Original `message_id` from `/add` |
| `app_id` | `string` | Scope where the buffered message lives |
| `project_id` | `string` | Scope where the buffered message lives |
| `session_id` | `string` | Matches the requested `filters.session_id` |
| `sender_id` | `string` | Original sender id from `/add` |
| `sender_name` | `string \| null` | Original sender name; `null` if not provided |
| `role` | `"user" \| "assistant" \| "tool"` | Original role |
| `content` | `string \| array<object>` | `string` for the single-text shorthand, `array` of opaque content items for the original multimodal payload (mirrors [MessageItem.content](#messageitem)) |
| `timestamp` | `string` | ISO-8601 with timezone offset — see [Conventions](#conventions) |
| `tool_calls` | `array<object> \| null` | Original tool_calls payload if any |
| `tool_call_id` | `string \| null` | Original tool_call_id if any |

#### cURL example

```bash
curl -X POST http://127.0.0.1:8000/api/v2/memory/search \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "alice",
    "app_id": "default",
    "project_id": "default",
    "query": "Where do I like to climb?",
    "top_k": 5
  }'
```

Response (real capture):

```json
{
    "request_id": "3a8ad3dcd2484fe4a05ccbd89d438704",
    "data": {
        "episodes": [
            {
                "id": "alice_ep_20260528_00000001",
                "user_id": "alice",
                "app_id": "default",
                "project_id": "default",
                "session_id": "demo-002",
                "timestamp": "2026-05-28T11:30:36Z",
                "sender_ids": ["alice"],
                "summary": "On May 28, 2026 at 11:30 AM UTC, Alice shared that she loves climbing in Yosemite every spring...",
                "subject": "Alice's Outdoor Activities and Daily Routine on May 28, 2026: Climbing, Coffee, and Biking",
                "episode": "On May 28, 2026 at 11:30 AM UTC, Alice shared that she loves climbing in Yosemite every spring, highlighting a recurring seasonal activity. She also mentioned that her favorite coffee shop is Blue Bottle located in SOMA, indicating a preferred local spot. Additionally, Alice stated that she bikes to work most days, describing a habitual mode of transportation.",
                "type": "Conversation",
                "score": 0.6298899054527283,
                "atomic_facts": [
                    {
                        "id": "alice_af_20260528_00000001",
                        "content": "Alice said she loves climbing in Yosemite every spring.",
                        "score": 0.6298899054527283
                    }
                ]
            }
        ],
        "profiles": [],
        "agent_cases": [],
        "agent_skills": [],
        "unprocessed_messages": []
    }
}
```

### POST /api/v2/memory/get

Paginated listing of memory records of a given kind for a single
owner. No ranking — ordering is `sort_by` × `sort_order` only. Used
for UI browsing, exports, or filtered scans.

#### Request body

| Field | Type | Required | Default | Constraints |
|---|---|---|---|---|
| `user_id` | `string \| null` | XOR with `agent_id` | `null` | `minLength=1` if set |
| `agent_id` | `string \| null` | XOR with `user_id` | `null` | `minLength=1` if set |
| `app_id` | `string` *(ScopeId)* | no | `"default"` | — |
| `project_id` | `string` *(ScopeId)* | no | `"default"` | — |
| `memory_type` | [GetMemoryType](#getmemorytype) | yes | — | — |
| `page` | `integer` | no | `1` | `≥ 1` |
| `page_size` | `integer` | no | `20` | `1..100` |
| `sort_by` | `"timestamp" \| "updated_at"` | no | `"timestamp"` | — |
| `sort_order` | `"asc" \| "desc"` | no | `"desc"` | — |
| `filters` | [FilterNode](#filternode-filter-dsl) `\| null` | no | `null` | — |

**`user_id` / `agent_id`** — **Exactly one** must be set, and it must
match the track implied by `memory_type` (`"episode"` / `"profile"`
require `user_id`; `"agent_case"` / `"agent_skill"` require
`agent_id`).

**`app_id` / `project_id`** — Scope identifiers.

**`memory_type`** — Which item kind to list; see
[GetMemoryType](#getmemorytype). The route populates exactly one of
the four arrays in `data` based on this value.

> `"profile"` is a single-row lookup, not a list: an owner has at most
> one profile. `page`, `page_size`, `filters`, `sort_by` and
> `sort_order` are accepted for it but have no effect — the response
> carries that one profile (or an empty array), with `total_count` in
> `{0, 1}` whatever page was requested.

**`page`** — 1-indexed page number. Together with `page_size`
determines the window. The response's `total_count` reports how many
items match the request before paging.

**`page_size`** — Items per page, in `1..100`. Default 20.

**`sort_by`** — Column to sort by. Note: for `memory_type` values
where `"timestamp"` does not apply (e.g. `"profile"` has no
timestamp, `"agent_skill"` is a named entity), the server silently
falls back to `"updated_at"`.

**`sort_order`** — `"desc"` (newest first, default) or `"asc"`.

**`filters`** — Optional [FilterNode](#filternode-filter-dsl) for
predicate-based filtering before pagination.

#### Response body

`200 OK` returns a SuccessEnvelope wrapping `GetData`. The four
arrays are always present so client code can iterate without
branching on `memory_type`; exactly one is populated.

| Field | Type | Notes |
|---|---|---|
| `episodes` | `array<GetEpisodeItem>` | Populated when `memory_type="episode"` |
| `profiles` | `array<GetProfileItem>` | Populated when `memory_type="profile"` |
| `agent_cases` | `array<GetAgentCaseItem>` | Populated when `memory_type="agent_case"` |
| `agent_skills` | `array<GetAgentSkillItem>` | Populated when `memory_type="agent_skill"` |
| `total_count` | `integer` | Total matching records **before** paging |
| `count` | `integer` | Number of items in **this page** (`len(items)`) |

#### GetEpisodeItem

Same shape as [SearchEpisodeItem](#searchepisodeitem) **minus**
`score` and `atomic_facts` (listing is unranked and does not nest
sub-facts).

| Field | Type | Notes |
|---|---|---|
| `id` | `string` | `<user_id>_ep_<YYYYMMDD>_<NNN>` |
| `user_id` | `string \| null` | Owner |
| `app_id` | `string` | Scope |
| `project_id` | `string` | Scope |
| `session_id` | `string` | Originating session |
| `timestamp` | `string` | ISO-8601 with timezone offset, identical to `/search` (both go through the same `from_iso_format` path) — see [Conventions](#conventions) |
| `sender_ids` | `array<string>` | Distinct `sender_id`s in the underlying messages |
| `summary` | `string` | Short summary |
| `subject` | `string` | One-line subject |
| `episode` | `string` | Full extracted narrative |
| `type` | `"Conversation"` | — |

> **`id` is unique within `(app_id, project_id)`.** The same `id` can appear
> under two projects (each project keeps its own daily files and sequence);
> address a row globally by `(app_id, project_id, id)`. Within one response
> ids never repeat, because a request is always scoped to one project.

#### GetProfileItem

| Field | Type | Notes |
|---|---|---|
| `id` | `string` | Profile id |
| `user_id` | `string \| null` | Owner |
| `app_id` | `string` | Scope |
| `project_id` | `string` | Scope |
| `profile_data` | `object` | Free-form structured profile fields |

#### GetAgentCaseItem

Same shape as [SearchAgentCaseItem](#searchagentcaseitem) **minus**
`score`.

| Field | Type | Notes |
|---|---|---|
| `id` | `string` | `<agent_id>_ac_<YYYYMMDD>_<NNN>` |
| `entry_id` | `string` | `ac_<YYYYMMDD>_<NNN>`; the id a skill's `source_case_ids` refers to |
| `agent_id` | `string` | Owner |
| `app_id` | `string` | Scope |
| `project_id` | `string` | Scope |
| `session_id` | `string` | Originating run |
| `task_intent` | `string` | What the agent was trying to do |
| `approach` | `string` | How it went about it |
| `quality_score` | `number` | Quality assessed at extraction time |
| `key_insight` | `string \| null` | One-line takeaway, if any |
| `timestamp` | `string` | ISO-8601 UTC |

#### GetAgentSkillItem

Same shape as [SearchAgentSkillItem](#searchagentskillitem) **minus**
`score`.

| Field | Type | Notes |
|---|---|---|
| `id` | `string` | `<agent_id>_sk_<YYYYMMDD>_<NNN>` |
| `agent_id` | `string` | Owner |
| `app_id` | `string` | Scope |
| `project_id` | `string` | Scope |
| `name` | `string` | Skill name |
| `description` | `string` | One-line description |
| `content` | `string` | Full skill body |
| `confidence` | `number` | Extraction confidence in `[0, 1]` |
| `maturity_score` | `number` | Maturity assessed at clustering, in `[0, 1]` |
| `source_case_ids` | `array<string>` | `entry_id` values of the `agent_case` items that contributed to this skill |

#### cURL example

```bash
curl -X POST http://127.0.0.1:8000/api/v2/memory/get \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "alice",
    "app_id": "default",
    "project_id": "default",
    "memory_type": "episode"
  }'
```

Response (real capture):

```json
{
    "request_id": "b8ba1255440147a684885f2084f6a25d",
    "data": {
        "episodes": [
            {
                "id": "alice_ep_20260528_00000001",
                "user_id": "alice",
                "app_id": "default",
                "project_id": "default",
                "session_id": "demo-002",
                "timestamp": "2026-05-28T11:30:36Z",
                "sender_ids": ["alice"],
                "summary": "On May 28, 2026 at 11:30 AM UTC, Alice shared that she loves climbing in Yosemite every spring...",
                "subject": "Alice's Outdoor Activities and Daily Routine on May 28, 2026: Climbing, Coffee, and Biking",
                "episode": "On May 28, 2026 at 11:30 AM UTC, Alice shared that she loves climbing in Yosemite every spring, highlighting a recurring seasonal activity. She also mentioned that her favorite coffee shop is Blue Bottle located in SOMA, indicating a preferred local spot. Additionally, Alice stated that she bikes to work most days, describing a habitual mode of transportation.",
                "type": "Conversation"
            }
        ],
        "profiles": [],
        "agent_cases": [],
        "agent_skills": [],
        "total_count": 1,
        "count": 1
    }
}
```

### POST /api/v2/ome/trigger

Manually trigger a registered OME strategy.

#### Request body

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | `string` | yes | — | Strategy name (e.g. `reflect_episodes`) |
| `timeout` | `float` | no | `120.0` | Max seconds to wait for completion |
| `force` | `bool` | no | `false` | Bypass the `enabled` gate in `ome.toml` |

#### Response body

`200 OK` returns the body below **bare** — no `request_id` / `data`
envelope. This is the one business-prefixed endpoint that behaves like
the operational probes; clients that share a response parser with
`/memory/*` must special-case it.

| Field | Type | Notes |
|---|---|---|
| `status` | `"ok" \| "timeout" \| "not_dispatched"` | `ok` = every dispatched run settled — a dead-lettered run still counts as settled (see `runs[*].error`); `timeout` = at least one run had not settled when `timeout` elapsed; `not_dispatched` = no strategy was dispatched (see below) |
| `name` | `string` | Echoes the requested strategy name |
| `dispatched` | `int` | Number of strategy routes enqueued. `0` iff `status == "not_dispatched"` |
| `runs` | `list[RunSummary]` | One entry per strategy run *attempt*, not per dispatched route: `{run_id: string, status: string, error?: string}`. A strategy that retried before settling contributes multiple entries sharing one `event_id`. `status` is one of `running` / `success` / `failed` / `dead_letter` / `crashed`. Includes dead-lettered runs |

**`not_dispatched`** means every subscriber was rejected by one of the
five dispatch gates (event-class subscription / `_routes_to` / `enabled` /
`applies_to` / `Counter`). The most common cause is forgetting `"force": true` on a
strategy that is `enabled=false` in `ome.toml` — e.g. triggering
`reflect_episodes` without `force` while it is disabled in config
returns `{"status": "not_dispatched", "dispatched": 0, "runs": []}`
instead of an error.

> `status: "ok"` means all dispatched strategy runs settled — including
> runs that dead-lettered (their errors are in `runs[*].error`). It does
> **not** mean the LanceDB index has caught up. Markdown is written
> synchronously; the index syncs asynchronously (see
> [Eventual consistency](#eventual-consistency)).
>
> If you need read-your-write semantics, poll `GET /health`'s
> `cascade.pending` field until it reads `0` on two consecutive samples
> (a single zero can be a false convergence — the watcher-input window
> can briefly report an empty queue between md write and enqueue).

See [docs/openapi.json](openapi.json) for the exact generated schema
(`TriggerResponse` / `RunSummary`) behind this table.

#### Errors

- `404` — strategy name not found in the OME registry.

#### cURL example

```bash
curl -X POST http://127.0.0.1:8000/api/v2/ome/trigger \
  -H 'Content-Type: application/json' \
  -d '{"name": "reflect_episodes", "force": true}'
```

---

### Knowledge endpoints

The knowledge base subsystem (`/api/v2/knowledge/*`) provides document
upload, CRUD, and hybrid search. These endpoints are fully documented
in their own reference: **[docs/knowledge.md](knowledge.md)**.

Summary of available routes:

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v2/knowledge/documents` | Upload and extract a document |
| `GET` | `/api/v2/knowledge/documents` | List documents (paginated) |
| `GET` | `/api/v2/knowledge/documents/{doc_id}` | Get a single document |
| `PUT` | `/api/v2/knowledge/documents/{doc_id}` | Replace a document |
| `PATCH` | `/api/v2/knowledge/documents/{doc_id}` | Partial update |
| `DELETE` | `/api/v2/knowledge/documents/{doc_id}` | Delete a document |
| `GET` | `/api/v2/knowledge/topics/{topic_id}` | Get a single topic |
| `POST` | `/api/v2/knowledge/search` | Hybrid search over topics |
| `GET` | `/api/v2/knowledge/categories` | List taxonomy categories |

---

## OpenAPI spec source

This document mirrors the OpenAPI 3.x spec that FastAPI auto-generates
from the Pydantic DTOs. **The spec is not exposed by default** — the
`/openapi.json`, `/docs` (Swagger UI), and `/redoc` endpoints only
mount when the server starts with `ENV=DEV` set.

```bash
ENV=DEV everos server start
# In another terminal:
curl http://127.0.0.1:8000/openapi.json | python -m json.tool
# Or interactively in a browser:
open http://127.0.0.1:8000/docs
```

This document and the spec are kept in sync by hand. Where they
disagree, the **spec is the structural ground truth** (field names,
required flags, value constraints). This document carries the
**business semantics** the spec cannot — when to use which method,
the eventual-consistency caveat, why a field is shaped the way it is.

---

For higher-level context (cascade design, DDD layering, on-disk
layout), see [architecture.md](architecture.md) and
[storage_layout.md](storage_layout.md).
