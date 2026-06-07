# EverOS HTTP API (v1)

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
  - [POST /api/v1/memory/add](#post-apiv1memoryadd)
  - [POST /api/v1/memory/flush](#post-apiv1memoryflush)
  - [POST /api/v1/memory/search](#post-apiv1memorysearch)
  - [POST /api/v1/memory/get](#post-apiv1memoryget)
- [OpenAPI spec source](#openapi-spec-source)

## Overview

### Base URL and versioning

| Setting | Default | Override |
|---|---|---|
| Host | `127.0.0.1` (loopback only) | `EVEROS_API__HOST` env var or `--host` flag |
| Port | `8000` | `EVEROS_API__PORT` env var or `--port` flag |
| Version prefix | `/api/v1` | — |

All business endpoints documented here live under `/api/v1/memory/`.
The operational endpoints `GET /health` and `GET /metrics` exist but
are intentionally outside this reference — they are runtime probes for
deployment, not part of the application contract.

### Content type

All `POST` endpoints require `Content-Type: application/json`. Request
and response bodies are UTF-8 JSON.

### Authentication

EverOS ships **no built-in authentication**. The server binds to
`127.0.0.1` by default; place your own gateway or auth layer in front
before exposing the API on any other interface. See
[../SECURITY.md](../SECURITY.md) for the threat model.

### Response envelope

Successful (`200 OK`) responses always wrap the payload in:

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
the new entry exists on disk. The LanceDB vector / BM25 / scalar index
is rebuilt by the in-process **cascade coroutine asynchronously**.

That means `/search` and `/get` may not see a record immediately after
the `/flush` that produced it. Typical sync latency is sub-second, but
under load it can reach ~10–15 seconds. If you need read-your-write
semantics, retry with backoff. The markdown file is durable
regardless; index lag never loses data.

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
`everos.component.utils.datetime`
(`get_utc_now` / `ensure_utc` / `UtcDatetime` for storage;
`get_now_with_timezone` / `to_display_tz` for display). See
[datetime.md](datetime.md) for the design rationale.

#### Other conventions

- **Server-generated IDs** follow `<owner>_<kind>_<YYYYMMDD>_<NNN>`,
  e.g. `alice_ep_20260528_00000001` for an episode, `alice_af_...`
  for an atomic fact. See
  [storage_layout.md §4](storage_layout.md) for the encoding.
- **All endpoints are POST** for `/api/v1/memory/*` even when the
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
    "code": "HTTP_ERROR",
    "message": "Value error, exactly one of user_id / agent_id must be provided",
    "timestamp": "2026-06-01T12:24:46+00:00",
    "path": "/api/v1/memory/search"
  }
}
```

| HTTP | `error.code` | `error.message` | When |
|---|---|---|---|
| `415 Unsupported Media Type` | `HTTP_ERROR` | the parse-failure reason | `/add` only — a `ContentItem` could not be parsed (unsupported modality for the configured multimodal LLM, or a payload that cannot be fetched / dispatched) |
| `422 Unprocessable Entity` | `HTTP_ERROR` | the **first** validation error (see below) | Request-body validation failure. Also covers `/search` / `/get` filter-DSL compile errors — the compile reason rides in `message` |
| `500 Internal Server Error` | `SYSTEM_ERROR` | `"Internal server error"` (fixed; internal details are logged, never leaked) | Unhandled exception caught by the global handler |

### error object

| Field | Type | Description |
|---|---|---|
| `code` | `string` | `"HTTP_ERROR"` for 4xx (validation / business / `HTTPException`); `"SYSTEM_ERROR"` for 5xx |
| `message` | `string` | Human-readable reason. For `422`, **only the first** validation error is surfaced, formatted `"<msg>: <dotted-loc>"` with the leading `body` segment stripped (e.g. `"Field required: messages"`); a model-level validator with no field location surfaces just `"<msg>"` (e.g. the XOR example above) |
| `timestamp` | `string` | ISO-8601 with timezone offset (display tz) |
| `path` | `string` | Request path, e.g. `/api/v1/memory/add` |

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
message is the response to.

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
node is compiled to a LanceDB `where` clause server-side. Compile
errors surface as `422` with the offending field / operator in
`error.message`.

#### Shape

A node is a JSON object whose keys are one of:

| Key | Value | Notes |
|---|---|---|
| `AND` | `array<FilterNode>` | All child nodes must match. Omit if not needed |
| `OR` | `array<FilterNode>` | At least one child node must match. Omit if not needed |
| *<allowed field>* | scalar or operator map | Predicate on that field — see [Allowed fields](#filter-allowed-fields) and [Operators](#filter-operators) |

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
| `"hybrid"` *(default)* | Reciprocal-rank fuse of BM25 + vector + optional scalar filter in a single LanceDB query |
| `"agentic"` | Iterative cluster-path retrieval driven by a cross-encoder rerank loop; higher quality at higher latency / cost |

`"hybrid"` is the default because it balances recall and precision
with one LanceDB roundtrip. `"agentic"` calls the LLM in a loop and
should be reserved for offline or background workflows.

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

### POST /api/v1/memory/add

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

#### cURL example

```bash
TS=$(date +%s)
curl -X POST http://127.0.0.1:8000/api/v1/memory/add \
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

### POST /api/v1/memory/flush

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

`/flush` is synchronous with respect to markdown persistence: by the
time the response returns, the new entry is on disk. LanceDB index
sync is still asynchronous — see
[Eventual consistency](#eventual-consistency).

#### cURL example

```bash
curl -X POST http://127.0.0.1:8000/api/v1/memory/flush \
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

### POST /api/v1/memory/search

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

**`radius`** — Optional **cosine-similarity threshold** in `[0.0, 1.0]`.
Candidates whose score is below this value are dropped — use it to cut
a long tail of weak matches. Three-level fallback for the effective
radius:

1. Caller-supplied `radius` (including the literal `0.0`) always wins.
2. With `top_k=-1` and no caller-supplied `radius`, a server-side
   default radius kicks in.
3. With `top_k>0` and no caller-supplied `radius`, no threshold is
   applied (`null`).

**`include_profile`** — When `user_id` is set, also fetch the user's
profile and include it in `data.profiles`. The profile is not
ranked; `score` is `null`. Ignored when `agent_id` is set.

**`enable_llm_rerank`** — Opt-in LLM rerank pass for
`method: "hybrid"`. Applies to `agent_case` and `agent_skill` fusion
only; the episode hierarchy path has built-in fact eviction and
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
| `source_case_ids` | `array<string>` | `agent_case` ids that contributed to this skill |
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
curl -X POST http://127.0.0.1:8000/api/v1/memory/search \
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

### POST /api/v1/memory/get

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
| `source_case_ids` | `array<string>` | `agent_case` ids that contributed to this skill |

#### cURL example

```bash
curl -X POST http://127.0.0.1:8000/api/v1/memory/get \
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
