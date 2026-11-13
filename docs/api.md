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
  - [Changing and forgetting a memory](#changing-and-forgetting-a-memory)
  - [POST /api/v2/memory/update](#post-apiv2memoryupdate)
  - [GET /api/v2/memory/history](#get-apiv2memoryhistory)
  - [POST /api/v2/memory/forget](#post-apiv2memoryforget)
  - [POST /api/v2/memory/forget/undo](#post-apiv2memoryforgetundo)
  - [POST /api/v2/memory/delete](#post-apiv2memorydelete)
  - [POST /api/v2/memory/delete_by_ids](#post-apiv2memorydelete_by_ids)
  - [POST /api/v2/ome/trigger](#post-apiv2ometrigger)
  - [Groups](#groups)
- [OpenAPI spec source](#openapi-spec-source)

## Overview

### Base URL and versioning

| Setting | Default | Override |
|---|---|---|
| Host | `127.0.0.1` (loopback only) | `EVEROS_API__HOST` env var or `--host` flag |
| Port | `8000` | `EVEROS_API__PORT` env var or `--port` flag |
| Version prefix | `/api/v2` | — |

Business endpoints live under `/api/v2/memory/`, `/api/v2/groups/`,
`/api/v2/projects/`, `/api/v2/tasks/` and `/api/v2/ome/`. EverOS 2.0 keeps
chat memory only: there are no knowledge-base routes (`/api/v2/knowledge/...`
answers `404`) and no agent memory (see
[`/memory/search`](#post-apiv2memorysearch)). The operational endpoints
`GET /health`, `GET /metrics` and
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
`capabilities.decider` and the `decider` block name the multi-round decider
(`model`, `base_url` without credentials) and say whether it answered one
request (`max_tokens = 1`) sent in the background when the server started
(`reachable`, `probe_error`, `probed_at`; `reachable` is `null` while that
probe runs). Like `embed`, that verdict is a startup snapshot. The searches
themselves keep it current: `rounds` and `failed_rounds` count the decider
rounds since start, and `last_failure` / `last_failure_at` name the newest
round that fell back, so a decider that stops answering later shows there
without `/health` calling a model. The other capability flags are
`multimodal_llm` and `parser`; there is no `rerank` flag (2.0 has no rerank
provider). `disabled_features` lists at most two features the configured
providers leave off: `vector_recall` (no embedder: multi-round search
recalls by keyword only) and `multimodal_upload` (no multimodal LLM or no
parser: a non-text chat item is not read into text).

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

Successful responses from the business endpoints (`/memory/*`) wrap the
payload in the envelope below (`200 OK`, or `202 Accepted` for a queued
`/add` or `/flush`). `POST /api/v2/ome/trigger` and the operational
endpoints return their body bare — see each section.

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
**after** the extraction call returns: profiles, scenes (when switched on)
and a group's profile. `/flush` returns once the episode is on disk; the
`extract_user_profile` strategy then runs its own LLM call, so `/get` with
`memory_type: "profile"` can legitimately return the profile as it was
before that conversation (or `total_count: 0` for a new user) for seconds
after a flush that returned `"extracted"`. A member's profile in a group
waits longer: it is updated every `[groups] member_profile_every` segments
and on each flush of the group. `GET /health`'s `cascade.pending` does
**not** cover this window — it only tracks the md → index queue, which is
empty while the strategy is still thinking. Poll `/get` with a budget
instead. The same "seconds later" rule for every OME-produced kind is
summarised in [how-memory-works.md](how-memory-works.md).

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
  e.g. `alice_ep_20260528_00000001` for an episode, `alice_im_...`
  for a picture memory. See
  [storage_layout.md §4](storage_layout.md) for the encoding.
- **All endpoints are POST** for `/api/v2/memory/*` even when the
  semantics look like a read (`/search`, `/get`) — the request bodies
  are too rich (filters, methods, paging) to encode in a query string.

### ScopeId: `app_id` and `project_id`

`app_id` and `project_id` partition memory at the disk layer. Every
write lands under
`~/.everos/<app>/<project>/users/<user_id>/...` (or
`.../groups/<group_id>/...` for a group). The default scope materialises on disk as
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
    "message": "no memory 'amy_ep_20261112_00000099'",
    "timestamp": "2026-11-12T09:30:00+00:00",
    "path": "/api/v2/memory/history"
  }
}
```

### error.code values

`error.code` is a machine-readable `ErrorCode` enum. Clients can switch
on this value to decide retry / display / routing behaviour without
parsing the human-readable `message` field.

| `error.code` | HTTP | Retryable? | When |
|---|---|---|---|
| `NOT_FOUND` | `404` | No | Requested resource does not exist -- or exists but the asker may not see it (a group picture outside their membership windows answers exactly as a missing one) |
| `FORBIDDEN` | `403` | No | The asker may not read the scope or perform the action: a group read by someone who is not a current member (or a group that does not exist), a personal read whose `asker_id` is another user, a body `asker_id` that disagrees with the gateway identity header, a roster change by a member who is not an owner or admin |
| `CONFLICT` | `409` | No | Operation conflicts with existing state (e.g. a session bound to another group; a roster event that conflicts with the member's open interval; a group undo whose forget a later forget rewrote) |
| `VERSION_CONFLICT` | `409` | No | `/memory/update` named a version that is no longer the current one; `message` names the current version. Read it (`/memory/history`) and decide again |
| `MEMORY_FORGOTTEN` | `409` | No | `/memory/update` named a forgotten memory; undo the forget first |
| `INVALID_INPUT` | `422` | No | Request-body validation failure. Also covers `/search` / `/get` filter-DSL compile errors — the compile reason rides in `message` — and, on `/add` and `/flush`, a session whose buffered messages the model could not segment on three consecutive calls: the buffer is moved to the `quarantine` track of `unprocessed_buffer` so the session can continue, and `message` names the session and the number of messages set aside |
| `BAD_REQUEST` | `400` | No | Path traversal attempt or other malformed input |
| `UNSUPPORTED_FORMAT` | `415` | No | File format or modality not supported (e.g. unsupported `ContentItem` type, missing `ext` for `base64`) |
| `PAYLOAD_TOO_LARGE` | `413` | No | An `async_mode` request body exceeds `memorize.async_queue.max_payload_bytes`; the batch was not accepted |
| `QUEUE_SATURATED` | `429` | Yes | The memorize queue is at its per-session or global cap; the batch was not accepted — retry later or keep it client-side |
| `SESSION_BUSY` | `503` | Yes | A forget of a conversation (`/forget`, `/forget/session`, `/delete` with a `session_id`) while an `/add` holds the session past `delete_wait_seconds`; nothing was forgotten |
| `EXTERNAL_SERVICE_UNAVAILABLE` | `503` | **Yes** | An external service (LLM, embedding) returned an error or timed out, or the local database stayed locked by another writer past `[sqlite] busy_timeout_ms` (message: `"database is busy"`) |
| `CAPABILITY_UNAVAILABLE` | `503` | No | A required server-side capability is missing (e.g. `everos[multimodal]` extra not installed, LibreOffice absent) — requires admin action, not retry |
| `CONFIGURATION_ERROR` | `500` | No | A required configuration is missing or invalid (e.g. embedding model not set) |
| `INTERNAL_ERROR` | `500` | No | Unhandled exception (internal details are logged, never leaked) |

### error object

| Field | Type | Description |
|---|---|---|
| `code` | `string` | One of the `ErrorCode` values listed above |
| `message` | `string` | Human-readable reason. For `INVALID_INPUT` from request validation, **only the first** validation error is surfaced, formatted `"<msg>: <dotted-loc>"` with the leading `body` segment stripped (e.g. `"Field required: messages"`); a model-level validator with no field location surfaces just `"<msg>"` (e.g. `"Value error, exactly one of user_id / group_id / group_ids must be provided"`) |
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
`user_id: "<sender_id>"` reach it. For `role: "assistant"`, the field is
informational (the LLM sees it during extraction but it is not used as an
indexing key).

**`sender_name`** — Optional human-readable display name. Lets the LLM
use a real name during extraction without changing the indexing key.
For example, set `sender_id: "u_42"` (stable internal id) and
`sender_name: "Alice"` (what the LLM sees).

**`role`** — Speaker role; one of:
- `"user"` — content originates from a human (or human-acting client).
- `"assistant"` — content from the AI assistant.
- `"tool"` — output of a tool call; pair with `tool_call_id`.

Memory is chat memory. A tool row (`role: "tool"`) and an assistant message
that carries `tool_calls` are accepted, and counted in `message_count`, but
dropped before segmentation: they are never buffered, segmented or extracted,
whatever text they carry. The user's and the assistant's plain messages of
the same batch are memorized as usual.

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
assistant emitted in this turn (OpenAI Chat Completions shape). Such a
message is dropped before segmentation (see `role`).

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
| `name` | `string` | yes | Function name the assistant called |
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
| `"llm_multiround"` *(default, and the only one)* | Multi-round retrieval: every round recalls each current sub-query by BM25 + vector against the configured derived index backend, fuses each sub-query's two lists with reciprocal-rank fusion, and lets a decider model select the core memories, stop, or issue follow-up sub-queries. Episodes, scenes and picture memories compete in one candidate pool. |

The keyword, vector, hybrid and agentic methods were removed in 2.0
together with atomic facts. A request naming one of them, or any other
value, is refused with `422 INVALID_INPUT` and a message that says only
`"llm_multiround"` is supported. Multi-round retrieval calls the
`[decider]` model (falling back to `[llm]`) once per round; without an
`[embedding]` provider it recalls by BM25 alone.

### GetMemoryType

| Value | Track | Returned in `data.<plural>` |
|---|---|---|
| `"episode"` | user, group | `data.episodes` — [GetEpisodeItem](#getepisodeitem) |
| `"profile"` | user | `data.profiles` — [GetProfileItem](#getprofileitem) |
| `"scene"` | user, group | `data.scenes` — the owner's current scene versions ([scenes.md](scenes.md)), in the [GetEpisodeItem](#getepisodeitem) shape with `type: "Scene"`; `"episode"` never lists a scene |
| `"image_memory"` | group | `data.images` — the group's picture memories |
| `"member_profile"` | group | `data.profiles` — one member's profile in each group of the scope; needs `member_id` |
| `"group_profile"` | group | `data.group_profile` — the group-profile items the asker can see, newest first; every version with `include_superseded` |

`memory_type` must match the requested owner kind: `"episode"` /
`"profile"` / `"scene"` take `user_id`; a group scope (`group_id` /
`group_ids`) takes `"episode"`, `"scene"`, `"image_memory"`,
`"member_profile"` and `"group_profile"`. The mismatching combinations are
rejected with `422`. EverOS 2.0 has no agent memory: `"agent_case"` and
`"agent_skill"` are refused with `422` like an `agent_id`
([`/memory/get`](#post-apiv2memoryget)).

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
| `session_id` | `string` | yes | — | length 1–128; one line: no line breaks or other control characters, no leading or trailing whitespace (stored as one trimmed markdown line; anything else would read back as another id) |
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
| `rejected` | `integer` | Pictures refused at ingest: over an `[images]` limit (bytes, per message, per request), not a picture, or unreadable |
| `failed` | `integer` | Non-text items the parser could not read into text (its model call failed) |

**`rejected` / `failed`** — A dropped item never fails the request: the rest
of its message is kept, and the item adds nothing to the memory. A refused
picture is not sent to any model. Both are `0` on a clean batch. A file
format the parser does not handle at all still answers `415`, as before (see
[multimodal.md](multimodal.md#errors-and-limits)). A
queued batch (`202`) is ingested later; its counts are logged
(`memorize.ingest_dropped`).

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
        "status": "accumulated",
        "rejected": 0,
        "failed": 0
    }
}
```

### POST /api/v2/memory/flush

Force the boundary detector to decide **now** for the given session
buffer. The LLM runs extraction (one call) regardless of whether the
heuristic would have tripped on its own. Useful at the end of a chat
to make sure pending context becomes durable memory.

#### Request body

| Field | Type | Required | Default | Constraints |
|---|---|---|---|---|
| `session_id` | `string` | yes | — | length 1–128; one line: no line breaks or other control characters, no leading or trailing whitespace (stored as one trimmed markdown line; anything else would read back as another id) |
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

Multi-round retrieval over a user's or a group's memory (see
[SearchMethod](#searchmethod)). Returns the selected episodes and picture
memories, ranked together, plus optional profiles.

#### Request body

| Field | Type | Required | Default | Constraints |
|---|---|---|---|---|
| `user_id` | `string \| null` | one owner scope | `null` | `minLength=1` if set |
| `group_id` / `group_ids` | `string` / `array<string>` | one owner scope | `null` | see [group-memory.md](group-memory.md) |
| `asker_id` | `string \| null` | for a group scope | `null` | `minLength=1` if set |
| `app_id` | `string` *(ScopeId)* | no | `"default"` | — |
| `project_id` | `string` *(ScopeId)* | no | `"default"` | — |
| `query` | `string` | yes | — | `minLength=1` |
| `method` | [SearchMethod](#searchmethod) | no | `"llm_multiround"` | only `"llm_multiround"` |
| `top_k` | `integer` | no | `-1` | `-1` or `1..100` |
| `include_profile` | `boolean` | no | `false` | — |
| `include_groups` | `boolean` | no | `false` | with `user_id` only; see below |
| `include_group_ids` | `array<string> \| null` | no | `null` | with `include_groups`; `1..16` ids |
| `images` | `array` | no | `[]` | at most 4 pictures that are part of the question |
| `filters` | [FilterNode](#filternode-filter-dsl) `\| null` | no | `null` | — |

**Owner scope** — **Exactly one** of `user_id`, `group_id` and
`group_ids` must be set. EverOS 2.0 has no agent memory: an `agent_id` or a
`context_group_id` is refused with `422 INVALID_INPUT`, whose message reads
`"Value error, EverOS 2.0 has no agent memory: name a user (user_id) or a
group; agent_id, context_group_id and the agent_case / agent_skill memory
types are not accepted"`. `/memory/get` refuses them the same way, and so
does `/memory/delete` an `agent_id`.

**`include_groups`** — Off by default; a request without it is answered
exactly as before. With `user_id`, also search the user's groups: every
group of `app_id` / `project_id` the user currently belongs to joins the
personal memory in one candidate pool, read exactly as a group search by the
user reads it (inside the user's membership windows; superseded and
forgotten records left out; a group the user left adds nothing).
`include_group_ids` narrows it to those of the user's groups it names; an id
that is not one of them adds nothing. Every returned episode and picture
memory then names its owner: `owner_type` `"user"`, or `"group"` with
`group_id` (group records carry `source` and `audience`, as in a group
search). `scope` lists the groups searched with the user's windows
(`owner_type: "user"`; `group_ids: []` when the user belongs to none). The
question names the user (`"<display name>: <question>"`) once a group is
included. Attached besides the personal profile: with `include_profile`, the
user's own profile in each group (never another member's), and the groups'
current decisions in `group_profile.decisions` (at most `[search]
group_profile_cap`, newest first; `include_group_profile: false` leaves them
out). No constraint block, `coverage` or link expansion. With a group scope,
or with `audience: "group"`, it is `422`. See
[group-memory.md](group-memory.md#88-a-personal-search-over-the-users-groups).

**`app_id` / `project_id`** — Scope identifiers; results never cross
scopes.

**`query`** — The question the multi-round loop starts from. Round 0
recalls it by BM25 and by its embedding; later rounds recall the
decider's sub-queries the same way.

**`method`** — Optional; `"llm_multiround"` is the default and the only
accepted value. `"keyword"`, `"vector"`, `"hybrid"`, `"agentic"` and any
other value return `422 INVALID_INPUT`.

**`top_k`** — Maximum number of items to return (episodes and pictures
share it). `-1` (default) caps at 100; `1..100` is an explicit cap.
Values of `0` or outside `-1` / `1..100` are rejected with `422`.

**`include_profile`** — For a user scope, also fetch the user's
profile into `data.profiles`; for a group scope, the asker's and the
named members' member-in-group profiles. Profiles are not ranked;
`score` is `null`.

**`filters`** — Optional filter-DSL node; see
[FilterNode](#filternode-filter-dsl). Applied **before** ranking, so
it does not perturb the ranker.

The `radius`, `min_score` and `enable_llm_rerank` fields of 1.x tuned the
removed methods; a request that still sends them is refused as an unknown
field (`422`).

#### Response body

`200 OK` returns a SuccessEnvelope wrapping `SearchData`. Every array is
always present so client code can iterate without branching on the
owner scope; arrays that do not apply stay `[]`.

| Field | Type | Notes |
|---|---|---|
| `episodes` | `array<SearchEpisodeItem>` | The selected episodes (and group scenes) |
| `images` | `array<SearchImageItem>` | Picture memories ranked in the same pool as the episodes. Show an answering model `model_path`: the `[images] model_edge` copy (1024 px), on disk when the response returns (`GET /api/v2/memory/images/{image_id}?variant=model` serves it by id). `original_path` is the picture as sent; `thumbnail_path` the decider's copy |
| `profiles` | `array<SearchProfileItem>` | Populated **only** when `include_profile=true` |
| `unprocessed_messages` | `array<UnprocessedMessageDTO>` | Populated **only** when `filters.session_id` is a top-level eq scalar; otherwise stays `[]`. Buffer rows have no owner attribution — boundary detection runs before owner inference |
| `scope` | `object \| null` | For a group scope: the groups, the asker and the asker's windows; for `include_groups`, the groups searched alongside the user |
| `degraded` | `array<SearchDegradation>` | What did not run as configured; `[]` when the search did. See below |

#### SearchDegradation

One entry per multi-round round the decider did not decide. The search
still answers `200`: the round's candidates were ranked, but which of them
are core was not the decider's choice, and the loop stopped at that round.
Count or exclude such answers; the reason is also in the server log
(`llm_multiround_decider_fallback`) and, when tracing, in the round trace
(`decider_error`).

| Field | Type | Notes |
|---|---|---|
| `component` | `"decider"` | The part that did not run |
| `round` | `integer` | The round, 0-based |
| `outcome` | `"fallback" \| "salvaged"` | `fallback`: every attempt failed and the round's core is the top `[decider] fallback_core` candidates by fused score. `salvaged`: every attempt failed to parse and the core was read from a reply cut short (`[decider] salvage_truncated_core`) |
| `reason` | `string` | The last attempt's error, e.g. `NotFoundError: Error code: 404 ...` (credential-shaped text masked) |
| `attempts` | `integer` | Requests sent for the round |

A group scope's `coverage` reads `partial`, not `complete`, when the loop
ended on such a round.

#### SearchEpisodeItem

Conversation episode (or group scene) selected by the search. `score` is
the multi-round fusion score: the highest reciprocal-rank-fusion score
the episode reached across the sub-queries that recalled it.

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
| `type` | `"Conversation" \| "Scene"` | `"Scene"` for a scene version ([scenes.md](scenes.md)): a narrative over several of the owner's episodes, ranked with them (`session_id` is `null`) |
| `score` | `number` | Fusion score for this episode |

> **`id` is unique within `(app_id, project_id)`.** The same `id` can appear
> under two projects (each project keeps its own daily files and sequence);
> address a row globally by `(app_id, project_id, id)`. Within one response
> ids never repeat, because a request is always scoped to one project.

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

#### UnprocessedMessageDTO

A raw message still sitting in the boundary-detection buffer — sent to
`/add` but not yet carved into an episode. Returned **only**
when the request's `filters` contains `session_id` as a top-level eq
scalar (`{"session_id": "<sid>"}`); compound shapes (`AND` / `OR`
combinators, operator maps such as `{"eq": ...}` / `{"in": ...}`) do
not trigger the lookup because there is no defensible buffer-scope
mapping for them. Buffer rows have no `user_id` attribution, so
`session_id` is the only meaningful query dimension.

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
    "method": "llm_multiround",
    "top_k": 5
  }'
```

Response (abridged):

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
                "score": 0.01639344262295082
            }
        ],
        "images": [],
        "profiles": [],
        "unprocessed_messages": [],
        "degraded": []
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
| `user_id` | `string \| null` | one owner scope | `null` | `minLength=1` if set |
| `group_id` / `group_ids` | `string` / `array<string>` | one owner scope | `null` | see [Groups](#groups) |
| `asker_id` | `string \| null` | for a group scope | `null` | `minLength=1` if set |
| `app_id` | `string` *(ScopeId)* | no | `"default"` | — |
| `project_id` | `string` *(ScopeId)* | no | `"default"` | — |
| `memory_type` | [GetMemoryType](#getmemorytype) | yes | — | — |
| `member_id` | `string \| null` | with `"member_profile"` | `null` | `minLength=1` if set |
| `include_superseded` | `boolean` | no | `false` | with `"group_profile"` only |
| `page` | `integer` | no | `1` | `≥ 1` |
| `page_size` | `integer` | no | `20` | `1..100` |
| `sort_by` | `"timestamp" \| "updated_at"` | no | `"timestamp"` | — |
| `sort_order` | `"asc" \| "desc"` | no | `"desc"` | — |
| `filters` | [FilterNode](#filternode-filter-dsl) `\| null` | no | `null` | — |
| `include_groups` | `boolean` | no | `false` | with `user_id` and `memory_type` `"episode"` or `"scene"` |
| `include_group_ids` | `array<string> \| null` | no | `null` | with `include_groups`; `1..16` ids |

**Owner scope** — **Exactly one** of `user_id`, `group_id` and `group_ids`
must be set, and it must match `memory_type` (see
[GetMemoryType](#getmemorytype)). An `agent_id`, a `context_group_id` and the
`agent_case` / `agent_skill` memory types are refused with `422`: EverOS 2.0
has no agent memory (the message is quoted under
[`/memory/search`](#post-apiv2memorysearch)).

**`include_groups`** — Off by default. With `user_id` and `memory_type`
`"episode"` or `"scene"`, also list the records of the user's groups (every
group of the space the user currently belongs to, inside the user's
membership windows; `include_group_ids` narrows them) in the same list:
one index query sorts, counts and pages both, and every item names its owner
(`owner_type` `"user"`, or `"group"` with `group_id` and `source`). Other
kinds, a group scope and `audience: "group"` answer `422`.

**`app_id` / `project_id`** — Scope identifiers.

**`memory_type`** — Which item kind to list; see
[GetMemoryType](#getmemorytype). The route populates exactly one of
the arrays in `data` based on this value.

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
where `"timestamp"` does not apply (`"profile"` has no timestamp), the
server silently falls back to `"updated_at"`.

**`sort_order`** — `"desc"` (newest first, default) or `"asc"`.

**`filters`** — Optional [FilterNode](#filternode-filter-dsl) for
predicate-based filtering before pagination.

#### Response body

`200 OK` returns a SuccessEnvelope wrapping `GetData`. Every array is
always present so client code can iterate without branching on
`memory_type`; exactly one is populated.

| Field | Type | Notes |
|---|---|---|
| `episodes` | `array<GetEpisodeItem>` | Populated when `memory_type="episode"` |
| `scenes` | `array<GetEpisodeItem>` | Populated when `memory_type="scene"` |
| `images` | `array<object>` | Populated when `memory_type="image_memory"` |
| `profiles` | `array<GetProfileItem>` | Populated when `memory_type="profile"` or `"member_profile"` |
| `group_profile` | `array<object>` | Populated when `memory_type="group_profile"` |
| `total_count` | `integer` | Total matching records **before** paging |
| `count` | `integer` | Number of items in **this page** (`len(items)`) |

#### GetEpisodeItem

Same shape as [SearchEpisodeItem](#searchepisodeitem) **minus**
`score` (listing is unranked).

| Field | Type | Notes |
|---|---|---|
| `id` | `string` | `<user_id>_ep_<YYYYMMDD>_<NNN>` |
| `user_id` | `string \| null` | Owner |
| `app_id` | `string` | Scope |
| `project_id` | `string` | Scope |
| `session_id` | `string \| null` | Originating session; `null` for an episode merged from several sessions (by the Reflection of an earlier 2.0 build) |
| `timestamp` | `string` | ISO-8601 with timezone offset, identical to `/search` (both go through the same `from_iso_format` path) — see [Conventions](#conventions) |
| `sender_ids` | `array<string>` | Distinct `sender_id`s in the underlying messages |
| `summary` | `string` | Short summary |
| `subject` | `string` | One-line subject |
| `episode` | `string` | Full extracted narrative |
| `type` | `"Conversation" \| "Scene"` | `"Scene"` only under `memory_type: "scene"` |

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

Response (abridged):

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
        "scenes": [],
        "images": [],
        "profiles": [],
        "group_profile": [],
        "total_count": 1,
        "count": 1
    }
}
```

### Changing and forgetting a memory

EverOS 2.0 never edits or deletes a memory in place. A change writes a **new
version** that names the one it replaced; a forget **marks** what it reaches,
and its undo takes the mark away. Every old version stays in markdown, so
`GET /memory/history` can show it -- a default search or get shows only the
current versions.

A memory is named by the id a read returned: `<owner>_<entry_id>` (or the
bare entry id) for an episode (`ep_` / a group's `gep_`), a picture memory
(`im_` / `gim_`) or a scene version (`sc_` / `gsc_`), and the `item_id`
(`pi_<hex>`) for a profile item -- of `user.md`, or of a member's profile in a
group. A user's memory is named by `user_id`; a group's by `group_id` with
`asker_id`, under the forget rule: owners and admins may change or forget any
record of the group, a member those resting on their own messages alone. A
personal memory is changed and read by its owner: an `asker_id` (or the
identity header) naming someone else is `403`.

| Route | What |
|---|---|
| `POST /memory/update` | write a person's new version of one memory |
| `GET /memory/history` | every version of one memory, current first |
| `POST /memory/forget` | forget by target; a dry run unless `dry_run: false` |
| `POST /memory/forget/undo` | undo one forget by the `forget_id` it answered |
| `POST /memory/forget/session` (+ `/undo`) | alias: a user's `session` target, at once |
| `POST /memory/images/revise` | alias: `/update` of a picture memory, by its hash |
| `POST /memory/delete`, `/delete_by_ids` | alias: a forget, at once (2.0 has no hard delete) |

The command line runs the same service functions in-process:
`everos memory history|update|forget|undo` ([cli.md](cli.md)).

### POST /api/v2/memory/update

Changes one memory: writes a new version that supersedes the one the caller
read.

| Field | Type | Required | Default | What |
|---|---|---|---|---|
| `app_id`, `project_id` | ScopeId | no | `"default"` | — |
| `user_id` \| `group_id` | `string` | exactly one | — | whose memory |
| `asker_id` | `string` | for a group | — | who asks; recorded as `edited_by` (a user's defaults to the user) |
| `id` | `string` | yes | — | the memory, any version's id |
| `expected_version` | `string` | no | `id` | the version the caller read |
| `subject`, `summary`, `content` | `string` | — | stored text | an episode's title, preview and body; a scene's title, summary and narrative |
| `description`, `ocr` | `string` | — | stored text | a picture memory's description and the text in it; a profile item's `description` |
| `hold` | `bool` | no | `true` | keep the new version from the model's later updates |

A field the kind does not take is `422`. The new version keeps what the old
one rests on -- its time, conversation, segment, speakers, pictures -- and
adds `supersedes`, `edited_by`, `edited_at` and `held`; the old one is marked
superseded by it. An episode, picture memory or scene version is a new entry
in today's daily file, embedded and indexed by the cascade (a search returns
the new version, and never the old one, once the index has it; usually within
a second). A profile item is a new item with a fresh `item_id`, `supersedes`
and `since`; the old item moves to the profile's `history` with
`superseded_by`, as when the model gives a fact a new value.

**Held.** A person's version is held unless `hold` is false: the model's later
updates leave it as it is.

- The next **profile update** (one-to-one or a member's) still shows the model
  a held item, so it does not add the fact again, and undoes whatever its
  reply did to it: a new value, a retirement, a merge. A new value the model
  wrote for a held item stays as an item of its own, linked to nothing, so
  what the conversation said is not lost; the profile's summary is the
  model's and may mention it.
- The next **scene update** writes no narrative over a held version (no model
  call): an episode joining the scene still joins its cluster. Once released,
  the next join writes the narrative over the person's, with every episode
  that joined meanwhile (at most `[scenes] max_new_episodes` read).
- An episode and a picture memory are not rewritten by a model after they are
  written, so their hold only marks whose text it is. A **forget** reaches a
  held version like any other: a picture forget writes the version again
  without the picture's reference, a message forget re-extracts the segment.

`hold: false` alone releases a held version: a new version with the same text
for an episode, a picture memory or a scene (nothing in a daily log is edited
in place); for a profile item the flag only, under the same `item_id`. An
update that changes nothing answers `changed: false` and writes nothing.

```json
{
  "request_id": "<32-char hex>",
  "data": {
    "kind": "episode",
    "id": "amy_ep_20261112_00000003",
    "supersedes": "amy_ep_20261110_00000007",
    "changed": true,
    "held": true,
    "edited_by": "amy",
    "edited_at": "2026-11-12T09:30:00+08:00",
    "member_id": null
  }
}
```

| Status | `error.code` | When |
|---|---|---|
| `404` | `NOT_FOUND` | no such memory (a group record the asker cannot see answers the same) |
| `409` | `VERSION_CONFLICT` | `expected_version` (default `id`) is not the current version; `message` names it |
| `409` | `MEMORY_FORGOTTEN` | the memory is forgotten |
| `403` | `FORBIDDEN` | someone else's personal memory; a group asker who may not change the record |

### GET /api/v2/memory/history

Every version of one memory, current first, then the older ones newest
first -- whichever of its ids is asked.

Query: `id` (required), `user_id` or `group_id` (exactly one), `asker_id` (a
group's), `app_id`, `project_id`.

Each version: `id`; `state` -- `current`, `superseded` (a newer version
replaced it), `merged` (the Reflection of an earlier 2.0 build merged it
with others into one episode), `forgotten`;
`timestamp` (the memory's own time: when what it says happened); `next` (the
version that replaced it); `supersedes`; `edited_by` / `edited_at` / `held`
(a version a person wrote); `replaced_at` and `reason` (a profile value);
`sources` (conversation, segment, speakers, pictures; a scene's member
episodes; times there in epoch ms); `content`. `current` is `null` when the
memory is forgotten or its value was retired.

Read from markdown, never from the index. A user's own history lists the
user's forgotten versions as `forgotten` (and a value a live forget took out
of `user.md`). Inside a group only the versions inside the asker's membership
windows are listed, and none a live forget marked or replaced -- they may say
what was forgotten -- so a record with none answers `404`.

### POST /api/v2/memory/forget

Forgets, reversibly. **A dry run by default**: the answer says what the real
run would touch, and nothing changes; send `"dry_run": false` to forget. A
real forget answers with its `forget_id`, the undo token.

A user's body: `user_id`, `asker_id` (optional), `target`, `reason`,
`dry_run`.

| `target.kind` | Fields | Forgets |
|---|---|---|
| `record` | `record_ids` (1..100) | the records by id, of any kind: an episode, a picture memory (its current version), a scene (its current version, and its cluster, so it does not grow back), a profile item or a replaced value |
| `session` | `session_id` | the conversation's current episodes and picture memories |
| `message` | `session_id`, `message_ids` (1..100) | messages of one of the user's conversations (one bound to no group), by the id the memory holds -- the server's, `m_<session_id>_<timestamp ms>_<index in its batch, 3 digits>`, not the client's `message_id` (a group's batch keeps the client's): each segment that held one is extracted again without it (one model call per segment that keeps text; a segment left without text is forgotten), a picture only those messages carried is forgotten, and a buffered one is never read |
| `owner` | — | every current episode, picture memory and scene of the user, and the whole profile: items, replaced values, summary |
| `condition` | `memory_types` (`episode`, `image_memory`, `scene`, `profile`), `start` and/or `end` (epoch ms, `[start, end)`) | the memories of those kinds whose time falls in the range (a profile item by when it was observed) |
| `image` | `image_id` (content hash) | the picture's memories; the episodes and scenes that name it are written again without the reference |

A forget **follows the sources**: a scene none of whose episodes is alive any
more is forgotten, and one that keeps a live episode keeps its narrative,
which may still say what a forgotten episode or message said (forget the
scene by its id to drop it; a group's scene is written again); a picture file no live memory of the user refers to is
taken out of circulation (the bytes stay); a profile item, or a replaced
value, goes once every segment it rests on is forgotten (or every picture it
was read off). A profile that loses an item loses its summary too, which is
written over every item and may repeat it: the next profile update writes it
again from the items left, and the undo puts it back while none has (a
member's profile in a group does the same). A forget marks and erases the
index rows at once: nothing leaves markdown, and the forgotten versions stay
readable through the user's `/memory/history`. Every forget is recorded in
the forget log (`system.db`) with what it marked or wrote and what it took
out of profile documents, each value with its place, who asked and why. A
conversation sent again after its forget (or continued) is new material: its
new memories are not forgotten, and the profile reads them (a `message`
target forgets those message ids for good, until its undo).

```json
{
  "data": {
    "forget_id": "fg_0123456789abcdef",
    "dry_run": false,
    "user_id": "amy",
    "target": {"kind": "session", "session_id": "trip-chat"},
    "episodes": ["amy_ep_20261110_00000002"],
    "new_episodes": [],
    "image_memories": ["amy_im_20261110_00000001"],
    "images": ["9f2c…"],
    "scenes": [],
    "new_scenes": [],
    "profile_items": ["pi_3f9a2c71d0be"]
  }
}
```

A group's body (`group_id`, `asker_id`, `target` of kind `message`,
`member`, `session`, `record` or `image`, `reason`, `dry_run`) forgets
inside the group ([group-memory.md](group-memory.md), section 5.5), also a dry
run by default; two people's implicit group takes the same body. One engine
serves both: a forget marks (in a group too), a picture's forget is a text
edit with no model call, and a message's re-extracts the segments that keep
text. The picture body of 1.x (`user_id` + `image_id`, no `target`) is a
user's `image` target run at once.

`404` for a record id that names nothing; `503` `SESSION_BUSY` when a
conversation it touches stays busy (nothing was forgotten).

### POST /api/v2/memory/forget/undo

Undoes one forget: `user_id` + `forget_id` (any target of the user), or
`group_id` + `asker_id` + `forget_id` (the forget's own authorisation). The
marks it wrote go back to what they were; a version it wrote in place of
another (a text edit, a re-extracted episode, a scene narrated again) is
superseded by the version it replaced, written again as the newest -- nothing
in a daily log is removed. Profile values come back in their places, by id
(a forget logged before places were kept puts them back by description), and
so do the values other live forgets took because of it (a profile item
resting on its segments); an entry another live forget also marked stays
forgotten until that one is undone too.

**Order.** `409` `CONFLICT` while a version this forget wrote is no longer
current -- a later forget, undo or edit replaced it: undo that later forget
first. Forgets of different records undo in any order. `applied: false` when
it was undone already. The picture body (`user_id` + `image_id`) undoes the
picture's newest forget, and `/forget/session/undo` the conversation's
newest.

### POST /api/v2/memory/delete

**EverOS 2.0 has no hard delete.** The route stays for the callers that send
it -- everos-mcp's `forget_session`, the evaluation harnesses'
`--replace-sessions` -- and answers with a forget of the same scope, run at
once and reversible by the `forget_ids` it returns.

| Request sets | Forget |
|---|---|
| `user_id` | target `owner` of the user |
| `user_id` + `session_id` | target `session` of the user |
| `session_id` | target `session` of every user whose memory holds the conversation, one forget each |
| `agent_id` (± `session_id`) | refused, `422`: EverOS 2.0 has no agent memory |
| `group_id` + `asker_id` + `session_id` or `user_id` | the group's delete: an owner's or admin's forget of the session or the member's messages ([group-memory.md](group-memory.md), section 15) |

A session bound to a group without `group_id` is `422`. Nothing leaves
markdown, and the conversation's segments and buffer stay in `system.db`:
sending the conversation again writes new memories beside the forgotten ones.

```json
{
  "request_id": "<32-char hex>",
  "data": {"filters": ["session_id"], "count": 7, "forget_ids": ["fg_0123456789abcdef"]}
}
```

`count` is the number of memories taken out of default reads: episodes,
picture memories, scenes and profile items. `503` `SESSION_BUSY` as for a
forget of the conversation.

### POST /api/v2/memory/delete_by_ids

**A forget**, too: the episodes by id (`memory_type: "episode"`, `memory_ids`
`<owner_id>_ep_<YYYYMMDD>_<n>`, 1..50; a `reason` whose `code` is kept and
whose `note` is never logged), target `record` per owner, run at once. One
malformed id rejects the whole request (`422`). Ids that match nothing are
skipped; sending the same request again forgets nothing more. The answer is
Cloud's shape with the forgets' ids:

```json
{"data": {"memory_type": "episode", "success": true, "forget_ids": ["fg_…"]}}
```

### POST /api/v2/ome/trigger

Manually trigger a registered OME strategy.

#### Request body

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | `string` | yes | — | Strategy name (e.g. `extract_user_profile`) |
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
`applies_to` / `Counter`). A manual trigger sends a `ManualTick`, which
reaches only a strategy that subscribes to it or runs on a `Cron` schedule,
and only with `"force": true` when the strategy is `enabled=false` in
`ome.toml`. Every strategy EverOS 2.0 ships runs on an event instead -- a
new episode (`extract_user_profile`, `update_user_scenes`), a group segment
or flush (`update_member_profiles`, `update_group_scenes`,
`update_group_profile`) or a group's idle window
(`update_group_profile_idle`) -- so triggering one of them returns
`{"status": "not_dispatched", "dispatched": 0, "runs": []}` instead of an
error.

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
  -d '{"name": "extract_user_profile"}'
# {"status": "not_dispatched", "name": "extract_user_profile", "dispatched": 0, "runs": []}
```

---

### Groups

Group chats are the second memory owner next to users
([group-memory.md](group-memory.md)). A group has a roster of membership
intervals (`[joined_at, left_at)`, epoch ms) kept as an append-only log in
`groups/<group_id>/group.md` and mirrored into `system.db`.

| Route | Who may | What |
|---|---|---|
| `PUT /api/v2/groups/{group_id}` | anyone to create; owners and admins to update | Create the group (a roster that names no owner makes the asker its owner) or update `name` / `description` / `extra` / `history_visible_from`; `roster` events are appended |
| `POST /api/v2/groups/{group_id}/roster` | owners and admins; a member for their own `leave` | Append `join` / `leave` / `update` events. The open buffers of the group's sessions are cut first, so no segment spans the change |
| `GET /api/v2/groups/{group_id}?project_id=&asker_id=&include_history=` | current members | Meta and current roster; every interval with `include_history=true` |

A roster event: `op`, `member_id`, optional `kind` (`human` / `assistant`),
`display_name`, `aliases`, `role` (`owner` / `admin` / `member` /
`assistant`), `title`, `at` (epoch ms, default now). A `join` for a member who
is in the group, or a `leave` / `update` for one who is not, is `409`.

`/memory/add` takes `group_id` (the first batch binds the session; later
batches inherit it; another group is `409`), `roster` (events applied before
the messages; for an existing group the `asker_id` must be an owner or admin)
and `messages[].message_id`. A sender who is not a member joins at their first
message (`[groups] implicit_join`); an unknown group is created
(`[groups] implicit_create`). Group messages never become per-sender personal
memory. A batch without `group_id` whose session holds two or more people
(role `user` senders) is their implicit group: `implicit_<hash>`, the same for
every session between the same people, created with them (and the assistant)
as members and history visible from creation; a person who first speaks in
such a session later is `409`, and a client never names an `implicit_` group
on `/memory/add`, `PUT /groups/{id}` or the roster route (`422`). See
[group-memory.md, section 7.9](group-memory.md#79-implicit-groups-two-people-are-a-group-too).

`/memory/search` and `/memory/get` take `group_id` or `group_ids` (one pool)
with `asker_id` and `audience` (`asker`, default; `group`: only what every
current member can see). The asker must be a current member of each group, and
sees only records whose sources lie inside one of their membership windows;
the search response carries a `scope` block with those windows. A user's own
search can cover the user's groups too (`include_groups`, off by default; see
[`/memory/search`](#post-apiv2memorysearch)); `/memory/get` takes it for
`episode` and `scene` listings, paged as one list. A group's
records are changed with `/memory/update`, read with `/memory/history` and
forgotten with `/memory/forget` and a group body
([Changing and forgetting a memory](#changing-and-forgetting-a-memory)); a
group-scoped `/memory/delete` is an owner's or admin's forget.

**What a group session writes.** Each segment becomes one group episode under
`groups/<group_id>/episodes/`, with `sender_ids` (who spoke), `subject_ids`
(the members it names, else the human senders) and `origin` (`member`,
`assistant` or `mixed`). Each picture becomes one picture memory with
`sender_ids = [poster]`. In the background, each member who spoke gets a
profile in the group, `groups/<group_id>/members/<member_id>.md`. It is
updated after `[groups] member_profile_every` segments and on every flush.
With `[scenes] groups = true` (off by default), the group's scenes are
written to `groups/<group_id>/scenes/`, as a user's are with `[scenes] users`.

**Group results.** A group episode carries `owner_type: "group"`, `group_id`
and `source: {memcell_id, from, to}`: its memcell and the epoch ms of the
segment's first and last message. A scene has `type: "Scene"` and
`memcell_id: null`. With `include_profile: true` a group search attaches the
asker's member-in-group profile and those of up to three members the query
names by roster name or alias, in each group:
`{owner_type: "group", group_id, member_id, profile_data}`. Items whose
sources lie outside the asker's windows are left out, and so is the summary
then. An item that replaced earlier values of the same fact carries `since`
(epoch ms), `supersedes` and `previous`: the values it replaced, newest
first, at most two, each `{item_id, description, until, source_start,
source_end}`. A replaced value outside the asker's windows, or forgotten, is
not listed, and an item left with none carries none of the three
([group-memory.md](group-memory.md), section 7.4). A personal profile is never
attached. The multi-round search starts from `"<display name>: <question>"`.

**Identity.** EverOS has no authentication: `asker_id` is the caller's
assertion. A deployment behind a gateway sets `[access] identity_header`; the
asker is then read from that header and a body `asker_id` that differs is
`403`. A one-to-one request without `asker_id` is read as asked by its owner.

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
