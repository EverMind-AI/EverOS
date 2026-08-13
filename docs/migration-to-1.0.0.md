# EverOS 1.0.0 Migration Notes

EverOS 1.0.0 is a fresh architecture. Historical issues, examples, and
plugins may reference APIs and infrastructure that no longer exist in
the current open-source repository.

## Current 1.0.0 Contract

The supported local OSS HTTP API lives under:

```text
POST /api/v1/memory/add
POST /api/v1/memory/flush
POST /api/v1/memory/search
POST /api/v1/memory/get
```

Use [api.md](api.md) for the canonical request and response schemas.

The storage stack is:

```text
Markdown files + SQLite + LanceDB
```

MongoDB, Elasticsearch, Milvus, Redis, Kafka, longjob workers, and the
old Docker Compose stack are not part of EverOS 1.0.0.

## Legacy API Mapping

These pre-1.0.0 routes are no longer supported by the OSS repo:

| Legacy route | EverOS 1.0.0 replacement |
|---|---|
| `POST /api/v1/memories` | `POST /api/v1/memory/add` |
| `POST /api/v1/memories/group` | `POST /api/v1/memory/add` with `app_id` / `project_id` scoping |
| `GET /api/v1/memories/search` | `POST /api/v1/memory/search` |
| `POST /api/v1/memories/search` | `POST /api/v1/memory/search` |
| `GET /api/v1/memories` | `POST /api/v1/memory/get` |
| `POST /api/v1/memories/get` | `POST /api/v1/memory/get` |
| `/api/v3/agentic/*` | `POST /api/v1/memory/*` |

The 1.0.0 API also changed memory type names. For example,
`episodic_memory` is now `episode` in `/get`; `/search` returns typed
arrays such as `episodes`, `profiles`, `agent_cases`, and
`agent_skills`.

## Integration Guidance

For new integrations:

- Batch messages into one `/api/v1/memory/add` request instead of
  sending one flat message object per HTTP call.
- Use `/flush` when a demo or test needs immediate extraction.
- Use `/search` for ranked recall and `/get` for paginated browsing.
- Treat old OpenClaw and EverMem Cloud plugin examples as archived
  references unless they have been explicitly updated to the 1.0.0 API.

## Benchmark Guidance

The current LoCoMo reproduction path is documented in
[benchmarks/README.md](../benchmarks/README.md). The benchmark driver uses
the server API: add, flush, search, answer, and evaluate.

Old HyperMem / pre-1.0.0 evaluation pipeline reports should not be used
as 1.0.0 bug reports unless they can be reproduced with the current
benchmark commands.
