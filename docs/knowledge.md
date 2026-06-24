# Knowledge Base

The Knowledge module turns unstructured documents (Markdown, PDF, DOCX, …)
into a searchable topic library. Upload a file, and EverOS extracts a
structured topic tree via LLM, indexes it for keyword + vector search,
and keeps the original file for reference.

## Quick start

> The examples below assume EverOS is running on the default port 8000.
> See [README](../README.md) or [QUICKSTART](../QUICKSTART.md) to start
> the server.

```bash
# Upload a document
curl -s -X POST http://localhost:8000/api/v1/knowledge/documents \
  -F "file=@my-report.pdf" \
  -F "title=Q1 Engineering Report" \
  | jq .data
# → { "doc_id": "d_a1b2c3d4e5f6", "category_id": "Technology", "topic_count": 8, ... }

# Search
curl -s -X POST http://localhost:8000/api/v1/knowledge/search \
  -H "Content-Type: application/json" \
  -d '{"query": "performance bottleneck", "method": "hybrid"}' \
  | jq '.data.hits[:3] | .[] | {topic_name, score}'
```

## Three-tier hierarchy

Knowledge is organized into three levels, from broadest to most granular:

```
L0  Category      ← taxonomy bucket (e.g., "Technology", "Finance")
L1  Document      ← one uploaded file = one document
L2  Topic         ← LLM-extracted section with content
```

Each level corresponds to a different granularity of API:

| Level | Endpoint | Returns |
|-------|----------|---------|
| L0 | `GET /categories` | `category_id`, `description`, `document_count` |
| L1 | `GET /documents` | `doc_id`, `title`, `category_id`, `topic_count`, `created_at` |
| L1 (detail) | `GET /documents/{id}` | Full detail: summary, source info, original file path, topic list |
| L2 | `GET /topics/{id}` | Full topic: content, labels, tree position |

## Storage layout

Every document is a self-contained directory. Markdown files are the
single source of truth; SQLite and LanceDB are derived indexes built
automatically by the cascade daemon.

```
~/.everos/<app>/<project>/knowledge/
├── .taxonomy.md                              ← category definitions (YAML)
├── Technology/
│   └── Q1_Engineering_Report_d_a1b2c3d4e5f6/
│       ├── index.md                          ← document metadata + summary
│       ├── 1_Performance_Analysis.md         ← topic with full content
│       ├── 2_Infrastructure_Costs.md
│       ├── 3_Team_Velocity.md
│       └── _original/                        ← original uploaded file
│           └── my-report.pdf
└── Finance/
    └── Budget_Review_d_f6e5d4c3b2a1/
        ├── index.md
        ├── 1_Revenue.md
        └── _original/
            └── budget.xlsx
```

### Storage roles

```
Markdown (source of truth)  +  SQLite (structured state)  +  LanceDB (vector + BM25 index)
```

| Store | What it holds | Role |
|-------|---------------|------|
| Markdown | Document metadata, summaries, topic content, original files | Single source of truth; human-readable and editable |
| SQLite | Document rows, topic rows (with content), change queue | Structured queries, paginated lists, count aggregation |
| LanceDB | Topic vectors, BM25 tokens, scalar fields | Search index (fully rebuildable from Markdown) |

Even if SQLite and LanceDB data is corrupted, as long as the Markdown
files are intact, the indexes can be fully rebuilt via the cascade daemon.

### Markdown format

**index.md** (document root):

```yaml
---
type: knowledge_document
id: d_a1b2c3d4e5f6
doc_id: d_a1b2c3d4e5f6
category_id: Technology
title: Q1 Engineering Report
schema_version: 1
source_name: my-report.pdf
source_type: file
---
This report covers Q1 engineering outcomes including performance
analysis, infrastructure costs, and team velocity metrics.
```

The body is an LLM-generated summary of the entire document.

**Topic files** (e.g., `1_Performance_Analysis.md`):

```yaml
---
type: knowledge_topic
id: d_a1b2c3d4e5f6_1
node_id: d_a1b2c3d4e5f6_1
doc_id: d_a1b2c3d4e5f6
category_id: Technology
topic_index: 1
topic_name: Performance Analysis
topic_path: Q1 Engineering Report > Performance Analysis
summary: Analysis of API latency, database query times, and caching hit rates.
depth: 1
parent_node_id: d_a1b2c3d4e5f6_0
children_node_ids: []
content_labels: ["performance", "latency", "caching"]
schema_version: 1
---
The P99 API latency dropped from 450ms to 120ms after the Redis
caching layer was deployed in week 6. Database query times improved
by 40% following the index optimization sprint...
```

The body is the full extracted content for this topic.

> The taxonomy file uses `kind` (not `type`) in its frontmatter to
> distinguish it from document and topic files, which use `type`.

### Original file preservation

The `_original/` subdirectory stores the uploaded binary file unchanged.
Users can locate the original via the `original_file_path` field returned
by `GET /documents/{doc_id}`.

The underscore prefix follows the Jekyll/Eleventy convention for
non-content directories that the cascade daemon should skip.

Lifecycle:
- **POST** (create) — writes `_original/<filename>`
- **PUT** (replace) — clears the old directory and writes the new file
- **DELETE** — removes the entire document directory including `_original/`
- **PATCH** (category change) — moves the whole directory; `_original/` follows

## Taxonomy

Categories are defined in `.taxonomy.md` at the knowledge root. EverOS
ships with 20 default categories:

| Category | Description |
|----------|-------------|
| Technology | CS, software, AI/ML, cloud, cybersecurity |
| Science | Physics, chemistry, biology, astronomy |
| Medical | Clinical medicine, drugs, public health |
| Finance | Securities, banking, accounting, fintech |
| Legal | Laws, contracts, compliance, IP |
| Education | Teaching, curriculum, e-learning |
| Business | Strategy, marketing, operations, HR |
| Engineering | Mechanical, civil, electrical engineering |
| Arts | Visual arts, music, literature, film |
| Sports | Athletics, fitness, sports science |
| Travel | Tourism, hospitality, transportation |
| Food | Culinary, nutrition, food safety |
| Environment | Climate, ecology, sustainability |
| Politics | Government, international relations, policy |
| History | Historical events, civilizations, historiography |
| Psychology | Cognitive science, behavioral psychology, mental health |
| Agriculture | Farming, crop science, agribusiness |
| RealEstate | Property development, urban planning, housing |
| Media | Journalism, social media, PR |
| Others | Fallback for unclassified documents |

### Customization

Edit `.taxonomy.md` directly to add, remove, or rename categories:

```yaml
---
kind: knowledge_taxonomy
categories:
  - id: Technology
    description: Computer science, software engineering, AI/ML.
  - id: InternalOps
    description: Company-specific operational procedures and runbooks.
  - id: CustomerSuccess
    description: Customer onboarding, support playbooks, case studies.
---
```

Taxonomy changes are **hot-reloaded** — no server restart needed. The
system reads `.taxonomy.md` from disk on every upload and category list
request, so edits take effect immediately.

When a document is uploaded, the LLM selects the best-matching category
from this list. If no category matches, the document falls back to
`Others`. You can also specify `category_id` explicitly in the upload
request to bypass LLM classification.

## API reference

All endpoints are under `/api/v1/knowledge`. Responses use the envelope
format `{"request_id": "...", "data": {...}}`. The `request_id` field is
omitted from examples below for brevity.

### Upload a document

```
POST /documents
Content-Type: multipart/form-data
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | yes | The document to upload |
| `title` | string | yes | Human-readable title |
| `source_type` | string | no | Provenance type (`"file"`, `"url"`, …) |
| `category_id` | string | no | Skip LLM classification; use this category |
| `app_id` | string | no | Tenant app (default: `"default"`) |
| `project_id` | string | no | Tenant project (default: `"default"`) |

**Response** (201):

```json
{
  "data": {
    "doc_id": "d_a1b2c3d4e5f6",
    "category_id": "Technology",
    "topic_count": 8,
    "source_name": "my-report.pdf",
    "md_path": "/home/user/.everos/default_app/default_project/knowledge/Technology/Q1_Report_d_a1b2c3d4e5f6",
    "original_file_path": "/home/user/.everos/.../Q1_Report_d_a1b2c3d4e5f6/_original/my-report.pdf"
  }
}
```

`original_file_path` is the absolute path to the preserved upload, or `null`
when no binary was stored (e.g. an empty filename).

**Example — Python**:

```python
from pathlib import Path

import httpx


async def upload_document(file_path: str, title: str) -> dict:
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        with open(file_path, "rb") as f:
            resp = await client.post(
                "/api/v1/knowledge/documents",
                files={"file": (Path(file_path).name, f)},
                data={"title": title},
            )
        resp.raise_for_status()
        return resp.json()["data"]
```

**Example — curl**:

```bash
curl -X POST http://localhost:8000/api/v1/knowledge/documents \
  -F "file=@report.pdf" \
  -F "title=Quarterly Report" \
  -F "category_id=Finance"
```

### Replace a document

```
PUT /documents/{doc_id}
Content-Type: multipart/form-data
```

Same fields as POST. Atomic operation: if extraction fails, the old
document is restored from backup.

### Update metadata

```
PATCH /documents/{doc_id}
Content-Type: application/json
```

```json
{
  "title": "Updated Title",
  "category_id": "Finance"
}
```

Returns `updated_fields: ["title", "category_id"]`. Changing `category_id`
moves the document directory to the new category folder.

### Delete a document

```
DELETE /documents/{doc_id}
```

Returns 204 if the document did not exist (idempotent), or 200 with
`deleted_topics` count.

### List documents

```
GET /documents?page=1&page_size=20&sort_by=created_at&sort_order=desc
```

Optional filter: `category_id=Technology`. `sort_by` accepts `created_at`
(default), `updated_at`, or `title`; `sort_order` is `asc` or `desc`.

```json
{
  "data": {
    "documents": [
      {
        "doc_id": "d_a1b2c3d4e5f6",
        "category_id": "Technology",
        "title": "Q1 Engineering Report",
        "topic_count": 8,
        "created_at": "2026-06-24T10:00:00Z"
      }
    ],
    "total": 42,
    "page": 1,
    "page_size": 20
  }
}
```

### Get document detail

```
GET /documents/{doc_id}
```

Returns full metadata, summary, original file path, and topic overview list.

```json
{
  "data": {
    "doc_id": "d_a1b2c3d4e5f6",
    "category_id": "Technology",
    "title": "Q1 Engineering Report",
    "summary": "This report covers Q1 engineering outcomes...",
    "source_name": "my-report.pdf",
    "source_type": "file",
    "original_file_path": "/home/user/.everos/.../Q1_Report_d_a1b2c3d4e5f6/_original/my-report.pdf",
    "topics": [
      {
        "topic_id": "d_a1b2c3d4e5f6_1",
        "topic_name": "Performance Analysis",
        "topic_path": "Q1 Engineering Report > Performance Analysis",
        "depth": 1,
        "summary": "Analysis of API latency..."
      }
    ],
    "created_at": "2026-06-24T10:00:00Z",
    "updated_at": "2026-06-24T10:00:00Z"
  }
}
```

`original_file_path` is `null` for documents created before the original
file preservation feature, or when no file was attached.

### Get topic detail

```
GET /topics/{topic_id}
```

Returns the full topic content, tree structure, and labels.

```json
{
  "data": {
    "topic_id": "d_a1b2c3d4e5f6_1",
    "doc_id": "d_a1b2c3d4e5f6",
    "category_id": "Technology",
    "topic_name": "Performance Analysis",
    "topic_path": "Q1 Engineering Report > Performance Analysis",
    "depth": 1,
    "summary": "Analysis of API latency, database query times...",
    "content": "The P99 API latency dropped from 450ms to 120ms...",
    "content_labels": ["performance", "latency", "caching"],
    "parent_topic_id": "d_a1b2c3d4e5f6_0",
    "children_topic_ids": [],
    "created_at": "2026-06-24T10:00:00Z",
    "updated_at": "2026-06-24T10:00:00Z"
  }
}
```

### Search

```
POST /search
Content-Type: application/json
```

```json
{
  "query": "performance bottleneck",
  "method": "hybrid",
  "top_k": 10,
  "include_content": true,
  "score_threshold": 0.5
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | — | Search query (required, 1–2000 chars) |
| `method` | string | `"hybrid"` | `"keyword"`, `"vector"`, or `"hybrid"` |
| `top_k` | int | 10 | Max results (1–100) |
| `include_content` | bool | false | Include full topic content in results |
| `score_threshold` | float | null | Drop results below this score |

**Search methods**:

- **keyword** — BM25 sparse retrieval over tokenized summary + content
- **vector** — Dense ANN over embedded summary vectors (requires embedding provider)
- **hybrid** — Parallel keyword + vector, fused with Reciprocal Rank Fusion (RRF), then cross-encoder reranking

All three methods embed the query and apply cross-encoder reranking, so
knowledge search requires **both** an embedding and a rerank provider —
there is no provider-free fallback (this is by design: no silent
degradation). The two failure modes map to distinct status codes:

- **Provider not configured** → `500 CONFIGURATION_ERROR` (a required
  setting is missing; retrying will not help — set `EVEROS_EMBEDDING__*` /
  `EVEROS_RERANK__*`).
- **Provider configured but failing/timing out at call time** →
  `503 EXTERNAL_SERVICE_UNAVAILABLE` (transient; retryable).

**Response**:

```json
{
  "data": {
    "hits": [
      {
        "topic_id": "d_a1b2c3d4e5f6_1",
        "category_id": "Technology",
        "topic_name": "Performance Analysis",
        "topic_path": "Q1 Engineering Report > Performance Analysis",
        "depth": 1,
        "summary": "Analysis of API latency...",
        "content": "The P99 API latency dropped...",
        "score": 0.92,
        "retrieval_method": "hybrid",
        "document": {
          "doc_id": "d_a1b2c3d4e5f6",
          "title": "Q1 Engineering Report",
          "summary": "This report covers..."
        }
      }
    ],
    "total": 3,
    "took_ms": 245.6
  }
}
```

### List categories

```
GET /categories
```

```json
{
  "data": {
    "categories": [
      {"category_id": "Technology", "description": "Computer science...", "document_count": 12},
      {"category_id": "Finance", "description": "Securities...", "document_count": 5},
      {"category_id": "Others", "description": "Fallback...", "document_count": 0}
    ]
  }
}
```

## Search pipeline

```
query ─→ embed ─→ keyword (BM25) ─┐
                   vector  (ANN) ──┤─→ RRF fusion ─→ rerank ─→ top_k
```

1. **Embed** — the query is embedded using the configured embedding provider
2. **Recall** — dual-channel retrieval from LanceDB:
   - BM25 channel: keyword matching on `summary_tokens` + `content_tokens`
   - ANN channel: nearest-neighbor search on the `vector` column
   - In `hybrid` mode, both channels run in parallel
3. **Fuse** — Reciprocal Rank Fusion merges the two candidate lists
4. **Rerank** — cross-encoder reranker rescores the top candidates
5. **Filter** — drop results below `score_threshold` and limit to `top_k`

### Configuration

Search tuning parameters in `src/everos/config/default.toml`:

```toml
[knowledge.search]
recall_n = 200       # initial recall pool size per channel
rerank_n = 50        # candidates sent to reranker
mass_top_m = 50      # category-aware retrieve pool
lambda = 0.1         # category boost weight
top_k_cap = 100      # hard cap on returned results
```

Override via environment variables:

```bash
export EVEROS_KNOWLEDGE__SEARCH__RECALL_N=500
export EVEROS_KNOWLEDGE__SEARCH__RERANK_N=100
```

## Cascade sync

The cascade daemon watches the knowledge Markdown directory for file
changes and keeps SQLite + LanceDB in sync.

```
md file written
  → FSEvents / watchdog detects change
  → worker picks up from queue (≤1s poll interval)
  → handler dispatched by file type:
      index.md    → KnowledgeDocumentHandler → SQLite upsert (metadata)
      N_topic.md  → KnowledgeTopicHandler    → tokenize + embed + SQLite + LanceDB upsert
```

The topic handler uses a SHA-256 content digest to skip unchanged files —
re-embedding only happens when the content actually changes.

Typical latency from file write to search availability: **1–3 seconds**.

## Supported file formats

EverOS accepts text-based files natively. Binary formats require the
`everos[multimodal]` extra (depends on LibreOffice for document conversion).

| Category | Formats | Requires `[multimodal]` |
|----------|---------|:-----------------------:|
| Text | `.txt`, `.md`, `.csv`, `.tsv`, `.vtt` | No |
| Documents | `.pdf`, `.docx`, `.doc`, `.rtf`, `.odt`, `.pages` | Yes |
| Spreadsheets | `.xlsx`, `.xls`, `.ods`, `.numbers` | Yes |
| Presentations | `.pptx`, `.ppt`, `.odp`, `.key` | Yes |
| Web | `.html`, `.htm`, `.eml` | Yes |
| Images (OCR) | `.png`, `.jpg`, `.webp`, `.tiff`, `.bmp`, `.svg` | Yes |
| Audio (transcription) | `.mp3`, `.wav`, `.m4a`, `.amr`, `.aiff`, `.aac`, `.ogg`, `.flac` | Yes |

```bash
pip install everos[multimodal]
```

## Error handling

| HTTP | Error code | Scenario |
|------|-----------|----------|
| 404 | `NOT_FOUND` | Document or topic does not exist |
| 409 | `CONFLICT` | `doc_id` already exists (use PUT to replace) |
| 415 | `UNSUPPORTED_FORMAT` | File format not parseable |
| 422 | `INVALID_INPUT` | Empty/oversized query, empty title, invalid ID format |
| 500 | `CONFIGURATION_ERROR` | Embedding or rerank provider not configured |
| 503 | `EXTERNAL_SERVICE_UNAVAILABLE` | Configured embedding/rerank provider failing at call time |
| 503 | `CAPABILITY_UNAVAILABLE` | `everos[multimodal]` not installed |

All error responses include a human-readable `message` field:

```json
{
  "error": {
    "code": "NOT_FOUND",
    "message": "Document 'd_abc123' not found"
  }
}
```

## Multi-tenancy

All endpoints accept `app_id` and `project_id` parameters (default:
`"default"`). Data is fully isolated per tenant pair:

```bash
# Tenant A uploads
curl -X POST .../documents -F "file=@a.pdf" -F "title=A" \
  -F "app_id=tenant_a" -F "project_id=proj_1"

# Tenant B cannot see Tenant A's data
curl .../documents?app_id=tenant_b&project_id=proj_1
# → { "documents": [], "total": 0 }
```

Storage paths, SQLite rows, and LanceDB indexes are all scoped by
`app_id` + `project_id`.

## End-to-end walkthrough

A complete workflow from upload to search:

```bash
BASE=http://localhost:8000/api/v1/knowledge

# 1. List available categories
curl -s "$BASE/categories" | jq '[.data.categories[] | .category_id]'
# → ["Technology", "Science", "Medical", ..., "Others"]

# 2. Upload a document
DOC_ID=$(curl -s -X POST "$BASE/documents" \
  -F "file=@architecture-guide.md" \
  -F "title=System Architecture Guide" \
  | jq -r .data.doc_id)
echo "Created: $DOC_ID"

# 3. View document detail (with topic list)
curl -s "$BASE/documents/$DOC_ID" | jq '{
  title: .data.title,
  category: .data.category_id,
  topics: [.data.topics[] | .topic_name],
  original: .data.original_file_path
}'

# 4. Read a topic — pick the first from the detail response
TOPIC_ID=$(curl -s "$BASE/documents/$DOC_ID" \
  | jq -r '.data.topics[0].topic_id')
curl -s "$BASE/topics/$TOPIC_ID" | jq '{
  name: .data.topic_name,
  path: .data.topic_path,
  content: .data.content[:200],
  labels: .data.content_labels
}'

# 5. Search (index is typically ready within 1–3 seconds)
sleep 3
curl -s -X POST "$BASE/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how to handle service failures",
    "method": "hybrid",
    "top_k": 5,
    "include_content": true
  }' | jq '.data | {
    total,
    took_ms,
    top_hit: .hits[0] | {topic_name, score, content: .content[:100]}
  }'

# 6. Move document to a different category
curl -s -X PATCH "$BASE/documents/$DOC_ID" \
  -H "Content-Type: application/json" \
  -d '{"category_id": "Engineering"}' \
  | jq .data.updated_fields
# → ["category_id"]

# 7. Clean up
curl -s -X DELETE "$BASE/documents/$DOC_ID" | jq .
```
