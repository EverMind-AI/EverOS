# Multimodal Memory

EverOS turns non-text content — images, PDFs, audio, office documents,
HTML, email — into the **same structured, searchable memory** as plain
text. You attach the asset to a message at ingest time; a vision/audio
capable LLM parses it into text, and from there it flows through the
identical extraction → markdown → index pipeline as any text turn. The
result is fully retrievable with the same `/search` stack.

## Table of contents

- [How it works](#how-it-works)
- [Prerequisites](#prerequisites)
  - [Install the extra](#install-the-extra)
  - [LibreOffice (office documents only)](#libreoffice-office-documents-only)
  - [Configure the multimodal LLM](#configure-the-multimodal-llm)
- [Supported modalities](#supported-modalities)
- [Sending multimodal content](#sending-multimodal-content)
  - [Payload: `uri` vs `base64`](#payload-uri-vs-base64)
  - [Example: image by URL](#example-image-by-url)
  - [Example: mixed text + image in one turn](#example-mixed-text--image-in-one-turn)
  - [Example: inline PDF via base64](#example-inline-pdf-via-base64)
  - [Example: local file via `file://`](#example-local-file-via-file)
  - [Calling from Python (plain HTTP)](#calling-from-python-plain-http)
- [Configuration reference](#configuration-reference)
- [Errors and limits](#errors-and-limits)
- [Searching multimodal memory](#searching-multimodal-memory)

## How it works

```
POST /api/v1/memory/add
  messages[].content = [ ContentItem, ContentItem, ... ]
        │
        │  text items      → used verbatim
        │  non-text items  → multimodal LLM (everalgo-parser)
        ▼
  parsed text merged back into the session buffer (in original order)
        │
        ▼
  boundary detector → extraction LLM → MemCell
        │
        ▼
  markdown (truth)  +  SQLite (state)  +  LanceDB (vector + BM25)
        │
        ▼
  retrievable via /search and /get like any text memory
```

Each **non-text** `ContentItem` is routed through the parser, which calls
a separate, vision/audio capable LLM (configured independently from the
main extraction `[llm]`, so parsing can target a multimodal endpoint
without changing boundary or extraction behaviour). Visual/audio formats
(image / pdf / audio / office) always go through that LLM; a few
text-bearing formats can be parsed without it (e.g. a plain email with no
inline images). The parser returns text; that text takes the place of the
asset in the message buffer. Nothing downstream of the parser
knows or cares that the content originated as an image or PDF — the raw
bytes are **not** persisted past extraction (the episode and memory cell
store only the parsed text).

## Prerequisites

### Install the extra

Multimodal parsing lives behind an optional dependency group so the base
install stays lean:

```bash
uv pip install 'everos[multimodal]'    # or: pip install 'everos[multimodal]'
```

This pulls in `everalgo-parser[svg]` — the `[svg]` bundle adds `cairosvg`
so SVG works out of the box.

### LibreOffice (office documents only)

Office formats (`.doc` / `.docx` / `.ppt` / `.pptx` / `.xls` / `.xlsx`)
are converted to PDF before being fed to the multimodal LLM. The parser
shells out to `soffice`, LibreOffice's headless renderer, so LibreOffice
must be present on the **server** host:

```bash
brew install --cask libreoffice          # macOS
sudo apt-get install -y libreoffice       # Debian / Ubuntu
```

Without LibreOffice, **office uploads return `503`**
(`CAPABILITY_UNAVAILABLE`) with a clear error message; image / PDF /
audio / HTML / email parsing is unaffected.

### Configure the multimodal LLM

The parser uses its own LLM section, independent from `[llm]`. The model
must accept OpenAI `image_url` parts. Configure these in `everos.toml`
(under `[multimodal]`) or via env vars:

```bash
EVEROS_MULTIMODAL__MODEL=google/gemini-3-flash-preview
EVEROS_MULTIMODAL__API_KEY=<your key>
EVEROS_MULTIMODAL__BASE_URL=https://openrouter.ai/api/v1
```

The default targets Gemini via OpenRouter so a single key covers both
chat extraction and multimodal parsing. See
[Configuration reference](#configuration-reference) for the full list.

## Supported modalities

| `type`  | Typical formats | Payload | Notes |
|---|---|---|---|
| `text`  | — | `text` | Plain text; the string shorthand also maps here |
| `image` | PNG / JPG / GIF / WebP / SVG | `uri` or `base64` | SVG via the bundled `cairosvg` |
| `pdf`   | PDF | `uri` or `base64` | — |
| `audio` | MP3 / WAV / … | `uri` or `base64` | Endpoint must accept audio parts |
| `doc`   | DOC / DOCX / PPT / PPTX / XLS / XLSX | `uri` or `base64` | **Requires LibreOffice** (converted to PDF first) |
| `html`  | HTML | `uri` or `base64` | To inline HTML as plain text instead, send it as `type: "text"` |
| `email` | EML / MSG | `uri` or `base64` | — |

A **non-text** item must carry a fetchable/decodable payload (`uri` or
`base64`). A non-text item that only carries `text` returns `415` — the
parser has nothing to parse.

## Sending multimodal content

Multimodal input is a `content` **array** of `ContentItem` objects on a
[MessageItem](api.md#messageitem). A bare string `content` is shorthand
for a single text item; switch to the array form when you mix text with
non-text assets. Field-level rules are in
[api.md → ContentItem](api.md#contentitem); the essentials:

| Field | Purpose |
|---|---|
| `type` | One of the modalities above |
| `text` | The literal text — **only** for `type: "text"` |
| `uri`  | `http(s)://` (fetched server-side) or `file://` (read from the server fs) |
| `base64` | Inline payload, plain base64 (no `data:` prefix) |
| `ext`  | Extension hint (`"pdf"`, `"png"`, …); effectively required for `base64` |
| `name` | Display filename for logs |

Carry the payload in exactly **one** of `text` / `uri` / `base64`.

### Payload: `uri` vs `base64`

| | `uri` (`http(s)://`) | `base64` |
|---|---|---|
| Where the bytes live | Fetched transiently at parse time | Held verbatim in the SQLite session buffer until flush |
| Wire size | URL only | ~4/3× the raw size (base64 inflation) |
| Best for | Large assets, S3/OSS presigned URLs | Small assets, or when no reachable URL exists |

**Prefer `uri` for anything large.** A multi-MB base64 blob becomes
multi-MB of SQLite buffer text for the buffer's lifetime and slows
request parsing. The bytes are never persisted past extraction either
way — only the parsed text is.

### Example: image by URL

```bash
TS=$(($(date +%s) * 1000))     # v1 contract: timestamp in ms
curl -X POST http://127.0.0.1:8000/api/v1/memory/add \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\": \"mm-001\",
    \"messages\": [
      {
        \"sender_id\": \"alice\",
        \"role\": \"user\",
        \"timestamp\": $TS,
        \"content\": [
          { \"type\": \"image\", \"uri\": \"https://example.com/whiteboard.png\" }
        ]
      }
    ]
  }"
```

### Example: mixed text + image in one turn

```json
{
  "session_id": "mm-001",
  "messages": [
    {
      "sender_id": "alice",
      "role": "user",
      "timestamp": 1748390400000,
      "content": [
        { "type": "text",  "text": "Here's the whiteboard from today's planning session." },
        { "type": "image", "uri": "https://example.com/whiteboard.png", "name": "whiteboard.png" }
      ]
    }
  ]
}
```

### Example: inline PDF via base64

```json
{
  "session_id": "mm-001",
  "messages": [
    {
      "sender_id": "alice",
      "role": "user",
      "timestamp": 1748390400000,
      "content": [
        { "type": "text", "text": "Quarterly report attached." },
        { "type": "pdf",  "base64": "JVBERi0xLjQK...", "ext": "pdf", "name": "q3.pdf" }
      ]
    }
  ]
}
```

`ext` is effectively **required** for `base64` payloads — it drives
modality dispatch. Without it the server falls back to MIME inference and
otherwise `415`s.

### Example: local file via `file://`

A `file://` URI is read from the **server's** local filesystem (the path
must be reachable by the server process), guardrailed by size and an
optional allowlist:

```json
{ "type": "pdf", "uri": "file:///srv/uploads/q3.pdf" }
```

Guardrails (a violation surfaces as `415`):

- the resolved path (symlinks followed) must be an existing regular file;
- size ≤ `EVEROS_MULTIMODAL__FILE_URI_MAX_BYTES` (default 50 MiB);
- if `EVEROS_MULTIMODAL__FILE_URI_ALLOW_DIRS` is set, the path must lie
  within one of the listed roots (unset = any readable file, the
  local-first default — confine this when exposing the API beyond
  loopback).

### Calling from Python (plain HTTP)

There is no EverOS Python client; call the HTTP API directly with any
HTTP library:

```python
import httpx

httpx.post(
    "http://127.0.0.1:8000/api/v1/memory/add",
    json={
        "session_id": "mm-001",
        "messages": [
            {
                "sender_id": "alice",
                "role": "user",
                "timestamp": 1748390400000,
                "content": [
                    {"type": "text", "text": "Here's the whiteboard from today's meeting."},
                    {"type": "image", "uri": "https://example.com/whiteboard.png"},
                ],
            }
        ],
    },
)
```

## Configuration reference

All fields bind from the environment via the parent `Settings`
(`EVEROS_MULTIMODAL__<FIELD>`) or the `[multimodal]` TOML section.

| Env var | Default | Meaning |
|---|---|---|
| `EVEROS_MULTIMODAL__MODEL` | `google/gemini-3-flash-preview` | Parsing model; must accept `image_url` parts |
| `EVEROS_MULTIMODAL__API_KEY` | — | API key for the multimodal endpoint |
| `EVEROS_MULTIMODAL__BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible base URL |
| `EVEROS_MULTIMODAL__MAX_CONCURRENCY` | `4` | Cap on parallel multimodal calls within one extraction |
| `EVEROS_MULTIMODAL__FILE_URI_MAX_BYTES` | `52428800` (50 MiB) | Max size of a `file://` asset |
| `EVEROS_MULTIMODAL__FILE_URI_ALLOW_DIRS` | `[]` (any) | JSON list of allowlisted base dirs for `file://` URIs |

## Errors and limits

Three failure classes behave differently:

**Format errors** — the uploaded file format is invalid or not
recognized. These abort the batch with `415` (`UNSUPPORTED_FORMAT`):

| Condition | HTTP | `error.code` |
|---|---|---|
| Non-text item carries only `text` (no `uri` / `base64`) | `415` | `UNSUPPORTED_FORMAT` |
| Extension / modality the parser has no handler for | `415` | `UNSUPPORTED_FORMAT` |
| `base64` without a resolvable `ext` / MIME to dispatch on | `415` | `UNSUPPORTED_FORMAT` |
| `file://` fails a guardrail (missing / non-regular / too large / outside allowlist) | `415` | `UNSUPPORTED_FORMAT` |

**Capability errors** — the server is missing a required dependency.
These abort the batch with `503` (`CAPABILITY_UNAVAILABLE`). Unlike
transient errors, retrying will not help — admin action is required:

| Condition | HTTP | `error.code` |
|---|---|---|
| `everos[multimodal]` extra not installed | `503` | `CAPABILITY_UNAVAILABLE` |
| Office document but no LibreOffice (`soffice`) on host | `503` | `CAPABILITY_UNAVAILABLE` |

**Transient LLM errors** — the multimodal LLM call failed. These
degrade gracefully — the request still returns `200`, the affected
item is marked `parse_status="failed"` and contributes no text, and the
rest of the batch extracts normally:

| Condition | HTTP | Result |
|---|---|---|
| Multimodal LLM call fails (timeout / rate-limit / model rejects) | `200` | That item is skipped; the rest of the batch still extracts |

All error responses use the standard error envelope — see
[api.md → Errors](api.md#errors).

## Searching multimodal memory

Nothing special is required. Because parsed text is folded into the same
episodes and memory cells as text turns, every retrieval method works
across multimodal-derived memory unchanged:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/memory/search \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "alice",
    "query": "whiteboard from the planning session",
    "method": "hybrid"
  }'
```

`keyword`, `vector`, `hybrid` (default), and `agentic` all apply — see
[api.md → SearchMethod](api.md#searchmethod).
