# EverOS Memory for DeepSeek Harness

Automatic cross-session memory for DeepSeek Harness (DSH), backed by a local
[EverOS](https://github.com/EverMind-AI/EverOS) service.

The plugin follows a three-stage lifecycle:

1. **Recall** — once at the start of each user turn, search the EverOS user and agent
   tracks in parallel and append a source-attributed recall message. When the user has
   switched sessions, pending memory from the previous session is committed first.
2. **Capture** — at every turn stopping boundary, durably append newly committed user,
   assistant, tool-call, and tool-result events to EverOS's SQLite buffer without an LLM
   extraction call.
3. **Flush** — batch extraction after an idle window, a token/message threshold, a session
   switch, a maximum delay, or shutdown. Session disposal remains the final safety net.

EverOS failures are fail-open: they are logged, but never reject the user's DSH step or
turn.

## Requirements

- Node.js `^22.19.0 || >=24.0.0`
- DeepSeek Harness `0.1.0-rc.6` or newer within the `0.1.x` line
- EverOS installed and initialized

```bash
uv tool install everos
everos init
```

EverOS 1.2.3 supports the plugin's LLM-only Tier 1 path with keyword recall. Deferred
capture batching additionally requires an EverOS build that supports
`defer_extraction`; until that capability is included in a tagged release, install
EverOS from the same checkout as this example. With 1.2.3, capture remains functional,
but `/add` uses the eager boundary and extraction path.

Embedding, rerank, and multimodal credentials are optional. EverOS owns provider and
storage configuration; this plugin does not accept or store API keys.

For a local source checkout, install the current tree and provide the LLM key through the
environment before starting EverOS:

```bash
uv tool install --editable ../..
everos init
export EVEROS_LLM__API_KEY='<your OpenRouter key>'
everos server start
```

The generated `~/.everos/everos.toml` already contains the default OpenRouter model and
base URL. Keep secrets out of the repository and use your normal secret manager for
persistent setup.

## Install

For local development from the EverOS repository:

```bash
cd examples/dsh
npm ci
npm run ci
dsh plugin --profile web add .
```

After the package is published:

```bash
dsh plugin --profile web add @evermind-ai/dsh-plugin
```

The package declares `dsh.bundle.patch`, so a repository URL ending in
`/tree/main/examples/dsh` is also suitable for DSH plugin discovery. A source install
runs the package's `prepare` build, so pnpm may require the user to approve that build in
the profile's `allowBuilds` list. The npm package ships prebuilt output and needs no such
approval.

## Configuration

The bundled patch reads the most common values from environment variables:

```bash
export EVEROS_DSH_BASE_URL=http://127.0.0.1:8000
export EVEROS_DSH_USER_ID=alice
export EVEROS_DSH_AGENT_ID=dsh
export EVEROS_DSH_RECALL_METHOD=keyword
export EVEROS_DSH_FLUSH_IDLE_MS=30000
export EVEROS_DSH_FLUSH_TOKEN_THRESHOLD=12000
export EVEROS_DSH_FLUSH_MESSAGE_THRESHOLD=50
export EVEROS_DSH_FLUSH_MAX_DELAY_MS=300000
export EVEROS_DSH_START_COMMAND='everos server start'
export EVEROS_DSH_DIR=/path/to/EverOS
```

Every option can also be set in the plugin row of the DSH Cordis profile:

```yaml
- id: everos-memory
  name: '@evermind-ai/dsh-plugin'
  config:
    baseUrl: http://127.0.0.1:8000
    apiVersion: auto
    appId: dsh
    userId: alice
    agentId: dsh
    recallMethod: keyword
    queryN: 3
    queryMaxChars: 2000
    recallTopK: 5
    recallMaxChars: 12000
    recallTimeoutMs: 5000
    captureTimeoutMs: 15000
    captureMaxChars: 50000
    flushIdleMs: 30000
    flushTokenThreshold: 12000
    flushMessageThreshold: 50
    flushMaxDelayMs: 300000
    flushOnSessionSwitch: true
    autoStart: true
    startCommand: everos server start
```

| Option | Default | Meaning |
| --- | --- | --- |
| `baseUrl` | `http://127.0.0.1:8000` | EverOS server root |
| `apiVersion` | `auto` | Try `/api/v2`, then fall back to `/api/v1` on 404 |
| `appId` | `dsh` | EverOS application partition |
| `projectId` | workspace-derived | Optional fixed project partition |
| `userId` | operating-system account | User-memory owner |
| `agentId` | DSH agent preset | Agent-memory owner |
| `recallMethod` | `keyword` | EverOS retrieval method; `keyword` supports LLM-only Tier 1 |
| `queryN` | `3` | Direct user messages blended into a recall query |
| `queryMaxChars` | `2000` | Recall-query character budget |
| `recallTopK` | `5` | Result limit per owner track |
| `recallMaxChars` | `12000` | Maximum injected memory block |
| `recallTimeoutMs` | `5000` | Timeout for each search |
| `captureTimeoutMs` | `15000` | Timeout for add and flush requests |
| `captureMaxChars` | `50000` | Per-message capture limit |
| `flushIdleMs` | `30000` | Debounced flush after no newly captured turn activity |
| `flushTokenThreshold` | `12000` | Approximate buffered-token threshold for an immediate flush |
| `flushMessageThreshold` | `50` | Buffered-message threshold for an immediate flush |
| `flushMaxDelayMs` | `300000` | Maximum age of a non-empty buffer before flushing |
| `flushOnSessionSwitch` | `true` | Commit other pending sessions in the workspace before recall |
| `autoStart` | `true` | Start EverOS when a loopback endpoint is unavailable |
| `startCommand` | `everos server start` | Shell-free auto-start command |
| `everosDir` | process directory | Working directory for auto-start |

## Scope and identity mapping

- `app_id` defaults to `dsh`.
- `project_id` is the workspace directory name plus a stable hash of its absolute path.
  Two repositories with the same directory name therefore remain separate.
- `session_id` is the DSH session id, normalized to the EverOS 128-character contract.
- `user_id` comes from explicit config, `USER`/`USERNAME`, or the OS account.
- `agent_id` comes from explicit config or the DSH agent preset. Sessions using the same
  preset can learn shared agent cases and skills.

All derived identifiers are deterministic and path-safe.

## Context and capture policy

Recall is injected with DSH provenance:

```text
{ kind: "plugin", plugin: "everos-memory", form: "recall" }
```

Recalled content is fenced as untrusted historical evidence. Stored fence tokens are
neutralized before injection, which prevents a recalled item from escaping the memory
block. Plugin-generated context is never captured as direct user input, so recalled
memory does not recursively save itself.

Capture includes:

- direct user messages;
- visible assistant text;
- assistant tool calls with raw JSON arguments;
- tool results linked by call id;
- safe image metadata such as media type and dimensions.

Raw model reasoning and image bytes are intentionally excluded. DSH attachment ids are
opaque and are not treated as file paths or bearer URLs.

## Operational behavior

- Writes are serialized per DSH session.
- Capture uses `defer_extraction: true`: raw turns are durable immediately, while the
  expensive boundary and memory LLM work is batched. An EverOS build with
  `defer_extraction` support is required for this optimization; EverOS 1.2.3 ignores the
  request field and retains its eager `/add` behavior.
- Starting a new DSH session establishes a read-after-write barrier for other pending
  sessions in the same workspace, so the first recall sees the latest committed memory.
- A cursor based on DSH event sequence numbers captures only new live events, including
  additional steps in the same turn.
- Add requests are split at EverOS's 500-message API limit.
- A resumed DSH session starts capture at `Session.firstLiveSeq`; historical seed events
  are not re-ingested.
- Auto-start never invokes a shell and is restricted to loopback URLs. A process started
  by the plugin is stopped during plugin disposal; an existing EverOS process is not.
- If another EverOS process owns the OME lock, the plugin keeps polling and connects to
  that process when it becomes healthy.
- Keyword recall is the safe default for EverOS Tier 1. Users who configure embedding
  and rerank capabilities may explicitly select `vector`, `hybrid`, or `agentic`.

## Privacy and trust boundary

EverOS is local by default, but the captured trajectory may include source snippets,
commands, and tool output. Review the EverOS storage and model-provider configuration
before using the plugin with sensitive repositories. Keep credentials in approved secret
stores and avoid printing them into agent-visible tool output.

The plugin deliberately exposes no model-callable memory write tool. Memory is derived
from the durable DSH trajectory, while EverOS remains the single storage and extraction
authority.

## Development

```bash
npm ci
npm run ci
```

The tests use mocked HTTP and DSH-shaped event logs; they require no provider credentials
and do not start an EverOS process.

## Current limitations

- DSH is still a release candidate, so its plugin APIs may change before a stable release.
- The adapter records image metadata, not attachment bytes.
- The EverOS add API has no idempotency key. An ambiguous network failure followed by a
  later retry may produce at-least-once capture semantics.
- Recall currently has no management UI or explicit remember/forget tools.
