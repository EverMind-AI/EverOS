# Connect MCP clients

EverOS exposes its memory to MCP (Model Context Protocol) clients — Claude Code, Codex,
Claude Desktop, Cursor and any other MCP host — through the official **everos-mcp**
server: [PyPI `everos-mcp`](https://pypi.org/project/everos-mcp/) ·
[source](https://github.com/EverMind-AI/EverOS-MCP).

`everos-mcp` is a separate process that runs next to your EverOS server. It speaks MCP to
the host over stdio and calls this server's `/api/v2/memory/*` endpoints. It does not start
or stop EverOS for you.

- [Prerequisites](#prerequisites)
- [Add the server to your client](#add-the-server-to-your-client)
- [What the assistant gets](#what-the-assistant-gets)
- [Behaviour on a self-hosted server](#behaviour-on-a-self-hosted-server)
- [Remote hosts](#remote-hosts)

## Prerequisites

1. `everos init` done and the LLM / embedding API keys filled in (`everos.toml`).
2. `everos server start` running — default `http://127.0.0.1:8000`. Start it before the
   host launches the MCP server, and keep it running.
3. [`uv`](https://docs.astral.sh/uv/getting-started/installation/) installed; `uvx` runs
   `everos-mcp` without a manual install.

No API key is needed when `EVEROS_BASE_URL` points at your own server.

## Add the server to your client

```bash
# Claude Code
claude mcp add everos -e EVEROS_BASE_URL=http://127.0.0.1:8000 -- uvx everos-mcp

# Codex
codex mcp add everos --env EVEROS_BASE_URL=http://127.0.0.1:8000 -- uvx everos-mcp
```

Claude Desktop, Cursor and other JSON-configured clients:

```json
{
  "mcpServers": {
    "everos": {
      "command": "uvx",
      "args": ["everos-mcp"],
      "env": { "EVEROS_BASE_URL": "http://127.0.0.1:8000" }
    }
  }
}
```

Optional environment variables: `EVEROS_USER_ID` (memory owner, default `default-user`),
`EVEROS_APP_ID` / `EVEROS_PROJECT_ID` (scope, default `default`). The full list is in the
[everos-mcp README](https://github.com/EverMind-AI/EverOS-MCP#configuration).

## What the assistant gets

Seven tools, backed by this server's `/api/v2/memory/*` endpoints: `get_profile`,
`add_memory`, `search_memory`, `list_memories`, `record_trajectory`,
`recall_agent_experience` and `forget_session`. The
server also ships an autonomy protocol through MCP `instructions`, so capable hosts load the
profile at session start, save new facts and recall context without being asked. The current tool list and the protocol are
documented in the [everos-mcp README](https://github.com/EverMind-AI/EverOS-MCP#tools);
this page only covers what is specific to a self-hosted EverOS.

## Behaviour on a self-hosted server

- **`forget_session` forgets the session's memories.** It calls
  [`POST /api/v2/memory/delete`](api.md#post-apiv2memorydelete) with the session id,
  which EverOS 2.0 answers with a forget (it has no hard delete): every user's memories
  of that conversation are marked forgotten and leave every default read, reversibly
  (the answer's `forget_ids` undo it through `POST /api/v2/memory/forget/undo`). While
  the session is still being memorized the server answers `SESSION_BUSY` ("nothing was
  forgotten, retry later"), which the tool passes on to the model.
- **New memories become searchable shortly after saving.** `add_memory` returns once
  extraction has written the Markdown; the search index catches up a moment later.
  `list_memories` shows new entries sooner than `search_memory`.
- **The two agent-experience tools have no agent memory behind them.** EverOS 2.0
  keeps chat memory only. `record_trajectory` sends the trajectory to `/memory/add`,
  which drops its tool rows (`role: "tool"`, and the assistant messages that carry
  `tool_calls`) before segmentation: the plain user and assistant messages (the task
  request, the final answer) are memorized as the user's chat, and no case or skill is
  distilled. `recall_agent_experience` searches with `agent_id`, which the server refuses
  (`422 INVALID_INPUT`: "EverOS 2.0 has no agent memory"), so the tool fails.

## Remote hosts

To serve MCP over HTTP instead of stdio (hosts on another machine, or OAuth-only hosts),
run `everos-mcp --transport http` on the machine that can reach your EverOS server and put
it behind your gateway. Configuration is described in the
[everos-mcp README](https://github.com/EverMind-AI/EverOS-MCP#configuration).
