# EverOS OpenClaw Plugin

Persistent memory for **OpenClaw** through normal conversation.

This plugin keeps the current OpenClaw `context-engine` architecture and connects it to a self-hosted EverOS backend powered by [EverMemOS](https://github.com/EverMind-AI/EverMemOS).

## What it does

- Recalls relevant memory before each reply through `assemble()`
- Saves new conversation content after each turn through `afterTurn()`
- Works through normal natural-language chat
- Does not require manual `memory_store` or `memory_search` tool calls

Important:

- This is a `context-engine` plugin
- This is not a `memory` slot plugin
- To avoid conflicts, installation sets `plugins.slots.memory = "none"`

## Quick start

Recommended install:

```bash
npx --yes --package @evermind-ai/openclaw-plugin everos-install
```

The installer:

- reuses `~/.openclaw/openclaw.json` if it already exists
- adds the plugin path to `plugins.load.paths`
- adds `evermind-ai-everos` to `plugins.allow`
- sets `plugins.slots.contextEngine = "evermind-ai-everos"`
- sets `plugins.slots.memory = "none"`
- creates or updates the plugin entry with sane defaults

After install:

```bash
openclaw gateway restart
```

Then verify with natural language:

```text
Remember: I like espresso.
What coffee do I like?
```

## Backend

Default backend URL:

```text
http://localhost:1995
```

Health check:

```bash
curl http://localhost:1995/health
```

If you have not started the EverOS backend yet:

```bash
git clone https://github.com/EverMind-AI/EverMemOS.git
cd EverMemOS
docker compose up -d
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
cp env.template .env
# edit .env
uv run python src/run.py
```

## How natural-language memory works

Runtime flow:

1. The user sends a normal message.
2. `assemble()` searches the EverOS backend for relevant memory.
3. Matching memory is injected as context.
4. OpenClaw replies normally.
5. `afterTurn()` saves the new turn back to the EverOS backend.

This means the user experience is:

- "Remember: I prefer dark mode"
- later: "What UI style do I prefer?"

without any explicit memory tool usage.

## OpenClaw config

Expected config shape:

```json
{
  "plugins": {
    "allow": ["evermind-ai-everos"],
    "slots": {
      "memory": "none",
      "contextEngine": "evermind-ai-everos"
    },
    "entries": {
      "evermind-ai-everos": {
        "enabled": true,
        "config": {
          "baseUrl": "http://localhost:1995",
          "userId": "everos-user",
          "groupId": "everos-group",
          "topK": 5,
          "memoryTypes": ["episodic_memory", "profile", "agent_skill", "agent_case"],
          "retrieveMethod": "hybrid"
        }
      }
    }
  }
}
```

## Configuration

| Field | Default | Description |
| --- | --- | --- |
| `baseUrl` | `http://localhost:1995` | EverOS backend URL |
| `userId` | `everos-user` | Memory owner identity |
| `groupId` | `everos-group` | Shared memory namespace |
| `topK` | `5` | Max retrieved entries |
| `memoryTypes` | `["episodic_memory", "profile", "agent_skill", "agent_case"]` | Memory types to search |
| `retrieveMethod` | `hybrid` | Retrieval mode |

## Manual install

```bash
npm install -g @evermind-ai/openclaw-plugin
everos-install
```

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Plugin not loading | Check `plugins.allow`, `plugins.load.paths`, and `plugins.slots.contextEngine` |
| Backend connection failed | Verify `baseUrl` and run `curl <baseUrl>/health` |
| Memory not recalled | Check backend data and try a more specific query |
| Memory not saved | Verify the EverOS backend write API is healthy |
| Conflict with another memory plugin | Ensure `plugins.slots.memory = "none"` |

## Related files

- `index.js`: plugin entry point (register)
- `src/engine.js`: ContextEngine lifecycle hooks
- `src/convert.js`: OpenClaw message to EverMemOS format conversion
- `src/api.js`: EverOS backend REST client
- `src/messages.js`: message normalization and turn collection
- `src/prompt.js`: memory search response parsing and prompt building
- `src/subagent-assembler.js`: subagent context assembly
- `src/subagent-tracker.js`: subagent lifecycle tracking
- `bin/install.js`: installer and config bootstrap
- `openclaw.plugin.json`: plugin metadata and config schema

## License

Apache-2.0
