# EverMemOS OpenClaw Plugin

An OpenClaw memory plugin that gives AI Agents long-term memory powered by [EverMemOS](https://github.com/EverMind-AI/EverMemOS).

## Features

- **Memory retrieval**: Before each Agent run, automatically searches for relevant memories (episodes, user profiles, cases, skills) and injects them as context
- **Memory persistence**: After each Agent run, automatically saves the conversation (including tool calls) to EverMemOS for future retrieval
- **Flexible retrieval strategies**: Supports keyword, vector, hybrid, RRF, and agentic retrieval
- **Tool-call aware**: Captures the full tool call lifecycle and persists it alongside the conversation

## Prerequisites

- A running [EverMemOS](https://github.com/EverMind-AI/EverMemOS) instance that supports `agent_case` and `agent_skill`
- An Agent runtime that supports the OpenClaw plugin interface

## Installation

Add the plugin directory to `plugins.load.paths` and enable it under `plugins.entries` in `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "slots": {
      "memory": "evermemos-openclaw-plugin"
    },
    "load": {
      "paths": ["~/.openclaw/extensions/evermemos-openclaw-plugin"]
    },
    "entries": {
      "evermemos-openclaw-plugin": {
        "enabled": true,
        "config": {
          "baseUrl": "http://localhost:1995",
          "userId": "evermemos-user",
          "groupId": "evermemos-group",
          "topK": 5,
          "memoryTypes": ["episodic_memory", "profile", "agent_skill", "agent_case"],
          "retrieveMethod": "hybrid"
        }
      }
    }
  }
}
```

> **Note**: `plugins.slots.memory` is required to replace the built-in `memory-core` plugin. Without it, OpenClaw will keep `memory-core` as the active memory slot and disable this plugin.

Then restart the OpenClaw gateway for the changes to take effect.

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `baseUrl` | string | `http://localhost:1995` | EverMemOS server base URL |
| `userId` | string | `evermemos-user` | Identity used for memory ownership and as message sender |
| `groupId` | string | `evermemos-group` | Group id |
| `topK` | integer | `5` | Maximum number of memory entries to retrieve |
| `memoryTypes` | string[] | `["episodic_memory", "profile", "agent_skill", "agent_case"]` | EverMemOS memory types to search (see below) |
| `retrieveMethod` | string | `hybrid` | Retrieval strategy used by EverMemOS (see below) |

### Memory types

| Value | Description |
|-------|-------------|
| `episodic_memory` | Past conversation episodes |
| `profile` | User profile and preferences |
| `agent_case` | Similar historical cases |
| `agent_skill` | Agent skill knowledge |

### Retrieval strategies

| Value | Description |
|-------|-------------|
| `keyword` | Full-text keyword search |
| `vector` | Semantic vector search |
| `hybrid` | Keyword + vector fusion |
| `rrf` | Reciprocal Rank Fusion |
| `agentic` | Agent-driven adaptive retrieval |

### Example

```json
{
  "plugin": "evermemos-openclaw-plugin",
  "config": {
    "baseUrl": "http://localhost:1995",
    "userId": "evermemos-user",
    "groupId": "evermemos-group",
    "topK": 5,
    "memoryTypes": ["episodic_memory", "profile", "agent_skill", "agent_case"],
    "retrieveMethod": "hybrid"
  }
}
```

## Project structure

```
├── index.js                  # Plugin entry — lifecycle hooks
├── package.json
├── openclaw.plugin.json      # Plugin metadata and config schema
└── src/
    ├── config.js             # Config parsing and validation
    ├── memory-api.js         # EverMemOS REST API calls
    ├── formatter.js          # Memory parsing and context formatting
    ├── message-utils.js      # Message collection and format conversion
    └── http-client.js        # HTTP client with timeout and retry
```

## Acknowledgements

This plugin was inspired by and references the implementation of [MemOS-Cloud-OpenClaw-Plugin](https://github.com/MemTensor/MemOS-Cloud-OpenClaw-Plugin).