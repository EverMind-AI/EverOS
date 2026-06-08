# Chat Agent Integration Guide

Recommended patterns for integrating a Chat Agent (LLM-based assistant)
with EverOS persistent memory. Covers the write path (ingesting
conversations), the read path (recalling memories on demand), and an
official tool schema for function-calling agents.

## Architecture: On-Demand Search vs Per-Turn RAG

| Pattern | How it works | Trade-off |
|---|---|---|
| **Per-turn RAG** | Every turn, retrieve memories and inject into the LLM context window before generating a response. | Simple but pollutes context with irrelevant memories; burns tokens on every turn. |
| **On-demand search** | The agent decides *when* to recall by calling a memory search tool. | Token-efficient; closer to how human memory works (you don't recall everything every sentence). |

**Recommendation: on-demand search.** Keep short-term context (the last
*n* turns) in the LLM `messages` array as working memory. Long-term
memory is retrieved only when the agent determines it needs historical
context.

## Write Path

Ingest every conversation turn automatically. Do not wait for the agent
to decide what to remember.

```
POST /api/v1/memory/add
{
  "session_id": "chat-abc123",
  "messages": [
    {
      "sender_id": "user_42",
      "role": "user",
      "content": "I prefer dark mode for all my apps",
      "timestamp": 1740564000000
    }
  ]
}
```

When you need to trigger memory extraction immediately (e.g. end of
conversation), call flush:

```
POST /api/v1/memory/flush
{
  "session_id": "chat-abc123"
}
```

Extraction also fires automatically when the buffer reaches a size
threshold. Calling flush is optional but useful when you want memories
available for search right away.

See [POST /api/v1/memory/add](api.md#post-apiv1memoryadd) for the
full request schema.

## Read Path

When the agent needs to recall past context, have it call the search
tool:

```
POST /api/v1/memory/search
{
  "user_id": "user_42",
  "query": "dark mode preferences",
  "filters": {
    "timestamp": {
      "gte": 1740480000000,
      "lt": 1740566400000
    }
  }
}
```

See [POST /api/v1/memory/search](api.md#post-apiv1memorysearch) for
the full request schema.

### Time-Range Filtering

For natural-language time references ("what we discussed yesterday about
X"), resolve the spoken time window to concrete `timestamp` bounds in
the `filters` field:

- Use Unix epoch milliseconds **or** ISO-8601 strings.
- `gte` / `lt` operators bracket the window.
- Timestamps reflect **when the conversation happened**, not when the
  memory was extracted. If your extraction pipeline is async (flush-
  based), propagate the original conversation timestamp.

```json
{
  "filters": {
    "AND": [
      {"timestamp": {"gte": 1740480000000, "lt": 1740566400000}},
      {"session_id": {"eq": "chat-abc123"}}
    ]
  }
}
```

### Retrieval Methods

| Method | When to use |
|---|---|
| `hybrid` (default) | General-purpose — combines BM25 + vector search. Best starting point. |
| `keyword` | When the query is exact-match friendly (e.g. function names, error codes). |
| `vector` | When semantic similarity matters more than keyword overlap. |
| `agentic` | When you want the system to run multi-step retrieval with LLM sufficiency checks. |

## Official Tool Schema

The following OpenAI-compatible tool definition exposes memory search as
a function the agent can call. Fields align with the `/search` endpoint
documented in [api.md](api.md#post-apiv1memorysearch).

```json
{
  "type": "function",
  "function": {
    "name": "memory_search",
    "description": "Search the user's long-term memory for relevant past conversations, facts, and context. Use when the user references previous sessions, asks about past decisions, or when historical context would improve your response.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "Search query — what to look for in past memories. Be specific."
        },
        "top_k": {
          "type": "integer",
          "default": 10,
          "description": "Maximum number of results to return. -1 for server default."
        },
        "filters": {
          "type": "object",
          "description": "Optional filters for time range, session, or other dimensions.",
          "properties": {
            "timestamp": {
              "type": "object",
              "description": "Time range filter. Use gte/lt with Unix epoch ms or ISO-8601 strings.",
              "properties": {
                "gte": {"type": ["integer", "string"], "description": "Start of time range (inclusive)"},
                "lt": {"type": ["integer", "string"], "description": "End of time range (exclusive)"}
              }
            },
            "session_id": {
              "type": "object",
              "properties": {
                "eq": {"type": "string", "description": "Filter to a specific session"}
              }
            }
          }
        }
      },
      "required": ["query"]
    }
  }
}
```

### MCP Tool Reference

For Claude Code and other MCP-compatible agents, a reference
implementation is available at
[`use-cases/claude-code-plugin/skills/memory-tools.md`](../use-cases/claude-code-plugin/skills/memory-tools.md).
That document describes the `evermem_search` tool and when to use it.

## Key Integration Points

1. **Write automatic, read agent-initiated.** Every turn goes through
   `/add`; the agent calls `/search` only when it needs context.

2. **Session scoping.** Use `session_id` to group turns from one
   conversation. The `/search` endpoint can filter by session.

3. **Owner scoping.** Pass `user_id` for user-facing agents or
   `agent_id` for autonomous agents. Results never cross owner
   boundaries.

4. **App / project scoping.** Use `app_id` and `project_id` to
   isolate memories across different products or environments.
