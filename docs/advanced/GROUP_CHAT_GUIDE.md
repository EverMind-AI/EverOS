# Group Chat Conversations Guide

[Home](../../README.md) > [Docs](../README.md) > [Advanced](.) > Group Chat Guide

## Overview

EverMemOS supports organizing conversations into **groups** using `group_id` and `group_name`. This allows you to:

- **Separate messages** into logical groups for better organization
- **Filter and retrieve memories** by group for targeted searches
- **Generate better summaries** within each group's context
- **Isolate memory contexts** between different groups

This guide covers how to leverage group-based memory management for various use cases.

---

## Core Concepts

### Group ID and Group Name

| Field | Description | Example |
|-------|-------------|---------|
| `group_id` | Unique identifier for the group | `"team_engineering"`, `"project_alpha"` |
| `group_name` | Human-readable display name | `"Engineering Team"`, `"Project Alpha"` |

**Key Benefits:**

1. **Memory Isolation** - Memories from different groups are separated, preventing cross-contamination
2. **Targeted Retrieval** - Query memories from a specific group without irrelevant results
3. **Contextual Summaries** - Generate summaries that understand the group's context and participants
4. **Scalable Organization** - Manage thousands of conversations across multiple groups

---

## When to Use Group ID

Starting from v1.2.0, `group_id` and `group_name` are **truly optional**. When omitted, the API automatically creates a default group based on the `sender` field.

### Scenarios Where Group ID is NOT Needed

| Use Case | Description |
|----------|-------------|
| **Knowledge Base** | Ingesting documents or content from a single source where cross-message correlation isn't needed |
| **Persona Building** | Building user profiles/personas from individual interactions |
| **Simple Q&A** | Single-user chatbot interactions without complex context |

**Example without group_id:**
```json
{
  "message_id": "msg_001",
  "create_time": "2025-01-15T10:00:00+00:00",
  "sender": "user_001",
  "content": "I prefer dark roast coffee in the morning"
}
```

### Scenarios Where Group ID is Recommended

| Use Case | Description |
|----------|-------------|
| **Multi-User Conversations** | Group chats where multiple participants interact and context between messages matters |
| **User + AI Assistant** | Conversations between a user and AI where you want correlated episodic memories |
| **Project/Topic Organization** | When you need to query and organize memories by logical groupings |
| **Meeting Transcripts** | Multi-participant discussions where speaker context is important |

**Why it matters:** Episodic memories are extracted from multiple related messages. When messages share a `group_id`, the system can build richer context by understanding the relationships between messages from different senders. Without a `group_id`, each sender's messages are processed independently.

**Example with group_id:**
```json
{
  "message_id": "msg_001",
  "create_time": "2025-01-15T10:00:00+00:00",
  "sender": "user_001",
  "content": "What should we use for the database?",
  "group_id": "project_alpha",
  "group_name": "Project Alpha Discussion"
}
```

---

## Use Cases

### 1. Team/Department Conversations

Organize conversations by team or department within an organization.

```json
{
  "group_id": "dept_engineering",
  "group_name": "Engineering Department"
}
```

**Benefits:**
- Keep engineering discussions separate from marketing, sales, etc.
- Generate team-specific insights and summaries
- Track team decisions and action items

### 2. Project-Based Conversations

Group all conversations related to a specific project.

```json
{
  "group_id": "project_mobile_app_v2",
  "group_name": "Mobile App v2.0 Development"
}
```

**Benefits:**
- All project discussions, decisions, and context in one place
- Query project-specific knowledge: "What was decided about the login flow?"
- Generate project progress summaries

### 3. Channel-Based Conversations (Slack/Discord Style)

Mirror your communication platform's channel structure.

```json
{
  "group_id": "channel_general",
  "group_name": "#general"
}
```

```json
{
  "group_id": "channel_random",
  "group_name": "#random"
}
```

**Benefits:**
- Maintain channel context when building AI assistants
- Search within specific channels
- Channel-specific summaries and insights

### 4. Customer Support Conversations

Group support tickets or customer interactions.

```json
{
  "group_id": "support_ticket_12345",
  "group_name": "Ticket #12345 - Login Issue"
}
```

**Benefits:**
- Track full context of a support case
- Generate case summaries for handoffs
- Query similar past issues

### 5. Meeting Transcripts

Organize meeting notes and transcripts.

```json
{
  "group_id": "meeting_weekly_standup_2025_02",
  "group_name": "Weekly Standup - February 2025"
}
```

**Benefits:**
- Query across all standups: "What blockers were mentioned this month?"
- Generate meeting summaries automatically
- Track action items across meetings

---

## Data Format

### Complete Group Chat Structure

```json
{
  "version": "1.0.0",
  "conversation_meta": {
    "group_id": "team_001",
    "name": "Engineering Team",
    "scene": "group_chat",
    "scene_desc": {},
    "user_details": {
      "alice": {
        "full_name": "Alice Smith",
        "role": "user",
        "custom_role": "Tech Lead",
        "extra": {"department": "Engineering"}
      },
      "bob": {
        "full_name": "Bob Jones",
        "role": "user",
        "custom_role": "Senior Engineer"
      }
    },
    "default_timezone": "+00:00"
  },
  "conversation_list": [
    {
      "message_id": "msg_001",
      "create_time": "2025-02-01T10:00:00+00:00",
      "sender": "alice",
      "sender_name": "Alice Smith",
      "role": "user",
      "type": "text",
      "content": "Let's discuss the new API design"
    },
    {
      "message_id": "msg_002",
      "create_time": "2025-02-01T10:01:00+00:00",
      "sender": "bob",
      "sender_name": "Bob Jones",
      "role": "user",
      "type": "text",
      "content": "I think we should use REST with OpenAPI spec"
    }
  ]
}
```

### Key Fields

| Field | Required | Description |
|-------|----------|-------------|
| `group_id` | No | Unique identifier for filtering and retrieval (see [When to Use Group ID](#when-to-use-group-id)) |
| `name` | No | Human-readable group name |
| `scene` | No | Scene type: `assistant` (1:1 with AI) or `group_chat` (group chat) |
| `user_details` | No | Participant information for context |

---

## Processing Group Chats

### Step 1: Prepare Your Data

Create a JSON file following the GroupChatFormat specification.

### Step 2: Process the Group Chat

```bash
uv run python src/bootstrap.py src/run_memorize.py \
  --input your_group_chat.json \
  --scene group_chat \
  --api-url http://localhost:1995/api/v1/memories
```

**Parameters:**
- `--input`: Path to your GroupChatFormat JSON file (required)
- `--scene`: Memory extraction scene - `group_chat` or `assistant` (required)
- `--api-url`: Memory API endpoint (required unless using `--validate-only`)
- `--validate-only`: Only validate the input file format without processing

### Step 3: Verify Processing

Check that memories were extracted:

```bash
curl -X GET "http://localhost:1995/api/v1/memories/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was discussed?",
    "group_id": "team_001",
    "memory_types": ["episodic_memory"],
    "retrieve_method": "rrf"
  }'
```

---

## Retrieving Group Memories

### Filter by Group ID

The primary way to retrieve group-specific memories:

```python
import requests

response = requests.get(
    "http://localhost:1995/api/v1/memories/search",
    json={
        "query": "What decisions were made about the API?",
        "group_id": "team_001",  # Filter to this group only
        "memory_types": ["episodic_memory"],
        "retrieve_method": "rrf",
        "top_k": 10
    }
)

memories = response.json()
result = memories.get("result", {})
for group in result.get("memories", []):
    print(f"Group: {group}")
```

### Retrieve Method Options

| Method | Description |
|--------|-------------|
| `keyword` | BM25 keyword retrieval |
| `vector` | Vector semantic retrieval |
| `hybrid` | Combined keyword + vector |
| `rrf` | RRF fusion retrieval (keyword + vector) |
| `agentic` | LLM-guided multi-round retrieval |

### Memory Type Options

| Type | Description |
|------|-------------|
| `episodic_memory` | Conversation episodes and events |
| `profile` | User profile information |
| `foresight` | Prospective memory |
| `event_log` | Atomic facts extracted from episodes |

---

## Generating Group Summaries

EverMemOS can generate contextual summaries within a group because it understands:

- **Who participated** - User details and roles
- **What was discussed** - Full conversation context
- **When it happened** - Temporal relationships
- **Key decisions** - Extracted from conversation flow

### Example: Query for Group Summary

```python
response = requests.get(
    "http://localhost:1995/api/v1/memories/search",
    json={
        "query": "Summarize the key decisions and action items",
        "group_id": "team_001",
        "memory_types": ["episodic_memory", "event_log"],
        "retrieve_method": "agentic"  # Use agentic mode for better synthesis
    }
)
```

---

## Best Practices

### 1. Consistent Group ID Naming

Use a consistent naming convention:

```
# Good: Clear, hierarchical naming
team_engineering
project_mobile_v2
tenant_acme_corp
channel_general

# Avoid: Inconsistent or unclear naming
eng
proj1
abc123
```

### 2. Include User Details

Providing user details improves memory quality:

```json
"user_details": {
  "alice": {
    "full_name": "Alice Smith",
    "custom_role": "Tech Lead"  // Helps understand context
  }
}
```

### 3. Use Appropriate Scene Types

- Use `group_chat` for multi-person group chats
- Use `assistant` for 1:1 conversations with an AI assistant

### 4. Batch Related Messages

Process conversations in logical batches rather than individual messages for better context understanding.

---

## Example Files

- **[Chinese Sample](../../data/group_chat_zh.json)** - Chinese language example
- **[English Sample](../../data/group_chat_en.json)** - English language example
- **[Format Specification](../../data_format/group_chat/group_chat_format.md)** - Complete format reference

---

## See Also

- [Group Chat Format Specification](../../data_format/group_chat/group_chat_format.md) - Complete data format reference
- [Batch Operations Guide](../usage/BATCH_OPERATIONS.md) - Processing multiple messages
- [Memory Retrieval Strategies](RETRIEVAL_STRATEGIES.md) - Optimizing search
- [Conversation Metadata Control](METADATA_CONTROL.md) - Fine-grained metadata management
- [API Documentation](../api_docs/memory_api.md) - Complete API reference
