# Settings Control Guide

[Home](../../README.md) > [Docs](../README.md) > [Advanced](.) > Settings

## Overview

EverCore uses **settings** to provide context for memory extraction and retrieval. Properly configured settings enable:

- **Better memory extraction** - Understanding who said what and in what context
- **Accurate retrieval filtering** - Querying memories by user, group, or time range
- **Contextual summaries** - Generating summaries that understand participants and relationships
- **Multi-tenant isolation** - Separating memories between different groups or organizations

This guide explains when and how to control metadata for optimal results.

---

## When to Control Metadata

### 1. Multi-User Conversations

**Use Case:** Group chats, team discussions, meetings with multiple participants

**Why:** Without `user_details`, EverCore cannot distinguish between speakers or understand their roles.

```json
{
  "session_meta": {
    "user_details": {
      "alice": {
        "full_name": "Alice Smith",
        "role": "user",
        "custom_role": "Tech Lead",
        "department": "Engineering"
      },
      "bob": {
        "full_name": "Bob Jones",
        "role": "user",
        "custom_role": "Product Manager"
      }
    }
  }
}
```

**Benefits:**
- Memory extraction attributes facts to the correct person
- Retrieval can filter by specific user within a group
- Summaries understand organizational context

### 2. AI Assistant Conversations

**Use Case:** 1:1 conversations between a user and an AI assistant

**Why:** The `scene` and `scene_desc` fields tell EverCore which messages are from the AI (to potentially exclude from personal memory extraction).

```json
{
  "session_meta": {
    "scene": "solo",
    "scene_desc": {
      "description": "Project discussion group chat"
    },
    "user_details": {
      "user_123": {
        "full_name": "John Doe",
        "role": "user"
      },
      "assistant_001": {
        "full_name": "AI Assistant",
        "role": "assistant"
      }
    }
  }
}
```

**Benefits:**
- AI responses can be handled differently from user messages
- Personal memories focus on what the user shared, not AI responses

### 3. Cross-Timezone Teams

**Use Case:** Distributed teams working across different timezones

**Why:** The `timezone` ensures timestamps are interpreted correctly when timezone info is missing from individual messages.

```json
{
  "session_meta": {
    "timezone": "America/Los_Angeles",
    "user_details": {
      "dev_sf": {"full_name": "SF Developer"},
      "dev_tokyo": {"full_name": "Tokyo Developer"}
    }
  }
}
```

**Benefits:**
- Temporal queries ("What was discussed yesterday?") work correctly
- Memory ordering is accurate across timezones

### 4. Categorized Conversations

**Use Case:** Organizing conversations by type (work, social, family, etc.)

**Why:** The `scene` field categorizes conversations for better context understanding and potential filtering.

```json
{
  "session_meta": {
    "scene": "team",
    "tags": ["project-alpha", "backend", "Q1-2025"]
  }
}
```

**Benefits:**
- Memory extraction understands the context (professional vs casual)
- Tags enable additional filtering and organization

### 5. Default Configuration Fallback

**Use Case:** Setting organization-wide defaults that apply when specific group config is missing

**Why:** EverCore supports a default configuration that applies when a specific `scene` config is not found.

```python
# Save default config
requests.put(
    "http://localhost:1995/api/v1/settings",
    json={
        "scene": "team",
        "scene_desc": {},
        "timezone": "UTC",
        "llm_custom_setting": {}
    }
)
```

**Benefits:**
- New groups automatically inherit sensible defaults
- Reduces configuration overhead for common settings

---

## Settings Fields Reference

### Settings (`session_meta`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `scene` | string | No | Scene type: `solo` (1:1 with AI) or `team` (group chat) |
| `scene_desc` | object | No | Scene-specific details (e.g., `description` for solo scene) |
| `timezone` | string | No | IANA timezone name (e.g., `America/New_York`) |
| `llm_custom_setting` | object | No | Custom LLM settings for memory extraction |

### User Details (`user_details`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `full_name` | string | No | User's display name |
| `role` | string | No | `user` (human) or `assistant` (AI) |
| `custom_role` | string | No | Job title or position (e.g., "Tech Lead") |
| `department` | string | No | Department or team name |
| `email` | string | No | Email address |
| `extra` | object | No | Additional custom fields |

### Message Metadata

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message_id` | string | Yes | Unique identifier for the message |
| `create_time` | string | Yes | Message timestamp (ISO 8601 with timezone) |
| `sender` | string | Yes | User ID of the sender (must exist in `user_details`) |
| `sender_name` | string | No | Override display name for this message |
| `role` | string | No | `user` or `assistant` (overrides `user_details` role) |
| `refer_list` | array | No | Referenced message IDs or objects |

---

## API Operations

### Store Message with Metadata

When storing a single message, you can include group and sender metadata:

```python
import requests

response = requests.post(
    "http://localhost:1995/api/v0/memories",
    json={
        "message_id": "msg_001",
        "create_time": "2025-02-01T10:00:00+00:00",
        "sender": "user_123",
        "sender_name": "John",  # Optional display name
        "content": "I prefer Python for backend development",
        "group_id": "team_engineering",
        "group_name": "Engineering Team",
        "role": "user"
    }
)
```

### Search with Metadata Filters

Filter search results by user or group:

```python
# Search within a specific group
response = requests.get(
    "http://localhost:1995/api/v0/memories/search",
    json={
        "query": "What programming languages are preferred?",
        "group_id": "team_engineering",
        "user_id": "user_123",  # Optional: filter to specific user
        "retrieve_method": "rrf",
        "start_time": "2025-01-01T00:00:00+00:00",
        "end_time": "2025-02-01T00:00:00+00:00"
    }
)
```

### Manage Settings

#### Get Settings

```python
# Get current settings
response = requests.get(
    "http://localhost:1995/api/v1/settings"
)
```

#### Save/Update Settings (Upsert)

The V1 Settings API uses PUT for both creation and updates (upsert semantics):

```python
response = requests.put(
    "http://localhost:1995/api/v1/settings",
    json={
        "scene": "team",
        "scene_desc": {
            "description": "Backend engineering team discussions"
        },
        "timezone": "America/Los_Angeles",
        "llm_custom_setting": {}
    }
)
```

**Response format (success):**
```json
{
  "data": {
    "scene": "team",
    "scene_desc": {"description": "..."},
    "timezone": "America/Los_Angeles",
    "llm_custom_setting": {}
  }
}
```

### Delete Memories with Metadata Filters

```python
# Delete all memories for a specific user in a group
response = requests.delete(
    "http://localhost:1995/api/v0/memories",
    json={
        "user_id": "user_123",
        "group_id": "team_engineering"
    }
)
```

---

## Use Cases

### 1. Customer Support System

Track support conversations with customer context:

```json
{
  "session_meta": {
    "scene": "solo",
    "scene_desc": {"description": "Support conversation with customer"},
    "timezone": "UTC",
    "llm_custom_setting": {}
  }
}
```

### 2. Meeting Transcription

Capture meeting context with participant roles:

```json
{
  "session_meta": {
    "scene": "team",
    "scene_desc": {
      "description": "Daily Standup - Feb 1, 2025"
    },
    "timezone": "America/New_York",
    "llm_custom_setting": {}
  }
}
```

### 3. Personal AI Assistant

Track personal conversations with the AI:

```json
{
  "session_meta": {
    "scene": "solo",
    "scene_desc": {"description": "Personal assistant conversation"},
    "timezone": "UTC",
    "llm_custom_setting": {}
  }
}
```

---

## Best Practices

### 1. Always Provide User Details

Even for simple conversations, providing `user_details` improves memory quality:

```json
"user_details": {
  "user_123": {"full_name": "John Doe"}
}
```

### 2. Use Consistent User IDs

Use the same `sender` ID across all messages from the same person. The ID in messages must match keys in `user_details`.

### 3. Include Timezone Information

Always include timezone in message timestamps or set `timezone` in settings:

```json
"create_time": "2025-02-01T10:00:00-05:00"
```

### 4. Use Appropriate Scene Types

- **`solo`**: Use for 1:1 human-AI conversations
- **`team`**: Use for multi-person group chats and meetings

### 5. Leverage Tags for Organization

Tags provide additional filtering and categorization without affecting the core metadata structure.

---

## See Also

- [Team Chat Guide](TEAM_CHAT_GUIDE.md) - Multi-participant conversations
- [Conversation Format Specification](../../data_format/conversation/conversation_format.md) - Complete schema reference
- [Batch Operations](../usage/BATCH_OPERATIONS.md) - Processing conversations in batch
- [API Documentation](../api_docs/memory_api.md) - Complete API reference
