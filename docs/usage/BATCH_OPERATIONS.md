# Batch Operations Guide

[Home](../../README.md) > [Docs](../README.md) > [Usage](.) > Batch Operations

This guide explains how to efficiently process multiple messages using EverMemOS's batch operations.

---

## Table of Contents

- [Overview](#overview)
- [Group Chat Format](#group-chat-format)
- [Batch Storage Script](#batch-storage-script)
- [Data Format Specification](#data-format-specification)
- [Examples](#examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

EverMemOS supports batch processing for efficiently storing multiple messages at once. This is particularly useful for:

- Processing historical conversation data
- Importing chat logs from other platforms
- Group chat conversations with multiple participants
- Bulk data migration

---

## Group Chat Format

EverMemOS uses a standardized **GroupChatFormat** for batch operations. This format supports:

- Conversation metadata (group info, user details)
- Multi-speaker conversations
- Timestamps and message IDs

For complete format specifications, see [Group Chat Format Specification](../../data_format/group_chat/group_chat_format.md).

---

## Batch Storage Script

### Basic Usage

```bash
# Store group chat messages (Chinese data)
uv run python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat_zh.json \
  --api-url http://localhost:1995/api/v1/memories \
  --scene group_chat

# Store group chat messages (English data)
uv run python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat_en.json \
  --api-url http://localhost:1995/api/v1/memories \
  --scene group_chat

# Validate file format without storing
uv run python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat_en.json \
  --scene group_chat \
  --validate-only
```

### Script Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--input` | Yes | Path to the conversation data file (JSON format) |
| `--api-url` | No | API endpoint (default: http://localhost:1995/api/v1/memories) |
| `--scene` | Yes | Scene type: `assistant` or `group_chat` |
| `--validate-only` | No | Validate format without sending to API |

### Scene Parameter Explanation

The `--scene` parameter specifies the memory extraction strategy:

- **`assistant`** - Use for one-on-one conversations with AI assistant
- **`group_chat`** - Use for multi-person group discussions

**Important Note**: In your data files, you may see `scene` values like `work`, `company`, or `social` - these are internal scene descriptors in the data format. The `--scene` command-line parameter uses different values (`assistant`/`group_chat`) to specify which extraction pipeline to apply.

---

## Data Format Specification

### GroupChatFormat Structure

```json
{
  "version": "1.0.0",
  "conversation_meta": {
    "group_id": "group_001",
    "name": "Project Discussion Group",
    "description": "Team project planning and updates",
    "scene": "group_chat",
    "timezone": "Asia/Shanghai",
    "user_details": {
      "user_101": {
        "full_name": "Alice",
        "role": "Product Manager",
        "nickname": "Ali"
      },
      "user_102": {
        "full_name": "Bob",
        "role": "Engineer"
      }
    }
  },
  "conversation_list": [
    {
      "message_id": "msg_001",
      "create_time": "2025-02-01T10:00:00+00:00",
      "sender": "user_101",
      "content": "Good morning everyone, let's discuss the new feature"
    },
    {
      "message_id": "msg_002",
      "create_time": "2025-02-01T10:05:00+00:00",
      "sender": "user_102",
      "content": "Sure! I've prepared the technical spec"
    }
  ]
}
```

### Required Fields

**conversation_meta:**
- `group_id` (string) - Unique identifier for the conversation group
- `name` (string) - Human-readable name for the group
- `user_details` (object) - Map of user IDs to user information

**conversation_list:**
- `message_id` (string) - Unique identifier for each message
- `create_time` (string) - ISO 8601 timestamp with timezone
- `sender` (string) - User ID (must exist in user_details)
- `content` (string) - Message content

### Optional Fields

**conversation_meta:**
- `description` (string) - Group description
- `scene` (string) - Internal scene descriptor (group_chat or assistant)
- `timezone` (string) - Timezone for the conversation

**conversation_list:**
- `sender_name` (string) - Override sender's display name

---

## Examples

### Example 1: Simple Group Chat

```json
{
  "version": "1.0.0",
  "conversation_meta": {
    "group_id": "team_standup",
    "name": "Daily Standup",
    "user_details": {
      "alice": {"full_name": "Alice Smith"},
      "bob": {"full_name": "Bob Jones"}
    }
  },
  "conversation_list": [
    {
      "message_id": "msg_1",
      "create_time": "2025-02-01T09:00:00+00:00",
      "sender": "alice",
      "content": "Yesterday I completed the login feature"
    },
    {
      "message_id": "msg_2",
      "create_time": "2025-02-01T09:01:00+00:00",
      "sender": "bob",
      "content": "Great! I'm working on the dashboard today"
    }
  ]
}
```

###Example 2: Family Chat with Rich Metadata

```json
{
  "version": "1.0.0",
  "conversation_meta": {
    "group_id": "family_chat_001",
    "name": "Smith Family",
    "description": "Family group chat",
    "scene": "group_chat",
    "timezone": "America/New_York",
    "user_details": {
      "mom": {
        "full_name": "Jane Smith",
        "nickname": "Mom",
        "role": "Parent"
      },
      "dad": {
        "full_name": "John Smith",
        "nickname": "Dad",
        "role": "Parent"
      },
      "daughter": {
        "full_name": "Emily Smith",
        "age": 16
      }
    }
  },
  "conversation_list": [
    {
      "message_id": "fam_001",
      "create_time": "2025-02-01T18:00:00-05:00",
      "sender": "mom",
      "content": "Dinner is ready! Come down please.",
    },
    {
      "message_id": "fam_002",
      "create_time": "2025-02-01T18:02:00-05:00",
      "sender": "daughter",
      "content": "Coming! Just finishing homework."
    }
  ]
}
```

### Example 3: One-on-One Assistant Chat

```json
{
  "version": "1.0.0",
  "conversation_meta": {
    "group_id": "user_assistant_001",
    "name": "Personal Assistant",
    "scene": "assistant",
    "user_details": {
      "user_001": {
        "full_name": "Alex"
      }
    }
  },
  "conversation_list": [
    {
      "message_id": "chat_001",
      "create_time": "2025-02-01T10:00:00+00:00",
      "sender": "user_001",
      "content": "I love playing soccer on weekends"
    },
    {
      "message_id": "chat_002",
      "create_time": "2025-02-01T10:30:00+00:00",
      "sender": "user_001",
      "content": "My favorite team is Barcelona"
    }
  ]
}
```

**Command for assistant chat:**
```bash
uv run python src/bootstrap.py src/run_memorize.py \
  --input my_assistant_chat.json \
  --scene assistant
```

---

## Best Practices

### 1. Data Preparation

- **Validate before importing**: Use `--validate-only` to check format
- **Use consistent IDs**: Ensure message_id and user IDs are unique
- **Include timestamps**: Always use ISO 8601 format with timezone
- **Provide user details**: Include at least full_name for each user

### 2. Performance Optimization

- **Batch size**: Process 100-1000 messages at a time for optimal performance
- **Sequential processing**: Script processes messages sequentially to maintain order
- **Monitor progress**: Watch for errors in terminal output
- **Wait for indexing**: Allow 10-15 seconds after completion for search indexes to update

### 3. Data Quality

- **Clean content**: Remove formatting artifacts or special characters
- **Accurate timestamps**: Ensure chronological order
- **Complete metadata**: Fill in all available user information
- **Meaningful group IDs**: Use descriptive, stable identifiers

### 4. Scene Selection

- Use `assistant` for:
  - One-on-one conversations
  - Personal AI assistant chats
  - Individual user interactions

- Use `group_chat` for:
  - Multi-participant discussions
  - Team conversations
  - Family or social group chats

---

## Troubleshooting

### Validation Errors

**Problem**: `--validate-only` reports format errors

**Solutions:**
- Check JSON syntax is valid
- Verify all required fields are present
- Ensure timestamps are in ISO 8601 format
- Confirm sender IDs exist in user_details

### API Errors

**Problem**: Script reports API errors when storing

**Solutions:**
- Verify API server is running: `curl http://localhost:1995/health`
- Check API URL is correct (default: http://localhost:1995/api/v1/memories)
- Ensure .env has required API keys (LLM_API_KEY, VECTORIZE_API_KEY)
- Review error messages for specific issues

### Slow Processing

**Problem**: Batch processing is very slow

**Solutions:**
- This is normal for large batches (each message requires LLM extraction)
- Reduce batch size if memory issues occur
- Ensure Docker services have adequate resources
- Check LLM API rate limits

### Missing Memories

**Problem**: Messages processed but not searchable

**Solutions:**
- Wait 10-15 seconds for indexing to complete
- Verify Elasticsearch and Milvus are running
- Check MongoDB for stored data
- Ensure embeddings were created (requires VECTORIZE_API_KEY)

---

## See Also

- [Group Chat Format Specification](../../data_format/group_chat/group_chat_format.md) - Complete format reference
- [Usage Examples](USAGE_EXAMPLES.md) - Other usage methods
- [Demos](DEMOS.md) - Interactive demo walkthroughs
- [API Documentation](../api_docs/memory_api.md) - Memory API reference
- [Data Guide](../../data/README.md) - Sample data and format details
