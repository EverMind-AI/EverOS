# Conversation Format Specification

## Overview

This is an open-source group chat data format specification for standardizing the storage and exchange of group chat conversation data.

## Format Definition

For the complete format definition, please refer to [`conversation_format.py`](../conversation_format.py)

## Core Features

### 1. Separated Metadata and Message List

```json
{
  "version": "1.0.0",
  "session_meta": { ... },
  "conversation_list": [ ... ]
}
```

- **version**: Format version number (follows semantic versioning)
- **session_meta**: Session metadata
- **conversation_list**: Message list

### 2. Scene Types and Scene Descriptions

Two core conversation scenes are supported:

- **solo**: Human-AI assistant conversation scene, one-on-one dialogue, AI acts as a personal assistant
- **team**: Work group chat scene, multi-person group chat, team collaboration

**Solo Scene Example**:
```json
"session_meta": {
  "scene": "solo",
  "scene_desc": {
    "description": "User and AI assistant conversation about Beijing tourism, health management, and sports rehabilitation"
  },
  ...
}
```

**Team Scene Example**:
```json
"session_meta": {
  "scene": "team",
  "scene_desc": {},
  ...
}
```

- **scene**: Scene type identifier (`solo` or `team`)
- **scene_desc**: Scene description information
  - Solo scene: Contains `description` field, describing the conversation scene
  - Team scene: Usually an empty object, representing a multi-person collaboration scene

### 3. User Details

All user details are centrally stored in `session_meta.user_details`:

```json
"user_details": {
  "user_101": {
    "full_name": "Alex",
    "role": "user",
    "custom_role": "Tech Lead",
    "department": "Technology",
    "email": "alex@example.com"
  },
  "robot_001": {
    "full_name": "AI Assistant",
    "role": "assistant"
  }
}
```

**Field Descriptions**:
- `full_name`: User's display name (optional)
- `role`: User type role (`user` for human, `assistant` for AI) (optional)
- `custom_role`: User's job/position role (e.g., Tech Lead, Product Manager) (optional)
- `department`: Department (optional)
- `email`: Email address (optional)
- `extra`: Additional extended information (optional)

### 4. Message Structure

Each message uses user ID as `sender`, with an optional `sender_name` for readability:

```json
{
  "message_id": "msg_001",
  "create_time": "2025-02-01T10:00:00+00:00",
  "sender": "user_103",
  "sender_name": "Chen",
  "role": "user",
  "type": "text",
  "content": "Message content",
  "refer_list": []
}
```

### 4.1 Message Sender Role

The optional `role` field identifies the source of the message:

- **user**: Message from a human user
- **assistant**: Message from an AI assistant

This is compatible with OpenAI/mem0/memos message format.

**Human Message Example**:
```json
{
  "message_id": "msg_001",
  "sender": "user_101",
  "sender_name": "Alex",
  "role": "user",
  "type": "text",
  "content": "Can you help me summarize this document?"
}
```

**AI Response Example**:
```json
{
  "message_id": "msg_002",
  "sender": "robot_001",
  "sender_name": "AI Assistant",
  "role": "assistant",
  "type": "text",
  "content": "Here's the summary of the document..."
}
```

### 5. Timezone-Aware Timestamps

- Uses ISO 8601 format
- Recommended to include timezone information (e.g., `+00:00`)
- If a message doesn't have timezone information, it can be obtained from `session_meta.default_timezone`

### 6. Message Types

Multiple message types are supported:

- **text**: Text message
- **image**: Image message
- **file**: File message
- **audio**: Audio message
- **video**: Video message
- **link**: Link message
- **system**: System message

### 7. Message References

Flexible reference methods are supported, each element in `refer_list` can be:

**Method 1: String Reference (message_id only)**
```json
"refer_list": ["msg_002", "msg_005"]
```

**Method 2: Object Reference (only message_id is required, other fields are optional)**

Minimal form:
```json
"refer_list": [
  {
    "message_id": "msg_002"
  }
]
```

Including partial fields (e.g., content for preview):
```json
"refer_list": [
  {
    "message_id": "msg_002",
    "content": "Good morning. Let's align on the goal first. Is it an MVP for internal testing or direct customer pilot?"
  }
]
```

Including complete information:
```json
"refer_list": [
  {
    "message_id": "msg_002",
    "create_time": "2025-02-01T10:01:00+00:00",
    "sender": "user_102",
    "sender_name": "Betty",
    "type": "text",
    "content": "Good morning. Let's align on the goal first. Is it an MVP for internal testing or direct customer pilot?",
    "refer_list": []
  }
]
```

**Method 3: Mixed Usage**
```json
"refer_list": [
  "msg_001",
  {
    "message_id": "msg_002",
    "content": "Partial content..."
  }
]
```

**Usage Recommendations:**
- String reference: Most concise, suitable for scenarios with clear reference relationships
- Minimal object reference: Only message_id, suitable for scenarios requiring unified format but no additional information
- With preview content: Includes message_id + content, suitable for quick preview scenarios
- Complete object reference: Includes all fields, suitable for export, archiving, or use independent of original data

### 8. Extension Fields

Use `extra` field to store additional information:

```json
"extra": {
  "file_name": "UI_draft_v1.pdf",
  "file_size": 2048576,
  "file_type": "application/pdf"
}
```

## Example Files

- [`conversation_format_example.json`](./conversation_format_example.json) - Complete example file
- [`conversation_compatible.json`](./conversation_compatible.json) - Legacy format compatibility example

## Usage

### Python

```python
from data.conversation_format import ConversationFormat, validate_conversation_format
import json

# Read group chat data
with open('conversation_example.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Validate format
is_valid = validate_conversation_format(data)
print(f"Format validation: {'Passed' if is_valid else 'Failed'}")

# Access user information
user_details = data['session_meta']['user_details']
print(f"User list: {list(user_details.keys())}")

# Iterate through messages
for msg in data['conversation_list']:
    sender_info = user_details[msg['sender']]
    print(f"{sender_info['full_name']}: {msg['content']}")
```

### Creating New Conversation Data

```python
from data.conversation_format import create_example_conversation
import json

# Create example data
chat_data = create_example_conversation()

# Save to file
with open('my_chat.json', 'w', encoding='utf-8') as f:
    json.dump(chat_data, f, ensure_ascii=False, indent=2)
```

## Version History

- **1.0.0** (2025-02-01)
  - Initial version
  - Support for basic message types
  - Support for user details
  - Support for message references
  - Support for timezone-aware timestamps

## Contributing

Issues and Pull Requests are welcome to improve this format specification.

## License

Open source license to be determined
