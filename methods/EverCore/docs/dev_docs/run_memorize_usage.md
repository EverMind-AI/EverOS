# run_memorize.py Usage Documentation

## Overview

`run_memorize.py` is a conversation memory storage script that reads JSON files conforming to the `ConversationFormat` format and stores them item by item into the memory system via HTTP API.

## Features

- ✅ Read and validate JSON files in ConversationFormat format
- ✅ Support both `solo` and `team` scenarios
- ✅ Automatically save settings metadata
- ✅ Call memorize interface item by item to process messages
- ✅ Provide format validation mode
- ✅ Detailed logging output

## Usage

### 1. Basic Usage

Store memories via HTTP API (must specify scene):

```bash
python src/bootstrap.py src/run_memorize.py \
  --input data/team_chat.json \
  --api-url http://localhost:1995/api/v0/memories \
  --scene solo
```

### 2. Using team Scenario

```bash
python src/bootstrap.py src/run_memorize.py \
  --input data/team_chat.json \
  --api-url http://localhost:1995/api/v0/memories \
  --scene team
```

### 3. Format Validation Only

Validate whether the input file format is correct without performing storage (no API address needed):

```bash
python src/bootstrap.py src/run_memorize.py \
  --input data/team_chat.json \
  --scene solo \
  --validate-only
```

## Command-Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--input` | Yes | Input conversation JSON file path (ConversationFormat format) |
| `--scene` | Yes | Memory extraction scenario, only supports `solo` or `team` |
| `--api-url` | No* | memorize API address (required for non-validation mode) |
| `--validate-only` | No | Only validate input file format, do not perform storage |

*Note: When using `--validate-only`, no need to provide `--api-url`, otherwise it's required.

## Input File Format

The input file must conform to the `ConversationFormat` specification, see `data_format/conversation/conversation_format.py`.

### Format Example

```json
{
  "version": "1.0.0",
  "session_meta": {
    "name": "Smart Sales Assistant Project Team",
    "description": "Development discussion group for Smart Sales Assistant project",
    "group_id": "group_sales_ai_2025",
    "created_at": "2025-02-01T01:00:00Z",
    "default_timezone": "UTC",
    "user_details": {
      "user_101": {
        "full_name": "Alex",
        "role": "Tech Lead"
      },
      "user_102": {
        "full_name": "Betty",
        "role": "Product Manager"
      }
    },
    "tags": ["AI", "Sales", "Project Development"]
  },
  "conversation_list": [
    {
      "message_id": "msg_001",
      "create_time": "2025-02-01T02:00:00Z",
      "sender": "user_101",
      "sender_name": "Alex",
      "type": "text",
      "content": "Good morning everyone, let's discuss project progress today",
      "refer_list": []
    }
  ]
}
```

## Processing Flow

The script executes the following steps:

1. **Format Validation**
   - Read input JSON file
   - Validate whether it conforms to ConversationFormat specification
   - Output data statistics

2. **Save Settings**
   - Call `settings` interface
   - Save metadata such as scene, group information, user details
   - API address: `{base_url}/api/v1/settings`

3. **Process Messages Item by Item**
   - Call `memorize` interface sequentially for each message
   - Each message includes: message_id, create_time, sender, content, etc.
   - Automatically add group_id, group_name, scene information
   - API address: `{api_url}` (specified by `--api-url` argument)

4. **Output Results**
   - Display number of successfully processed messages
   - Display total number of saved memories

## Output Example

### Successful Output

```
🚀 Group Chat Memory Storage Script
======================================================================
📄 Input File: /path/to/team_chat.json
🔍 Validation Mode: No
🌐 API Address: http://localhost:1995/api/v0/memories
======================================================================
======================================================================
Validating Input File Format
======================================================================
Reading file: /path/to/team_chat.json
Validating ConversationFormat format...
✓ Format validation passed!

=== Data Statistics ===
Format Version: 1.0.0
Group Chat Name: Smart Sales Assistant Project Team
Group Chat ID: group_sales_ai_2025
Number of Users: 5
Number of Messages: 8
Time Range: 2025-02-01T02:00:00Z ~ 2025-02-01T02:05:00Z

======================================================================
Reading Group Chat Data
======================================================================
Reading file: /path/to/team_chat.json
Using simple direct single message format, processing item by item

======================================================================
Starting to Call memorize API Item by Item
======================================================================
Group Name: Smart Sales Assistant Project Team
Group ID: group_sales_ai_2025
Number of Messages: 8
API Address: http://localhost:1995/api/v0/memories

--- Saving Settings ---
Saving settings to: http://localhost:1995/api/v1/settings
Scene: solo, Group ID: group_sales_ai_2025
  ✓ Settings saved successfully
  Scene: solo

--- Processing Message 1/8 ---
  ✓ Successfully saved 1 memory

--- Processing Message 2/8 ---
  ⏳ Waiting for episode boundary

--- Processing Message 3/8 ---
  ✓ Successfully saved 2 memories

--- Processing Message 4/8 ---
  ⏳ Waiting for episode boundary

--- Processing Message 5/8 ---
  ⏳ Waiting for episode boundary

--- Processing Message 6/8 ---
  ✓ Successfully saved 1 memory

--- Processing Message 7/8 ---
  ⏳ Waiting for episode boundary

--- Processing Message 8/8 ---
  ✓ Successfully saved 2 memories

======================================================================
Processing Complete
======================================================================
✓ Successfully Processed: 8/8 messages
✓ Total Saved: 6 memories

======================================================================
✓ Processing Complete!
======================================================================
```

## Error Handling

### File Does Not Exist

```
Error: Input file does not exist: /path/to/file.json
```

### Format Validation Failed

```
✗ Format validation failed!
Please ensure input file conforms to ConversationFormat specification
```

### JSON Parsing Error

```
✗ JSON parsing failed: Expecting value: line 1 column 1 (char 0)
```

## Development Notes

### Core Dependencies

- `infra_layer.adapters.input.api.mapper.conversation_converter`: Format validation
- `httpx`: HTTP client (async requests)
- `core.observation.logger`: Logging utilities

### API Endpoints

The script calls two API endpoints:

1. **settings**: Save settings
   - Path: `{base_url}/api/v1/settings`
   - Method: PUT
   - Data: Contains metadata such as scene, group_id, user_details

2. **memorize**: Store single message memory
   - Path: `{api_url}` (specified by `--api-url` argument)
   - Method: POST
   - Data: Contains message_id, sender, content, scene, etc.

### Extension Suggestions

1. **Batch Processing**: Support processing multiple files in a directory
2. **Progress Display**: Add progress bar to show processing status
3. **Error Retry**: Add failure retry mechanism
4. **Concurrent Processing**: Support batch concurrent API calls (note: maintain message order)
5. **Result Export**: Export storage results as JSON file

## Common Questions

### Q1: Why is it recommended to start with bootstrap.py?

A: `bootstrap.py` automatically handles:
- Python path setup
- Environment variable loading
- Dependency injection container initialization
- Mock mode support

This ensures the script runs in a complete application context.

### Q2: What's the difference between solo and team scenarios?

A:
- **solo**: Assistant scenario, suitable for AI assistant and user one-on-one conversations
- **team**: Team scenario, suitable for multi-person group chat and team collaboration

Different scenarios affect memory extraction strategies and storage methods. Choose based on actual application scenario.

### Q3: Why does message processing show "Waiting for episode boundary"?

A: The memory system uses "Episode Boundary" to determine when to form complete memory fragments.
- Not every message immediately generates a memory
- The system waits for a complete conversation episode to end before extracting memories
- This is normal processing behavior, not a failure

### Q4: Can I not provide an API address?

A: No. The current version only supports calling via HTTP API, you must provide the `--api-url` argument (unless using `--validate-only` for format validation only).

### Q5: What to do if API call fails?

A: Check the following:
1. Ensure memory service is running
2. Confirm API address is correct (including port number)
3. View server logs to understand detailed error information
4. Confirm input data format is correct

## References

- [ConversationFormat Format Definition](../../data_format/conversation/conversation_format.py)
- [Memory API Documentation](../api_docs/memory_api.md)
- [Bootstrap Usage Documentation](./bootstrap_usage.md)
