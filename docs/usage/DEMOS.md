# Interactive Demos

[Home](../../README.md) > [Docs](../README.md) > [Usage](.) > Interactive Demos

This guide provides detailed walkthroughs of EverMemOS's interactive demos.

---

## Table of Contents

- [Overview](#overview)
- [Simple Demo (Recommended)](#simple-demo-recommended)
- [Full-Featured Demo](#full-featured-demo)
- [Demo Configuration](#demo-configuration)
- [Troubleshooting](#troubleshooting)

---

## Overview

EverMemOS provides two demo modes:

1. **Simple Demo** - Quick 2-step demo showing basic storage and retrieval
2. **Full Demo** - Complete workflow with memory extraction and interactive chat

Both demos require the API server to be running.

---

## Simple Demo (Recommended)

The fastest way to experience EverMemOS! Perfect for first-time users.

### Prerequisites

- Completed installation (see [Setup Guide](../installation/SETUP.md))
- API server running

### Quick Start

```bash
# Terminal 1: Start the API server
uv run python src/run.py --port 1995

# Terminal 2: Run the simple demo
uv run python src/bootstrap.py demo/simple_demo.py
```

### What It Does

1. **Stores 4 conversation messages** about sports hobbies:
   ```python
   messages = [
       "I love playing soccer on weekends",
       "I enjoy watching Barcelona matches",
       "My favorite sport is basketball",
       "I used to play tennis in high school"
   ]
   ```

2. **Waits 10 seconds** for indexing to complete

3. **Searches for memories** with 3 different queries:
   - "What sports does the user like?"
   - "Tell me about the user's hobbies"
   - "What does the user do on weekends?"

4. **Displays results** with relevance scores

### Expected Output

```
=== EverMemOS Simple Demo ===

Step 1: Storing conversation messages...
✓ Stored message 1: I love playing soccer on weekends
✓ Stored message 2: I enjoy watching Barcelona matches
✓ Stored message 3: My favorite sport is basketball
✓ Stored message 4: I used to play tennis in high school

Step 2: Waiting 10 seconds for indexing...
[Progress bar]

Step 3: Searching for relevant memories...

Query: "What sports does the user like?"
Results:
  1. [Score: 0.95] I love playing soccer on weekends
  2. [Score: 0.89] My favorite sport is basketball
  3. [Score: 0.82] I used to play tennis in high school

Query: "Tell me about the user's hobbies"
Results:
  1. [Score: 0.91] I love playing soccer on weekends
  2. [Score: 0.87] I enjoy watching Barcelona matches
  ...

✓ Demo completed successfully!
```

### Demo Code Location

See [`demo/simple_demo.py`](../../demo/simple_demo.py) for the complete source code.

### Perfect For

- ✓ First-time users
- ✓ Quick testing
- ✓ Verifying installation
- ✓ Understanding core concepts
- ✓ Demonstrating EverMemOS to others

---

## Full-Featured Demo

Experience the complete EverMemOS workflow: memory extraction from conversations followed by interactive chat with memory retrieval.

### Prerequisites

**1. Start the API Server:**

```bash
# Terminal 1: Start the API server (keep running)
uv run python src/run.py --port 1995
```

**2. Configure Environment:**

Ensure your `.env` file has the required API keys:
- `LLM_API_KEY` (or `OPENROUTER_API_KEY` or `OPENAI_API_KEY`)
- `VECTORIZE_API_KEY`

See [Configuration Guide](../installation/SETUP.md#environment-configuration) for details.

---

### Step 1: Extract Memories

Process sample conversation data and build the memory database.

```bash
# Terminal 2: Run the extraction script
uv run python src/bootstrap.py demo/extract_memory.py
```

#### What This Script Does

1. **Clears existing data** by calling `demo.tools.clear_all_data.clear_all_memories()`
   - Resets MongoDB, Elasticsearch, Milvus, and Redis to empty state
   - Ensures demo starts fresh

2. **Loads conversation data** from `data/assistant_chat_zh.json`
   - Sample conversations in Chinese
   - For English data, modify the `data_file` constant

3. **Processes each message** through the Memory API
   - Appends `scene="assistant"` to indicate one-on-one conversation
   - Streams entries to `http://localhost:1995/api/v1/memories`

4. **Creates memories in databases**
   - MemCells extracted from conversations
   - Episodes constructed from related MemCells
   - Profiles built from user information
   - Indexes created in Elasticsearch and Milvus

#### Configuration Options

Edit `demo/extract_memory.py` to customize:

```python
# API endpoint
base_url = "http://localhost:1995"

# Data file
data_file = "data/assistant_chat_zh.json"  # or assistant_chat_en.json

# Scene type
profile_scene = "assistant"  # or "group_chat"
```

#### Expected Output

```
Clearing all existing memories...
✓ Cleared MongoDB collections
✓ Cleared Elasticsearch indices
✓ Cleared Milvus collections
✓ Cleared Redis cache

Loading conversation data from data/assistant_chat_zh.json...
Found 150 messages

Processing messages:
[Progress bar] 150/150 messages processed

✓ Memory extraction completed
✓ 150 MemCells created
✓ 23 episodes constructed
✓ 5 profiles built

You can now run the chat demo!
```

#### For More Details

See [`demo/README.md`](../../demo/README.md) for comprehensive documentation.

---

### Step 2: Chat with Memory

Start the interactive chat demo to query extracted memories.

```bash
# Terminal 2: Run the chat program
uv run python src/bootstrap.py demo/chat_with_memory.py
```

#### How It Works

1. **Loads environment** via `python-dotenv`
2. **Verifies LLM keys** are available
3. **Connects to MongoDB** to enumerate groups with MemCells
4. **Invokes search API** for each user query
5. **Displays retrieved memories** before generating response

#### Interactive Workflow

##### 1. Select Language

```
Welcome to EverMemOS Chat Demo!
Select language / 选择语言:
  1. English
  2. 中文
Choice [1-2]:
```

##### 2. Select Scenario Mode

```
Select scenario mode:
  1. Assistant (one-on-one conversation)
  2. Group Chat (multi-speaker analysis)
Choice [1-2]:
```

##### 3. Select Conversation Group

```
Available conversation groups:
  1. Personal Assistant (150 messages)
  2. Work Discussion (85 messages)
  3. Family Chat (42 messages)
Select group [1-3]:
```

Groups are read from MongoDB. Run the extraction step first to populate groups.

##### 4. Select Retrieval Mode

```
Select retrieval mode:
  1. rrf (Hybrid - Recommended)
  2. embedding (Semantic search)
  3. bm25 (Keyword search)
  4. agentic (LLM-guided - Slower but more intelligent)
Choice [1-4]:
```

**Retrieval Modes:**
- **rrf** - Reciprocal Rank Fusion of semantic and keyword search (recommended)
- **embedding** - Pure semantic vector search
- **bm25** - Pure keyword search
- **agentic** - Multi-round LLM-guided retrieval (higher latency, better results)

##### 5. Start Chatting

```
You are now chatting with: Personal Assistant
Retrieval mode: rrf

Available commands:
  - help: Show available commands
  - clear: Clear conversation history
  - reload: Reload memories from database
  - exit: Exit the chat

You: What are my hobbies?

[Retrieved Memories]
1. [Episode] User mentioned loving soccer on weekends (2025-01-15)
2. [Episode] User enjoys watching Barcelona matches (2025-01-16)
3. [Profile] Sports: Soccer, Basketball, Tennis
Assistant: Based on your memories, you enjoy several sports including soccer, basketball,
and tennis. You particularly love playing soccer on weekends and watching Barcelona matches.
```

**Chat Commands:**
- `help` - Show available commands
- `clear` - Clear conversation history (keeps memories)
- `reload` - Reload memories from database
- `exit` - Exit the chat demo

---

## Demo Configuration

### Customizing Demo Data

You can use your own conversation data with the demos:

1. **Prepare your data** in the GroupChatFormat (see [Format Specification](../../data_format/group_chat/group_chat_format.md))
2. **Edit `demo/extract_memory.py`** to point to your data file
3. **Run the extraction script** to process your data
4. **Chat with your memories!**

### Demo Parameters

**Extraction Script:**
- `base_url` - API server endpoint (default: http://localhost:1995)
- `data_file` - Path to conversation data file
- `profile_scene` - Scene type: "assistant" or "group_chat"

**Chat Script:**
- Language selection (en/zh)
- Scenario mode (assistant/group_chat)
- Retrieval mode (rrf/embedding/bm25/agentic)

---

## Troubleshooting

### Demo Won't Start

**Problem**: Demo scripts fail to run

**Solutions:**
- Verify API server is running: `curl http://localhost:1995/health`
- Check .env file has required API keys
- Ensure Docker services are running: `docker-compose ps`
- Verify Python version: `python --version` (should be 3.10+)

### No Memories Found

**Problem**: Chat demo shows "No conversation groups found"

**Solutions:**
- Run the extraction script first: `uv run python src/bootstrap.py demo/extract_memory.py`
- Check MongoDB has data: Connect to MongoDB and verify collections
- Ensure extraction completed successfully (check terminal output)

### Retrieval Returns Empty Results

**Problem**: Search queries return no results

**Solutions:**
- Wait 10-15 seconds after storing messages (indexing delay)
- Verify Elasticsearch is running: `curl http://localhost:19200`
- Verify Milvus is running: `docker-compose ps`
- Check if embeddings were created (requires VECTORIZE_API_KEY)

### Chat Demo Errors

**Problem**: Chat demo crashes or shows errors

**Solutions:**
- Verify LLM API key is configured in .env
- Check API key has sufficient credits/quota
- Try a different retrieval mode (rrf is most reliable)
- Check logs for specific error messages

### Slow Performance

**Problem**: Demos are slow or timeout

**Solutions:**
- Use "rrf" or "keyword" instead of "agentic" mode
- Reduce `top_k` parameter (fewer results = faster)
- Check Docker container resource usage
- Ensure sufficient RAM (4GB minimum)

---

## See Also

- [Usage Examples](USAGE_EXAMPLES.md) - All usage methods
- [Batch Operations](BATCH_OPERATIONS.md) - Process multiple messages
- [Setup Guide](../installation/SETUP.md) - Installation and configuration
- [Demo README](../../demo/README.md) - Comprehensive demo documentation
- [Data Format](../../data/README.md) - Conversation data format specifications
