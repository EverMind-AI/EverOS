# Demo - EverMemOS Interactive Examples


This directory contains interactive demos showcasing the core functionality of EverMemOS.

## 📂 Directory Structure

```
demo/
├── chat_with_memory.py          # 🎯 Main: Interactive chat with memory
├── extract_memory.py            # 🎯 Main: Memory extraction (HTTP API)
├── simple_demo.py               # 🎯 Main: Quick start example
│
├── utils/                       # Utility module
│   ├── __init__.py
│   ├── memory_utils.py         # Shared utility functions
│   └── simple_memory_manager.py # Simple memory manager (HTTP API wrapper)
│
├── ui/                          # UI module
│   ├── __init__.py
│   └── i18n_texts.py           # Internationalization texts
│
├── chat/                        # Chat system components
│   ├── __init__.py
│   ├── orchestrator.py         # Chat application orchestrator
│   ├── session.py              # Session management
│   ├── ui.py                   # User interface
│   └── selectors.py            # Language/scenario/group selectors
│
├── tools/                       # Auxiliary tools
│   ├── clear_all_data.py       # Clear all memory data
│   ├── resync_memcells.py      # Resync memory cells
│   └── test_retrieval_comprehensive.py  # Retrieval testing tool
│
├── chat_history/                # 📁 Output: Chat logs (auto-generated)
│
├── README.md                    # 📖 Documentation (English)
```

**Notes**:
- All memory data is stored in databases (MongoDB, Elasticsearch, Milvus), no local `memcell_outputs/` directory
- `extract_memory.py` directly calls HTTP API without complex configuration classes
- Chat conversation history is saved in `chat_history/` directory

## 🎯 Core Scripts

### 1. `simple_demo.py` - Quick Start Example ⭐

**The simplest way to experience EverMemOS!** Just 67 lines of code demonstrating the complete memory workflow.

**What it demonstrates:**
- 💾 **Store**: Save conversation messages via HTTP API
- ⏳ **Index**: Wait for data to be indexed (MongoDB, Elasticsearch, Milvus)
- 🔍 **Search**: Retrieve relevant memories with natural language queries

**Code example:**
```python
from demo.utils import SimpleMemoryManager

# Create memory manager
memory = SimpleMemoryManager()

# Store conversations
await memory.store("I love playing soccer, often go to the field on weekends")
await memory.store("Soccer is a great sport! Which team do you like?", sender="Assistant")
await memory.store("I love Barcelona the most, Messi is my idol")

# Wait for indexing
await memory.wait_for_index(seconds=10)

# Search memories
await memory.search("What sports does the user like?")
await memory.search("What is the user's favorite team?")
```

**How to run:**

⚠️ **Important**: You must start the API server first!

```bash
# Terminal 1: Start the API server
uv run python src/run.py 

# Terminal 2: Run the simple demo
uv run python src/bootstrap.py demo/simple_demo.py
```

**Why this demo?**
- ✅ Minimal code - understand core concepts in seconds
- ✅ Complete workflow - storage → indexing → retrieval
- ✅ Friendly output - explanations for every step
- ✅ Real HTTP API - uses the same API as production

**Dependencies**: `utils/simple_memory_manager.py` (HTTP API wrapper)

### 2. `extract_memory.py` - Memory Extraction

Batch process conversation data and extract memories via HTTP API.

**Workflow**:
- Clears all existing memories (ensures clean starting state)
- Loads conversation files from `data/` directory (e.g., `data/assistant_chat_zh.json`)
- Sends each message to the API server (`/api/v1/memories`)
- Server-side automatically extracts MemCells, generates episodes and profiles
- All data is stored in databases (MongoDB, Elasticsearch, Milvus)

**Prerequisites**: API server must be running (`uv run python src/run.py `)

**Dependencies**: HTTP API, `clear_all_data` tool

### 3. `chat_with_memory.py` - Memory-Enhanced Chat

Command-line interface for conversing with memory-enabled AI agents.

**Features**:
- Interactive language selection (Chinese/English) and scenario selection (Assistant/Group Chat)
- Automatically load conversation groups from MongoDB
- Flexible retrieval mode selection (RRF/Embedding/BM25/Agentic)
- Real-time display of retrieved memories
- Auto-save conversation history

**Prerequisites**: Must run `extract_memory.py` first to extract memory data

**Dependencies**: `chat/` module, HTTP API

## 📦 Supporting Modules

### Utility Modules
- **`utils/simple_memory_manager.py`** - Simplified HTTP API wrapper for simple_demo
- **`utils/memory_utils.py`** - MongoDB connection and common utility functions

### UI Module
- **`ui/i18n_texts.py`** - Bilingual interface text resources (Chinese/English)

### Core Components
- **`chat/`** - Chat system implementation (orchestrator, session management, interface, selectors)
- **`tools/`** - Auxiliary tools (data cleanup, retrieval testing, etc.)

## 🚀 Quick Start

### ⚠️ Important: Memory Language Configuration

`MEMORY_LANGUAGE` controls the prompt language during memory extraction. **Must match your data file language and be set before starting the API server.**

| Data File | Setting |
|-----------|---------|
| `*_en.json` | `MEMORY_LANGUAGE=en` |
| `*_zh.json` | `MEMORY_LANGUAGE=zh` |

> **Note:** Setting `MEMORY_LANGUAGE` in `extract_memory.py` does NOT work - the server reads this at startup. If languages mismatch, you'll get mixed Chinese/English memories. To switch languages, restart the API server after changing the setting.

---

### Option A: Super Simple Mode (Recommended for Beginners) ⭐

The fastest way to experience EverMemOS! Just 2 terminals:

```bash
# Terminal 1: Start the API server (required)
uv run python src/run.py 

# Terminal 2: Run the simple demo
uv run python src/bootstrap.py demo/simple_demo.py
```

**What happens:**
1. 📝 Stores 4 conversation messages
2. ⏳ Waits 10 seconds for indexing (MongoDB → Elasticsearch → Milvus)
3. 🔍 Searches memories with 3 different queries
4. 📊 Shows results with relevance scores and explanations

**Note**: The API server (`src/run.py `) must be running in a separate terminal for the demo to work.

---

### Option B: Full Feature Mode

#### Step 1: Extract Memories

Run the extraction script to extract memories from conversation data:

```bash
# Start API server (if not already running)
uv run python src/run.py 

# In another terminal, run the extraction script
uv run python src/bootstrap.py demo/extract_memory.py
```

The script will:
- Clear all existing memory data
- Load `data/assistant_chat_zh.json` conversation file
- Send each message to the API server for memory extraction
- Store all memories in databases (MongoDB, Elasticsearch, Milvus)

> **💡 Tip**: `extract_memory.py` is straightforward and directly calls the HTTP API. You can modify the `data_file` and `profile_scene` variables in the script to use different data files.

#### Step 2: Start Conversation

Run the chat script to start conversing with AI:

```bash
# Ensure API server is still running
# In another terminal, run the chat program
uv run python src/bootstrap.py demo/chat_with_memory.py
```

**Interactive Selection**:
1. **Language**: Choose `[1] 中文` or `[2] English`
2. **Scenario**: Choose `[1] Assistant Mode` or `[2] Group Chat Mode`
3. **Group**: Select from available groups loaded from MongoDB
4. **Retrieval Mode**: Choose RRF (recommended), Embedding, BM25, or Agentic

**Chat Features**:
- 💬 Natural language conversation with memory-based context
- 🔍 Automatic retrieval of relevant memories (displays retrieval results)
- 📝 Conversation history auto-saved to `chat_history/` directory
- 🧠 Special commands for detailed information (`help`, `clear`, `reload`, `exit`)

---

## 📁 Data Files

The system uses sample conversation files from the `data/` directory:

| Scenario | Language | Filename |
|----------|----------|----------|
| Assistant Chat | Chinese | `data/assistant_chat_zh.json` |
| Assistant Chat | English | `data/assistant_chat_en.json` |
| Group Chat | Chinese | `data/group_chat_zh.json` |
| Group Chat | English | `data/group_chat_en.json` |

All data files follow the [GroupChatFormat](../data_format/group_chat/group_chat_format.md) specification. See [data documentation](../data/README.md) for details.

**Using Custom Data**:
Edit `extract_memory.py` and modify the `data_file` and `profile_scene` variables to point to your data file.

## 💬 Chat Commands

During chat sessions, the following commands are supported:

- **Normal Input**: Type questions directly, AI will answer based on memories
- `help` - Show help information
- `clear` - Clear current conversation history
- `reload` - Reload memories and profiles
- `exit` - Save conversation history and exit
- `Ctrl+C` - Interrupt and save

## ⚙️ Environment Configuration

Create a `.env` file in the project root (refer to `env.template`):

```bash
# LLM Configuration
LLM_MODEL=your_model
LLM_API_KEY=your_api_key
LLM_BASE_URL=your_base_url

# Embedding Model Configuration
EMB_BASE_URL=http://localhost:11000/v1/embeddings
EMB_MODEL=Qwen3-Embedding-4B

# MongoDB Configuration
MONGODB_URI=mongodb://admin:memsys123@localhost:27017
```

## 🔗 Related Documentation

- [Group Chat Format Specification](../data_format/group_chat/group_chat_format.md)
- [API Documentation](../docs/api_docs/memory_api.md)
- [Data Documentation](../data/README.md)

## 📖 Demo Data Overview

### Group Chat Scenario (group_chat_en.json / group_chat_zh.json)

**Project Context:** AI product work group documenting the complete development journey of "Smart Sales Assistant"

**Key Contents:**
- MVP development phase: RAG-based Q&A system
- Advanced feature iteration: Emotion recognition, memory system
- Team collaboration practices: Complete workflow from requirements to delivery

**Available in:** English and Chinese versions

**Good for exploring:** Team collaboration patterns, project management, technical solution evolution

### Assistant Scenario (assistant_chat_en.json / assistant_chat_zh.json)

**Conversation Context:** Personal health & lifestyle assistant documenting nearly 2 months of continuous interaction

**Key Contents:**
- Travel planning: Food recommendations, itinerary suggestions
- Health management: Weight monitoring, dietary guidance
- Exercise recovery: Training advice, post-injury rehabilitation

**Available in:** English and Chinese versions

**Good for exploring:** Personalized services, long-term memory accumulation, contextual understanding

## ❓ Recommended Questions

**Group Chat AI Scenario Examples:**
- What did Alex/Betty/... do in the emotion recognition project?
- Based on the emotion recognition project, what work capabilities does Alex/Betty/... demonstrate?
- What are the deliverable results of the emotion recognition project?
- How is the memory system project progressing?

**Assistant AI Scenario Examples:**
- Please recommend sports suitable for me.
- Please recommend food I might like.
- How is my health condition?


## 🔗 Related Documentation

- 📋 [Group Chat Format Specification](../data_format/group_chat/group_chat_format.md) - Data file format
- 🔌 [API Documentation](../docs/api_docs/memory_api.md) - API reference
- 📦 [Data Documentation](../data/README.md) - Sample data details
- 🏠 [Project Home](../README.md) - Project overview and architecture
- 📘 [Batch Memorization Guide](../docs/dev_docs/run_memorize_usage.md) - Advanced usage

## ❓ FAQ

### Q: Why are my extracted memories mixed with Chinese and English?
**A**: The `MEMORY_LANGUAGE` on the API server doesn't match your data file. Set it in `.env` to match your data (`en` or `zh`), then **restart the server** and re-run extraction.

### Q: Cannot connect to API server?
**A**: Ensure the API server is running first: `uv run python src/run.py `

### Q: How to use custom data with extract_memory.py?
**A**: Edit the script and modify these variables:
- `data_file`: Point to your JSON data file
- `profile_scene`: Set to `"assistant"` or `"group_chat"`
- `base_url`: API server address (default `http://localhost:1995`)

### Q: Where is data stored?
**A**: All memory data is stored via HTTP API to databases:
- **MongoDB**: Stores MemCells, episodes, profiles
- **Elasticsearch**: Keyword indexing (BM25)
- **Milvus**: Vector indexing (semantic retrieval)
- **Local files**: Only `chat_history/` directory saves conversation logs

### Q: What scenarios are supported?
**A**: Two scenarios are supported:
- **Assistant mode (assistant)**: One-on-one conversations, extract personalized profiles
- **Group chat mode (group_chat)**: Multi-participant conversations, extract group memories and member profiles

### Q: What is the data file format?
**A**: JSON format following the [GroupChatFormat](../data_format/group_chat/group_chat_format.md) specification. The project provides 4 sample files for reference.

### Q: How to view data in databases?
**A**: 
- **MongoDB**: Use MongoDB Compass or command-line queries
- **Retrieval test**: Run `demo/tools/test_retrieval_comprehensive.py`
- **Clear data**: Run `demo/tools/clear_all_data.py`

## 💡 Need Help?

- 🏠 See the main [README](../README.md) for project setup and architecture
- 💬 Open an issue on GitHub
- 📧 Contact project maintainers

---

**Happy exploring! 🧠✨**

