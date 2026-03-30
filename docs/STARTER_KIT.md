# EverMemOS Competition Starter Kit

[Home](../README.md) > [Docs](./README.md) > Starter Kit

Welcome to the EverMemOS Competition! This starter kit will help you get up and running quickly so you can focus on building amazing AI memory applications.

> **Join our [Discord](https://discord.gg/gYep5nQRZJ)** to brainstorm ideas, find teammates, and get help from the community!

---

## Table of Contents

- [Quick Start (5 Minutes)](#quick-start-5-minutes)
- [Competition Tracks](#competition-tracks)
- [How to Submit](#how-to-submit)
- [Evaluation Criteria](#evaluation-criteria)
- [API Cheatsheet](#api-cheatsheet)
- [Sample Data](#sample-data)
- [Example Projects](#example-projects)
- [Resources](#resources)
- [Tips for Success](#tips-for-success)

---

## Quick Start (5 Minutes)

### 1. Clone and Start Services

```bash
# Clone the repository
git clone https://github.com/EverMind-AI/EverMemOS.git
cd EverMemOS

# Start all services with Docker
docker compose up -d

# Verify services are running
curl http://localhost:1995/health
```

### 2. Store Your First Memory

```bash
curl -X POST "http://localhost:1995/api/v1/memories" \
  -H "Content-Type: application/json" \
  -d '{
    "message_id": "msg_001",
    "create_time": "2025-01-20T10:00:00+00:00",
    "sender": "user_001",
    "content": "I love building AI applications with memory!"
  }'
```

### 3. Search Memories

```bash
curl -X GET "http://localhost:1995/api/v1/memories/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does the user love?",
    "user_id": "user_001",
    "retrieve_method": "hybrid"
  }'
```

You're ready to build!

---

## Competition Tracks

### Track 1: Agent + Memory

Build intelligent agents with long-term, evolving memories.

| Idea | Description |
|------|-------------|
| Personal Digital Twin | AI that learns your preferences, habits, and communication style over time |
| Content Creator Copilot | Assistant that remembers your brand voice, past content, and audience feedback |
| CRM Copilot | Sales assistant with full customer interaction history and relationship context |
| Therapy/Health Companion | Supportive AI that tracks emotional patterns and wellness journey |
| Education AI | Tutor that adapts to learning style and remembers progress across subjects |
| Multi-Agent Collaboration | Team of specialized agents sharing knowledge through common memory |
| Customer Support Bot | Support agent with full case history and user preference memory |
| Game NPC System | NPCs with persistent memory of player interactions and world events |

### Track 2: Platform Plugins

Develop plugins or SDKs to integrate EverMemOS everywhere developers work.

| Idea | Description |
|------|-------------|
| VSCode Extension | Remember coding context, past solutions, and project decisions |
| Chrome Extension | Capture and recall browsing context, research sessions, and bookmarks |
| Slack/Discord Bot | Team memory bot that tracks decisions, action items, and discussions |
| Notion Integration | Sync notes and documents with conversational memory |
| Obsidian Plugin | Connect personal knowledge base with AI memory |
| LangChain Memory Backend | Drop-in memory class for LangChain applications |
| LlamaIndex Integration | Memory store connector for LlamaIndex pipelines |
| CLI Tool | Command-line interface for memory operations |

### Track 3: OS Infrastructure

Optimize the EverMemOS platform: core functionality, performance tuning, and system improvements.

| Idea | Description |
|------|-------------|
| Custom Retrieval Methods | Graph-based, time-weighted, or multi-modal retrieval strategies |
| Memory Extractors | Domain-specific extractors (medical, legal, technical) |
| Performance Optimization | Query optimization, caching strategies, batch processing |
| New Benchmarks | Evaluation datasets and metrics for specific use cases |
| Emotion-Aware Tagging | Extract and tag emotional context from conversations |
| Action Item Extraction | Automatically identify TODOs and follow-ups from conversations |
| Multi-Language Support | Improve memory extraction for non-English languages |
| Privacy Features | Memory anonymization, selective forgetting, access controls |

---

## How to Submit

> **Note:** A submission portal will be available soon. Stay tuned for updates!

### Required

| Item | Description |
|------|-------------|
| **GitHub Repository** | Public repo with your project code |
| **README.md** | Clear project introduction, setup instructions, and how EverMemOS is used |
| **Video Demo** | 3-5 minute video demonstrating your project and explaining the concept |

### Optional (Preferred)

| Item | Description |
|------|-------------|
| **Live Demo** | Deployed application or interactive demo link |

### Submission Checklist

- [ ] GitHub repository is public and accessible
- [ ] README includes project overview and setup instructions
- [ ] README explains how EverMemOS powers your solution
- [ ] Video demo is 3-5 minutes and covers functionality + concept
- [ ] (Optional) Live demo URL included in README

---

## Evaluation Criteria

Projects will be evaluated based on the following priorities:

### Innovation

We value creative and original approaches. Does your project solve a problem in a new way? Is the use of memory capabilities imaginative? Does it stand out from existing solutions?

### Technical Depth

We look for well-engineered projects with clean implementation, effective use of EverMemOS features, and appropriate technical sophistication for the problem being solved.

### Consumer Value

We prioritize projects that solve real problems for real users. Is it useful? Is it intuitive? Could it scale or be commercialized?

---

## API Cheatsheet

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/memories` | Store a message |
| GET | `/api/v1/memories` | Fetch memories by type |
| GET | `/api/v1/memories/search` | Search memories |
| DELETE | `/api/v1/memories` | Delete memories |

### Key Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `retrieve_method` | `keyword`, `vector`, `hybrid`, `rrf`, `agentic` | Search strategy |
| `memory_types` | `episodic_memory`, `profile`, `foresight`, `event_log` | Memory categories |
| `role` | `user`, `assistant` | Message sender type |

### Response Structure

```json
{
  "status": "ok",
  "message": "Success description",
  "result": {
    // Response data here
  }
}
```

---

## Sample Data

Pre-loaded sample conversations are available:

| File | Description |
|------|-------------|
| `data/group_chat_en.json` | English group chat example |
| `data/group_chat_zh.json` | Chinese group chat example |

Load sample data:
```bash
uv run python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat_en.json \
  --scene group_chat \
  --api-url http://localhost:1995/api/v1/memories
```

---

## Example Projects

| Project | Description |
|---------|-------------|
| [Game of Thrones Demo](https://github.com/EverMind-AI/evermem_got_demo) | Interactive demo comparing memory-augmented AI vs vanilla LLM responses using "A Game of Thrones" book content |

---

## Resources

### Documentation
- [API Documentation](./api_docs/memory_api.md) - Complete API reference
- [Architecture](./ARCHITECTURE.md) - System design overview
- [Group Chat Guide](./advanced/GROUP_CHAT_GUIDE.md) - Multi-user conversations

### Examples
- [Batch Operations](./usage/BATCH_OPERATIONS.md) - Processing multiple messages
- [Retrieval Strategies](./advanced/RETRIEVAL_STRATEGIES.md) - Optimizing search

### External
- [Paper](https://arxiv.org/abs/2601.02163) - Research paper
- [Discord](https://discord.gg/gYep5nQRZJ) - Community support

---

## Tips for Success

1. **Start Simple** - Get the basic flow working before adding complexity
2. **Use Hybrid Retrieval** - Best balance of speed and accuracy for most use cases
3. **Group Your Memories** - Use `group_id` for multi-user or multi-agent scenarios
4. **Check Pending Messages** - Memories may not extract immediately (boundary detection)
5. **Ask for Help** - Join our Discord for real-time support

---

Good luck and happy building!
