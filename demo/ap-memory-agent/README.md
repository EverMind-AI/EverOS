# AP Memory Agent

Accounts Payable automation agent built with **LangGraph + EverMemOS + Claude** for the [Memory Genesis Competition 2026](https://luma.com/n88icl03?tk=aawakR).

**Core demo moment:** An agent catches a fraudulent or duplicate invoice because it remembers what happened weeks ago.

## Demo Video

<a href="https://www.loom.com/share/9834fe7b5fca45c78c304f7101b15ac6">
  <img src="https://cdn.loom.com/sessions/thumbnails/9834fe7b5fca45c78c304f7101b15ac6-31131ab95b0f7652.jpg" width="600" />
</a>

[Watch the full demo on Loom](https://www.loom.com/share/9834fe7b5fca45c78c304f7101b15ac6)

## Architecture

```
Streamlit UI (invoice form)
        ↓
LangGraph Orchestrator
    ├── Node 1: invoice_agent     → normalise invoice, write episodic_memory, fetch vendor context
    ├── Node 2: risk_agent         → Claude reasons over invoice + memory context, returns risk flags
    ├── Node 3: approval_agent     → Claude makes approve/hold/reject decision with reasoning
    └── Node 4: memory_updater     → write resolution to episodic_memory, update profile if risk ≥ 40
                ↕                        ↕                    ↕                       ↕
                            EverMemOS (Shared Memory Bus)
```

### Memory Strategy

- **Invoice Agent:** Writes each invoice event as episodic memory; searches episodic + event_log (agentic retrieval); fetches profile separately; merges into `memory_context`.
- **Memory Updater:** Always writes resolution as episodic memory; conditionally writes profile when risk_score ≥ 40 or decision is hold/reject
- **Profile** is not searchable — fetched via GET /memories and merged manually

## Setup

1. **Prerequisites**
   - Python 3.11+
   - EverMemOS running locally (default: `http://localhost:1995`)

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your ANTHROPIC_API_KEY
   # EVERMEM_GROUP_ID is required for seeded vendor history
   ```

4. **Seed vendor history**
   ```bash
   cd demo/ap-memory-agent
   python seed_memory.py
   # Or: python seed_memory.py --clear   # Clear and re-seed
   ```
   From EverMemOS root: `
   `

   **Extraction timing:** EverMemOS uses boundary detection — messages are extracted into episodic memories when a "conversation episode" is complete. The seed sends user+assistant pairs with `sync_mode` to trigger extraction. Expect 1–2 minutes for the full seed. Episodic memories appear after boundary detection runs.

5. **Run the UI**
   ```bash
   cd demo/ap-memory-agent
   streamlit run streamlit_app.py
   ```
   From EverMemOS root: `uv run streamlit run demo/ap-memory-agent/streamlit_app.py`

## Demo Script (for judges)

1. Run `seed_memory.py` to populate vendor history
2. Open Streamlit UI
3. **Acme Corp** — Submit invoice `INV-2025-0042` for $500 → should **APPROVE** (clean history)
4. **Globex Supplies** — Submit invoice `INV-2025-0105` for $1,200 → should **HOLD** (prior dispute)
5. **Shadow LLC** — Submit invoice `INV-777` for $3,000 → should **REJECT** (duplicate caught by memory)

## Test Examples

### Streamlit UI (manual form)

| Test | Vendor Name | Invoice Number | Amount | Expected |
|------|-------------|----------------|--------|----------|
| 1 | Acme Corp | INV-2025-0042 | 500 | APPROVE |
| 2 | Globex Supplies | INV-2025-0105 | 1200 | HOLD |
| 3 | Shadow LLC | INV-777 | 3000 | REJECT |
| 4 | Acme Corp | INV-2025-0099 | 550 | APPROVE |

### CLI test script

```bash
uv run python demo/ap-memory-agent/test_demo.py
```

Runs all 4 test invoices through the graph and prints decisions.

## Project Structure

```
ap-memory-agent/
├── .env                    # API keys (never commit)
├── .env.example            # Template for env vars
├── requirements.txt       # Dependencies
├── ap_agent_graph.py      # LangGraph pipeline (4 nodes)
├── evermem_client.py      # EverMemOS API wrapper
├── seed_memory.py         # Pre-populate vendor history
├── streamlit_app.py       # Streamlit UI
└── README.md
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Claude API key | required |
| `EVERMEM_BASE_URL` | EverMemOS URL | `http://localhost:1995` |
| `EVERMEM_USER_ID` | Memory scope | `ap_agent_system` |
| `EVERMEM_GROUP_ID` | Group for seeded data | `eb6618c4d52d3bf9_group` |

## EverMemOS API Notes

- **POST /api/v1/memories** — Store message (MemorizeMessageRequest: message_id, create_time, sender, content, role)
- **GET /api/v1/memories** — Fetch by memory_type (profile, episodic_memory, etc.)
- **GET /api/v1/memories/search** — Search episodic_memory, event_log, foresight (profile NOT supported)
- **retrieve_method="agentic"** — Use for agent queries (LLM-guided retrieval)
