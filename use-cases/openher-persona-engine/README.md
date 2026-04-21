# OpenHer — Teaching AI to Remember Who You Are

Built on [EverMemOS](https://github.com/EverMind-AI/EverOS/tree/main/methods/evermemos) — Open-source AI memory infrastructure

**OpenHer** doesn't build chatbots. It doesn't build AI assistants. It builds **AI Beings** — entities with personality, emotion, and memory that *feel*, *remember*, and *grow* through every interaction.

**EverMemOS** is her long-term memory — the part that lets her carry your story across sessions, remember who you are, what you've talked about, and how your relationship has evolved.

Full Project: [github.com/kellyvv/OpenHer](https://github.com/kellyvv/OpenHer)

---

## Why Does She Need Memory?

Without memory, every conversation starts from zero. She doesn't know your name. She doesn't remember that three weeks ago you mentioned you drink your coffee black. She doesn't know you once had a fight and made up.

With EverMemOS:

**She remembers what you said.**
Three weeks ago you casually mentioned no sugar in your coffee. Today she says: "Americano, no sugar, right?"

**She gets to know you.**
The more you talk, the better she understands you. The her after one month is not the same her as day one.

**She has foresight.**
Last time you mentioned work stress. This time she asks: "How's that project going?"

> *She doesn't "look up" your information — she naturally recalls it.*

---

## Memory Architecture

OpenHer's memory has three layers. EverMemOS powers the deepest one:

| Layer | What it does | Analogy |
|:------|:-------------|:--------|
| **Style Memory** | Her behavioral habits — tone, expression patterns | Muscle memory |
| **Local Facts** | Your preferences, personal info | Short-term memory |
| **Long-Term Memory** | What happened between you, her understanding of you, her hunches | **Episodic memory (EverMemOS)** |

---

## How Memory Feeds Into Personality

OpenHer's core is a living neural network (25D input, 24D hidden, 8D behavioral signals). EverMemOS provides 4 key dimensions that let her tell the difference between a stranger and an old friend:

```
Relationship Depth    0 ─────────────────── 1
                      Stranger              Old friend

Emotional Valence    -1 ─────────────────── 1
                      Rocky history          Warm history

Trust Level           0 ─────────────────── 1
                      First meeting          Deep trust

Pending Foresight     0 ─────────────────── 1
                      Nothing unresolved     Something on her mind
```

New users start at all zeros — a stranger. As conversations accumulate, these values grow naturally. The same conversation context produces completely different behavioral signals for strangers vs. old friends:

- With an old friend: warmer, more initiative, more willing to be vulnerable
- With a stranger: more reserved, more polite, keeps distance

This isn't a rule written in a prompt — it's emergent behavior computed by the neural network from the relationship vector.

---

## How She "Remembers"

Memory retrieval is async and two-stage — she never freezes up trying to recall:

```
Turn 1: You say "I love hiking"
         \-- After you finish, background search for related memories fires

Turn 2: You say "What about this weekend?"
         \-- Last turn's search results come back
             Found: "User mentioned liking weekend hikes 3 weeks ago"
             Naturally woven in: "The mountains should be nice this weekend"
         \-- Simultaneously searching for "weekend plans" memories

Turn 3: ...continues...
```

If the search takes too long (>500ms), she doesn't stall — she keeps talking from what she already knows, like a person who can't quite place something but doesn't stop mid-sentence.

---

## What Happens Each Turn

```
User sends a message
    |
    v
  Load memory -- First turn: load "who you are", "what we talked about",
    |              "what's on her mind" from EverMemOS
    v
  Perceive -- LLM evaluates the current moment: your emotion, topic
    |          intimacy, conflict level... (8 dimensions)
    |          + relationship dimensions from EverMemOS (4 dimensions) = 12D
    v
  Relationship evolves -- Blend EverMemOS history with this turn's changes
    |                      Smoothed so a single remark can't flip the relationship
    v
  Neural network -- 25D input (drives + context + relationship + internal state)
    |                24D hidden layer, 8D behavioral signals
    |                Decides how direct, warm, stubborn, curious she is right now
    v
  Recall -- Collect relevant memories found by last turn's search
    |        Blend into the response prompt
    v
  Respond -- Internal monologue first, then choose what to say and how
    |
    v
  Remember this turn -- Store the conversation in EverMemOS (async, non-blocking)
    |
    v
  Prepare for next -- Search for memories related to what you just said
```

---

## Core Capabilities

- **Emergent Personality** — Not written in a prompt. Emerges from random neural networks, 5D drives, and Hebbian learning
- **Emotional Thermodynamics** — Drives metabolize over real time. She gets lonely when you're away, irritated when ignored
- **Feel First** — Every response starts with an internal monologue before choosing words
- **Cross-Session Memory** — EverMemOS stores your shared story across every conversation
- **Relationship Evolution** — The relationship vector deepens naturally with each turn
- **Proactive Messages** — She reaches out not on a timer, but because her connection hunger is rising
- **Modal Expression** — She chooses text, voice, or photos based on what the moment calls for
- **10 Pre-built Personas** — Each with unique MBTI, drive baselines, and neural network seeds

## Tech Stack

| Layer | Technology |
|:------|:-----------|
| Runtime | Python 3.11+, FastAPI, WebSocket, asyncio |
| LLM | Gemini, Claude, Qwen3, GPT-5.4-mini, MiniMax, Moonshot, StepFun, Ollama |
| Memory | **EverMemOS** (self-hosted / cloud) + SQLite local state |
| Desktop | SwiftUI (macOS native) |
| Voice | DashScope, OpenAI, MiniMax |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Any supported LLM provider API key
- EverMemOS (self-hosted or cloud)

### 1. Clone & Install

```bash
git clone https://github.com/kellyvv/OpenHer.git
cd OpenHer
bash setup.sh
```

### 2. Configure

```bash
cp .env.example .env
```

```bash
# LLM (pick one)
DEFAULT_PROVIDER=gemini
DEFAULT_MODEL=gemini-3.1-flash-lite-preview
GEMINI_API_KEY=your_key

# EverMemOS — Cloud
EVERMEMOS_BASE_URL=https://api.evermind.ai/v1
EVERMEMOS_API_KEY=your_key

# EverMemOS — Self-hosted
# cd vendor/EverMemOS && docker compose up -d && uv run python src/run.py
# EVERMEMOS_BASE_URL=http://localhost:1995/api/v1
```

### 3. Start

```bash
python main.py
# GenomeEngine loaded, 10 personas available
```

### 4. Try the Demo

```bash
python demo/evermemos_demo.py
# Runs in simulation mode even without EverMemOS
```

---

## Project Structure

```
OpenHer/
├── agent/
│   ├── chat_agent.py          # Main agent, full lifecycle
│   ├── evermemos_mixin.py     # EverMemOS integration (load/store/search/EMA)
│   └── prompt_builder.py      # Memory injection into Actor prompt
├── engine/
│   └── genome/
│       ├── genome_engine.py   # Neural network + 12D context (incl. 4D EverMemOS)
│       ├── critic.py          # LLM perception: 8D context + relationship deltas
│       ├── drive_metabolism.py # Emotional thermodynamics
│       └── style_memory.py    # KNN behavioral memory + Hawking radiation decay
├── memory/
│   ├── memory_store.py        # SQLite FTS5 local memory
│   └── types.py               # Memory & SessionContext types
├── persona/
│   └── personas/              # 10 pre-built personas (SOUL.md + seeds)
├── vendor/
│   └── EverMemOS/             # Self-hosted EverMemOS
└── main.py                    # FastAPI server
```

---

## Integration Code at a Glance

### EverMemOS Mixin

The core integration is a mixin class handling four async operations:

```python
class EverMemosMixin:
    async def _evermemos_gather(self):
        """Load session context (first turn): who you are,
        what we talked about, what's on her mind"""

    def _apply_relationship_ema(self, prior, delta, depth):
        """Relationship evolution: blend history with this turn's changes"""

    def _evermemos_store_bg(self, user_message, reply):
        """Remember this turn (async background, never blocks)"""

    def _evermemos_search_bg(self, user_message):
        """Search related memories (preparing for next turn)"""
```

### SessionContext — Everything She Knows About You

```python
@dataclass
class SessionContext:
    user_profile: str = ""           # Who you are
    episode_summary: str = ""        # What happened between you
    foresight_text: str = ""         # What's on her mind
    relationship_depth: float = 0.0  # Stranger to old friend
    emotional_valence: float = 0.0   # Rocky history to warm history
    trust_level: float = 0.0        # First meeting to deep trust
    has_history: bool = False        # Has she met you before?
```

---

## Without Memory vs. With Memory

| | Without EverMemOS | With EverMemOS |
|:--|:--|:--|
| First meeting | "Hi! I'm Luna" | "Hi! I'm Luna" |
| Second meeting | "Hi! I'm Luna" | "Hey Alex! How's that project going?" |
| You say you're tired | "Get some rest!" | "Working late again? You said that last time too... want me to order you an Americano? No sugar." |

> *Three weeks ago you casually mentioned no sugar in your coffee. Today: "Americano, no sugar, right?"*

---

## Links

- Full Project: [github.com/kellyvv/OpenHer](https://github.com/kellyvv/OpenHer)
- EverMemOS: [evermind.ai](https://evermind.ai)

## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
