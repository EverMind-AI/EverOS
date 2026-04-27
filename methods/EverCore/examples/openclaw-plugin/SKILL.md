---
name: evermind-ai-everos
version: 1.5.2
description: |
  Install and configure EverOS for OpenClaw natural-language memory.

  Use when users say:
  - "install everos"
  - "setup everos"
  - "install everos plugin"
  - "enable everos memory"
  - "remember my preferences in OpenClaw"
author: EverMind
keywords:
  - everos
  - context engine
  - persistent memory
  - openclaw
  - natural language memory
---

# EverOS

EverOS OpenClaw Plugin gives OpenClaw persistent memory through the **ContextEngine API**.

Important distinction:

- This is a `context-engine` plugin, not a `memory` slot plugin.
- Users do not need to call memory tools manually.
- Memory is triggered by normal conversation:
  - before reply: relevant memory is recalled and injected
  - after reply: new conversation content is saved back automatically

---

## Trigger phrases

Use this skill when the user wants to:

- install EverOS
- set up EverOS memory for OpenClaw
- enable long-term memory for OpenClaw
- make natural-language conversations persist across sessions
- configure the EverOS context engine

---

## When to use this skill

Use this skill when the user wants:

- persistent memory across OpenClaw sessions
- automatic recall and save during normal conversation
- a self-hosted EverOS backend
- memory without manually calling `memory_store` / `memory_search`

---

## When NOT to use this skill

Do not use this skill for:

- cloud-only memory onboarding
- troubleshooting unrelated OpenClaw issues
- temporary context that only matters in the current session

---

## What the user gets

Automatic lifecycle behavior:

| Hook | Trigger | What happens |
| --- | --- | --- |
| `bootstrap()` | Session starts | Backend health check and session state init |
| `assemble()` | Before each turn | Searches relevant memory and injects it as context |
| `afterTurn()` | After each turn | Saves new messages from the turn |
| `compact()` | Compaction check | Participates in token-budget decisions |
| `dispose()` | Session ends | Clears in-memory session state |

User-facing result:

- the user chats naturally
- EverOS recalls relevant past context automatically
- EverOS stores new useful context automatically

---

## Definition of Done

This task is not complete until all of the following are true:

1. the EverOS backend is reachable or the user has been explicitly told it still needs to be started
2. the plugin is installed
3. `openclaw.json` is configured correctly
4. OpenClaw has been restarted
5. the user has been given a natural-language verification step
6. the full final handoff message has been sent

---

## Onboarding

### Step 0 - Set expectations

Before changing anything, tell the user:

> OpenClaw will need a restart during setup. After restart, wait about 1 minute, then send a short natural-language message here to verify memory.

Do not skip this notice if you are going to restart OpenClaw automatically.

### Step 1 - Check the EverOS backend

Default backend:

```bash
curl -sf http://localhost:1995/health && echo "OK" || echo "NOT_RUNNING"
```

If the backend is healthy, continue.

If it is not healthy, tell the user plainly:

> The EverOS backend is not reachable yet. I can still finish plugin installation now, but automatic recall/save will not work until the backend is running.

If the user asks how to start the backend, give the shortest useful path:

```bash
git clone https://github.com/EverMind-AI/EverOS.git
cd EverOS
docker compose up -d
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
cp env.template .env
# edit .env
uv run python src/run.py
```

### Step 2 - Install plugin

Preferred one-shot install:

```bash
npx --yes --package @evermind-ai/openclaw-plugin everos-install
```

Manual alternative:

```bash
npm install -g @evermind-ai/openclaw-plugin
everos-install
```

What the installer does:

- adds the plugin path to `plugins.load.paths`
- adds `evermind-ai-everos` to `plugins.allow`
- sets `plugins.slots.contextEngine = "evermind-ai-everos"`
- sets `plugins.slots.memory = "none"` to avoid slot conflicts
- creates or updates `plugins.entries["evermind-ai-everos"]`

### Step 3 - Manual config fallback

If the installer is unavailable, patch `~/.openclaw/openclaw.json` manually.

Expected config shape:

```json
{
  "plugins": {
    "allow": ["evermind-ai-everos"],
    "slots": {
      "memory": "none",
      "contextEngine": "evermind-ai-everos"
    },
    "entries": {
      "evermind-ai-everos": {
        "enabled": true,
        "config": {
          "baseUrl": "http://localhost:1995",
          "userId": "everos-user",
          "groupId": "everos-group",
          "topK": 5,
          "memoryTypes": ["episodic_memory"],
          "retrieveMethod": "hybrid"
        }
      }
    }
  }
}
```

Merge-safe patch:

```bash
jq '
  .plugins = (.plugins // {}) |
  .plugins.load = (.plugins.load // {}) |
  .plugins.load.paths = ((.plugins.load.paths // []) + ["/path/to/everos-openclaw-plugin"] | unique) |
  .plugins.allow = ((.plugins.allow // []) + ["evermind-ai-everos"] | unique) |
  .plugins.slots = (.plugins.slots // {}) |
  .plugins.slots.memory = "none" |
  .plugins.slots.contextEngine = "evermind-ai-everos" |
  .plugins.entries = (.plugins.entries // {}) |
  .plugins.entries["evermind-ai-everos"].enabled = true |
  .plugins.entries["evermind-ai-everos"].config = (
    (.plugins.entries["evermind-ai-everos"].config // {}) + {
      "baseUrl": "http://localhost:1995",
      "userId": "everos-user",
      "groupId": "everos-group",
      "topK": 5,
      "memoryTypes": ["episodic_memory"],
      "retrieveMethod": "hybrid"
    }
  )
' ~/.openclaw/openclaw.json > tmp.json && mv tmp.json ~/.openclaw/openclaw.json
```

### Step 4 - Restart OpenClaw

Restart command:

```bash
openclaw gateway restart
```

Immediately before restart, tell the user:

> EverOS is installed. I am restarting OpenClaw now. After about 1 minute, send a short message so we can verify memory recall.

### Step 5 - Verify

Verification has two parts.

Backend:

```bash
curl http://localhost:1995/health
```

User-facing natural-language test:

> Say: "Remember: I like espresso."
>
> Then ask: "What coffee do I like?"

This is the preferred validation because it checks the real user flow instead of just config.

---

## Final handoff

After successful setup, send this handoff message in the user's language.
Do not remove sections.

```text
EverOS is ready.

-- WHAT YOU CAN DO NEXT --

From now on, you can use normal natural language to make OpenClaw remember information.
You do not need to call memory tools manually.

Examples:
- "Remember: I like espresso."
- "Remember: this project uses PostgreSQL by default."
- "My coding style prefers small functions and explicit naming."

Later you can ask:
- "What coffee do I like?"
- "What database does this project use by default?"

-- CURRENT CONNECTION --

EverOS backend:
BASE_URL: <base-url>

OpenClaw config file:
~/.openclaw/openclaw.json

-- RECOVERY --

1. Keep your EverOS backend data and configuration
2. Reinstall this plugin on the new machine
3. Write the same `baseUrl`, `userId`, and `groupId` back into `openclaw.json`
4. Restart OpenClaw to reconnect to the same memory space

-- BACKUP --

- Back up `~/.openclaw/openclaw.json`
- Back up the EverOS backend data directory or database
- Back up the EverOS `.env` and deployment configuration
```

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Plugin not loading | Check `plugins.allow`, `plugins.load.paths`, and `plugins.slots.contextEngine` |
| Backend unhealthy | Check `baseUrl` and ensure the EverOS backend is running |
| No recall | Verify the backend contains memories and the query is meaningful |
| No save | Verify `afterTurn()` is running and backend write API is reachable |
| Memory plugin conflict | Make sure `plugins.slots.memory = "none"` |

---

## API reference

Base: `http://localhost:1995`

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Health check |
| POST | `/api/v1/memories` | Save memory |
| GET | `/api/v1/memories/search` | Search memory |
| DELETE | `/api/v1/memories` | Delete memory |

---

## Communication style

When talking to users:

- say this is automatic natural-language memory
- do not describe it as a `memory` slot plugin
- keep the next step concrete: restart, then try one short memory sentence
- prefer real conversational verification over low-level API demos
