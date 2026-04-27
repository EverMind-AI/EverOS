---
description: Search EverMem for relevant memories from past sessions
arguments:
  - name: query
    description: The search query to find relevant memories
    required: true
---

Search EverMem Cloud for memories matching the user's query.

Run this command to search:

```bash
node "${CLAUDE_PLUGIN_ROOT}/commands/scripts/search-memories.js" "$ARGUMENTS"
```

After the search completes, summarize the key findings for the user. Highlight the most relevant memories and explain how they might be useful for their current work.
