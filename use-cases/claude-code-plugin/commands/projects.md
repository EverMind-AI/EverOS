---
description: View your Claude Code projects tracked by EverMem
---

# EverMem Projects
View all Claude Code projects that have been tracked by EverMem.

## Instructions

Show the user their projects stored in the local groups.jsonl file.

1. Read the groups file from the plugin's data directory
2. Aggregate entries by groupId (count sessions, find first/last seen)
3. Display the project table with statistics
4. If no groups file exists, explain that projects are tracked automatically

## Actions

Check and read the groups data file:

```bash
GROUPS_FILE="${CLAUDE_PLUGIN_ROOT}/data/groups.jsonl"
if [ -f "$GROUPS_FILE" ] && [ -s "$GROUPS_FILE" ]; then
  cat "$GROUPS_FILE"
else
  echo "NO_GROUPS_FILE"
fi
```

**Note:** The file uses JSONL format (one JSON object per line). Each line is a session start event.

Entry format: `{"keyId":"...","groupId":"...","name":"...","path":"...","timestamp":"..."}`

- `keyId`: SHA-256 hash (first 12 chars) of the API key - associates projects with accounts
- `groupId`: Short identifier (9 chars: project name prefix + path hash)

Aggregate by `keyId + groupId` when displaying:
- Count occurrences = sessionCount
- Earliest timestamp = firstSeen
- Latest timestamp = lastSeen

## Output Format

If projects exist:
```
📁 Claude Code Projects

| Project             | Group ID   | Sessions | Last Active |
|---------------------|------------|----------|-------------|
| evermem-claude-code | ever8d8d5  | 42       | just now    |
| my-react-app        | myrea1b2c3 | 12       | 2h ago      |

Total: 2 projects
```

If no projects file:
```
📁 Claude Code Projects

No projects tracked yet. Projects are automatically recorded when you start Claude Code sessions.

Each project directory creates a unique group ID for organizing memories.
```

## Notes

- Projects are identified by working directory path (hashed to 9-char ID)
- Each project has its own memory namespace in EverMem Cloud
- The groups.jsonl file is appended by the SessionStart hook
- Same project used with different API keys will appear as separate entries
- `keyId` is a SHA-256 hash (first 12 chars) of the API key - secure and unique
