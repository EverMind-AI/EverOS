---
description: View EverMem debug logs to troubleshoot memory saving and retrieval issues
---

# EverMem Debug Log Viewer

View the EverMem debug log to troubleshoot issues.

## Instructions

Show the user the recent debug log entries from `/tmp/evermem-debug.log`.

1. First check if debug mode is enabled by looking for `EVERMEM_DEBUG=1` in the plugin's `.env` file
2. Read the last 50 lines of the debug log file
3. If the file doesn't exist or is empty, inform the user how to enable debug mode

## Actions

1. Check debug mode status:
   ```bash
   grep "EVERMEM_DEBUG" /path/to/plugin/.env 2>/dev/null || echo "Not configured"
   ```

2. Show recent logs:
   ```bash
   tail -50 /tmp/evermem-debug.log 2>/dev/null || echo "No debug log found"
   ```

3. Format the output for the user, highlighting:
   - `[inject]` entries for memory retrieval
   - `[store]` entries for memory saving
   - Any errors or warnings

## Output Format

```
📋 EverMem Debug Log

Status: Debug mode [ENABLED/DISABLED]
Log file: /tmp/evermem-debug.log

--- Recent Entries ---
[timestamp] [inject] ...
[timestamp] [store] ...

--- Tips ---
• Enable debug: Add EVERMEM_DEBUG=1 to .env
• Clear log: > /tmp/evermem-debug.log
• Live view: tail -f /tmp/evermem-debug.log
```

## Additional Options

If the user specifies arguments:
- `clear` - Clear the debug log
- `live` - Show command for live monitoring
- `full` - Show more lines (100+)
- `inject` - Filter to show only [inject] entries
- `store` - Filter to show only [store] entries
