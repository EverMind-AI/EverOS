#!/bin/bash
# EverMem SessionStart Hook Wrapper
# Ensures npm dependencies are installed before running the hook

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Check if SDK is installed, if not install it silently
if [ ! -d "$PLUGIN_ROOT/node_modules/@anthropic-ai/claude-agent-sdk" ]; then
  (cd "$PLUGIN_ROOT" && npm install --silent 2>/dev/null) || true
fi

# Run the actual hook script, passing stdin through
exec node "$SCRIPT_DIR/session-context.js"
