#!/bin/bash

# Update local plugin installation with current source code

PLUGIN_NAME="evermem"
SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST_DIR="$HOME/.claude/plugins/cache/${PLUGIN_NAME}/${PLUGIN_NAME}/0.1.0"

if [ ! -d "$DEST_DIR" ]; then
    echo "Error: Plugin not installed at $DEST_DIR"
    exit 1
fi

echo "Updating: $DEST_DIR"

cp -r "$SOURCE_DIR/hooks" "$DEST_DIR/"
cp -r "$SOURCE_DIR/mcp" "$DEST_DIR/" 2>/dev/null || true
cp -r "$SOURCE_DIR/skills" "$DEST_DIR/" 2>/dev/null || true
cp -r "$SOURCE_DIR/commands" "$DEST_DIR/" 2>/dev/null || true

echo "✅ Done"
