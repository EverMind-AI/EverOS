#!/usr/bin/env bash
# Install EverOS memory hooks into the current project (.cursor/hooks/everos-memory).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-.cursor/hooks/everos-memory}"

mkdir -p "$TARGET"
cp -R "$ROOT/hooklib" "$TARGET/"
cp "$ROOT/hooks/"*.py "$TARGET/"
chmod +x "$TARGET"/*.py

ENV_TARGET="${2:-.env}"
if [[ ! -f "$ENV_TARGET" ]] && [[ -f "$ROOT/env.example" ]]; then
  cp "$ROOT/env.example" "$ENV_TARGET"
  echo "Created $ENV_TARGET from env.example — fill in EverOS settings if needed."
fi

HOOKS_JSON=".cursor/hooks.json"
if [[ ! -f "$HOOKS_JSON" ]]; then
  cp "$ROOT/hooks/hooks.json.example" "$HOOKS_JSON"
  echo "Created $HOOKS_JSON"
else
  echo ""
  echo "NOTE: $HOOKS_JSON already exists."
  echo "Merge the entries from:"
  echo "  $ROOT/hooks/hooks.json.example"
  echo "Paths assume hooks live at: $TARGET"
fi

echo ""
echo "Installed EverOS Cursor hooks to: $TARGET"
echo "Next: start EverOS (everos server start), enable hooks in Cursor, open a new Agent chat."
