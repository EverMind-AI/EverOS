#!/bin/bash
# Run openclaw from local source instead of installed package.
# Usage: set command in openclaw.yaml to this script's path:
#   command: ./src/agents/openclaw/openclaw_dev.sh
# Requires openclaw source at the path below.
# Set OPENCLAW_SRC to your local openclaw source directory
OPENCLAW_SRC="${OPENCLAW_SRC:-$(dirname "$0")/../../../openclaw}"
exec node "$OPENCLAW_SRC/openclaw.mjs" "$@"
