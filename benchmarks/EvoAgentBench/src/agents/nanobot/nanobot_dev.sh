#!/bin/bash
# Run nanobot from local source instead of installed package.
# Usage: set command in nanobot.yaml to this script's path:
#   command: ./src/agents/nanobot/nanobot_dev.sh
# Requires nanobot source at the PYTHONPATH below.
# Set NANOBOT_SRC to your local nanobot source directory
NANOBOT_SRC="${NANOBOT_SRC:-$(dirname "$0")/../../../nanobot}"
PYTHONPATH="$NANOBOT_SRC" exec python3 -m nanobot "$@"
