#!/bin/sh
# Setup tmux inside a container: install, start session, set options.
# Usage: setup-tmux.sh [proxy_url]
PROXY="$1"

# Configure apt proxy if provided
if [ -n "$PROXY" ]; then
    mkdir -p /etc/apt/apt.conf.d
    printf 'Acquire::http::Proxy "%s";\nAcquire::https::Proxy "%s";\n' "$PROXY" "$PROXY" \
        > /etc/apt/apt.conf.d/99proxy
fi

# Install tmux if not present
if ! command -v tmux >/dev/null 2>&1; then
    DEBIAN_FRONTEND=noninteractive apt-get update -qq && apt-get install -y -qq tmux 2>&1 \
        || yum install -y tmux 2>&1 \
        || apk add --no-cache tmux 2>&1 \
        || { echo "FAIL: cannot install tmux"; exit 1; }
fi

tmux -V >/dev/null 2>&1 || { echo "FAIL: tmux not found after install"; exit 1; }

# Start tmux session
export TERM=xterm-256color
export SHELL=/bin/bash
script -qc "tmux new-session -x 200 -y 100 -d -s main 'bash --login'" /dev/null \
    || { echo "FAIL: cannot start tmux session"; exit 1; }

tmux set-option -g history-limit 10000000
echo "OK"
