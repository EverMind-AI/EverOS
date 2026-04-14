#!/bin/sh
# tmux-run: send command, wait up to N seconds, capture NEW output only
CMD="$1"
WAIT="${2:-3}"
MAX=600
[ "$WAIT" -gt "$MAX" ] 2>/dev/null && WAIT=$MAX

# Special command: send Ctrl-C to recover from stuck states
if [ "$CMD" = "ctrl-c" ]; then
    tmux send-keys -t main C-c
    sleep 0.5
    tmux capture-pane -t main -p | tail -5
    exit 0
fi

# Syntax check: reject commands that would enter PS2 (unclosed quote/heredoc/bracket)
if [ -n "$CMD" ]; then
    if ! sh -n -c "$CMD" 2>/dev/null; then
        echo "[ERROR] Syntax error: unclosed quote, heredoc, or bracket. Use WRAPPER_PATH write to create files."
        exit 1
    fi
fi

# Send command with marker
MARK=""
if [ -n "$CMD" ]; then
    MARK="__MRK_$(date +%s)_$$__"
    tmux send-keys -t main "echo $MARK" Enter
    sleep 0.2
    tmux send-keys -t main "$CMD" Enter
    sleep 1
    W=1
else
    W=0
fi

# Wait loop: use pane_current_command to detect if command finished
while [ "$W" -lt "$WAIT" ]; do
    sleep 1
    W=$((W + 1))
    PANE_CMD=$(tmux display-message -t main -p '#{pane_current_command}' 2>/dev/null)
    case "$PANE_CMD" in
        bash|sh|zsh|fish|"") break ;;
    esac
done

# Check if command is still running after wait
PANE_CMD=$(tmux display-message -t main -p '#{pane_current_command}' 2>/dev/null)
case "$PANE_CMD" in
    bash|sh|zsh|fish|"") ;;
    *)
        echo "[STILL RUNNING: $PANE_CMD] (waited ${W}s)"
        echo "Latest output:"
        tmux capture-pane -t main -p | tail -5
        exit 0
        ;;
esac

# Capture output
TMPF="/tmp/.tmux_cap_$$"
if [ -n "$MARK" ]; then
    tmux capture-pane -t main -p -S -2000 > "$TMPF"
    MARK_LINE=$(grep -n "$MARK" "$TMPF" | tail -1 | cut -d: -f1)
    if [ -n "$MARK_LINE" ]; then
        tail -n "+$((MARK_LINE + 1))" "$TMPF" > "${TMPF}.out"
    else
        # Marker lost: fallback to last 50 lines of scrollback
        tmux capture-pane -t main -p -S -50 > "${TMPF}.out"
    fi
    LINES=$(wc -l < "${TMPF}.out")
    if [ "$LINES" -gt 60 ]; then
        OMIT=$((LINES - 55))
        head -5 "${TMPF}.out"
        echo "... ($OMIT lines omitted) ..."
        tail -50 "${TMPF}.out"
    else
        cat "${TMPF}.out"
    fi
    rm -f "$TMPF" "${TMPF}.out"
else
    tmux capture-pane -t main -p
fi
