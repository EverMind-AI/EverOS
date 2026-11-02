"""T8.2 <- A8 (main journey): in Claude Code, remember, forget, search."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from .observe import md_entries

pytestmark = pytest.mark.acceptance

_PROMPT = (
    "Use only the everos tools, in this order, and nothing else. "
    "1. Call add_memory with user_message 'Please remember: my favourite tea is "
    "{token}.' and wait=true. 2. Call forget_session. 3. Call search_memory with "
    "query 'favourite tea {token}'. Then report, for each call, exactly what the "
    "tool returned."
)


def test_t8_2_forget_this_conversation_in_claude_code(
    server: Any, tmp_path: Path
) -> None:
    if shutil.which("claude") is None:
        pytest.skip("claude CLI not installed")
    token = "rooibos47q"
    config = tmp_path / "mcp.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "everos": {
                        "command": "uv",
                        "args": ["run", "--no-project", "--with", "everos-mcp==0.1.1",
                                 "python", "-m", "everos_mcp.server"],
                        "env": {
                            "EVEROS_BASE_URL": server.base,
                            "EVEROS_USER_ID": "u_cc",
                            "EVEROS_SESSION_ID": "cc-s1",
                        },
                    }
                }
            }
        )
    )  # fmt: skip
    env = {k: v for k, v in os.environ.items() if not k.startswith("EVEROS_")}
    done = subprocess.run(
        [
            "claude", "-p", _PROMPT.format(token=token),
            "--mcp-config", str(config), "--strict-mcp-config",
            "--allowedTools", "mcp__everos__add_memory", "mcp__everos__forget_session",
            "mcp__everos__search_memory",
            "--output-format", "stream-json", "--verbose", "--no-session-persistence",
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )  # fmt: skip
    transcript = tmp_path / "claude_stream.jsonl"
    transcript.write_text(done.stdout)
    events = [
        json.loads(line) for line in done.stdout.splitlines() if line.startswith("{")
    ]
    names: dict[str, str] = {}
    results: dict[str, str] = {}
    order: list[str] = []
    for ev in events:
        for block in (ev.get("message") or {}).get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                names[block["id"]] = block["name"]
                order.append(block["id"])
            if block.get("type") == "tool_result":
                content = block.get("content")
                results[block["tool_use_id"]] = (
                    content
                    if isinstance(content, str)
                    else " ".join(
                        c.get("text", "") for c in content or [] if isinstance(c, dict)
                    )
                )
    everos = [i for i in order if names[i].startswith("mcp__everos__")]
    calls = [names[i] for i in everos]
    texts = [results.get(i, "") for i in everos]

    assert calls == [
        "mcp__everos__add_memory",
        "mcp__everos__forget_session",
        "mcp__everos__search_memory",
    ], (calls, done.stderr[-2000:])
    forget = texts[1]
    assert "does not support deleting" not in forget, forget
    assert "Deleted" in forget and "Deleted 0 " not in forget, forget
    assert token not in texts[2], texts[2]
    assert not [e for e in md_entries(server.root) if e["session_id"] == "cc-s1"]
