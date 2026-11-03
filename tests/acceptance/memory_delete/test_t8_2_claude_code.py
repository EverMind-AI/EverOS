"""T8.2 <- A8 (main journey): in Claude Code, remember, forget, search."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from .observe import md_entries, wait_idle

pytestmark = pytest.mark.acceptance

_REMEMBER = (
    "Use only the everos tools and nothing else. Call add_memory with "
    "user_message 'Please remember: my favourite tea is {token}.' and wait=true. "
    "Then report exactly what the tool returned."
)
_FORGET = (
    "Use only the everos tools, in this order, and nothing else. "
    "1. Call search_memory with query 'favourite tea {token}'. "
    "2. Call forget_session. 3. Call search_memory with query "
    "'favourite tea {token}'. Then report, for each call, exactly what the tool "
    "returned."
)


def _claude(prompt: str, config: Path, cwd: Path) -> tuple[list[str], list[str], str]:
    """Run one Claude Code turn; the everos tools it called, their results in
    order, and the raw stream."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("EVEROS_")}
    done = subprocess.run(
        [
            "claude", "-p", prompt,
            "--mcp-config", str(config), "--strict-mcp-config",
            "--allowedTools", "mcp__everos__add_memory", "mcp__everos__forget_session",
            "mcp__everos__search_memory",
            "--output-format", "stream-json", "--verbose", "--no-session-persistence",
        ],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
    )  # fmt: skip
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
    calls = [names[i].removeprefix("mcp__everos__") for i in everos]
    return (
        calls,
        [results.get(i, "") for i in everos],
        done.stdout + done.stderr[-2000:],
    )


def test_t8_2_forget_this_conversation_in_claude_code(
    server: Any, tmp_path: Path
) -> None:
    """Remember in one turn; once indexed, search (must find it), forget,
    search again (must not) in the next. The first search is the positive
    control: without it an empty search proves nothing."""
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
    calls1, texts1, raw1 = _claude(_REMEMBER.format(token=token), config, tmp_path)
    wait_idle(server.base, server.root)
    calls2, texts2, raw2 = _claude(_FORGET.format(token=token), config, tmp_path)
    (tmp_path / "claude_stream.jsonl").write_text(raw1 + raw2)
    print(f"T8.2 turn 1: {list(zip(calls1, texts1, strict=True))}")
    print(f"T8.2 turn 2: {list(zip(calls2, texts2, strict=True))}")

    assert calls1 == ["add_memory"], (calls1, raw1[-2000:])
    assert calls2 == ["search_memory", "forget_session", "search_memory"], (
        calls2,
        raw2[-2000:],
    )
    assert token in texts2[0], ("positive control: search found nothing", texts2[0])
    forget = texts2[1]
    assert "does not support deleting" not in forget, forget
    assert "Deleted" in forget and "Deleted 0 " not in forget, forget
    assert token not in texts2[2], texts2[2]
    assert not [e for e in md_entries(server.root) if e["session_id"] == "cc-s1"]
