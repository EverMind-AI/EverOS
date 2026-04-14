"""
Sandbox path layout + Node bridge invocation for the OpenClaw adapter.

The bridge is a Node script that speaks the BridgeCommand/BridgeResponse
JSON protocol (see openclaw_types.py). We provide both sync and async
wrappers so callers can pick based on context:

- run_bridge: blocking, used from add()/prepare() paths that run outside
  the asyncio event loop (Task 4 ingest + flush + index).
- arun_bridge: async, used from search() so concurrent per-question
  requests don't stall the event loop (Task 5).

All failure modes - non-zero exit, non-JSON stdout, ok=false - raise
BridgeError. Timeouts kill the entire process group so a stalled Node child
does not leak subprocesses.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, os.PathLike]


class BridgeError(RuntimeError):
    """Raised when the bridge returns a non-success response."""


class BridgeTimeout(BridgeError):
    """Raised when the bridge exceeds the configured timeout."""


def build_sandbox_paths(
    output_dir: PathLike, run_id: str, conversation_id: str
) -> dict[str, str]:
    """Construct the filesystem layout for a single conversation sandbox.

    Layout is deterministic so build_lazy_index() in Task 4 can rebuild the
    handle without re-running ingest.
    """
    base = (
        Path(output_dir)
        / "artifacts"
        / "openclaw"
        / run_id
        / "conversations"
        / conversation_id
    )
    return {
        "base_dir": str(base),
        "workspace_dir": str(base),
        "native_store_dir": str(base / "native_store"),
        "metrics_dir": str(base / "metrics"),
        "events_path": str(base / "events.jsonl"),
    }


def _parse_or_raise(stdout: str) -> dict[str, Any]:
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as err:
        raise BridgeError(
            f"bridge returned invalid json: {stdout[:500]!r}"
        ) from err

    if not parsed.get("ok", False):
        raise BridgeError(parsed.get("error") or "bridge returned ok=false")
    return parsed


def run_bridge(
    bridge_script: PathLike, payload: dict, timeout: float = 600.0
) -> dict[str, Any]:
    """Invoke the Node bridge synchronously.

    start_new_session=True puts the Node child in its own process group so
    on timeout we SIGKILL the whole group - Node sometimes spawns worker
    threads that survive a plain proc.kill().
    """
    bridge_script = Path(bridge_script)
    if not bridge_script.exists():
        raise BridgeError(f"bridge script not found: {bridge_script}")

    proc = subprocess.Popen(
        ["node", str(bridge_script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ},
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(json.dumps(payload), timeout=timeout)
    except subprocess.TimeoutExpired as err:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        raise BridgeTimeout(
            f"bridge timeout after {timeout}s for command "
            f"{payload.get('command')!r}"
        ) from err

    if proc.returncode != 0:
        raise BridgeError(stderr.strip() or stdout.strip() or "bridge exited non-zero")

    return _parse_or_raise(stdout)


async def arun_bridge(
    bridge_script: PathLike, payload: dict, timeout: float = 600.0
) -> dict[str, Any]:
    """Async counterpart of run_bridge.

    Uses asyncio.create_subprocess_exec so the event loop keeps servicing
    other searches while a single Node child is in flight.
    """
    bridge_script = Path(bridge_script)
    if not bridge_script.exists():
        raise BridgeError(f"bridge script not found: {bridge_script}")

    proc = await asyncio.create_subprocess_exec(
        "node",
        str(bridge_script),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
        env={**os.environ},
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(json.dumps(payload).encode()),
            timeout=timeout,
        )
    except asyncio.TimeoutError as err:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            await proc.wait()
        except Exception:
            pass
        raise BridgeTimeout(
            f"bridge timeout after {timeout}s for command "
            f"{payload.get('command')!r}"
        ) from err

    if proc.returncode != 0:
        msg = (
            stderr_b.decode().strip()
            or stdout_b.decode().strip()
            or "bridge exited non-zero"
        )
        raise BridgeError(msg)

    return _parse_or_raise(stdout_b.decode())
