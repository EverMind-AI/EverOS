"""Agent adapter base class.

Each agent (OpenClaw, nanobot, ...) implements this interface so the
runner can call any agent without knowing its specifics.

Framework calls these methods (all implemented in base class):
  - call_agent(prompt, session_id, timeout) -> dict
  - watch_session(session_id, stop_event, task_label)
  - collect_session(session_id, trial_dir) -> dict
  - should_retry(result) -> str | None

Subclasses override internal methods to customize behavior:
  - _build_cli_cmd (required)
  - _parse_watch_record (required)
  - _session_dir, _session_file, _parse_session_entry, _parse_extra (optional)
"""

import json
import logging
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path

from config import get_config

log = logging.getLogger("evoagentbench")

# ANSI colors for live watch output
C_RESET = "\033[0m"
C_DIM = "\033[2m"
C_CYAN = "\033[36m"
C_MAGENTA = "\033[35m"
C_YELLOW = "\033[33m"


class AgentAdapter(ABC):
    """Base class for agent adapters."""

    name: str = "base"

    # --- Common helpers -------------------------------------------------------

    def _command(self) -> str:
        """Return the CLI command to invoke this agent.

        Reads from config.yaml if this agent is the configured one,
        otherwise falls back to self.name.
        """
        cfg = get_config()
        if cfg.get("agent", {}).get("name") == self.name:
            return cfg["agent"].get("command", self.name)
        return self.name

    # --- Agent invocation -----------------------------------------------------

    @abstractmethod
    def _build_cli_cmd(self, prompt: str, session_id: str, timeout: int) -> list[str]:
        """Return the CLI command list to invoke the agent."""
        ...

    def _parse_extra(self, result: subprocess.CompletedProcess) -> dict:
        """Extract adapter-specific extra fields from the subprocess result.

        Override to add fields like response_json. Default returns empty dict.
        """
        return {}

    def setup_mcp(self, mcp_servers: dict, disabled_tools: list | None = None):
        """Configure MCP servers before calling the agent. Override per agent."""
        pass

    def teardown_mcp(self):
        """Clean up MCP config after calling the agent. Override per agent."""
        pass

    def _get_subprocess_env(self) -> dict | None:
        """Return custom env dict for the agent subprocess, or None to inherit."""
        return None

    def call_agent(self, prompt, session_id, timeout=3600) -> dict:
        """Invoke the agent CLI and return result dict.

        Returns: {"response", "completion_status", "elapsed_sec", "returncode", ...}
        """
        cmd = self._build_cli_cmd(prompt, session_id, timeout)
        log.info(f"  CLI: {' '.join(cmd[:5])}... (timeout={timeout}s)")

        start = time.time()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout + 60,
                env=self._get_subprocess_env(),
            )
            elapsed = time.time() - start

            d = {
                "response": result.stdout,
                "completion_status": "completed" if result.returncode == 0 else "error",
                "elapsed_sec": round(elapsed, 1),
                "method": "cli",
                "returncode": result.returncode,
                "stderr": (result.stderr or "")[:2000] or None,
            }
            d.update(self._parse_extra(result))
            return d

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            return {
                "response": "",
                "completion_status": "timeout",
                "elapsed_sec": round(elapsed, 1),
                "method": "cli",
                "error": f"subprocess timed out after {timeout+60}s",
            }

    # --- Session watching -----------------------------------------------------

    def _session_dir(self) -> Path:
        """Return the directory containing session JSONL files.

        Override this in subclasses. Default: ~/.{name}/sessions/
        """
        return Path.home() / f".{self.name}" / "sessions"

    def _session_file(self, session_id: str) -> Path:
        """Return the path to the session JSONL file.

        Default: _session_dir() / "{session_id}.jsonl"
        """
        return self._session_dir() / f"{session_id}.jsonl"

    @abstractmethod
    def _parse_watch_record(self, rec: dict, ts: str, prefix: str) -> None:
        """Parse one JSONL record and print live output. Called per record."""
        ...

    def watch_session(self, session_id, stop_event, task_label=""):
        """Tail the session JSONL file and print live progress."""
        session_file = self._session_file(session_id)
        prefix = f"[{task_label}]" if task_label else ""

        for _ in range(300):
            if session_file.exists():
                break
            if stop_event.is_set():
                return
            time.sleep(1)
        else:
            return

        seen = 0
        while not stop_event.is_set():
            try:
                with open(session_file) as f:
                    lines = f.readlines()
            except OSError:
                time.sleep(1)
                continue

            for line in lines[seen:]:
                seen += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = time.strftime("%H:%M:%S")
                self._parse_watch_record(rec, ts, prefix)

            time.sleep(2)

    # --- Session collection ---------------------------------------------------

    def _parse_session_entry(self, entry: dict, stats: dict) -> None:
        """Extract stats from one JSONL entry. Called per entry.

        Default: counts assistant turns. Override for token usage, etc.
        """
        if entry.get("role") == "assistant":
            stats["turns"] += 1

    def collect_session(self, session_id, trial_dir) -> dict:
        """Copy session JSONL to trial_dir and extract stats.

        Returns: {"turns", "input", "output", "total", "last_stop_reason"}
        """
        session_jsonl = self._session_file(session_id)
        stats = {"turns": 0, "input": 0, "output": 0, "total": 0,
                 "last_stop_reason": None}

        if not session_jsonl.exists():
            return stats

        shutil.copy2(session_jsonl, trial_dir / "session.jsonl")

        try:
            with open(session_jsonl) as f:
                for line in f:
                    entry = json.loads(line)
                    self._parse_session_entry(entry, stats)
        except Exception as e:
            log.warning(f"Failed to parse session JSONL: {e}")

        return stats

    # --- Retry ----------------------------------------------------------------

    def should_retry(self, result: dict) -> str | None:
        """Return a reason string if this result warrants a retry.

        No retry: passed, timeout, or agent completed normally (even if wrong).
        Retry: crashes, abnormal exits, and other unexpected failures.
        Total attempts capped by max_retries in runner.
        """
        reward = result.get("verifier_result", {}).get("reward", 0)
        if reward and reward > 0:
            return None

        status = result.get("agent_result", {}).get("completion_status")
        if status in ("timeout", "completed"):
            return None

        return "failed"

    # --- Helpers for subclass watch output ------------------------------------

    @staticmethod
    def _print_tool_call(ts: str, prefix: str, display: str):
        if len(display) > 200:
            display = display[:197] + "..."
        print(f"  {C_CYAN}{ts} {prefix} > {display}{C_RESET}", flush=True)

    @staticmethod
    def _print_tool_output(ts: str, prefix: str, text: str):
        lines_out = text.strip().split("\n")
        if len(lines_out) > 5:
            for ln in lines_out[:3]:
                print(f"  {C_DIM}{ts} {prefix}   {ln[:150]}{C_RESET}", flush=True)
            print(f"  {C_DIM}{ts} {prefix}   ... ({len(lines_out)} lines){C_RESET}", flush=True)
            print(f"  {C_DIM}{ts} {prefix}   {lines_out[-1][:150]}{C_RESET}", flush=True)
        else:
            for ln in lines_out[:5]:
                print(f"  {C_DIM}{ts} {prefix}   {ln[:150]}{C_RESET}", flush=True)
