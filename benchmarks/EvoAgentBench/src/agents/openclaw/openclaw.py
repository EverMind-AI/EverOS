"""OpenClaw agent adapter.

Uses a per-task temporary OPENCLAW_HOME so that concurrent tasks never
share or modify a global config file.  Model, MCP, and tool settings are
written into a disposable openclaw.json before each invocation.
"""

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

from agents.base import AgentAdapter, C_MAGENTA, C_YELLOW, C_RESET

log = logging.getLogger("evoagentbench")


class OpenClawAdapter(AgentAdapter):
    name = "openclaw"

    def __init__(self):
        self._temp_home = None

    # --- session path ----------------------------------------------------------

    def _session_dir(self) -> Path:
        if self._temp_home:
            return Path(self._temp_home) / ".openclaw" / "agents" / "main" / "sessions"
        return Path.home() / ".openclaw" / "agents" / "main" / "sessions"

    # --- MCP lifecycle ---------------------------------------------------------

    def _proxy_script(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent / "utils" / "openclaw_mcp_stdio_proxy.js"

    def _openclaw_node_modules_dir(self) -> Path | None:
        override = os.environ.get("OPENCLAW_MCP_PROXY_NODE_MODULES")
        if override:
            path = Path(override).expanduser()
            return path if path.exists() else None

        command_path = shutil.which(self._command())
        if command_path is None:
            return None

        command_path = Path(command_path).resolve()
        for parent in [command_path.parent, *command_path.parents]:
            for candidate in [
                parent / "node_modules",
                parent / "lib" / "node_modules" / "openclaw" / "node_modules",
            ]:
                if candidate.exists():
                    return candidate
        return None

    def _build_mcp_server_entry(self, cfg: dict) -> dict:
        """Build an MCP server config entry for openclaw.json."""
        if cfg.get("command"):
            entry = {"command": cfg["command"]}
            if cfg.get("args"):
                entry["args"] = list(cfg["args"])
            return entry

        url = cfg.get("url", "")
        if not url:
            raise RuntimeError("MCP config requires either command/args or url")

        node_modules = self._openclaw_node_modules_dir()
        if node_modules is None:
            raise RuntimeError(
                "Cannot locate OpenClaw node_modules for MCP stdio proxy. "
                "Set OPENCLAW_MCP_PROXY_NODE_MODULES or use a standard npm install."
            )

        proxy = self._proxy_script()
        if not proxy.exists():
            raise RuntimeError(f"MCP stdio proxy script missing: {proxy}")

        return {
            "command": shutil.which("node") or "node",
            "args": [
                str(proxy),
                "--url", url,
                "--transport", cfg.get("type", "auto"),
                "--node-modules", str(node_modules),
            ],
        }

    def _build_openclaw_config(self, mcp_servers=None, disabled_tools=None):
        """Build openclaw.json: global config + evoagentbench overrides + MCP."""
        from config import get_config
        agent_cfg = get_config().get("agent", {})

        # Read global openclaw config as base
        global_config_path = Path.home() / ".openclaw" / "openclaw.json"
        if global_config_path.exists():
            config = json.loads(global_config_path.read_text())
        else:
            config = {}

        # Strip gateway and auth sections — temp OPENCLAW_HOME doesn't have
        # the supporting files (auth-profiles.json, gateway process) so these
        # cause agent startup failures. Embedded mode with models.providers works.
        config.pop("gateway", None)
        config.pop("auth", None)

        # Override from evoagentbench agent config (if set)
        if agent_cfg.get("model"):
            config.setdefault("agents", {}).setdefault("defaults", {})
            config["agents"]["defaults"]["model"] = {"primary": agent_cfg["model"]}
        if agent_cfg.get("max_concurrent"):
            config.setdefault("agents", {}).setdefault("defaults", {})
            config["agents"]["defaults"]["maxConcurrent"] = agent_cfg["max_concurrent"]
        if agent_cfg.get("providers"):
            config.setdefault("models", {})
            config["models"]["providers"] = {name: dict(pcfg)
                                              for name, pcfg in agent_cfg["providers"].items()}

        # MCP servers
        if mcp_servers:
            mcp_section = {}
            for name, cfg in mcp_servers.items():
                mcp_section[name] = self._build_mcp_server_entry(cfg)
            config.setdefault("mcp", {})["servers"] = mcp_section

        # Tools config from agent yaml (e.g., profile, deny)
        if agent_cfg.get("tools"):
            config.setdefault("tools", {}).update(agent_cfg["tools"])

        # Disabled tools (from setup_mcp) — merge with existing deny list
        if disabled_tools:
            existing = config.setdefault("tools", {}).get("deny", [])
            config["tools"]["deny"] = list(set(existing) | set(disabled_tools))

        return config

    # --- Temp config (always created per task) ---

    def _ensure_temp_config(self):
        """Create per-task temp OPENCLAW_HOME: copy global + apply overrides."""
        if self._temp_home:
            return

        home_dir = Path(tempfile.mkdtemp(prefix="evoagentbench-openclaw-"))
        config_dir = home_dir / ".openclaw"
        config_dir.mkdir()

        config = self._build_openclaw_config()
        (config_dir / "openclaw.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False)
        )

        self._temp_home = str(home_dir)

    def _cleanup_temp_config(self):
        if self._temp_home:
            shutil.rmtree(self._temp_home, ignore_errors=True)
        self._temp_home = None

    # --- MCP lifecycle ---

    def setup_mcp(self, mcp_servers, disabled_tools=None):
        self._ensure_temp_config()
        # Rebuild config with MCP
        config = self._build_openclaw_config(mcp_servers, disabled_tools)
        config_path = Path(self._temp_home) / ".openclaw" / "openclaw.json"
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))

    def teardown_mcp(self):
        self._cleanup_temp_config()

    # --- subprocess env --------------------------------------------------------

    def _get_subprocess_env(self):
        self._ensure_temp_config()
        env = dict(os.environ)
        env["OPENCLAW_HOME"] = self._temp_home
        return env

    # --- CLI -------------------------------------------------------------------

    def _build_cli_cmd(self, prompt, session_id, timeout):
        return [
            self._command(), "agent",
            "--session-id", session_id,
            "--message", prompt,
            "--thinking", "high",
            "--timeout", str(timeout),
            "--json",
        ]

    def _parse_extra(self, result):
        """Extract response from openclaw JSON output.

        OpenClaw --json may write to stdout or stderr depending on the mode
        (gateway vs embedded).  The JSON block starts with '{' on its own line
        after gateway/plugin log lines.
        """
        for source in (result.stdout, result.stderr):
            if not source:
                continue
            start = source.rfind('\n{')
            if start < 0:
                start = 0 if source.startswith('{') else -1
            else:
                start += 1  # skip the newline
            if start < 0:
                continue
            try:
                data = json.loads(source[start:])
                payloads = data.get("payloads") or data.get("result", {}).get("payloads")
                if payloads:
                    text = payloads[0].get("text", "")
                    if text:
                        return {"response": text, "response_json": data}
            except (json.JSONDecodeError, TypeError):
                pass
        return {}

    # --- live watch ------------------------------------------------------------

    def _parse_watch_record(self, rec, ts, prefix):
        if rec.get("type") != "message":
            return
        msg = rec["message"]
        content = msg.get("content", [])
        if not isinstance(content, list):
            return

        for item in content:
            if not isinstance(item, dict):
                continue

            if item.get("type") == "toolCall":
                tool_name = item.get("name", "")
                args = item.get("arguments", {})
                if tool_name == "exec":
                    display = args.get("command", "").replace("\n", "\\n")
                else:
                    display = f"{tool_name}({json.dumps(args, ensure_ascii=False)[:150]})"
                self._print_tool_call(ts, prefix, display)

            if item.get("type") == "text" and msg.get("role") == "toolResult":
                self._print_tool_output(ts, prefix, item.get("text", ""))

            if item.get("type") == "thinking":
                first_line = item.get("thinking", "").strip().split("\n")[0][:120]
                if first_line:
                    print(f"  {C_MAGENTA}{ts} {prefix} 💭 {first_line}{C_RESET}", flush=True)

        stop_reason = msg.get("stopReason")
        if stop_reason and stop_reason not in ("end_turn", "toolUse", "tool_use"):
            print(f"  {C_YELLOW}{ts} {prefix} ⚠ stopReason={stop_reason}{C_RESET}", flush=True)

    # --- session stats ---------------------------------------------------------

    def _parse_session_entry(self, entry, stats):
        usage = entry.get("message", {}).get("usage")
        if usage:
            stats["input"] += usage.get("input", 0)
            stats["output"] += usage.get("output", 0)
            stats["total"] += usage.get("totalTokens", 0)
            stats["turns"] += 1
        sr = entry.get("message", {}).get("stopReason", "")
        if sr:
            stats["last_stop_reason"] = sr

    # --- retry -----------------------------------------------------------------

    def should_retry(self, result):
        """Don't retry toolUse/aborted (model won't converge) or timeouts."""
        stop = result.get("last_stop_reason", "")
        if stop in ("toolUse", "aborted"):
            return None
        elapsed = result.get("agent_result", {}).get("elapsed_sec", 0)
        from config import get_config
        retry_timeout = get_config().get("agent", {}).get("retry_timeout") or 3000
        if elapsed >= retry_timeout:
            return None
        return super().should_retry(result)


from config import register_agent
register_agent("openclaw", OpenClawAdapter)
