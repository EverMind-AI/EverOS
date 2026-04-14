"""Nanobot agent adapter."""

import json
import shutil
import tempfile
from pathlib import Path

from agents.base import AgentAdapter, C_MAGENTA, C_RESET


class NanobotAdapter(AgentAdapter):
    name = "nanobot"

    def __init__(self):
        self._temp_workspace = None
        self._temp_config = None

    def _session_dir(self) -> Path:
        if self._temp_workspace:
            return Path(self._temp_workspace) / "sessions"
        nanobot_config = Path.home() / ".nanobot" / "config.json"
        if nanobot_config.exists():
            workspace = Path(
                json.loads(nanobot_config.read_text())
                .get("agents", {}).get("defaults", {})
                .get("workspace", str(Path.home() / ".nanobot" / "workspace"))
            ).expanduser()
            return workspace / "sessions"
        return Path.home() / ".nanobot" / "workspace" / "sessions"

    def _session_file(self, session_id):
        safe_key = session_id.replace(":", "_")
        base = Path(self._temp_workspace) if self._temp_workspace else self._session_dir().parent
        return base / "sessions" / f"{safe_key}.jsonl"

    # --- Temp config (always created per task) ---

    def _ensure_temp_config(self):
        """Create per-task temp config: copy global + apply evoagentbench overrides."""
        if self._temp_workspace:
            return  # already created

        workspace_dir = Path(tempfile.mkdtemp(prefix="evoagentbench-nanobot-"))

        # Read global nanobot config as base
        global_config_path = Path.home() / ".nanobot" / "config.json"
        if global_config_path.exists():
            config = json.loads(global_config_path.read_text())
        else:
            config = {}

        # Point workspace to temp dir so sessions are written here
        config.setdefault("agents", {}).setdefault("defaults", {})
        config["agents"]["defaults"]["workspace"] = str(workspace_dir)

        # Override model/provider from evoagentbench agent config (if set)
        from config import get_config
        agent_cfg = get_config().get("agent", {})
        if agent_cfg.get("model"):
            config["agents"]["defaults"]["model"] = agent_cfg["model"]
        if agent_cfg.get("provider"):
            config["agents"]["defaults"]["provider"] = agent_cfg["provider"]
        if agent_cfg.get("providers"):
            config["providers"] = dict(agent_cfg["providers"])
        if agent_cfg.get("maxTokens") is not None:
            config["agents"]["defaults"]["maxTokens"] = agent_cfg["maxTokens"]
        if agent_cfg.get("contextWindowTokens") is not None:
            config["agents"]["defaults"]["contextWindowTokens"] = agent_cfg["contextWindowTokens"]
        if agent_cfg.get("temperature") is not None:
            config["agents"]["defaults"]["temperature"] = agent_cfg["temperature"]
        if agent_cfg.get("reasoningEffort") is not None:
            config["agents"]["defaults"]["reasoningEffort"] = agent_cfg["reasoningEffort"]
        if agent_cfg.get("tools"):
            config.setdefault("tools", {}).update(agent_cfg["tools"])

        config_path = workspace_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2))

        self._temp_workspace = str(workspace_dir)
        self._temp_config = str(config_path)

    def _cleanup_temp_config(self):
        if self._temp_workspace:
            shutil.rmtree(self._temp_workspace, ignore_errors=True)
        self._temp_workspace = None
        self._temp_config = None

    # --- MCP lifecycle ---

    def setup_mcp(self, mcp_servers, disabled_tools=None):
        self._ensure_temp_config()
        # Patch MCP into existing temp config
        config = json.loads(Path(self._temp_config).read_text())
        config.setdefault("tools", {})
        config["tools"]["mcpServers"] = mcp_servers
        if disabled_tools:
            config["tools"]["disabledTools"] = disabled_tools
        Path(self._temp_config).write_text(json.dumps(config, indent=2))

    def teardown_mcp(self):
        self._cleanup_temp_config()

    # --- CLI ---

    def _build_cli_cmd(self, prompt, session_id, timeout):
        self._ensure_temp_config()
        cmd = [
            self._command(), "agent",
            "--session", session_id,
            "--message", prompt,
            "--no-markdown",
            "--workspace", self._temp_workspace,
            "--config", self._temp_config,
        ]
        return cmd

    def _parse_watch_record(self, rec, ts, prefix):
        role = rec.get("role", "")

        if role == "assistant" and rec.get("reasoning_content"):
            first_line = rec["reasoning_content"].strip().split("\n")[0][:120]
            if first_line:
                print(f"  {C_MAGENTA}{ts} {prefix} 💭 {first_line}{C_RESET}", flush=True)

        if role == "assistant" and rec.get("tool_calls"):
            for tc in rec["tool_calls"]:
                func = tc.get("function", {})
                name = func.get("name", "")
                args_str = func.get("arguments", "")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {}
                if name == "exec":
                    display = (args.get("command", "") if isinstance(args, dict) else str(args))
                    display = display.replace("\n", "\\n")
                else:
                    display = f"{name}({str(args)[:150]})"
                self._print_tool_call(ts, prefix, display)

        if role == "tool":
            text = rec.get("content", "")
            if text:
                self._print_tool_output(ts, prefix, text)

    def should_retry(self, result):
        """Nanobot: also retry on zero turns (agent started but produced no output)."""
        turns = result.get("token_usage", {}).get("turns", 0)
        completion = result.get("agent_result", {}).get("completion_status", "")
        if turns == 0 and completion == "completed":
            return "zero_turns"
        return super().should_retry(result)


from config import register_agent
register_agent("nanobot", NanobotAdapter)
