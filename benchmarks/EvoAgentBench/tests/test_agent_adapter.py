"""Tests for AgentAdapter base class and its implementations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agents.base import AgentAdapter


# ---------------------------------------------------------------------------
# Test: _command() in base class
# ---------------------------------------------------------------------------

class TestBaseCommand:
    """_command() should be available in base and use self.name as default."""

    def _make_adapter(self):
        class Dummy(AgentAdapter):
            name = "dummy"

            def _build_cli_cmd(self, prompt, session_id, timeout):
                return [self._command(), "run", "--message", prompt]

            def _session_file(self, session_id):
                return Path(f"/tmp/{session_id}.jsonl")

            def _parse_watch_record(self, rec, ts, prefix):
                pass

            def _parse_session_entry(self, entry, info):
                pass

        return Dummy()

    def test_command_defaults_to_name(self):
        adapter = self._make_adapter()
        assert adapter._command() == "dummy"


# ---------------------------------------------------------------------------
# Test: subclasses don't duplicate _command()
# ---------------------------------------------------------------------------

class TestNoDuplicateCommand:
    """nanobot and openclaw should inherit _command() from base."""

    def test_nanobot_uses_base_command(self):
        from agents.nanobot.nanobot import NanobotAdapter
        # Should not override _command
        assert "_command" not in NanobotAdapter.__dict__

    def test_openclaw_uses_base_command(self):
        from agents.openclaw.openclaw import OpenClawAdapter
        assert "_command" not in OpenClawAdapter.__dict__


# ---------------------------------------------------------------------------
# Test: nanobot session_id has no hardcoded prefix
# ---------------------------------------------------------------------------

class TestNanobotSessionId:
    """Nanobot should not hardcode 'tbench:' prefix in session handling."""

    def test_no_hardcoded_tbench_prefix(self):
        source = (Path(__file__).resolve().parent.parent
                  / "src" / "agents" / "nanobot" / "nanobot.py").read_text()
        assert "tbench:" not in source, \
            "nanobot.py should not hardcode 'tbench:' prefix"
