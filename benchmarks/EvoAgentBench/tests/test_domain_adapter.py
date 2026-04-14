"""Tests for DomainAdapter base class and its implementations."""

import sys
from pathlib import Path

# Add src to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from domains.base import DomainAdapter


# ---------------------------------------------------------------------------
# Test: abstract methods are enforced
# ---------------------------------------------------------------------------

class TestAbstractEnforcement:
    """Only load_tasks, build_prompt, verify should be abstract."""

    def test_cannot_instantiate_without_abstract_methods(self):
        """Missing any abstract method should raise TypeError."""
        # Completely empty subclass — missing all 3 abstract methods
        class Empty(DomainAdapter):
            pass

        try:
            Empty()
            assert False, "Should have raised TypeError"
        except TypeError as e:
            msg = str(e)
            assert "load_tasks" in msg
            assert "build_prompt" in msg
            assert "verify" in msg

    def test_can_instantiate_with_only_abstract_methods(self):
        """Implementing only the 3 abstract methods should be enough."""
        class Minimal(DomainAdapter):
            name = "minimal"

            def load_tasks(self, args):
                return []

            def build_prompt(self, task, env_info):
                return "prompt"

            def verify(self, task, env_info, trial_dir, agent_result=None):
                return {"reward": 1.0}

        adapter = Minimal()
        assert adapter.name == "minimal"


# ---------------------------------------------------------------------------
# Test: default implementations
# ---------------------------------------------------------------------------

class TestDefaults:
    """Non-abstract methods should have sensible defaults."""

    def _make_adapter(self):
        class Minimal(DomainAdapter):
            name = "test"

            def load_tasks(self, args):
                return []

            def build_prompt(self, task, env_info):
                return ""

            def verify(self, task, env_info, trial_dir, agent_result=None):
                return {"reward": 0.0}

        return Minimal()

    def test_setup_returns_empty_dict(self):
        adapter = self._make_adapter()
        result = adapter.setup({"name": "t1"}, "agent", 1)
        assert result == {}

    def test_cleanup_is_noop(self):
        adapter = self._make_adapter()
        # Should not raise
        adapter.cleanup({"name": "t1"}, {})

    def test_get_agent_timeout_default(self):
        adapter = self._make_adapter()
        timeout = adapter.get_agent_timeout({"name": "t1"}, {})
        assert timeout == 3600

    def test_pre_task_trials_is_noop(self):
        adapter = self._make_adapter()
        adapter.pre_task_trials({"name": "t1"})

    def test_post_task_trials_is_noop(self):
        adapter = self._make_adapter()
        adapter.post_task_trials({"name": "t1"})

    def test_get_disk_cost_default(self):
        adapter = self._make_adapter()
        cost = adapter.get_disk_cost({"name": "t1"})
        assert cost == 100 * 1024 * 1024  # 100 MB


# ---------------------------------------------------------------------------
# Test: overriding optional methods works
# ---------------------------------------------------------------------------

class TestOverride:
    """Subclasses can override any optional method."""

    def test_override_setup_and_cleanup(self):
        class DockerBench(DomainAdapter):
            name = "docker_bench"
            cleaned = False

            def load_tasks(self, args):
                return [{"name": "t1"}]

            def build_prompt(self, task, env_info):
                return f"do {task['name']} in {env_info['container']}"

            def verify(self, task, env_info, trial_dir, agent_result=None):
                return {"reward": 1.0}

            def setup(self, task, agent_name, trial):
                return {"container": f"{task['name']}-{trial}"}

            def cleanup(self, task, env_info):
                self.cleaned = True

        adapter = DockerBench()
        env = adapter.setup({"name": "t1"}, "agent", 1)
        assert env == {"container": "t1-1"}
        assert adapter.build_prompt({"name": "t1"}, env) == "do t1 in t1-1"

        adapter.cleanup({"name": "t1"}, env)
        assert adapter.cleaned

    def test_override_get_disk_cost(self):
        class BigBench(DomainAdapter):
            name = "big"

            def load_tasks(self, args):
                return []

            def build_prompt(self, task, env_info):
                return ""

            def verify(self, task, env_info, trial_dir, agent_result=None):
                return {"reward": 0.0}

            def get_disk_cost(self, task):
                return 10 * 1024 * 1024 * 1024  # 10 GB

        adapter = BigBench()
        assert adapter.get_disk_cost({"name": "x"}) == 10 * 1024 * 1024 * 1024



# ---------------------------------------------------------------------------
# Test: runner calls lifecycle methods directly (no hasattr)
# ---------------------------------------------------------------------------

class TestRunnerIntegration:
    """Verify runner.py no longer uses hasattr for lifecycle methods."""

    def test_no_hasattr_in_runner(self):
        runner_path = Path(__file__).resolve().parent.parent / "src" / "runner.py"
        source = runner_path.read_text()
        assert "hasattr(domain" not in source, \
            "runner.py should not use hasattr checks — base class provides defaults"
