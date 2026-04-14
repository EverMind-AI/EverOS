"""Domain adapter base class.

Each domain (information_retrieval, reasoning, software_engineering, ...) implements this interface
so the runner can execute any domain without knowing its specifics.

Only load_tasks, build_prompt, and verify are required (abstractmethod).
Other lifecycle methods have sensible defaults and can be overridden as needed.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class DomainAdapter(ABC):
    """Base class for domain adapters."""

    name: str = "base"

    # --- Required interface (must override) ---

    @abstractmethod
    def load_tasks(self, args) -> list[dict]:
        """Load task list based on CLI args.

        Returns list of dicts, each with at least a 'name' key.
        """
        ...

    @abstractmethod
    def build_prompt(self, task: dict, env_info: dict) -> str:
        """Build the prompt to send to the agent."""
        ...

    @abstractmethod
    def verify(self, task: dict, env_info: dict, trial_dir: Path,
               agent_result: dict | None = None) -> dict:
        """Run verification and return {"reward": float, ...}.

        agent_result: dict from AgentResult.to_dict(), available for
        domains that verify based on agent response text.
        """
        ...

    # --- Optional lifecycle hooks (override as needed) ---

    def initialize(self, args):
        """Called once before any task runs. Use for shared infra (servers, etc).

        Default: no-op.
        """
        pass

    def finalize(self):
        """Called once after all tasks complete (even on error). Pair with initialize().

        Default: no-op.
        """
        pass

    def setup(self, task: dict, agent_name: str, trial: int) -> dict:
        """Prepare execution environment for a task.

        Returns env_info dict (contents are domain-specific).
        Default: no-op, returns empty dict.
        """
        return {}

    def cleanup(self, task: dict, env_info: dict):
        """Clean up environment (remove containers, temp files, etc).

        Default: no-op.
        """
        pass

    def get_agent_timeout(self, task: dict, env_info: dict) -> int:
        """Return agent timeout in seconds for this task.

        Default: 3600 (1 hour).
        """
        return 3600

    def pre_task_trials(self, task: dict):
        """Called once before all trials for a task (e.g. load Docker image).

        Default: no-op.
        """
        pass

    def post_task_trials(self, task: dict):
        """Called once after all trials for a task (e.g. remove Docker image).

        Default: no-op.
        """
        pass

    def get_disk_cost(self, task: dict) -> int:
        """Return estimated disk cost in bytes for a task.

        Used by disk-aware scheduler. Default: 100 MB.
        """
        return 100 * 1024 * 1024

    def pass_threshold(self) -> float:
        """Reward threshold for pass@1. Default: 0 (reward > 0 means passed)."""
        return 0.0

    def aggregate_metrics(self, results: list[dict]) -> dict:
        """Return domain-specific summary metrics from all result dicts.

        Called once after all tasks complete. Returned dict is written to
        summary.json under the "domain_metrics" key.
        Default: empty (no extra metrics).
        """
        return {}
