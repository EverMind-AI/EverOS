"""Omni-Math domain adapter.

Math competition problems. Agent solves problems step by step; verification
checks answer correctness.

Data source: local JSONL files from omni-math experiments.
"""

import json
import logging
from pathlib import Path

from domains.base import DomainAdapter
from domains.reasoning.evaluate import verify_answer
from config import get_domain_config, register_domain

log = logging.getLogger("evoagentbench")

_PROMPT_TEMPLATE = (Path(__file__).parent / "prompt.md").read_text()


def _cfg():
    return get_domain_config("reasoning")


def _data_dir() -> Path:
    """Return the data directory."""
    cfg = _cfg()
    path = cfg.get("data_dir")
    if not path:
        raise ValueError("No data_dir configured in reasoning.yaml.")
    return Path(path)


def _load_jsonl(path: Path) -> list[dict]:
    """Load records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class OmniMathAdapter(DomainAdapter):
    name = "reasoning"

    def __init__(self):
        self._dataset_cache = {}

    def _get_dataset(self, split: str = "test") -> list[dict]:
        """Load and cache problems for the given split (test/train)."""
        if split not in self._dataset_cache:
            ddir = _data_dir()
            path = ddir / f"{split}.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"Data not found: {path}")
            self._dataset_cache[split] = _load_jsonl(path)
            log.info(f"Loaded {len(self._dataset_cache[split])} {split} problems from {path}")
        return self._dataset_cache[split]

    def pass_threshold(self) -> float:
        return float(_cfg().get("pass_threshold", 0.0))

    def load_tasks(self, args) -> list[dict]:
        split = getattr(args, "split", None)
        if split and not split.isdigit():
            dataset = self._get_dataset(split)
        else:
            dataset = self._get_dataset()

        # Filter by task ID if specified
        if args.task:
            ids = [t.strip() for t in args.task.split(",")]
            int_ids = []
            for tid in ids:
                raw = tid.removeprefix("omni_")
                try:
                    int_ids.append(int(raw))
                except ValueError:
                    pass
            if int_ids:
                dataset = [r for r in dataset if r.get("_idx") in int_ids]
            else:
                dataset = [r for r in dataset if str(r.get("_idx")) in ids]

        if args.split and args.split.isdigit():
            dataset = dataset[:int(args.split)]

        tasks = []
        for rec in dataset:
            idx = rec.get("_idx", 0)
            domain = rec.get("domain", [])
            domain_str = " | ".join(domain) if isinstance(domain, list) else str(domain)

            tasks.append({
                "name": f"omni_{idx}",
                "task_id": str(idx),
                "problem": rec["problem"],
                "answer": rec.get("answer", ""),
                "solution": rec.get("solution", ""),
                "domain": domain_str,
                "difficulty": rec.get("difficulty", 0),
                "source": rec.get("source", ""),
                "problem_type": rec.get("problem_type", ""),
                "test_category": rec.get("test_category", ""),
            })

        log.info(f"Loaded {len(tasks)} tasks from omnimath")
        return tasks

    def get_agent_timeout(self, task: dict, env_info: dict) -> int:
        return int(_cfg().get("agent_timeout", 600))

    def build_prompt(self, task: dict, env_info: dict) -> str:
        return _PROMPT_TEMPLATE.format(
            problem=task["problem"],
        )

    def verify(self, task: dict, env_info: dict, trial_dir: Path,
               agent_result: dict | None = None) -> dict:
        """Verify the agent's answer against the expected answer."""
        import os

        cfg = _cfg()
        mode = cfg.get("verify_mode", "exact")
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        api_base = cfg.get("eval_api_base", "")
        model_owner = cfg.get("eval_model_owner", "openai")
        model_name = cfg.get("eval_model_name", "gpt-4o")
        # For local endpoints, use model name directly; for OpenRouter, use owner/name
        if api_base:
            model = model_name
        else:
            model = f"{model_owner}/{model_name}"

        # Extract agent output text
        agent_output = ""
        if agent_result:
            agent_output = agent_result.get("response", "") or agent_result.get("output", "") or agent_result.get("text", "")

        result = verify_answer(
            task, agent_output,
            mode=mode, api_key=api_key, model=model, api_base=api_base,
        )

        # Add task metadata to result
        result["task_id"] = task["task_id"]
        result["difficulty"] = task.get("difficulty", 0)
        result["domain"] = task.get("domain", "")
        result["test_category"] = task.get("test_category", "")
        result["source"] = task.get("source", "")

        # Save evaluation details
        verifier_dir = trial_dir / "verifier"
        verifier_dir.mkdir(parents=True, exist_ok=True)
        with open(verifier_dir / "eval_details.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return result

    def aggregate_metrics(self, results: list[dict]) -> dict:
        """Compute accuracy broken down by category, difficulty, and domain."""
        total = len(results)
        if total == 0:
            return {}

        correct = sum(1 for r in results if r.get("correct", False))

        # By test_category
        by_category = {}
        for r in results:
            cat = r.get("test_category", "unknown")
            by_category.setdefault(cat, {"total": 0, "correct": 0})
            by_category[cat]["total"] += 1
            if r.get("correct", False):
                by_category[cat]["correct"] += 1

        # By difficulty bucket
        by_difficulty = {}
        for r in results:
            diff = r.get("difficulty", 0)
            bucket = f"d{int(diff)}" if diff else "unknown"
            by_difficulty.setdefault(bucket, {"total": 0, "correct": 0})
            by_difficulty[bucket]["total"] += 1
            if r.get("correct", False):
                by_difficulty[bucket]["correct"] += 1

        return {
            "accuracy": f"{correct}/{total} ({100*correct/total:.1f}%)",
            "correct": correct,
            "total": total,
            "by_category": by_category,
            "by_difficulty": by_difficulty,
        }


register_domain("reasoning", OmniMathAdapter)
