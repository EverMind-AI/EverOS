"""LiveCodeBench domain adapter.

No Docker needed. Agent solves competitive programming problems.
Verification uses LiveCodeBench's official test runner.
"""

import json
import logging
import os
import re
import sys
from pathlib import Path

from domains.base import DomainAdapter
from config import get_domain_config, register_domain

log = logging.getLogger("evoagentbench")

_LCB_REPO = None


def _cfg():
    return get_domain_config("code_implementation")


def _lcb_repo() -> Path:
    global _LCB_REPO
    if _LCB_REPO is None:
        _LCB_REPO = Path(_cfg()["lcb_repo"])
    return _LCB_REPO


# ---------------------------------------------------------------------------
# Dataset loading & caching
# ---------------------------------------------------------------------------

def _load_dataset_from_hf(release_version: str, cache_path: Path) -> list[dict]:
    """Load from HuggingFace and cache locally as JSON."""
    old_endpoint = os.environ.get("HF_ENDPOINT")
    hf_mirror = _cfg().get("hf_endpoint")
    if hf_mirror:
        os.environ["HF_ENDPOINT"] = hf_mirror

    try:
        from datasets import load_dataset
        ds = load_dataset(
            "livecodebench/code_generation_lite",
            split="test",
            version_tag=release_version,
            trust_remote_code=True,
        )
    finally:
        if old_endpoint is not None:
            os.environ["HF_ENDPOINT"] = old_endpoint
        elif hf_mirror:
            os.environ.pop("HF_ENDPOINT", None)

    records = [dict(row) for row in ds]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(records, f, ensure_ascii=False)
    log.info(f"Cached {len(records)} problems to {cache_path}")
    return records


def _load_problems(release_version: str) -> list[dict]:
    """Load problems, using local cache if available."""
    cfg = _cfg()
    cache_dir = Path(cfg.get("cache_dir", "./data/livecode"))
    cache_path = cache_dir / f"{release_version}.json"

    if cache_path.exists():
        with open(cache_path) as f:
            records = json.load(f)
        log.info(f"Loaded {len(records)} problems from cache: {cache_path}")
    else:
        records = _load_dataset_from_hf(release_version, cache_path)

    lcb_repo = _lcb_repo()
    if str(lcb_repo) not in sys.path:
        sys.path.insert(0, str(lcb_repo))
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

    problems = []
    for rec in records:
        try:
            p = CodeGenerationProblem(**rec)
            problems.append(p)
        except Exception as e:
            log.warning(f"Skipping problem: {e}")

    return problems


# ---------------------------------------------------------------------------
# Split loading (unified cluster format)
# ---------------------------------------------------------------------------

def _load_split() -> dict:
    """Load task split file. Returns {split_name: [qid, ...]}.

    Supports both the unified cluster format:
        {"clusters": {"NAME": {"train": [...], "test": [...]}, ...}}
    and the flat format:
        {"train": [...], "test": [...]}
    """
    split_path = _cfg().get("split_file")
    if not split_path or not Path(split_path).exists():
        return {}
    with open(split_path) as f:
        raw = json.load(f)

    # Unified cluster format
    if "clusters" in raw and isinstance(raw["clusters"], dict):
        first_val = next(iter(raw["clusters"].values()), None)
        if isinstance(first_val, dict) and ("train" in first_val or "test" in first_val):
            # Standard: {"clusters": {"NAME": {"train": [...], "test": [...]}}}
            splits = {"train": [], "test": []}
            for cluster_name, cluster in raw["clusters"].items():
                splits["train"].extend(cluster.get("train", []))
                splits["test"].extend(cluster.get("test", []))
                for part in ("train", "test"):
                    splits[f"{cluster_name}_{part}"] = cluster.get(part, [])
            splits["all"] = splits["train"] + splits["test"]
            return splits

    # Flat format: {"train": [...], "test": [...]}
    if "train" in raw or "test" in raw:
        splits = {}
        if "train" in raw:
            splits["train"] = raw["train"]
        if "test" in raw:
            splits["test"] = raw["test"]
        splits["all"] = splits.get("train", []) + splits.get("test", [])
        return splits

    return {}


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def _find_python_blocks(text: str) -> list[str]:
    """Find all ```python code blocks in text."""
    return re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)


def _syntax_ok(code: str) -> bool:
    """Check if code compiles without SyntaxError."""
    try:
        compile(code, "<check>", "exec")
        return True
    except SyntaxError:
        return False


def _pick_best_block(blocks: list[str]) -> str:
    """From a list of code blocks, return the last one that compiles.
    Falls back to last block if none compile."""
    for block in reversed(blocks):
        code = block.strip()
        if code and _syntax_ok(code):
            return code
    return blocks[-1].strip() if blocks else ""


def _extract_code_from_text(text: str) -> str:
    """Extract code from plain text containing ```python blocks."""
    blocks = _find_python_blocks(text)
    if not blocks:
        blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    return _pick_best_block(blocks) if blocks else ""


def _extract_text_from_oc_response(response: str, response_json: dict | None = None) -> str:
    """Extract plain text from openclaw response (JSON payloads)."""
    if response_json and isinstance(response_json, dict) and "payloads" in response_json:
        parts = [p["text"] for p in response_json["payloads"]
                 if isinstance(p, dict) and p.get("text")]
        if parts:
            return "\n".join(parts)

    for marker in ['{"payloads"', '{']:
        idx = response.find(marker)
        if idx >= 0:
            try:
                data = json.loads(response[idx:])
                if isinstance(data, dict) and "payloads" in data:
                    parts = [p["text"] for p in data["payloads"]
                             if isinstance(p, dict) and p.get("text")]
                    if parts:
                        return "\n".join(parts)
            except (json.JSONDecodeError, TypeError):
                continue
    return response


def _extract_code_from_session(session_path: str | None) -> str:
    """Fallback: extract code from session.jsonl.

    For nanobot: model's raw assistant content has intact code blocks
                 (bypasses CLI --no-markdown line wrapping).
    For openclaw: write_file tool calls contain the actual code written.

    session_path: path to the agent's original session file (passed via
                  agent_result["_session_file"]).
    """
    if not session_path:
        return ""
    session_file = Path(session_path)
    if not session_file.exists():
        return ""

    all_blocks = []
    last_write = ""

    try:
        with open(session_file) as f:
            for line in f:
                rec = json.loads(line)

                # --- nanobot format: role-based records ---
                if rec.get("role") == "assistant":
                    content = rec.get("content") or ""
                    for block in _find_python_blocks(content):
                        all_blocks.append(block.strip())

                    for tc in rec.get("tool_calls", []):
                        fn = tc.get("function", {})
                        name = fn.get("name", "")
                        args_raw = fn.get("arguments", "{}")
                        try:
                            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                        except json.JSONDecodeError:
                            continue
                        if name == "write_file" and str(args.get("path", "")).endswith(".py"):
                            last_write = args.get("content", "")

                # --- openclaw format: message-based records ---
                if rec.get("type") == "message":
                    msg = rec.get("message", {})
                    for item in msg.get("content", []):
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") == "toolCall":
                            name = item.get("name", "")
                            args = item.get("arguments", {})
                            if name in ("write_file", "Write", "write"):
                                path = str(args.get("path", ""))
                                if path.endswith(".py"):
                                    last_write = args.get("content", "")
                        if item.get("type") == "text":
                            for block in _find_python_blocks(item.get("text", "")):
                                all_blocks.append(block.strip())
    except Exception:
        pass

    if last_write and _syntax_ok(last_write):
        return last_write

    meaningful = [b for b in all_blocks if len(b) > 50]
    if meaningful:
        code = _pick_best_block(meaningful)
        if code:
            return code

    return last_write


# ---------------------------------------------------------------------------
# Verification using LCB's test runner
# ---------------------------------------------------------------------------

def _verify_code(problem, code: str, timeout: int = 6) -> dict:
    """Run code against all test cases using LCB's check_correctness."""
    lcb_repo = _lcb_repo()
    if str(lcb_repo) not in sys.path:
        sys.path.insert(0, str(lcb_repo))
    from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness

    sample = problem.get_evaluation_sample()
    try:
        result_list, metadata = check_correctness(sample, code, timeout=timeout)
    except Exception as e:
        log.warning(f"check_correctness failed: {e}")
        return {"reward": 0.0, "error": str(e), "passed": 0, "total": 0}

    total = len(result_list)
    passed = sum(1 for r in result_list if r is True or r == 1)
    all_pass = all(r is True or r == 1 for r in result_list)

    return {
        "reward": 1.0 if all_pass else 0.0,
        "passed": passed,
        "total": total,
        "results": [int(r) if isinstance(r, bool) else r for r in result_list],
        "metadata": metadata if isinstance(metadata, dict) else {},
    }


# ---------------------------------------------------------------------------
# LiveCodeBench Adapter
# ---------------------------------------------------------------------------

class LiveCodeBenchAdapter(DomainAdapter):
    name = "code_implementation"

    def __init__(self):
        self._problems = {}  # question_id -> CodeGenerationProblem

    def _ensure_loaded(self):
        """Lazy-load problems on first access."""
        if self._problems:
            return
        cfg = _cfg()
        release = cfg.get("release_version", "release_v6")
        problems = _load_problems(release)

        from datetime import datetime
        start_date = cfg.get("start_date")
        end_date = cfg.get("end_date")
        if start_date:
            dt = datetime.strptime(start_date, "%Y-%m-%d")
            problems = [p for p in problems if p.contest_date >= dt]
        if end_date:
            dt = datetime.strptime(end_date, "%Y-%m-%d")
            problems = [p for p in problems if p.contest_date <= dt]

        for p in problems:
            self._problems[p.question_id] = p
        log.info(f"LiveCodeBench: {len(self._problems)} problems loaded")

    def load_tasks(self, args) -> list[dict]:
        """Return task list from loaded problems.

        Filtering: --task for explicit IDs, --split for split file or difficulty.
        """
        self._ensure_loaded()
        problems = list(self._problems.values())

        # 1. Split narrows the pool
        if args.split:
            splits = _load_split()
            if args.split in splits:
                ids = set(splits[args.split])
                problems = [p for p in problems if p.question_id in ids]
            elif args.split in ("easy", "medium", "hard"):
                problems = [p for p in problems if p.difficulty.value == args.split]
            elif args.split.isdigit():
                problems = problems[:int(args.split)]
            else:
                raise ValueError(
                    f"Unknown split: {args.split}. "
                    f"Available: {', '.join(sorted(splits.keys()))}")

        # 2. Task filters within the pool
        if args.task:
            task_ids = set(t.strip() for t in args.task.split(","))
            problems = [p for p in problems if p.question_id in task_ids]

        problems.sort(key=lambda p: (p.contest_date, p.question_id))

        tasks = []
        for p in problems:
            public_tests = []
            for t in p.public_test_cases:
                public_tests.append({"input": t.input, "output": t.output})

            tasks.append({
                "name": p.question_id,
                "title": p.question_title,
                "platform": p.platform.value,
                "difficulty": p.difficulty.value,
                "contest_date": p.contest_date.isoformat(),
                "starter_code": p.starter_code,
                "question_content": p.question_content,
                "public_tests": public_tests,
                "fn_name": p.metadata.get("func_name"),
            })

        return tasks

    def get_agent_timeout(self, task: dict, env_info: dict) -> int:
        return int(_cfg().get("agent_timeout", 600))

    _prompt_template = (Path(__file__).parent / "prompt.md").read_text()

    def build_prompt(self, task: dict, env_info: dict) -> str:
        starter = task.get("starter_code", "")
        if starter:
            format_section = (
                "### Format: You will use the following starter code to write "
                "the solution to the problem and enclose your code within delimiters.\n"
                f"```python\n{starter}\n```\n\n"
            )
        else:
            format_section = (
                "### Format: Read the inputs from stdin solve the problem and "
                "write the answer to stdout (do not directly test on the sample "
                "inputs). Enclose your code within delimiters as follows. Ensure "
                "that when the python program runs, it reads the inputs, runs the "
                "algorithm and writes output to STDOUT.\n"
                "```python\n# YOUR CODE HERE\n```\n\n"
            )

        return self._prompt_template.format(
            question=task["question_content"],
            format_section=format_section,
        )

    def verify(self, task: dict, env_info: dict, trial_dir: Path,
               agent_result: dict | None = None) -> dict:
        """Extract code from agent response and run against test cases."""
        agent_result = agent_result or {}
        code_source = "response"

        response_json = agent_result.get("response_json")
        text = _extract_text_from_oc_response(
            agent_result.get("response", ""), response_json)
        code = _extract_code_from_text(text)

        if not code or not _syntax_ok(code):
            session_code = _extract_code_from_session(agent_result.get("_session_file"))
            if session_code and _syntax_ok(session_code):
                code = session_code
                code_source = "session"

        verifier_dir = trial_dir / "verifier"
        verifier_dir.mkdir(parents=True, exist_ok=True)

        if not code:
            log.warning(f"No code extracted from response for {task['name']}")
            result = {"reward": 0.0, "error": "no_code_extracted",
                      "passed": 0, "total": 0}
        else:
            self._ensure_loaded()
            problem = self._problems.get(task["name"])
            if not problem:
                result = {"reward": 0.0, "error": "problem_not_found",
                          "passed": 0, "total": 0}
            else:
                test_timeout = int(_cfg().get("test_timeout", 6))
                result = _verify_code(problem, code, timeout=test_timeout)

        with open(verifier_dir / "details.json", "w") as f:
            json.dump({
                "code": code,
                "code_source": code_source,
                "reward": result["reward"],
                "passed": result.get("passed", 0),
                "total": result.get("total", 0),
                "results": result.get("results", []),
                "error": result.get("error"),
            }, f, indent=2, ensure_ascii=False)

        if code:
            with open(verifier_dir / "solution.py", "w") as f:
                f.write(code)

        return result

    def aggregate_metrics(self, results: list[dict]) -> dict:
        """Compute per-difficulty pass rates."""
        by_diff = {}
        for r in results:
            diff = r.get("task", {}).get("difficulty", "unknown")
            by_diff.setdefault(diff, {"total": 0, "passed": 0})
            by_diff[diff]["total"] += 1
            reward = r.get("verifier_result", {}).get("reward", 0)
            if reward > 0:
                by_diff[diff]["passed"] += 1

        metrics = {}
        for diff, counts in sorted(by_diff.items()):
            rate = counts["passed"] / counts["total"] if counts["total"] > 0 else 0
            metrics[f"pass@1_{diff}"] = round(rate, 4)
            metrics[f"count_{diff}"] = counts["total"]

        return metrics


register_domain("code_implementation", LiveCodeBenchAdapter)
