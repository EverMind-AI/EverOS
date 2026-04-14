"""BrowseComp-Plus domain adapter.

No Docker needed. Agent searches a local corpus via MCP to answer questions.
Verification: LLM Judge (semantic correctness) with exact-match fallback.

The MCP search server (FAISS) is managed automatically: started in
initialize() and stopped in finalize().  Set mcp_server.auto_start: false
in the yaml config to manage it yourself.
"""

import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from domains.base import DomainAdapter
from config import get_domain_config, register_domain


# ---------------------------------------------------------------------------
# Answer extraction and exact-match verification
# ---------------------------------------------------------------------------

def extract_answer(response_text: str) -> str:
    """Extract answer from agent response.

    Looks for 'Exact Answer: ...', 'ANSWER: ...', or 'extracted_final_answer: ...'.
    """
    for pattern in [
        r"Exact Answer:\s*(.+?)(?=\nConfidence:|\n\n|\Z)",
        r"ANSWER:\s*(.+?)(?=\nConfidence:|\n\n|\Z)",
        r"extracted_final_answer:\s*(.+?)(?=\nConfidence:|\n\n|\Z)",
    ]:
        match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
        if match:
            return " ".join(match.group(1).split())
    return ""


def verify_exact_match(predicted: str, ground_truth: str) -> dict:
    """Normalized exact match scoring."""
    def normalize(s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r'[^\w\s]', '', s)
        s = re.sub(r'\s+', ' ', s)
        return s.strip()

    match = normalize(predicted) == normalize(ground_truth)
    return {"reward": 1.0 if match else 0.0, "method": "exact_match"}

log = logging.getLogger("evoagentbench")

_PROMPT_TEMPLATE = (Path(__file__).parent / "prompt.md").read_text()


def _cfg():
    return get_domain_config("information_retrieval")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dataset() -> dict[str, dict]:
    """Load BrowseComp-Plus JSONL dataset, keyed by query_id."""
    records = {}
    with open(_cfg()["dataset_file"]) as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            qid = str(rec.get("query_id", i))
            rec["id"] = qid
            if "query" in rec and "problem" not in rec:
                rec["problem"] = rec["query"]
            records[qid] = rec
    return records


def _load_split() -> dict:
    """Load task split file. Returns {split_name: [qid, ...]}."""
    split_path = _cfg().get("split_file")
    if not split_path or not Path(split_path).exists():
        return {}
    with open(split_path) as f:
        raw = json.load(f)

    splits = {"train": [], "test": []}
    for cluster_name, cluster in raw.get("clusters", {}).items():
        splits["train"].extend(str(i) for i in cluster.get("train", []))
        splits["test"].extend(str(i) for i in cluster.get("test", []))
        for part in ("train", "test"):
            splits[f"{cluster_name}_{part}"] = [str(i) for i in cluster.get(part, [])]

    splits["all"] = splits["train"] + splits["test"]
    return splits


# ---------------------------------------------------------------------------
# MCP server helpers
# ---------------------------------------------------------------------------

def _check_health(url: str, timeout: float = 5) -> bool:
    """Return True if the MCP server health endpoint is reachable."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# BrowseComp-Plus Adapter
# ---------------------------------------------------------------------------

class BrowseCompPlusAdapter(DomainAdapter):
    name = "information_retrieval"

    def __init__(self):
        self._mcp_proc: subprocess.Popen | None = None

    # --- Domain-level lifecycle ---

    _START_SCRIPT = Path(__file__).parent.parent.parent / "utils" / "browsecomp-plus-tools" / "start_mcp.py"

    def _mcp_url(self) -> str:
        port = _cfg().get("mcp_server", {}).get("port", 9100)
        return f"http://localhost:{port}/mcp"

    def initialize(self, args):
        """Start the MCP search server if auto_start is enabled."""
        mcp_cfg = _cfg().get("mcp_server", {})
        url = self._mcp_url()

        if not mcp_cfg.get("auto_start", False):
            if not _check_health(url):
                log.warning(f"MCP server not reachable at {url}")
            return

        if _check_health(url):
            log.info(f"MCP server already running at {url}")
            return

        cfg_path = str(Path(__file__).parent / "information_retrieval.yaml")

        log.info(f"Starting MCP server on port {mcp_cfg.get('port', 9100)}...")
        self._mcp_log_file = open("/tmp/evoagentbench_mcp_server.log", "w")
        self._mcp_proc = subprocess.Popen(
            [sys.executable, str(self._START_SCRIPT), cfg_path],
            stdout=self._mcp_log_file, stderr=self._mcp_log_file,
            preexec_fn=os.setsid,
        )

        startup_timeout = mcp_cfg.get("startup_timeout", 120)
        deadline = time.time() + startup_timeout
        while time.time() < deadline:
            if self._mcp_proc.poll() is not None:
                self._mcp_log_file.close()
                err = Path("/tmp/evoagentbench_mcp_server.log").read_text()
                raise RuntimeError(
                    f"MCP server exited with code {self._mcp_proc.returncode}:\n"
                    f"{err[:2000]}"
                )
            if _check_health(url):
                log.info(f"MCP server ready at {url}")
                return
            time.sleep(2)

        self._mcp_proc.kill()
        raise RuntimeError(
            f"MCP server did not become ready within {startup_timeout}s. "
            f"Check /tmp/evoagentbench_mcp_server.log for details."
        )

    def finalize(self):
        """Stop the MCP search server if we started it."""
        if self._mcp_proc is None:
            return
        log.info("Stopping MCP server...")
        try:
            os.killpg(os.getpgid(self._mcp_proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            self._mcp_proc = None
            return
        try:
            self._mcp_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            log.warning("MCP server did not stop, sending SIGKILL")
            try:
                os.killpg(os.getpgid(self._mcp_proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
        self._mcp_proc = None

    # --- Task lifecycle ---

    def load_tasks(self, args) -> list[dict]:
        dataset = _load_dataset()

        # 1. Split narrows the pool
        if args.split:
            splits = _load_split()
            if args.split in splits:
                ids = splits[args.split]
            elif args.split.isdigit():
                ids = list(dataset.keys())[:int(args.split)]
            else:
                raise ValueError(
                    f"Unknown split: {args.split}. "
                    f"Available: {', '.join(sorted(splits.keys()))}"
                )
        else:
            ids = list(dataset.keys())

        # 2. Task filters within the pool
        if args.task:
            task_ids = set(t.strip() for t in args.task.split(","))
            ids = [i for i in ids if i in task_ids]

        return [
            {
                "name": qid,
                "problem": dataset[qid]["problem"],
                "answer": dataset[qid]["answer"],
                "topic": dataset[qid].get("problem_topic", ""),
            }
            for qid in ids if qid in dataset
        ]

    def setup(self, task: dict, agent_name: str, trial: int) -> dict:
        cfg = _cfg()
        return {
            "mcp_servers": {
                "bcp-search": {
                    "type": "sse",
                    "url": self._mcp_url(),
                },
            },
            "disabled_tools": cfg.get("disabled_tools", []),
        }

    def get_agent_timeout(self, task: dict, env_info: dict) -> int:
        return int(_cfg().get("agent_timeout", 600))

    def build_prompt(self, task: dict, env_info: dict) -> str:
        return _PROMPT_TEMPLATE.format(question=task["problem"])

    def verify(self, task: dict, env_info: dict, trial_dir: Path,
               agent_result: dict | None = None) -> dict:
        response = (agent_result or {}).get("response", "")
        predicted = extract_answer(response)
        ground_truth = task["answer"]
        gt_question = task["problem"]

        verifier_dir = trial_dir / "verifier"
        verifier_dir.mkdir(parents=True, exist_ok=True)

        # Primary: LLM Judge
        cfg = _cfg()
        judge_cfg = cfg.get("judge", {})

        judge_result = {}
        if judge_cfg.get("model") and judge_cfg.get("api_base"):
            from domains.information_retrieval.judge import call_judge
            judge_result = call_judge(
                question=gt_question,
                response=response,
                correct_answer=ground_truth,
                model=judge_cfg["model"],
                api_base=judge_cfg["api_base"],
                api_key=judge_cfg.get("api_key", "EMPTY"),
                max_tokens=judge_cfg.get("max_tokens", 4096),
                temperature=judge_cfg.get("temperature", 0.7),
                disable_thinking=judge_cfg.get("disable_thinking", True),
            )

        # Secondary: exact match
        exact_result = verify_exact_match(predicted, ground_truth)

        # Determine reward: judge takes priority
        if judge_result and not judge_result.get("parse_error"):
            reward = 1.0 if judge_result.get("correct") else 0.0
            method = "llm_judge"
        else:
            reward = exact_result["reward"]
            method = "exact_match"

        # Save details
        with open(verifier_dir / "details.json", "w") as f:
            json.dump({
                "predicted": predicted,
                "ground_truth": ground_truth,
                "judge_result": judge_result,
                "exact_match": exact_result,
                "reward": reward,
                "method": method,
            }, f, indent=2, ensure_ascii=False)

        return {
            "reward": reward,
            "method": method,
            "judge_correct": judge_result.get("correct"),
            "exact_match": exact_result["reward"] > 0,
        }

    def aggregate_metrics(self, results: list[dict]) -> dict:
        total = len(results)
        if total == 0:
            return {}

        judge_correct = sum(
            1 for r in results
            if r.get("verifier_result", r).get("judge_correct") is True
        )
        exact_correct = sum(
            1 for r in results
            if r.get("verifier_result", r).get("exact_match") is True
        )

        return {
            "judge_accuracy": round(judge_correct / total * 100, 2),
            "exact_match_accuracy": round(exact_correct / total * 100, 2),
            "total": total,
        }


register_domain("information_retrieval", BrowseCompPlusAdapter)
