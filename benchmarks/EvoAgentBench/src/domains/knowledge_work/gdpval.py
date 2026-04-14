"""GDPVal domain adapter.

No Docker needed. Agent creates deliverable files (Excel, PDF, Word, etc.)
in a workspace directory. Verification: LLM-based rubric scoring.

Data source: HuggingFace openai/gdpval dataset (loaded directly via datasets lib).
"""

import json
import logging
import os
import shutil
from pathlib import Path

from domains.base import DomainAdapter
from domains.knowledge_work.evaluate import evaluate_rubric
from config import get_domain_config, register_domain

log = logging.getLogger("evoagentbench")

_PROMPT_TEMPLATE = (Path(__file__).parent / "prompt.md").read_text()


def _cfg():
    return get_domain_config("knowledge_work")


def _load_dataset() -> list[dict]:
    """Load GDPVal dataset from HuggingFace (openai/gdpval).

    Uses the datasets library. Results are cached locally by HuggingFace.
    """
    from datasets import load_dataset

    cfg = _cfg()
    owner = cfg.get("hf_dataset_owner", "openai")
    name = cfg.get("hf_dataset_name", "gdpval")
    dataset_name = f"{owner}/{name}"
    ds = load_dataset(dataset_name, split="train")

    records = []
    for row in ds:
        rubric_json = row["rubric_json"]
        # Ensure rubric_json is a string for consistent handling
        if not isinstance(rubric_json, str):
            rubric_json = json.dumps(rubric_json, ensure_ascii=False)
        records.append({
            "task_id": row["task_id"],
            "sector": row["sector"],
            "occupation": row["occupation"],
            "prompt": row["prompt"],
            "reference_files": row.get("reference_files", []),
            "reference_file_urls": row.get("reference_file_urls", []),
            "deliverable_files": row.get("deliverable_files", []),
            "deliverable_file_urls": row.get("deliverable_file_urls", []),
            "rubric_json": rubric_json,
            "rubric_pretty": row.get("rubric_pretty", ""),
        })
    log.info(f"Loaded {len(records)} tasks from {dataset_name}")
    return records


class GDPValAdapter(DomainAdapter):
    name = "knowledge_work"

    def __init__(self):
        self._dataset = None

    def _get_dataset(self) -> list[dict]:
        if self._dataset is None:
            self._dataset = _load_dataset()
        return self._dataset

    def pass_threshold(self) -> float:
        return float(_cfg().get("pass_threshold", 0.6))

    def _load_split_ids(self, split: str) -> list[str] | None:
        """Load task IDs from split file (clusters format)."""
        split_file = _cfg().get("split_file")
        if not split_file or not Path(split_file).exists():
            return None
        with open(split_file) as f:
            raw = json.load(f)
        if split == "all":
            seen = set()
            ids = []
            for cluster in raw.get("clusters", {}).values():
                for part in ("train", "test"):
                    for tid in cluster.get(part, []):
                        if tid not in seen:
                            seen.add(tid)
                            ids.append(tid)
            return ids
        if split in raw and isinstance(raw[split], list):
            return raw[split]
        if "clusters" in raw:
            seen = set()
            ids = []
            for cluster in raw["clusters"].values():
                for tid in cluster.get(split, []):
                    if tid not in seen:
                        seen.add(tid)
                        ids.append(tid)
            return ids if ids else None
        return None

    def load_tasks(self, args) -> list[dict]:
        dataset = self._get_dataset()

        # 1. Split narrows the pool
        if args.split:
            split_ids = self._load_split_ids(args.split)
            if split_ids is not None:
                id_set = set(split_ids)
                dataset = [r for r in dataset if r["task_id"] in id_set]
            elif args.split.isdigit():
                dataset = dataset[:int(args.split)]
            else:
                raise ValueError(
                    f"Unknown split: {args.split}. "
                    f"Available: train, test, all (or a number for first N)")

        # 2. Task filters within the pool
        if args.task:
            ids = [t.strip() for t in args.task.split(",")]
            dataset = [r for r in dataset if r["task_id"] in ids
                       or r["task_id"][:8] in ids]

        tasks = []
        for rec in dataset:
            tid = rec["task_id"]
            short_id = tid[:8]
            tasks.append({
                "name": short_id,
                "task_id": tid,
                "sector": rec["sector"],
                "occupation": rec["occupation"],
                "prompt": rec["prompt"],
                "reference_files": rec.get("reference_files", []),
                "reference_file_urls": rec.get("reference_file_urls", []),
                "deliverable_files": rec.get("deliverable_files", []),
                "deliverable_file_urls": rec.get("deliverable_file_urls", []),
                "rubric_json": rec.get("rubric_json", "[]"),
                "rubric_pretty": rec.get("rubric_pretty", ""),
            })
        return tasks

    def setup(self, task: dict, agent_name: str, trial: int) -> dict:
        """Create workspace directory and prepare reference files."""
        cfg = _cfg()
        job_dir = task.get("_job_dir")
        if job_dir:
            workspace_root = Path(job_dir) / "workspaces"
        else:
            workspace_root = Path(cfg.get("workspace_dir", "./jobs/workspaces"))
        workspace = workspace_root / f"{task['name']}_t{trial}"
        workspace.mkdir(parents=True, exist_ok=True)

        # Copy reference files to workspace if available locally
        ref_dir = Path(cfg.get("reference_dir", ""))
        ref_paths = []
        ref_section_parts = []

        if ref_dir.exists():
            task_ref_dir = ref_dir / task["task_id"]
            if task_ref_dir.exists():
                for f in task_ref_dir.iterdir():
                    dest = workspace / f.name
                    shutil.copy2(f, dest)
                    ref_paths.append(str(dest))
                    ref_section_parts.append(f"- `{dest}` ({f.name})")

        # If no local files, download from URLs and cache
        if not ref_paths and task.get("reference_file_urls"):
            from urllib.request import urlretrieve
            cache_dir = ref_dir / task["task_id"] if ref_dir else workspace / ".ref_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            for url, rel_path in zip(task["reference_file_urls"], task["reference_files"]):
                fname = Path(rel_path).name
                cached = cache_dir / fname
                dest = workspace / fname
                if not cached.exists():
                    try:
                        urlretrieve(url, str(cached))
                    except Exception as e:
                        log.warning(f"  Download error for {fname}: {e}")
                        continue
                if not dest.exists():
                    shutil.copy2(str(cached), str(dest))
                ref_paths.append(str(dest))
                ref_section_parts.append(f"- `{dest}` ({fname})")

        return {
            "workspace_dir": str(workspace),
            "reference_paths": ref_paths,
            "reference_section": "\n".join(ref_section_parts) if ref_section_parts
                                 else "No reference files for this task.",
        }

    def get_agent_timeout(self, task: dict, env_info: dict) -> int:
        return int(_cfg().get("agent_timeout", 1800))

    def build_prompt(self, task: dict, env_info: dict) -> str:
        return _PROMPT_TEMPLATE.format(
            prompt=task["prompt"],
            reference_section=env_info.get("reference_section", "None"),
            workspace_dir=env_info["workspace_dir"],
        )

    def verify(self, task: dict, env_info: dict, trial_dir: Path,
               agent_result: dict | None = None) -> dict:
        """Evaluate deliverables in workspace against rubric (ClawWork LLMEvaluator).

        Note: openclaw system files and reference files are filtered in evaluate.py.
        """
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            log.error("  OPENROUTER_API_KEY not set, skipping evaluation")
            return {"reward": 0.0, "error": "missing_api_key"}

        cfg = _cfg()
        workspace = Path(env_info["workspace_dir"])
        model_owner = cfg.get("eval_model_owner", "openai")
        model_name = cfg.get("eval_model_name", "gpt-4o")
        model = f"{model_owner}/{model_name}"
        meta_prompts_dir = cfg.get("meta_prompts_dir", "")

        result = evaluate_rubric(task, workspace, api_key,
                                 meta_prompts_dir=meta_prompts_dir, model=model)

        # Save evaluation details
        verifier_dir = trial_dir / "verifier"
        verifier_dir.mkdir(parents=True, exist_ok=True)
        with open(verifier_dir / "eval_details.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return result

    def cleanup(self, task: dict, env_info: dict):
        """Optionally clean up workspace (keep for now for debugging)."""
        pass


register_domain("knowledge_work", GDPValAdapter)
