#!/usr/bin/env python3
"""Run evaluation with skills injected into prompts.

Three skill sources (mutually exclusive):

1. **Skill cache** (--skill-cache): reuse previously searched/saved skills JSON.
2. **API search** (--api-url + --user-id): query EverCore v1 search API per task.
3. **Static files** (--skills-dir): load SKILL.md files from disk.

Injection strategy is determined by domain_info:
  - "prompt_append" (default): monkey-patch build_prompt to append skills.
  - "task_field" (reasoning): set task[skill_field] for built-in injection.

Usage:
    # API search
    python eval_with_skills.py --api-url http://localhost:1997 --user-id extract_abc123

    # Static files
    python eval_with_skills.py --skills-dir src/skill_evolution/evermemos/skills

    # Reuse cached skills
    python eval_with_skills.py --skill-cache jobs/prev-run/skill_cache.json
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_EVAL_DIR))

_env_file = _PROJECT_ROOT / ".env"
if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

import httpx
from config import get_agent, get_domain, get_config, load_config
from extract_skills import (load_splits_from_adapter, load_task_clusters,
                            resolve_split_file)
from domain_info import BENCHMARK_DESCRIPTORS, DomainInfo, get_task_query
from runner import run_all


# ---------------------------------------------------------------------------
# Skill source: Static files
# ---------------------------------------------------------------------------

def _load_skill_dir(skill_dir: Path) -> str:
    if not skill_dir.exists():
        return ""
    parts = []
    for sub_dir in sorted(skill_dir.iterdir()):
        skill_file = sub_dir / "SKILL.md"
        if not skill_file.exists():
            continue
        content = skill_file.read_text()
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                content = content[end + 3:].strip()
        parts.append(content)
    return "\n\n".join(parts)


def load_cluster_skills(skills_dir: Path, cluster_name: str) -> str:
    """Load GLOBAL + cluster-specific skills from disk."""
    parts = []
    global_text = _load_skill_dir(skills_dir / "GLOBAL")
    if global_text:
        parts.append(global_text)
    cluster_text = _load_skill_dir(skills_dir / cluster_name)
    if cluster_text:
        parts.append(cluster_text)
    return "\n\n".join(parts)


def load_skills_from_files(skills_dir: Path, task_clusters: dict) -> dict:
    """Load skills from disk for all tasks. Returns {tid: skills_text}."""
    task_skills = {}
    for cluster_name, task_ids in task_clusters.items():
        skills_text = load_cluster_skills(skills_dir, cluster_name)
        if not skills_text:
            print(f"  SKIP {cluster_name}: no skills in {skills_dir / cluster_name}")
            continue
        for tid in task_ids:
            task_skills[tid] = skills_text
        print(f"  {cluster_name}: {len(task_ids)} tasks, skills loaded")
    return task_skills


# ---------------------------------------------------------------------------
# Skill source: EverCore v1 search API
# ---------------------------------------------------------------------------

async def _search_one(client, api_url, user_id, tid, question, top_k, method):
    if not question.strip():
        return tid, ""
    resp = await client.post(f"{api_url}/api/v1/memories/search", json={
        "query": question, "method": method,
        "memory_types": ["agent_memory"],
        "filters": {"user_id": user_id},
        "top_k": top_k,
    })
    resp.raise_for_status()
    data = resp.json().get("data", {})
    skills = data.get("agent_memory", {}).get("skills", [])
    parts = []
    for skill in skills:
        name = skill.get("name", "")
        content = skill.get("content", "")
        if content:
            parts.append(f"### {name}\n{content}" if name else content)
    return tid, "\n\n".join(parts)


async def search_all_skills(api_url, user_id, task_questions, top_k, parallel,
                             method="vector"):
    sem = asyncio.Semaphore(parallel)
    results = {}
    done = [0]
    total = len(task_questions)

    async def _bounded(tid, question):
        async with sem:
            tid, text = await _search_one(client, api_url, user_id, tid, question, top_k, method)
            done[0] += 1
            if done[0] % 20 == 0 or done[0] == total:
                print(f"    Searched {done[0]}/{total}")
            return tid, text

    async with httpx.AsyncClient(timeout=600.0) as client:
        tasks = [_bounded(tid, q) for tid, q in task_questions.items()]
        for coro in asyncio.as_completed(tasks):
            try:
                tid, text = await coro
                if text:
                    results[tid] = text
            except Exception as e:
                print(f"    WARNING: search failed: {e}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_eval_config() -> dict:
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    if cfg_path.exists():
        import yaml
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    eval_cfg = _load_eval_config()

    parser = argparse.ArgumentParser(description="Evaluate with skills injected")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--split-file", default=None)
    # Skill source: API
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--search-method", default="vector")
    # Skill source: static
    parser.add_argument("--skills-dir", default=None)
    # Skill source: cache
    parser.add_argument("--skill-cache", default=None, help="Reuse saved skill_cache.json")
    # Common
    parser.add_argument("--task", default=None)
    parser.add_argument("--clusters", nargs="*")
    parser.add_argument("--split", default=None)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--job", default=None)
    args = parser.parse_args()

    config_path = args.config or str(_PROJECT_ROOT / "config.yaml")
    load_config(config_path)
    cfg = get_config()
    domain_name = args.domain or cfg["domain"]["name"]
    split = args.split or eval_cfg.get("split_eval", "test")

    domain = get_domain(domain_name)

    # Resolve task IDs
    if args.task:
        all_task_ids = [t.strip() for t in args.task.split(",")]
        task_clusters = {"default": all_task_ids}
    else:
        split_file = resolve_split_file(args.split_file, domain_name)
        if split_file:
            task_clusters = load_task_clusters(split_file, split, args.clusters)
        else:
            print(f"  No split file for {domain_name}, loading from adapter")
            task_clusters = load_splits_from_adapter(domain, split)
        all_task_ids = list(dict.fromkeys(
            tid for ids in task_clusters.values() for tid in ids
        ))

    # --- Determine skill source ---
    use_api = bool(args.api_url or args.user_id)
    if use_api and not args.user_id:
        skills_dir = Path(args.skills_dir or eval_cfg.get("skills_dir", "src/skill_evolution/evermemos/skills"))
        meta_files = sorted(skills_dir.glob("metadata_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if meta_files:
            meta = json.loads(meta_files[0].read_text())
            args.user_id = meta.get("user_id")
            if not args.api_url:
                args.api_url = meta.get("api_url")
            print(f"  Auto-loaded user_id={args.user_id} from {meta_files[0].name}")
        if not args.user_id:
            print("ERROR: --user-id required for API search mode.")
            return

    api_url = args.api_url or eval_cfg.get("api_url", "http://localhost:1997")

    if args.skill_cache:
        task_skills = json.loads(Path(args.skill_cache).read_text())
        print(f"Skill source: cache ({args.skill_cache}), {len(task_skills)} tasks")

    elif use_api:
        print(f"Skill source: EverCore search ({args.search_method}, top_k={args.top_k})")
        bench_cfg = {}
        try:
            from config import get_domain_config
            bench_cfg = get_domain_config(domain_name)
        except Exception:
            pass

        tmp_args = argparse.Namespace(
            task=",".join(all_task_ids), split=None, trials=1,
            parallel=1, max_retries=0, live=False, disk_budget=None,
        )
        tasks_list = domain.load_tasks(tmp_args)
        task_questions = {}
        for t in tasks_list:
            query = get_task_query(domain_name, t, bench_cfg)
            if query:
                task_questions[str(t["name"])] = query

        print(f"  Tasks: {len(task_questions)}")
        task_skills = asyncio.run(
            search_all_skills(api_url, args.user_id, task_questions,
                              args.top_k, args.parallel, args.search_method)
        )
        print(f"  {len(task_skills)}/{len(task_questions)} tasks with skills\n")

    else:
        skills_dir = Path(args.skills_dir or eval_cfg.get("skills_dir", "src/skill_evolution/evermemos/skills"))
        print(f"Skill source: static files ({skills_dir})")
        task_skills = load_skills_from_files(skills_dir, task_clusters)

    print(f"Coverage: {len(task_skills)}/{len(all_task_ids)} tasks with skills")
    if not task_skills:
        print("No tasks with skills. Exiting.")
        return

    # --- Injection strategy (from registry) ---
    info = BENCHMARK_DESCRIPTORS.get(domain_name, DomainInfo())

    if info.skill_injection == "task_field":
        print(f"  Injection: task['{info.skill_field}'] (built-in)")
    else:
        _original = domain.__class__.build_prompt

        def _patched(self, task, env_info):
            prompt = _original(self, task, env_info)
            text = task_skills.get(str(task["name"]))
            if text:
                prompt += (
                    "\n\n## Domain-Specific Strategies\n\n"
                    "The following strategies are specifically designed for this type of task. "
                    "You MUST apply them:\n\n" + text
                )
            return prompt

        domain.__class__.build_prompt = _patched

    # --- Run ---
    agent = get_agent(cfg["agent"]["name"])
    from datetime import datetime
    import uuid
    job_name = args.job or f"evermemos-{domain_name}-{datetime.now().strftime('%m%d_%H%M')}-{uuid.uuid4().hex[:4]}"
    job_dir = Path(cfg["job_dir"]) / job_name
    job_dir.mkdir(parents=True, exist_ok=True)

    cache_path = job_dir / "skill_cache.json"
    with open(cache_path, "w") as f:
        json.dump(task_skills, f, ensure_ascii=False, indent=2)

    run_args = argparse.Namespace(
        task=",".join(all_task_ids), split=None,
        trials=cfg.get("trials", 1), parallel=args.parallel,
        max_retries=cfg.get("max_retries", 2), live=args.live,
        disk_budget=None,
    )

    tasks = domain.load_tasks(run_args)

    if info.skill_injection == "task_field":
        injected = 0
        for t in tasks:
            tid = str(t["name"])
            if tid in task_skills:
                t[info.skill_field] = task_skills[tid]
                injected += 1
        print(f"  Injected skills into {injected} tasks")

    print(f"Running {len(tasks)} tasks ({len(task_skills)} with skills)\n")
    run_all(tasks, domain, agent, job_dir, run_args)


if __name__ == "__main__":
    main()
