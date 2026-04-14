"""Summary generation utilities.

Collects result.json files from a job directory, computes pass@1,
avg turns/tokens/elapsed metrics, prints summary, and saves summary.json.
"""

import json
import logging
import math
from pathlib import Path

log = logging.getLogger("evoagentbench")


def print_summary(job_dir: Path, trials: int = 1, domain=None):
    """Print final summary with metrics and save summary.json."""
    print(f"{'─' * 70}")

    # --- Collect results ---
    task_results = {}  # task_name -> list of per-trial dicts
    agents_seen = set()
    for rf in job_dir.glob("*/result.json"):
        # Skip retry backup directories (e.g. task__trial_1_retry1)
        if "_retry" in rf.parent.name:
            continue
        with open(rf) as f:
            r = json.load(f)
        agents_seen.add(r.get("agent", "unknown"))
        task_name = r.get("task_name", rf.parent.name)

        # Support both nested and flat verifier_result format
        vr = r.get("verifier_result", {})
        nested = vr.get("rewards", {}).get("reward")
        reward = nested if nested is not None else vr.get("reward", 0)

        token_usage = r.get("token_usage", {})
        if task_name not in task_results:
            task_results[task_name] = []
        task_results[task_name].append({
            "reward": reward,
            "turns": token_usage.get("turns", 0),
            "input_tokens": token_usage.get("input", 0),
            "output_tokens": token_usage.get("output", 0),
            "total_tokens": token_usage.get("total", 0),
            "elapsed": r.get("agent_result", {}).get("elapsed_sec", 0),
            "verifier_result": vr,
        })

    total_tasks = len(task_results)
    if total_tasks == 0:
        print("\n  No results found.")
        return

    if len(agents_seen) > 1:
        log.warning(f"  Mixed agents in job dir: {agents_seen}")

    # --- Per-task aggregation ---
    threshold = domain.pass_threshold() if domain is not None else 0.0
    per_task = {}
    for task, results in sorted(task_results.items()):
        n = len(results)
        avg_reward = sum(r["reward"] for r in results) / n
        passed = sum(1 for r in results if r["reward"] > threshold)
        per_task[task] = {
            "trials": n,
            "passed": passed,
            "pass@1": round(passed / n, 4),
            "avg_reward": round(avg_reward, 4),
            "avg_turns": round(sum(r["turns"] for r in results) / n, 1),
            "avg_tokens": round(sum(r["total_tokens"] for r in results) / n),
            "avg_elapsed_sec": round(sum(r["elapsed"] for r in results) / n, 1),
        }

    # --- Global metrics ---
    pass1_values = [pt["pass@1"] for pt in per_task.values()]
    pass1_mean = sum(pass1_values) / len(pass1_values)
    if len(pass1_values) > 1:
        pass1_var = sum((x - pass1_mean) ** 2 for x in pass1_values) / (len(pass1_values) - 1)
        pass1_se = math.sqrt(pass1_var / len(pass1_values))
    else:
        pass1_se = 0.0

    all_results = [r for results in task_results.values() for r in results]
    total_trials = len(all_results)
    avg_turns = sum(r["turns"] for r in all_results) / total_trials
    avg_input = sum(r["input_tokens"] for r in all_results) / total_trials
    avg_output = sum(r["output_tokens"] for r in all_results) / total_trials
    avg_total = sum(r["total_tokens"] for r in all_results) / total_trials
    avg_elapsed = sum(r["elapsed"] for r in all_results) / total_trials

    # --- Build summary ---
    summary = {
        "tasks": total_tasks,
        "trials_per_task": trials,
        "pass@1": {"mean": round(pass1_mean, 4), "stderr": round(pass1_se, 4)},
        "avg_turns": round(avg_turns, 1),
        "avg_tokens": {
            "input": round(avg_input),
            "output": round(avg_output),
            "total": round(avg_total),
        },
        "avg_elapsed_sec": round(avg_elapsed, 1),
    }

    # --- Domain-specific metrics ---
    if domain is not None:
        bm = domain.aggregate_metrics(all_results)
        if bm:
            summary["domain_metrics"] = bm

    summary["per_task"] = per_task

    # --- Print ---
    print()
    print(f"  pass@1: {pass1_mean*100:.1f}% +/- {pass1_se*100:.1f}%")
    print(f"  avg turns: {avg_turns:.1f}  |  avg tokens: {avg_total:.0f}  |  avg elapsed: {avg_elapsed:.1f}s")
    print(f"  ({total_trials} total trials)")

    failed_tasks = sorted(t for t, pt in per_task.items() if pt["passed"] == 0)
    if failed_tasks:
        print(f"  Never passed ({len(failed_tasks)}): {', '.join(failed_tasks)}")

    summary_file = job_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary saved to {summary_file}")
    print()
