"""Generic trial execution and scheduling.

Handles: single trial execution, retry logic, multi-trial loops,
parallel/sequential scheduling, and disk-aware scheduling.
"""

import json
import logging
import shutil
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("evoagentbench")


# ---------------------------------------------------------------------------
# Single trial execution
# ---------------------------------------------------------------------------

def run_task_once(task, domain, agent, job_dir, trial=1, attempt=1, live=False):
    """Execute one trial: setup → agent → verify → cleanup."""
    task_name = task["name"]
    session_id = f"{domain.name}-{task_name}-{uuid.uuid4().hex[:6]}"
    trial_dir = job_dir / f"{task_name}__trial_{trial}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    env_info = {}
    mcp_setup_done = False
    result = {
        "task_name": task_name,
        "agent": agent.name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "trial": trial,
        "attempt": attempt,
        "agent_result": {},
        "verifier_result": {"reward": 0.0},
        "exception_info": None,
    }

    try:
        # 1. Setup environment
        task["_job_dir"] = job_dir
        env_info = domain.setup(task, agent.name, trial)

        # 2. Build prompt and call agent
        prompt = domain.build_prompt(task, env_info)
        agent_timeout = domain.get_agent_timeout(task, env_info)

        log.info(f"[{task_name}] Running {agent.name}...")

        mcp_servers = env_info.get("mcp_servers")
        if mcp_servers:
            agent.setup_mcp(mcp_servers, env_info.get("disabled_tools"))
            mcp_setup_done = True

        # Prepare agent temp config before watcher starts, so session
        # path points to the correct directory.
        if hasattr(agent, '_ensure_temp_config'):
            agent._ensure_temp_config()

        stop_watch = threading.Event()
        watcher = None
        if live:
            watcher = threading.Thread(
                target=agent.watch_session,
                args=(session_id, stop_watch, task_name),
                daemon=True,
            )
            watcher.start()

        try:
            agent_result = agent.call_agent(prompt, session_id, timeout=agent_timeout)
        finally:
            stop_watch.set()
            if watcher:
                watcher.join(timeout=3)

        result["agent_result"] = agent_result
        result["agent_result"]["_session_file"] = str(agent._session_file(session_id))

        # 3. Run verification
        log.info(f"[{task_name}] Running verification...")
        verifier_result = domain.verify(task, env_info, trial_dir,
                                                  agent_result=result["agent_result"])
        result["verifier_result"] = verifier_result

        reward = verifier_result.get("reward", 0.0)
        elapsed = agent_result.get("elapsed_sec", 0)
        mark = "\u2713" if reward > 0 else "\u2717"
        log.info(f"[{task_name}] {mark} reward={reward:.2f} ({elapsed:.0f}s)")

    except Exception as e:
        log.error(f"[{task_name}] ERROR: {e}")
        result["exception_info"] = {
            "type": type(e).__name__,
            "message": str(e),
        }

    finally:
        # Cleanup environment first — must run even if saving results fails
        try:
            domain.cleanup(task, env_info)
        except Exception as ce:
            log.warning(f"[{task_name}] cleanup failed: {ce}")

        result["ended_at"] = datetime.now(timezone.utc).isoformat()
        result["session_id"] = session_id

        # Collect session and extract stats
        session_stats = agent.collect_session(session_id, trial_dir)
        result["last_stop_reason"] = session_stats.pop("last_stop_reason", None)
        result["token_usage"] = session_stats

        # Teardown MCP and clean up temp config after session is collected
        if mcp_setup_done:
            agent.teardown_mcp()
        elif hasattr(agent, '_cleanup_temp_config'):
            agent._cleanup_temp_config()

        # Truncate response for storage only (verify already used full text)
        save_result = result.copy()
        save_result["agent_result"] = result.get("agent_result", {}).copy()
        resp = save_result["agent_result"].get("response", "")
        if len(resp) > 10000:
            save_result["agent_result"]["response"] = resp[:10000] + f"\n...(truncated, {len(resp)} chars)"

        result_file = trial_dir / "result.json"
        with open(result_file, "w") as f:
            json.dump(save_result, f, indent=2, ensure_ascii=False)

        reward_val = result["verifier_result"].get("reward", 0.0)
        verifier_dir = trial_dir / "verifier"
        verifier_dir.mkdir(parents=True, exist_ok=True)
        (verifier_dir / "reward.txt").write_text(str(reward_val))

    return result


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def run_task_with_retry(task, domain, agent, job_dir, trial=1, live=False, max_retries=2):
    """Run a single trial with retry logic."""
    task_name = task["name"]

    for attempt in range(1, max_retries + 2):
        result = run_task_once(task, domain, agent, job_dir,
                               trial=trial, attempt=attempt, live=live)

        retry_reason = agent.should_retry(result)
        if retry_reason and attempt <= max_retries:
            log.warning(f"[{task_name}] trial {trial} retry {attempt}/{max_retries}: {retry_reason}")
            old_trial = job_dir / f"{task_name}__trial_{trial}"
            backup = job_dir / f"{task_name}__trial_{trial}_retry{attempt}"
            if old_trial.exists():
                if backup.exists():
                    shutil.rmtree(backup)
                old_trial.rename(backup)
            continue
        break

    return result


# ---------------------------------------------------------------------------
# Multi-trial loop for one task
# ---------------------------------------------------------------------------

def run_task_trials(task, domain, agent, job_dir, trials,
                    live=False, max_retries=2, on_trial_done=None):
    """Run all trials for one task. Calls pre/post_task_trials hooks."""
    task_name = task["name"]
    results = []

    domain.pre_task_trials(task)
    try:
        for k in range(1, trials + 1):
            trial_dir = job_dir / f"{task_name}__trial_{k}"
            # Skip completed trials
            if (trial_dir / "result.json").exists():
                try:
                    with open(trial_dir / "result.json") as f:
                        r = json.load(f)
                    results.append(r)
                    continue
                except (json.JSONDecodeError, OSError) as e:
                    log.warning(f"[{task_name}] Corrupt result.json in trial {k}, re-running: {e}")

            result = run_task_with_retry(task, domain, agent, job_dir,
                                         trial=k, live=live, max_retries=max_retries)
            results.append(result)

            reward = result.get("verifier_result", {}).get("reward", 0)
            elapsed = result.get("agent_result", {}).get("elapsed_sec", 0)
            if on_trial_done:
                on_trial_done(task_name, k, reward, elapsed)
    finally:
        domain.post_task_trials(task)

    return results


# ---------------------------------------------------------------------------
# Scheduling: sequential, parallel, disk-aware
# ---------------------------------------------------------------------------

def _run_disk_aware(tasks, run_fn, budget_bytes, max_workers=32, cost_fn=None):
    """Schedule tasks to maximize parallelism within a disk budget."""
    import asyncio

    if cost_fn is None:
        cost_fn = lambda t: 100 * 1024 * 1024  # 100MB default
    task_costs = {t["name"]: cost_fn(t) for t in tasks}
    task_map = {t["name"]: t for t in tasks}
    pending_names = sorted(task_map.keys(), key=lambda n: task_costs[n], reverse=True)

    async def _schedule():
        nonlocal pending_names
        disk_used = 0
        active = 0
        done = asyncio.Event()

        async def _worker(name, cost):
            nonlocal disk_used, active
            try:
                await asyncio.to_thread(run_fn, task_map[name])
            except Exception as e:
                log.error(f"[{name}] Unhandled exception: {e}")
            finally:
                disk_used -= cost
                active -= 1
                done.set()

        log.info(f"[scheduler] budget={budget_bytes/1e9:.0f}G workers={max_workers} tasks={len(pending_names)}")

        def _dispatch(name, cost):
            nonlocal disk_used, active
            pending_names.remove(name)
            disk_used += cost
            active += 1
            log.info(f"[scheduler] +{name} cost={cost/1e9:.1f}G "
                     f"used={disk_used/1e9:.1f}/{budget_bytes/1e9:.0f}G "
                     f"active={active}/{max_workers}")
            asyncio.create_task(_worker(name, cost))

        while pending_names or active > 0:
            for n in list(pending_names):
                if active >= max_workers:
                    break
                cost = task_costs[n]
                if disk_used + cost <= budget_bytes or disk_used == 0:
                    _dispatch(n, cost)
                    break

            filled = True
            while filled and pending_names and active < max_workers:
                filled = False
                for n in reversed(list(pending_names)):
                    cost = task_costs[n]
                    if disk_used + cost <= budget_bytes:
                        _dispatch(n, cost)
                        filled = True
                        break

            if pending_names or active > 0:
                done.clear()
                await done.wait()

    asyncio.run(_schedule())


def run_all(tasks, domain, agent, job_dir, args):
    """Main scheduling entry point."""
    domain.initialize(args)
    try:
        _run_all_inner(tasks, domain, agent, job_dir, args)
    finally:
        domain.finalize()


def _run_all_inner(tasks, domain, agent, job_dir, args):
    trials = getattr(args, 'trials', 1) or 1

    # Filter tasks that still need work
    tasks_to_run = []
    for task in tasks:
        all_done = all(
            (job_dir / f"{task['name']}__trial_{k}" / "result.json").exists()
            for k in range(1, trials + 1)
        )
        if not all_done:
            tasks_to_run.append(task)

    todo = len(tasks_to_run)
    done_count = len(tasks) - todo

    print(f"{'=' * 60}")
    print(f"  {domain.name} Evaluation ({agent.name})")
    print(f"  Split: {getattr(args, 'split', 'N/A')}, Tasks: {len(tasks)}, Trials: {trials} (pass@{trials})")
    print(f"  Todo: {todo}, Done: {done_count}")
    print(f"  Parallel: {args.parallel}")
    print(f"  Job dir: {job_dir}")
    print(f"{'=' * 60}")

    if todo == 0:
        print("All tasks already completed. Nothing to do.")
        from utils.summary import print_summary
        print_summary(job_dir, trials)
        return

    progress_lock = threading.Lock()
    task_count = [done_count]

    def _on_trial(task_name, trial_num, reward, elapsed):
        with progress_lock:
            mark = "\u2705" if reward > 0 else "\u274c"
            label = f"{task_name}/t{trial_num}" if trials > 1 else task_name
            print(f"  {mark} {label:<35s} reward={reward:.2f}  {elapsed:>5.0f}s")
            sys.stdout.flush()

    max_retries = getattr(args, 'max_retries', 2)

    def run_one_task(task):
        task_agent = type(agent)()
        results = run_task_trials(task, domain, task_agent, job_dir, trials,
                                  live=args.live, max_retries=max_retries,
                                  on_trial_done=_on_trial)
        with progress_lock:
            task_count[0] += 1
            print(f"  {'─' * 40} [{task_count[0]}/{len(tasks)}]")
            sys.stdout.flush()
        return results

    print(f"\n{'─' * 70}")

    disk_budget = getattr(args, 'disk_budget', None)

    if disk_budget and disk_budget > 0:
        _run_disk_aware(tasks_to_run, run_one_task, disk_budget,
                        max_workers=getattr(args, 'parallel', 0) or 32,
                        cost_fn=lambda t: domain.get_disk_cost(t))
    elif args.parallel <= 1:
        for task in tasks_to_run:
            run_one_task(task)
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(run_one_task, t): t for t in tasks_to_run}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    log.error(f"Unhandled exception: {e}")

    from utils.summary import print_summary
    print_summary(job_dir, trials, domain=domain)
