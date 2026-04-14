#!/usr/bin/env python3
"""Evaluation entry point.

Reads config.yaml for agent, domain, output, and run settings.
Reads domain yaml for task selection (split, task filter, etc).
CLI args override both.

Usage:
    python run.py                                # use config.yaml defaults
    python run.py --agent nanobot                # override agent
    python run.py --task build-pmars --live      # override task
    python run.py --split test --parallel 2      # override settings
"""

import argparse
import logging
from pathlib import Path

# Load .env from project root if it exists
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    import os
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="EvoAgentBench - Evaluation Runner")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--agent", default=None, help="Override agent name")
    parser.add_argument("--domain", default=None, help="Override domain name")
    parser.add_argument("--split", default=None, help="Override task split")
    parser.add_argument("--task", default=None, help="Specific task(s), comma-separated")
    parser.add_argument("--job", default=None, help="Job name")
    parser.add_argument("--trials", type=int, default=None, help="Trials per task (pass@k)")
    parser.add_argument("--parallel", type=int, default=None, help="Max concurrent tasks")
    parser.add_argument("--live", action="store_true", default=None, help="Show live agent output")
    parser.add_argument("--disk-budget", default=None, dest="disk_budget", help="Disk budget: auto/25G/10240M")
    cli = parser.parse_args()

    # Load global config
    from config import load_config, get_config, get_domain, get_agent, get_domain_config
    load_config(cli.config, agent_override=cli.agent)
    cfg = get_config()

    # Resolve agent and domain names
    agent_name = cfg["agent"]["name"]
    domain_name = cli.domain or cfg["domain"]["name"]

    # Load domain config for task selection
    bench_cfg = get_domain_config(domain_name)

    # Task selection: from domain config, CLI overrides
    # When --task is given without --split, don't apply default split filter
    task = cli.task or bench_cfg.get("task")
    if cli.split:
        split = cli.split
    elif task:
        split = None
    else:
        split = bench_cfg.get("split", "train")

    # Run settings: from global config, CLI overrides
    trials = cli.trials or cfg["trials"]
    parallel = cli.parallel or cfg["parallel"]
    max_retries = cfg["max_retries"]
    live = cli.live if cli.live is not None else cfg.get("live", False)
    disk_budget_str = cli.disk_budget or cfg.get("disk_budget")

    # Parse disk budget
    disk_budget = None
    if disk_budget_str:
        from utils.docker import get_docker_root, parse_disk_budget
        disk_budget = parse_disk_budget(str(disk_budget_str), get_docker_root())

    # Build args for runner
    args = argparse.Namespace(
        split=split, task=task, trials=trials, parallel=parallel,
        live=live, disk_budget=disk_budget, max_retries=max_retries,
    )

    # Get domain and agent
    domain = get_domain(domain_name)
    agent = get_agent(agent_name)

    # Load tasks
    tasks = domain.load_tasks(args)

    # Setup job directory
    from pathlib import Path
    from datetime import datetime
    import uuid
    job_name = cli.job or f"{agent.name}-{domain_name}-{datetime.now().strftime('%m%d_%H%M')}-{uuid.uuid4().hex[:4]}"
    job_dir = Path(cfg["job_dir"]) / job_name
    job_dir.mkdir(parents=True, exist_ok=True)

    # Run
    from runner import run_all
    run_all(tasks, domain, agent, job_dir, args)


if __name__ == "__main__":
    main()
