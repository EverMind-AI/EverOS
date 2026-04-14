#!/usr/bin/env python3
"""Extract skills from train sessions using EverMemOS (v1 API).

Sends session trajectories to EverMemOS, triggers clustering, waits for skill
stabilization, and saves extracted skills as SKILL.md files.

Supports split files in two formats:

1. Cluster format:
    {"clusters": {"CLUSTER_A": {"train": [...], "test": [...]}, ...}}

2. Flat format (auto-wrapped into a "default" cluster):
    {"train": [...], "test": [...]}

For domains without a split file, tasks are loaded from the domain adapter.

Usage:
    python extract_skills.py --job-dir jobs/reasoning-XXX --api-url http://localhost:1997
    python extract_skills.py --domain information_retrieval --job-dir jobs/bcp-XXX
    python extract_skills.py --job-dir jobs/... --feedback --success-only
"""

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import httpx
from config import get_domain, get_domain_config, get_config, load_config

log = logging.getLogger("evoagentbench")


# ---------------------------------------------------------------------------
# Split file loading
# ---------------------------------------------------------------------------

def load_task_clusters(split_file: str | Path, split_name: str,
                       clusters: list[str] | None = None) -> dict[str, list[str]]:
    """Load {cluster_name: [task_id, ...]} from a split file.

    Supports cluster format and flat format (auto-wrapped into "default").
    """
    with open(split_file) as f:
        data = json.load(f)

    if "clusters" in data:
        result = {}
        for name, splits in data["clusters"].items():
            if clusters and name not in clusters:
                continue
            task_ids = splits.get(split_name, [])
            if task_ids:
                result[name] = [str(tid) for tid in task_ids]
        return result

    if split_name in data:
        return {"default": [str(tid) for tid in data[split_name]]}

    raise ValueError(
        f"Split file has neither 'clusters' key nor '{split_name}' key. "
        f"Keys found: {list(data.keys())}"
    )


def load_splits_from_adapter(domain, split_name: str) -> dict[str, list[str]]:
    """Fallback: load task IDs from domain adapter (no split file needed)."""
    tmp_args = argparse.Namespace(
        task=None, split=split_name, trials=1,
        parallel=1, max_retries=0, live=False, disk_budget=None,
    )
    tasks = domain.load_tasks(tmp_args)
    return {"default": [str(t["name"]) for t in tasks]}


# ---------------------------------------------------------------------------
# Session loading & normalization
# ---------------------------------------------------------------------------

def load_session_messages(path: Path) -> list:
    """Load and normalize session messages from a JSONL file.

    Supports nanobot (flat role/content) and openclaw (nested message blocks).

    Reasoning/thinking handling:
      - Multi-turn (has tool calls): reasoning dropped (tool selection noise).
      - Single-turn (no tools): reasoning chunked into simulated tool_call/
        tool_result pairs so EverMemOS can extract MemCells from them.
    """
    messages = []
    with open(path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("role"):
                messages.append(_normalize_nanobot_message(entry))
            elif entry.get("type") == "message" and entry.get("message"):
                msg = _normalize_openclaw_message(entry["message"])
                if msg:
                    messages.append(msg)

    has_tools = any(m.get("tool_calls") or m["role"] == "tool" for m in messages)

    if has_tools:
        for m in messages:
            m.pop("_reasoning", "")
    else:
        expanded = []
        for m in messages:
            reasoning = m.pop("_reasoning", "")
            if not reasoning:
                expanded.append(m)
                continue
            chunks = _split_thinking_to_blocks(reasoning, block_size=1000)
            if len(chunks) <= 1:
                m["content"] = (reasoning + "\n\n" + m["content"]) if m["content"] else reasoning
                expanded.append(m)
            else:
                for i, chunk in enumerate(chunks):
                    call_id = f"think_{i:03d}"
                    expanded.append({
                        "role": "assistant", "content": "",
                        "tool_calls": [{"id": call_id, "type": "function",
                                        "function": {"name": "reasoning_step",
                                                     "arguments": json.dumps({"step": i + 1, "total": len(chunks)})}}],
                    })
                    expanded.append({"role": "tool", "content": chunk, "tool_call_id": call_id})
                if m["content"].strip():
                    expanded.append({"role": "assistant", "content": m["content"]})
        messages = expanded

    return messages


def _split_thinking_to_blocks(text: str, block_size: int = 1000) -> list[str]:
    if not text or len(text) <= block_size:
        return [text] if text else []
    return [text[i:i + block_size] for i in range(0, len(text), block_size)]


def _normalize_nanobot_message(msg: dict) -> dict:
    result = {"role": msg["role"]}
    content = msg.get("content")
    tool_calls = msg.get("tool_calls")
    reasoning = msg.get("reasoning_content")

    if content:
        result["content"] = str(content)
    elif tool_calls:
        parts = [f"[call {tc.get('function', {}).get('name', '?')}(...)]" for tc in tool_calls]
        result["content"] = " ".join(parts)
    else:
        result["content"] = ""

    if reasoning:
        result["_reasoning"] = str(reasoning)
    if tool_calls:
        result["tool_calls"] = tool_calls
    if msg.get("tool_call_id"):
        result["tool_call_id"] = msg["tool_call_id"]
    return result


def _normalize_openclaw_message(msg: dict) -> dict | None:
    role = msg.get("role", "")
    if not role:
        return None

    content_blocks = msg.get("content", [])
    if not isinstance(content_blocks, list):
        return {"role": role, "content": str(content_blocks)}

    text_parts, thinking_parts, tool_calls = [], [], []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "thinking":
            thinking_parts.append(block.get("thinking", ""))
        elif btype in ("toolCall", "toolUse", "tool_use"):
            tool_calls.append({
                "id": block.get("id", block.get("toolCallId", "")),
                "type": "function",
                "function": {
                    "name": block.get("name", block.get("toolName", "")),
                    "arguments": json.dumps(block.get("arguments", block.get("input", {}))),
                },
            })

    normalized_role = "tool" if role == "toolResult" else role
    content = "\n".join(text_parts)
    reasoning = "\n".join(thinking_parts)
    if not content.strip() and not tool_calls and not reasoning.strip():
        return None
    result = {"role": normalized_role, "content": content}
    if reasoning:
        result["_reasoning"] = reasoning
    if tool_calls:
        result["tool_calls"] = tool_calls
    tool_call_id = msg.get("toolCallId") or msg.get("tool_call_id")
    if tool_call_id:
        result["tool_call_id"] = tool_call_id
    return result


def find_sessions(job_dir: Path, task_ids: list[str]) -> dict[str, Path]:
    found = {}
    for tid in task_ids:
        session_file = job_dir / f"{tid}__trial_1" / "session.jsonl"
        if session_file.exists():
            found[tid] = session_file
    return found


# ---------------------------------------------------------------------------
# Task feedback
# ---------------------------------------------------------------------------

def load_task_feedback(job_dir: Path, task_id: str) -> dict:
    """Load evaluation feedback from result.json + verifier/."""
    trial_dir = job_dir / f"{task_id}__trial_1"
    reward, status, feedback_text = 0.0, "unknown", ""

    result_file = trial_dir / "result.json"
    if result_file.exists():
        try:
            data = json.loads(result_file.read_bytes())
            reward = float(data.get("verifier_result", {}).get("reward", 0.0))
            status = data.get("agent_result", {}).get("completion_status", "unknown")
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    verifier_dir = trial_dir / "verifier"
    if verifier_dir.is_dir():
        for fname in ("eval_details.json", "details.json"):
            f = verifier_dir / fname
            if not f.exists():
                continue
            try:
                details = json.loads(f.read_bytes())
                feedback_text = details.get("feedback", "")
                if not feedback_text:
                    jr = details.get("judge_result", {})
                    if isinstance(jr, dict):
                        feedback_text = jr.get("reasoning") or ""
                break
            except (json.JSONDecodeError, ValueError):
                continue

    return {"reward": reward, "status": status, "feedback": feedback_text}


def _format_feedback(fb: dict) -> str:
    outcome = "SUCCESS" if fb["reward"] > 0 else "FAILURE"
    parts = [
        "## Task Evaluation Result",
        f"- Outcome: {outcome} (reward: {fb['reward']})",
    ]
    if fb["status"] != "unknown":
        parts.append(f"- Completion: {fb['status']}")
    if fb["feedback"]:
        parts.append(f"- Feedback: {fb['feedback']}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# EverMemOS v1 API
# ---------------------------------------------------------------------------

async def send_session_v1(client, messages, session_id, user_id, base_url):
    ts_base = int(datetime.now(timezone.utc).timestamp() * 1000)
    api_msgs = []
    for i, msg in enumerate(messages):
        api_msg = {
            "message_id": f"{session_id}_{i:04d}",
            "role": msg.get("role", "user"),
            "content": msg.get("content") or "",
            "timestamp": ts_base + i,
        }
        if msg.get("tool_calls"):
            api_msg["tool_calls"] = msg["tool_calls"]
        if msg.get("tool_call_id"):
            api_msg["tool_call_id"] = msg["tool_call_id"]
        api_msgs.append(api_msg)

    resp = await client.post(f"{base_url}/api/v1/memories/agent", json={
        "user_id": user_id, "session_id": session_id, "messages": api_msgs,
    })
    resp.raise_for_status()
    resp = await client.post(f"{base_url}/api/v1/memories/agent/flush", json={
        "user_id": user_id, "session_id": session_id,
    })
    resp.raise_for_status()


async def flush_clustering(client, user_id, base_url) -> dict:
    resp = await client.post(f"{base_url}/api/v1/memories/agent/flush-clustering",
                             json={"user_id": user_id})
    resp.raise_for_status()
    return resp.json().get("data", {})


async def fetch_skills_v1(client, user_id, base_url) -> list[dict]:
    all_skills, page = [], 1
    while True:
        resp = await client.post(f"{base_url}/api/v1/memories/get", json={
            "memory_type": "agent_skill", "page": page, "page_size": 100,
            "rank_by": "timestamp", "rank_order": "desc",
            "filters": {"user_id": user_id},
        })
        resp.raise_for_status()
        data = resp.json().get("data", {})
        skills = data.get("agent_skills", [])
        all_skills.extend(skills)
        if len(all_skills) >= data.get("total_count", 0) or not skills:
            break
        page += 1
    return all_skills


def _skills_fingerprint(skills: list[dict]) -> str:
    parts = [f"{s.get('id')}:{s.get('cluster_id')}:{s.get('content', '')}"
             for s in sorted(skills, key=lambda x: x.get("id", ""))]
    return hashlib.md5("|".join(parts).encode()).hexdigest()


async def wait_for_skills_stable(client, user_id, base_url,
                                  poll_interval=120, max_wait=1800) -> list[dict]:
    prev_fp, elapsed = None, 0
    while elapsed < max_wait:
        skills = await fetch_skills_v1(client, user_id, base_url)
        fp = _skills_fingerprint(skills)
        print(f"    Poll: {len(skills)} skills, fp={fp[:8]}, elapsed={elapsed}s")
        if skills and fp == prev_fp:
            return skills
        prev_fp = fp
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
    log.warning(f"Skills not stable after {max_wait}s, returning latest")
    return await fetch_skills_v1(client, user_id, base_url)


# ---------------------------------------------------------------------------
# Skill saving
# ---------------------------------------------------------------------------

def save_skills(skills: list[dict], output_dir: Path):
    from collections import defaultdict
    by_cluster = defaultdict(list)
    for skill in skills:
        by_cluster[skill.get("cluster_id") or "unclustered"].append(skill)

    for cluster_id, cluster_skills in by_cluster.items():
        cluster_dir = output_dir / cluster_id
        cluster_dir.mkdir(parents=True, exist_ok=True)
        for i, skill in enumerate(cluster_skills, 1):
            slug = skill.get("name", f"skill_{i}").lower().replace(" ", "_").replace("/", "-").replace(".", "")[:60]
            skill_dir = cluster_dir / slug
            skill_dir.mkdir(parents=True, exist_ok=True)
            desc = skill.get("description", "")
            content = skill.get("content", "")
            md = f"---\nname: {slug}\ndescription: >\n  {desc}\nalways: true\n---\n\n{content}"
            (skill_dir / "SKILL.md").write_text(md)
        print(f"    {cluster_id}: {len(cluster_skills)} skills")

    print(f"  Saved {len(skills)} skills across {len(by_cluster)} clusters to {output_dir}")


# ---------------------------------------------------------------------------
# Resolve helpers
# ---------------------------------------------------------------------------

def resolve_split_file(args_split_file: str | None,
                       domain_name: str) -> Path | None:
    """Resolve split file: CLI arg > domain config > None."""
    if args_split_file:
        p = Path(args_split_file)
        if not p.exists():
            raise FileNotFoundError(f"Split file not found: {p}")
        return p

    bench_cfg = get_domain_config(domain_name)
    split_file = bench_cfg.get("split_file")
    if split_file:
        p = Path(split_file)
        if p.exists():
            return p

    return None


def _load_eval_config() -> dict:
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    if cfg_path.exists():
        import yaml
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _resolve_job_dir(job_dir_str: str, global_cfg: dict) -> Path:
    if job_dir_str == "latest":
        jobs_root = Path(global_cfg.get("job_dir", "./jobs"))
        candidates = [p for p in jobs_root.iterdir() if p.is_dir()]
        if not candidates:
            raise FileNotFoundError(f"No job directories found in {jobs_root}")
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return Path(job_dir_str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    eval_cfg = _load_eval_config()

    parser = argparse.ArgumentParser(description="Extract skills via EverMemOS (v1 API)")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--job-dir", default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--api-url", default=None, help="EverMemOS API (default: http://localhost:1997)")
    parser.add_argument("--clusters", nargs="*")
    parser.add_argument("--split", default=None, help="Which split (default: train)")
    parser.add_argument("--parallel", type=int, default=16, help="Concurrent session sends")
    parser.add_argument("--poll-interval", type=int, default=120)
    parser.add_argument("--success-only", action="store_true", help="Only send successful sessions")
    parser.add_argument("--feedback", action="store_true", help="Append task feedback to sessions")
    args = parser.parse_args()

    config_path = args.config or str(_PROJECT_ROOT / "config.yaml")
    load_config(config_path)
    cfg = get_config()
    domain_name = args.domain or cfg["domain"]["name"]

    api_url = args.api_url or eval_cfg.get("api_url", "http://localhost:1997")
    output_dir_str = args.output_dir or eval_cfg.get("skills_dir", "src/skill_evolution/evermemos/skills")
    split = args.split or eval_cfg.get("split_extract", "train")
    job_dir_str = args.job_dir or eval_cfg.get("job_dir", "latest")

    split_file = resolve_split_file(args.split_file, domain_name)
    job_dir = _resolve_job_dir(job_dir_str, cfg)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    if split_file:
        task_clusters = load_task_clusters(split_file, split, args.clusters)
    else:
        print(f"  No split file for {domain_name}, loading from adapter")
        domain = get_domain(domain_name)
        task_clusters = load_splits_from_adapter(domain, split)

    all_sessions = {}
    for cluster_name, task_ids in task_clusters.items():
        sessions = find_sessions(job_dir, task_ids)
        missing = set(task_ids) - set(sessions.keys())
        if missing:
            print(f"  WARNING: {cluster_name} missing {len(missing)} sessions")
        all_sessions.update(sessions)

    if not all_sessions:
        print("No sessions found. Run train split first.")
        return

    user_id = f"extract_{uuid.uuid4().hex[:6]}"

    metadata_path = output_dir / f"metadata_{user_id}.json"
    with open(metadata_path, "w") as f:
        json.dump({
            "user_id": user_id, "api_url": api_url, "domain": domain_name,
            "job_dir": str(job_dir), "sessions": len(all_sessions),
        }, f, indent=2)

    print(f"EverMemOS Skill Extraction (v1 API)")
    print(f"  Domain: {domain_name}")
    print(f"  API: {api_url}")
    print(f"  Job: {job_dir}")
    print(f"  Sessions: {len(all_sessions)}")
    print(f"  User ID: {user_id}")
    print(f"  Feedback: {args.feedback}, Success only: {args.success_only}")

    async with httpx.AsyncClient(timeout=1800.0) as client:
        try:
            await client.post(f"{api_url}/api/v1/memories/get", json={
                "memory_type": "agent_skill", "page_size": 1,
                "filters": {"user_id": "health_check"},
            })
        except Exception as e:
            print(f"\n  ERROR: Cannot connect to EverMemOS at {api_url}: {e}")
            sys.exit(1)

        print(f"\n  Phase 1: Sending {len(all_sessions)} sessions...")
        sem = asyncio.Semaphore(args.parallel)
        sent = skipped = 0

        async def _send_one(tid, session_path):
            nonlocal sent, skipped
            messages = load_session_messages(session_path)
            if not messages:
                return

            feedback = load_task_feedback(job_dir, tid)
            if args.success_only and feedback["reward"] <= 0:
                skipped += 1
                return

            if args.feedback:
                messages.append({"role": "assistant", "content": _format_feedback(feedback)})

            async with sem:
                await send_session_v1(client, messages, f"session_{tid}", user_id, api_url)
            sent += 1
            if sent % 20 == 0:
                print(f"    Sent {sent}/{len(all_sessions)}")

        await asyncio.gather(*[_send_one(tid, p) for tid, p in all_sessions.items()],
                             return_exceptions=True)
        print(f"    Sent {sent}/{len(all_sessions)}"
              f"{f', skipped {skipped}' if skipped else ''}")

        print(f"\n  Phase 2: Triggering clustering...")
        result = await flush_clustering(client, user_id, api_url)
        print(f"    Clusters: {len(result.get('cluster_ids', []))}")

        print(f"\n  Phase 3: Waiting for skills (poll every {args.poll_interval}s)...")
        skills = await wait_for_skills_stable(client, user_id, api_url,
                                               poll_interval=args.poll_interval)
        print(f"    Final: {len(skills)} skills")

        if skills:
            save_skills(skills, output_dir)

    print(f"\nDone! User ID: {user_id}")


if __name__ == "__main__":
    asyncio.run(main())
