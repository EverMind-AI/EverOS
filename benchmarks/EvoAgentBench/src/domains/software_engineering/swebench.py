"""SWE-bench Verified domain adapter.

Docker-based: loads pre-built instance images from tar, runs agent via
tmux wrapper, verifies with official swebench eval harness.
"""

import json
import logging
import os
import time
import traceback
from pathlib import Path, PurePosixPath

import docker
import pandas as pd

from domains.base import DomainAdapter
from config import get_domain_config, register_domain
from utils.docker import (
    ensure_image, setup_container_tmux, create_wrapper_script,
    remove_container, remove_image,
)

from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
from swebench.harness.constants import (
    DOCKER_PATCH, DOCKER_WORKDIR, DOCKER_USER,
    KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION,
)
from swebench.harness.docker_utils import copy_to_container, exec_run_with_timeout
from swebench.harness.grading import get_eval_report

log = logging.getLogger("evoagentbench")

# Git apply strategies (same as official swebench harness)
GIT_APPLY_CMDS = [
    "git apply -v",
    "git apply -v --3way",
    "patch --batch --fuzz=5 -p1 -i",
]


def _cfg():
    return get_domain_config("software_engineering")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _instance_id_to_tar(instance_id: str, tar_dir: str) -> Path | None:
    """Map instance_id to tar file path.

    Naming: astropy__astropy-12907 → sweb.eval.x86_64.astropy_1776_astropy-12907.tar
    Convention: replace '__' with '_1776_' in instance_id.
    """
    id_docker = instance_id.replace("__", "_1776_").lower()
    tar_file = Path(tar_dir) / f"sweb.eval.x86_64.{id_docker}.tar"
    if tar_file.exists():
        return tar_file
    return None


def _get_docker_client():
    """Get a cached Docker client."""
    if not hasattr(_get_docker_client, "_client"):
        _get_docker_client._client = docker.from_env()
    return _get_docker_client._client


def _docker_exec_in_tmux(container_name: str, cmd: str, timeout: int = 30):
    """Execute a command inside tmux session via Docker SDK."""
    container = _get_docker_client().containers.get(container_name)
    tmux_cmd = f"tmux send-keys -t main '{cmd}' Enter"
    container.exec_run(["bash", "-c", tmux_cmd], user="root")
    time.sleep(3)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _run_verification(test_spec: TestSpec, container_name: str,
                      pred: dict, trial_dir: Path, timeout: int) -> dict:
    """Run official swebench eval: git diff → apply patch → run tests → grade."""
    verifier_dir = trial_dir / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    instance_id = test_spec.instance_id

    client = _get_docker_client()
    container = client.containers.get(container_name)

    try:
        # 1. Get agent's diff
        git_diff = container.exec_run(
            "git -c core.fileMode=false diff",
            workdir=DOCKER_WORKDIR, user=DOCKER_USER,
        )
        agent_patch = git_diff.output.decode("utf-8", errors="replace").strip()
        (verifier_dir / "agent_patch.diff").write_text(agent_patch)
        pred[KEY_PREDICTION] = agent_patch

        if not agent_patch:
            log.warning(f"  Agent produced no changes")
            return {"reward": 0.0, "error": "No patch generated"}

        # 2. Reset and re-apply patch
        container.exec_run("git checkout -- .", workdir=DOCKER_WORKDIR, user=DOCKER_USER)

        patch_file = verifier_dir / "patch.diff"
        patch_file.write_text(agent_patch)
        copy_to_container(container, patch_file, PurePosixPath(DOCKER_PATCH))

        applied = False
        for git_apply_cmd in GIT_APPLY_CMDS:
            val = container.exec_run(
                f"{git_apply_cmd} {DOCKER_PATCH}",
                workdir=DOCKER_WORKDIR, user=DOCKER_USER,
            )
            if val.exit_code == 0:
                applied = True
                log.info(f"  Patch applied with: {git_apply_cmd}")
                break
        if not applied:
            log.warning(f"  Failed to apply patch")
            return {"reward": 0.0, "error": "Patch apply failed"}

        # 3. Run eval script
        eval_file = verifier_dir / "eval.sh"
        eval_file.write_text(test_spec.eval_script)
        copy_to_container(container, eval_file, PurePosixPath("/eval.sh"))

        log.info(f"  Running eval script (timeout={timeout}s)...")
        test_output, timed_out, total_runtime = exec_run_with_timeout(
            container, "/bin/bash /eval.sh", timeout
        )
        log.info(f"  Test runtime: {total_runtime:.1f}s")

        test_output_path = verifier_dir / "test_output.txt"
        test_output_path.write_text(test_output)

        if timed_out:
            return {"reward": 0.0, "error": f"Tests timed out ({timeout}s)"}

        # 4. Grade with official get_eval_report()
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=str(test_output_path),
            include_tests_status=True,
        )
        resolved = report.get(instance_id, {}).get("resolved", False)

        (verifier_dir / "eval_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False)
        )
        reward = 1.0 if resolved else 0.0
        (verifier_dir / "reward.txt").write_text(str(reward))
        return {"reward": reward, "resolved": resolved}

    except Exception as e:
        log.error(f"  Verification error: {e}\n{traceback.format_exc()}")
        return {"reward": 0.0, "error": str(e)}


# ---------------------------------------------------------------------------
# SWEBench Adapter
# ---------------------------------------------------------------------------

class SWEBenchAdapter(DomainAdapter):
    name = "software_engineering"

    def __init__(self):
        self._dataset = None
        self._test_specs = {}

    def _load_dataset(self) -> pd.DataFrame:
        if self._dataset is None:
            self._dataset = pd.read_parquet(_cfg()["parquet_file"])
        return self._dataset

    def _get_test_spec(self, instance_id: str) -> TestSpec:
        if instance_id not in self._test_specs:
            df = self._load_dataset()
            matched = df[df["instance_id"] == instance_id]
            if matched.empty:
                raise ValueError(f"instance_id '{instance_id}' not found in dataset")
            row = matched.iloc[0].to_dict()
            # Parse JSON string fields for swebench compatibility
            for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
                if isinstance(row.get(key), str):
                    row[key] = json.loads(row[key])
            self._test_specs[instance_id] = make_test_spec(row)
        return self._test_specs[instance_id]

    def _image_name(self, instance_id: str) -> str:
        """Get the Docker image name for loading from tar.

        Tar images use naming: swebench/sweb.eval.x86_64.{id_docker}:latest
        """
        id_docker = instance_id.replace("__", "_1776_").lower()
        return f"swebench/sweb.eval.x86_64.{id_docker}:latest"

    def _load_split_ids(self, split: str) -> list[str] | None:
        """Load instance IDs from split_file if configured."""
        split_file = _cfg().get("split_file")
        if not split_file:
            return None
        with open(split_file) as f:
            raw = json.load(f)
        if split == "all":
            train = raw.get("train", [])
            test = raw.get("test", [])
            if train or test:
                return train + test
            if "clusters" in raw:
                seen = set()
                ids = []
                for cluster in raw["clusters"].values():
                    for part in ("train", "test"):
                        for iid in cluster.get(part, []):
                            if iid not in seen:
                                seen.add(iid)
                                ids.append(iid)
                return ids
            return []
        if split in raw and isinstance(raw[split], list):
            return raw[split]
        if "clusters" in raw:
            seen = set()
            ids = []
            for cluster in raw["clusters"].values():
                for iid in cluster.get(split, []):
                    if iid not in seen:
                        seen.add(iid)
                        ids.append(iid)
            return ids
        return None

    def load_tasks(self, args) -> list[dict]:
        df = self._load_dataset()
        # 1. Split narrows the pool
        if args.split:
            split_ids = self._load_split_ids(args.split)
            if split_ids is not None:
                missing = set(split_ids) - set(df["instance_id"])
                if missing:
                    log.warning(f"split_file has {len(missing)} IDs not in parquet: {list(missing)[:5]}")
                df = df[df["instance_id"].isin(split_ids)]
            elif args.split.isdigit():
                df = df.head(int(args.split))

        # 2. Task filters within the pool
        if args.task:
            task_ids = set(t.strip() for t in args.task.split(","))
            df = df[df["instance_id"].isin(task_ids)]
        return [{"name": row["instance_id"],
                 "problem_statement": row["problem_statement"],
                 "repo": row["repo"],
                 "hints_text": row.get("hints_text", "")}
                for _, row in df.iterrows()]

    def pre_task_trials(self, task: dict):
        """Ensure instance image exists: local → tar → pull."""
        instance_id = task["name"]
        image = self._image_name(instance_id)
        tar_dir = _cfg().get("tar_dir")
        tar_file = _instance_id_to_tar(instance_id, tar_dir) if tar_dir else None
        ensure_image(image, tar_file=tar_file)

    def post_task_trials(self, task: dict):
        remove_image(self._image_name(task["name"]))

    def setup(self, task: dict, agent_name: str, trial: int) -> dict:
        instance_id = task["name"]
        image = self._image_name(instance_id)
        container_name = f"swebench-{agent_name}-{instance_id}-t{trial}"

        # Start container via Docker SDK
        client = _get_docker_client()
        # Remove existing container if any
        try:
            old = client.containers.get(container_name)
            old.remove(force=True)
        except docker.errors.NotFound:
            pass

        container = client.containers.create(
            image=image,
            name=container_name,
            user=DOCKER_USER,
            detach=True,
            command="tail -f /dev/null",
        )
        container.start()
        log.info(f"[{instance_id}] Container started: {container.short_id}")

        # Setup tmux
        log.info(f"[{instance_id}] Setting up tmux...")
        setup_container_tmux(container_name)
        wrapper_path = create_wrapper_script(instance_id, container_name)

        # Activate conda env
        _docker_exec_in_tmux(container_name,
                             "source /opt/miniconda3/bin/activate && conda activate testbed && cd /testbed")

        test_spec = self._get_test_spec(instance_id)

        return {
            "container_name": container_name,
            "wrapper_path": wrapper_path,
            "test_spec": test_spec,
            "pred": {
                KEY_INSTANCE_ID: instance_id,
                KEY_MODEL: agent_name,
                KEY_PREDICTION: "",
            },
        }

    def get_agent_timeout(self, task: dict, env_info: dict) -> int:
        return int(_cfg().get("agent_timeout", 1800))

    _prompt_template = (Path(__file__).parent / "prompt.md").read_text()

    def build_prompt(self, task: dict, env_info: dict) -> str:
        prompt = self._prompt_template.format(
            wrapper_path=env_info["wrapper_path"],
            timeout_min=self.get_agent_timeout(task, env_info) // 60,
            problem=task["problem_statement"],
            repo=task["repo"],
        )
        hints = task.get("hints_text", "")
        if hints:
            prompt += f"\n## Hints\n\n{hints}\n"
        return prompt

    def verify(self, task: dict, env_info: dict, trial_dir: Path,
               agent_result: dict | None = None) -> dict:
        return _run_verification(
            env_info["test_spec"],
            env_info["container_name"],
            env_info["pred"],
            trial_dir,
            int(_cfg().get("verify_timeout", 1800)),
        )

    def cleanup(self, task: dict, env_info: dict):
        wrapper_path = env_info.get("wrapper_path")
        if wrapper_path:
            try:
                os.unlink(wrapper_path)
            except OSError:
                pass
        container_name = env_info.get("container_name")
        if container_name:
            remove_container(container_name)


register_domain("software_engineering", SWEBenchAdapter)
