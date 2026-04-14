"""Docker utility functions.

Low-level Docker operations: image loading, container lifecycle,
command execution, file copy, tmux setup, and wrapper script creation.

Shared by Docker-based domains (software_engineering, etc).
"""

import http.client
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from config import get_config

log = logging.getLogger("evoagentbench")

_SCRIPTS_DIR = Path(__file__).resolve().parent / "docker_scripts"


# ---------------------------------------------------------------------------
# Disk budget helpers
# ---------------------------------------------------------------------------

def get_docker_root() -> str:
    """Get Docker storage root via 'docker info'. Falls back to '/'."""
    cfg = get_config()
    try:
        r = subprocess.run(
            [cfg["docker"], "info", "--format", "{{.DockerRootDir}}"],
            capture_output=True, text=True, timeout=10,
        )
        root = r.stdout.strip()
        if root and os.path.isdir(root):
            return root
    except Exception as e:
        log.warning(f"Failed to get Docker root: {e}")
    return "/"


def parse_disk_budget(value: str, docker_root: str = "/") -> int:
    """Parse disk budget string to bytes. Supports 'auto', '25G', '10240M'."""
    if value.lower() == "auto":
        st = os.statvfs(docker_root)
        free = st.f_bavail * st.f_frsize
        budget = max(free - 1 * (1 << 30), 1 << 30)
        print(f"  Disk budget: auto-detected {free / (1<<30):.1f}G free on {docker_root}, using {budget / (1<<30):.1f}G")
        return budget
    value = value.strip().upper()
    if value.endswith("G"):
        return int(float(value[:-1]) * (1 << 30))
    elif value.endswith("M"):
        return int(float(value[:-1]) * (1 << 20))
    else:
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"Invalid disk budget format: '{value}'. Use 'auto', '25G', or '10240M'.")


# ---------------------------------------------------------------------------
# Basic Docker CLI wrappers
# ---------------------------------------------------------------------------

def _docker(*args, timeout=120, check=True, stdin_data=None, **kwargs):
    """Run a docker CLI command via subprocess."""
    cfg = get_config()
    cmd = [cfg["docker"]] + list(args)
    return subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=timeout, check=check, input=stdin_data, **kwargs,
    )


def image_exists(image_name: str) -> bool:
    """Check if a Docker image exists locally."""
    r = _docker("images", "-q", image_name, timeout=10, check=False)
    return bool(r.stdout.strip())


def ensure_image(image_name: str, tar_file: Path | None = None,
                 build_fn=None):
    """Ensure a Docker image exists locally.

    Priority: local cache → tar file → build_fn → pull from registry.
    """
    if image_exists(image_name):
        log.info(f"  Image {image_name} already exists locally")
        return
    if tar_file and tar_file.exists():
        log.info(f"  Loading image from tar: {tar_file}")
        load_image_from_tar(tar_file)
    elif build_fn:
        log.info(f"  Building image {image_name}...")
        build_fn()
    else:
        log.info(f"  Pulling image {image_name}...")
        pull_image(image_name)


def load_image_from_tar(tar_file: str | Path, retries: int = 2):
    """Load a Docker image from a tar file, with simple retry on failure."""
    tar_file = Path(tar_file)
    if not tar_file.exists():
        raise FileNotFoundError(f"Image tar not found: {tar_file}")
    cfg = get_config()
    for attempt in range(1, retries + 2):
        with open(tar_file, "rb") as f:
            r = subprocess.run(
                [cfg["docker"], "load"], stdin=f,
                capture_output=True, timeout=300,
            )
        if r.returncode == 0:
            return
        stderr = r.stderr.decode().strip() if r.stderr else ""
        if attempt <= retries:
            log.warning(f"  docker load failed (attempt {attempt}/{retries+1}): {stderr}, retrying...")
            time.sleep(3)
        else:
            raise RuntimeError(f"docker load failed after {retries+1} attempts: {stderr}")


def pull_image(image_name: str):
    """Pull a Docker image from registry."""
    _docker("pull", image_name, timeout=600)


def _get_container_id(container_name: str) -> str:
    r = _docker("inspect", "--format", "{{.Id}}", container_name, timeout=10)
    return r.stdout.strip()


def _unix_socket_connect():
    """Connect to the Docker daemon via Unix socket."""
    import socket
    cfg = get_config()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(cfg["docker_sock"])
    return sock


def docker_exec(container_name: str, cmd: str, timeout=300) -> subprocess.CompletedProcess:
    """Execute command inside container via Docker API (detached exec + poll)."""
    cid = _get_container_id(container_name)

    sess = f"e{int(time.time() * 1e9)}{os.getpid()}"
    edir = f"/tmp/_dw/{sess}"

    inner = f"#!/bin/sh\nmkdir -p '{edir}'\n"
    inner += f"( {cmd} ) > '{edir}/out' 2> '{edir}/err'\necho $? > '{edir}/rc'\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(inner)
        tmp_script = f.name
    os.chmod(tmp_script, 0o755)

    script_name = f"/tmp/_dw_run_{sess}.sh"
    try:
        _docker("cp", tmp_script, f"{cid}:{script_name}", timeout=30, check=False)
    finally:
        os.unlink(tmp_script)

    payload = json.dumps({
        "AttachStdout": False, "AttachStderr": False,
        "Cmd": ["sh", script_name],
    }).encode()

    conn = http.client.HTTPConnection("localhost")
    conn.sock = _unix_socket_connect()
    try:
        conn.request("POST", f"/containers/{cid}/exec", body=payload,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp_data = json.loads(resp.read().decode())
        exec_id = resp_data.get("Id", "")
    finally:
        conn.close()

    if not exec_id:
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="exec create failed")

    start_payload = json.dumps({"Detach": True}).encode()
    conn2 = http.client.HTTPConnection("localhost")
    conn2.sock = _unix_socket_connect()
    try:
        conn2.request("POST", f"/exec/{exec_id}/start", body=start_payload,
                      headers={"Content-Type": "application/json"})
        conn2.getresponse().read()
    finally:
        conn2.close()

    tmpdir = tempfile.mkdtemp(prefix="_dw_res_")
    deadline = time.time() + timeout
    stdout_text = ""
    stderr_text = ""
    rc = 1

    try:
        while time.time() < deadline:
            r = _docker("cp", f"{cid}:{edir}/rc", f"{tmpdir}/rc", timeout=10, check=False)
            if r.returncode == 0:
                _docker("cp", f"{cid}:{edir}/out", f"{tmpdir}/out", timeout=30, check=False)
                _docker("cp", f"{cid}:{edir}/err", f"{tmpdir}/err", timeout=30, check=False)

                rc_path = Path(tmpdir) / "rc"
                out_path = Path(tmpdir) / "out"
                err_path = Path(tmpdir) / "err"

                if rc_path.exists():
                    rc = int(rc_path.read_text().strip() or "1")
                if out_path.exists():
                    stdout_text = out_path.read_text()
                if err_path.exists():
                    stderr_text = err_path.read_text()
                break
            time.sleep(1)
        else:
            stderr_text = f"docker_exec timed out after {timeout}s"
            rc = 124
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        try:
            cleanup_payload = json.dumps({
                "AttachStdout": False, "AttachStderr": False,
                "Cmd": ["rm", "-rf", edir, script_name],
            }).encode()
            conn3 = http.client.HTTPConnection("localhost")
            conn3.sock = _unix_socket_connect()
            try:
                conn3.request("POST", f"/containers/{cid}/exec", body=cleanup_payload,
                              headers={"Content-Type": "application/json"})
                cr = json.loads(conn3.getresponse().read().decode())
            finally:
                conn3.close()
            if cr.get("Id"):
                conn4 = http.client.HTTPConnection("localhost")
                conn4.sock = _unix_socket_connect()
                try:
                    conn4.request("POST", f"/exec/{cr['Id']}/start",
                                  body=json.dumps({"Detach": True}).encode(),
                                  headers={"Content-Type": "application/json"})
                    conn4.getresponse().read()
                finally:
                    conn4.close()
        except Exception as e:
            log.warning(f"docker_exec cleanup failed: {e}")

    return subprocess.CompletedProcess(args=cmd, returncode=rc, stdout=stdout_text, stderr=stderr_text)


def start_container(task_name: str, container_name: str, task_config: dict):
    """Start a detached Docker container for a task, forwarding proxy env vars."""
    image = task_config.get("environment", {}).get("docker_image")
    if not image:
        raise ValueError(f"No docker_image specified in task config for '{task_name}'")
    cpus = str(task_config.get("environment", {}).get("cpus", 1))
    memory = task_config.get("environment", {}).get("memory", "2G")

    _docker("rm", "-f", container_name, timeout=30, check=False)

    env_args = []
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy"):
        val = os.environ.get(var)
        if val:
            env_args += ["-e", f"{var}={val}"]

    try:
        _docker(
            "run", "-d",
            "--name", container_name,
            f"--cpus={cpus}",
            f"--memory={memory}",
            *env_args,
            image,
            "sh", "-c", "sleep infinity",
            timeout=300,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        _docker("rm", "-f", container_name, timeout=30, check=False)
        raise

    docker_exec(container_name, "mkdir -p /logs/verifier /logs/agent /logs/artifacts", timeout=30)


def docker_cp(src: str, dst: str, timeout=120, check=True):
    _docker("cp", src, dst, timeout=timeout, check=check)


def remove_container(container_name: str):
    _docker("rm", "-f", container_name, timeout=30, check=False)


def remove_image(image_name: str):
    """Remove a Docker image by name."""
    log.info(f"  Removing image {image_name}")
    r = _docker("rmi", image_name, timeout=30, check=False)
    if r.returncode != 0:
        log.warning(f"  Failed to remove image {image_name}: {r.stderr.strip()}")


# ---------------------------------------------------------------------------
# Container tmux setup
# ---------------------------------------------------------------------------

def _deploy_file(container_name: str, local_path: Path, container_path: str):
    """Copy a local file into the container and make it executable."""
    cid = _get_container_id(container_name)
    _docker("cp", str(local_path), f"{cid}:{container_path}", timeout=30)
    docker_exec(container_name, f"chmod +x {container_path}", timeout=10)


def setup_container_tmux(container_name: str):
    """Install tmux inside container and start a tmux session named 'main'."""
    _deploy_file(container_name, _SCRIPTS_DIR / "tmux-setup.sh", "/tmp/tmux-setup.sh")
    _deploy_file(container_name, _SCRIPTS_DIR / "tmux-run.sh", "/usr/local/bin/tmux-run")

    proxy_url = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY", "")
    setup_cmd = f"sh /tmp/tmux-setup.sh {proxy_url}" if proxy_url else "sh /tmp/tmux-setup.sh"
    r = docker_exec(container_name, setup_cmd, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"Container tmux setup failed: {r.stderr or r.stdout}")

    time.sleep(2)


def create_wrapper_script(task_name: str, container_name: str) -> str:
    """Create a host-side shell script that calls tmux-proxy.py with container args."""
    cfg = get_config()
    cid = _get_container_id(container_name)
    wrapper_path = f"/tmp/{container_name}-exec"
    wrapper_py = str(_SCRIPTS_DIR / "tmux-proxy.py")

    import sys
    python = sys.executable or "python3"
    script_content = (
        f'#!/bin/sh\n'
        f'exec {python} {wrapper_py} --docker {cfg["docker"]} '
        f'--sock {cfg["docker_sock"]} --cid {cid} "$@"\n'
    )

    with open(wrapper_path, "w") as f:
        f.write(script_content)
    os.chmod(wrapper_path, 0o755)
    return wrapper_path
