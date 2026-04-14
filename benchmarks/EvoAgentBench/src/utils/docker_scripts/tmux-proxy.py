#!/usr/bin/env python3
"""Host-side proxy that executes commands inside a Docker container.

Supports two modes:
  - write: pipe stdin into a file inside the container
  - exec:  run a command inside the container and capture output

Usage:
  tmux-proxy.py --docker /usr/bin/docker --sock /var/run/docker.sock --cid <id> write /path
  tmux-proxy.py --docker /usr/bin/docker --sock /var/run/docker.sock --cid <id> tmux-run "cmd" 5
"""
import argparse
import json
import http.client
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time


def _conn(sock_path):
    """Open an HTTP connection over a Unix socket."""
    c = http.client.HTTPConnection("localhost")
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_path)
    c.sock = s
    return c


def docker_api_exec(sock, cid, cmd, detach=True):
    """Create and start a Docker exec instance via the API."""
    c = _conn(sock)
    c.request("POST", f"/containers/{cid}/exec",
              body=json.dumps({"AttachStdout": not detach, "AttachStderr": not detach, "Cmd": cmd}),
              headers={"Content-Type": "application/json"})
    d = json.loads(c.getresponse().read())
    eid = d.get("Id")
    if not eid:
        return None
    c2 = _conn(sock)
    c2.request("POST", f"/exec/{eid}/start",
               body=json.dumps({"Detach": detach}),
               headers={"Content-Type": "application/json"})
    c2.getresponse().read()
    return eid


def do_write(docker, cid, dest):
    """Read stdin and copy it into a file inside the container."""
    tmp = tempfile.NamedTemporaryFile(delete=False, prefix="_dw_w_")
    try:
        if not os.isatty(0):
            tmp.write(sys.stdin.buffer.read())
        tmp.close()
        rc = subprocess.run([docker, "cp", tmp.name, f"{cid}:{dest}"],
                            capture_output=True).returncode
        size = os.path.getsize(tmp.name)
        if rc == 0:
            print(f"Written {size} bytes to {dest}")
        else:
            print(f"Failed to write {dest}", file=sys.stderr)
        sys.exit(rc)
    finally:
        os.unlink(tmp.name)


def do_exec(docker, sock, cid, args):
    """Run a command inside the container, poll for completion, and print output."""
    sess = f"e{int(time.time() * 1e9)}{os.getpid()}"
    edir = f"/tmp/_dw/{sess}"
    inner = f'mkdir -p {edir} && "$@" > {edir}/out 2> {edir}/err; echo $? > {edir}/rc'
    cmd = ["sh", "-c", inner, "sh"] + args
    docker_api_exec(sock, cid, cmd, detach=True)

    tmpd = tempfile.mkdtemp(prefix="_dw_r_")
    for _ in range(3250):
        if subprocess.run([docker, "cp", f"{cid}:{edir}/rc", f"{tmpd}/rc"],
                          capture_output=True).returncode == 0:
            subprocess.run([docker, "cp", f"{cid}:{edir}/out", f"{tmpd}/out"], capture_output=True)
            subprocess.run([docker, "cp", f"{cid}:{edir}/err", f"{tmpd}/err"], capture_output=True)
            out_f = os.path.join(tmpd, "out")
            err_f = os.path.join(tmpd, "err")
            rc_f = os.path.join(tmpd, "rc")
            if os.path.exists(out_f):
                sys.stdout.buffer.write(open(out_f, "rb").read())
            if os.path.exists(err_f):
                sys.stderr.buffer.write(open(err_f, "rb").read())
            rc = open(rc_f).read().strip() if os.path.exists(rc_f) else "1"
            shutil.rmtree(tmpd, ignore_errors=True)
            docker_api_exec(sock, cid, ["rm", "-rf", edir], detach=True)
            sys.exit(int(rc))
        time.sleep(0.2)

    shutil.rmtree(tmpd, ignore_errors=True)
    print("exec timed out after 650s", file=sys.stderr)
    sys.exit(124)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker", required=True)
    parser.add_argument("--sock", required=True)
    parser.add_argument("--cid", required=True)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not args.cmd:
        sys.exit(0)

    if args.cmd[0] == "write":
        dest = args.cmd[1] if len(args.cmd) > 1 else "/dev/null"
        do_write(args.docker, args.cid, dest)
    else:
        do_exec(args.docker, args.sock, args.cid, args.cmd)


if __name__ == "__main__":
    main()
