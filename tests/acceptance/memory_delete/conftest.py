"""Start real ``everos server`` processes, a proxy, and clean them up."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from .proxy import Proxy

pytestmark = pytest.mark.acceptance

_LLM = "https://openrouter.ai/api/v1"
_EMB = "https://api.deepinfra.com/v1/openai"
_LIFETIME_S = 1800
"""Hard cap on one server process: a hung test cannot leave it running."""
_EVEROS = Path(sys.executable).parent / "everos"


def _keys() -> dict[str, str]:
    keys = {
        "llm": os.environ.get("OPENROUTER_API_KEY", ""),
        "emb": os.environ.get("DEEPINFRA_API_KEY", ""),
    }
    missing = [name for name, key in keys.items() if not key]
    if missing:
        pytest.skip(f"real-server acceptance needs credentials: {missing}")
    return keys


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass
class Server:
    """One ``everos server start`` process over ``root``."""

    root: Path
    proxy: Proxy
    env_extra: dict[str, str] = field(default_factory=dict)
    port: int = 0
    proc: subprocess.Popen[bytes] | None = None
    log_path: Path | None = None
    _watchdog: threading.Timer | None = None
    exit_codes: list[int] = field(default_factory=list)
    everos: Path = _EVEROS
    """The ``everos`` executable (another checkout's, for a baseline run)."""

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def env(self) -> dict[str, str]:
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("EVEROS_") and k != "VIRTUAL_ENV"
        }
        env |= {
            "EVEROS_LLM__API_KEY": "via-proxy",
            "EVEROS_LLM__BASE_URL": f"{self.proxy.base}/llm",
            "EVEROS_EMBEDDING__API_KEY": "via-proxy",
            "EVEROS_EMBEDDING__BASE_URL": f"{self.proxy.base}/emb",
            "EVEROS_RERANK__API_KEY": "",
        }
        return env | self.env_extra

    def start(self, **env_extra: str) -> Server:
        self.env_extra |= env_extra
        if not (self.root / "everos.toml").exists():
            subprocess.run(
                [str(self.everos), "init", "--root", str(self.root)],
                env=self.env(),
                check=True,
                capture_output=True,
            )
        self.port = free_port()
        self.log_path = (
            self.root.parent / f"server-{self.root.name}-{len(self.exit_codes)}.log"
        )
        log = self.log_path.open("wb")
        self.proc = subprocess.Popen(
            [
                str(self.everos),
                "server",
                "start",
                "--root",
                str(self.root),
                "--port",
                str(self.port),
            ],
            env=self.env(),
            cwd=self.root.parent,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._watchdog = threading.Timer(_LIFETIME_S, self.kill)
        self._watchdog.daemon = True
        self._watchdog.start()
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited early: {self.log_text()[-2000:]}")
            try:
                if httpx.get(f"{self.base}/health", timeout=2).status_code == 200:
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        raise TimeoutError("server never became healthy")

    def wait_exit(self, timeout: float = 60) -> int:
        assert self.proc is not None
        code = self.proc.wait(timeout=timeout)
        self.exit_codes.append(code)
        self._release()
        return code

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.exit_codes.append(self.proc.wait(timeout=60))
        except subprocess.TimeoutExpired:
            self.kill()
        self._release()

    def kill(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            self.exit_codes.append(self.proc.wait(timeout=30))
        self._release()

    def _release(self) -> None:
        if self._watchdog is not None:
            self._watchdog.cancel()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:  # the port is really free again
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", self.port)) != 0:
                    return
            time.sleep(0.2)

    def log_text(self) -> str:
        return self.log_path.read_text(errors="replace") if self.log_path else ""

    # ── API ──────────────────────────────────────────────────────────────

    def post(
        self, path: str, body: dict[str, Any], *, timeout: float = 300
    ) -> httpx.Response:
        return httpx.post(f"{self.base}/api/v2{path}", json=body, timeout=timeout)

    def add(
        self, session: str, messages: list[dict[str, Any]], **extra: Any
    ) -> httpx.Response:
        return self.post(
            "/memory/add", {"session_id": session, "messages": messages, **extra}
        )

    def flush(self, session: str, **extra: Any) -> httpx.Response:
        return self.post("/memory/flush", {"session_id": session, **extra})

    def delete(self, **scope: Any) -> httpx.Response:
        return self.post("/memory/delete", scope)

    def delete_by_ids(self, ids: list[str], **extra: Any) -> httpx.Response:
        return self.post(
            "/memory/delete_by_ids",
            {"memory_type": "episode", "memory_ids": ids, **extra},
        )


@pytest.fixture
def proxy() -> Iterator[Proxy]:
    keys = _keys()
    p = Proxy({"llm": _LLM, "emb": _EMB}, keys)
    p.start()
    yield p
    p.stop()


@pytest.fixture
def server(tmp_path: Path, proxy: Proxy) -> Iterator[Server]:
    """A started server on a fresh root (``server.root``)."""
    root = tmp_path / "root"
    root.mkdir()
    srv = Server(root=root, proxy=proxy)
    srv.start()
    yield srv
    srv.stop()


_T0 = 1_762_041_600_000  # 2025-11-02T00:00:00Z


def chat(
    user: str, token: str, *, assistant: str | None = None, n: int = 2
) -> list[dict[str, Any]]:
    """A short conversation carrying ``token`` (a search keyword / control)."""
    now = int(time.time() * 1000)
    msgs = []
    for i in range(n):
        msgs.append(
            {
                "sender_id": user,
                "role": "user",
                "timestamp": now + i * 2000,
                "content": (
                    f"I want to remember {token}: my favourite place to visit"
                    f" is the {token} museum, item {i}."
                ),
            }
        )
        if assistant:
            msgs.append(
                {
                    "sender_id": assistant,
                    "role": "assistant",
                    "timestamp": now + i * 2000 + 1000,
                    "content": f"Noted, the {token} museum it is.",
                }
            )
    return msgs


@pytest.fixture
def make_server(tmp_path: Path, proxy: Proxy) -> Iterator[Any]:
    """``make_server(name)``: a server over ``tmp_path/name``, initialised
    (``everos.toml``) but not started -- seed first, then ``.start()``."""
    made: list[Server] = []

    def make(name: str = "root") -> Server:
        root = tmp_path / name
        root.mkdir(exist_ok=True)
        srv = Server(root=root, proxy=proxy)
        if not (root / "everos.toml").exists():
            subprocess.run(
                [str(_EVEROS), "init", "--root", str(root)],
                env=srv.env(),
                check=True,
                capture_output=True,
            )
        made.append(srv)
        return srv

    yield make
    for srv in made:
        srv.stop()
