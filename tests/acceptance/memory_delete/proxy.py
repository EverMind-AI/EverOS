"""An HTTP proxy in front of the LLM and the embedding provider.

``/llm/...`` forwards to the LLM base URL, ``/emb/...`` to the embedding one.
A rule matches a request by target and a regex over its body and then holds
it (until released), holds the upstream's reply (``hold_reply``: the answer is
ready, the caller gets it the moment the test releases it) or delays it;
``held()`` reports what is held right now, which is how a test knows its
window is open before it acts.
"""

from __future__ import annotations

import asyncio
import random
import re
import socket
import threading
import time
from dataclasses import dataclass, field

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route


@dataclass
class Rule:
    target: str  # "llm" | "emb"
    pattern: str
    mode: str  # "hold" | "hold_reply" | "delay" | "jitter" (0..delay) | "drop"
    delay: float = 0.0
    times: int = 1_000_000
    hits: int = 0
    since: int = 0  # len(Proxy.bodies) when the rule was added
    release: threading.Event = field(default_factory=threading.Event)


class Proxy:
    """Run with :meth:`start`; stop with :meth:`stop`."""

    def __init__(self, upstream: dict[str, str], keys: dict[str, str]) -> None:
        self.upstream = upstream
        self.keys = keys
        self.rules: list[Rule] = []
        self._held: dict[int, str] = {}
        self._lock = threading.Lock()
        self._n = 0
        self.log: list[tuple[str, int, float]] = []  # (target, status, seconds)
        self.bodies: list[tuple[str, str]] = []  # (target, request body)
        self.port = _free_port()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def add(self, rule: Rule) -> Rule:
        with self._lock:
            rule.since = len(self.bodies)
            self.rules.append(rule)
        return rule

    def clear(self) -> None:
        with self._lock:
            for rule in self.rules:
                rule.release.set()
            self.rules.clear()

    def held(self, target: str | None = None) -> int:
        with self._lock:
            return sum(1 for t in self._held.values() if target in (None, t))

    def wait_held(self, target: str | None = None, *, timeout: float = 120) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.held(target):
                return True
            time.sleep(0.05)
        return False

    def start(self) -> None:
        app = Starlette(
            routes=[
                Route("/{target}/{path:path}", self._handle, methods=["POST", "GET"])
            ]
        )
        config = uvicorn.Config(
            app, host="127.0.0.1", port=self.port, log_level="warning"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started and time.monotonic() < deadline:
            time.sleep(0.02)

    def stop(self) -> None:
        self.clear()
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)

    def _match(self, target: str, body: str) -> Rule | None:
        with self._lock:
            for rule in self.rules:
                if (
                    rule.target == target
                    and rule.hits < rule.times
                    and re.search(rule.pattern, body, re.S)
                ):
                    rule.hits += 1
                    return rule
        return None

    async def _hold(self, rule: Rule, token: int, target: str) -> None:
        with self._lock:
            self._held[token] = target
        try:
            # Released from the test's thread: wait for it off the loop.
            await asyncio.to_thread(rule.release.wait)
        finally:
            with self._lock:
                self._held.pop(token, None)

    async def _handle(self, request: Request) -> Response:
        target = request.path_params["target"]
        path = request.path_params["path"]
        body = await request.body()
        text = body.decode("utf-8", "replace")
        self.bodies.append((target, text))
        rule = self._match(target, text)
        with self._lock:
            self._n += 1
            token = self._n
        started = time.monotonic()
        if rule is not None and rule.mode in ("hold", "drop"):
            await self._hold(rule, token, target)
            if rule.mode == "drop":
                return Response(
                    status_code=500, content=b'{"error": "dropped by proxy"}'
                )
        elif rule is not None and rule.mode in ("delay", "jitter"):
            await asyncio.sleep(
                rule.delay if rule.mode == "delay" else random.uniform(0, rule.delay)
            )
        url = f"{self.upstream[target].rstrip('/')}/{path}"
        headers = {
            "content-type": request.headers.get("content-type", "application/json")
        }
        headers["authorization"] = f"Bearer {self.keys[target]}"
        async with httpx.AsyncClient(timeout=180) as client:
            upstream = await client.request(
                request.method, url, content=body, headers=headers
            )
        self.log.append((target, upstream.status_code, time.monotonic() - started))
        if rule is not None and rule.mode == "hold_reply":
            await self._hold(rule, token, target)
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
        )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# What each call looks like (read from real request bodies, 2026-11-02):
BOUNDARY = r"boundary detection expert"
EPISODE = r"You write episodic memories"


def llm_call(kind: str, token: str) -> str:
    """Regex for one operator's LLM call that carries ``token``."""
    return rf"{kind}.*{token}"


def cascade_embedding(token: str) -> str:
    """The cascade embeds an entry's texts in one batch (two or more inputs);
    clustering embeds the episode alone (one input)."""
    return rf'"input":\s*\[\s*"[^"]*{token}[^"]*"\s*,'


def single_embedding(token: str) -> str:
    return rf'"input":\s*\[\s*"[^"]*{token}[^"]*"\s*\]'
