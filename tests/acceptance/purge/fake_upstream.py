"""A local stand-in for the LLM, embedding and rerank providers.

``/llm/chat/completions`` answers by the reply model each call names in
``response_format.json_schema.name`` -- the same name the in-process fakes
key their scripts on (:mod:`tests.fakes.llm`) -- so the server's operators
get the replies the unit tests already use. A request carrying a scope token
(``zq`` + 6 lowercase letters / digits) gets the token written into its reply
("Kyoto" -> "Kyoto <token>"), so what the server stores and embeds downstream
still says which scope it came from and a proxy rule can match it.

``/emb/embeddings`` hashes each word into one of ``dim`` signed buckets
(sha256), so equal texts embed equally and texts sharing most words land
close: clustering and Reflection see the same structure real models give
them. ``/rerank/rerank`` scores documents by words shared with the query.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from tests.fakes.llm import (
    AGENT_TRACK_OPERATORS,
    KNOWLEDGE_OPERATORS,
    USER_TRACK_OPERATORS,
)

_SCRIPTS = Path(__file__).parents[2] / "fakes" / "llm_scripts"
TOKEN = re.compile(r"\bzq[a-z0-9]{6}\b")
_WORD = re.compile(r"\w+")


def _script_files() -> dict[str, Path]:
    """Reply model name -> the default script answering it."""
    files: dict[str, Path] = {}
    for track, operators in (
        ("agent_track", AGENT_TRACK_OPERATORS),
        ("knowledge", KNOWLEDGE_OPERATORS),
        ("user_track", USER_TRACK_OPERATORS),  # last: owns the shared boundary
    ):
        for operator, name in operators.items():
            files[name] = _SCRIPTS / track / f"{operator}.json"
    return files


@dataclass
class Override:
    """Answer calls whose body matches ``pattern`` with ``reply``."""

    pattern: str
    reply: str
    name: str | None = None  # only calls naming this reply model
    times: int = 1_000_000
    hits: int = 0


def embed(text: str, dim: int) -> list[float]:
    """Signed feature hashing of ``text``'s words, unit length."""
    vec = [0.0] * dim
    for word in _WORD.findall(text.lower()):
        digest = hashlib.sha256(word.encode()).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        vec[index] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:  # no words: a fixed direction, never a zero vector
        return [1.0] + [0.0] * (dim - 1)
    return [v / norm for v in vec]


class FakeUpstream:
    """Run with :meth:`start`; stop with :meth:`stop`."""

    def __init__(self, *, dim: int = 64) -> None:
        self.dim = dim
        self.scripts = _script_files()
        self.overrides: list[Override] = []
        self.names: list[str] = []  # reply model of every LLM call, in order
        self.unknown_names: list[str] = []  # ones no script answers
        self.first_llm_body: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self.port = _free_port()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def override(self, item: Override) -> Override:
        with self._lock:
            self.overrides.append(item)
        return item

    def start(self) -> None:
        app = Starlette(
            routes=[
                Route("/llm/chat/completions", self._chat, methods=["POST"]),
                Route("/emb/embeddings", self._embeddings, methods=["POST"]),
                Route("/rerank/rerank", self._rerank, methods=["POST"]),
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
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)

    # ── handlers ─────────────────────────────────────────────────────────

    def _reply_for(self, name: str, text: str) -> str | None:
        with self._lock:
            for item in self.overrides:
                if (
                    item.name in (None, name)
                    and item.hits < item.times
                    and re.search(item.pattern, text, re.S)
                ):
                    item.hits += 1
                    return item.reply
        path = self.scripts.get(name)
        if path is None:
            return None
        reply = path.read_text(encoding="utf-8")
        token = TOKEN.search(text)
        return reply.replace("Kyoto", f"Kyoto {token.group(0)}") if token else reply

    async def _chat(self, request: Request) -> JSONResponse:
        raw = await request.body()
        text = raw.decode("utf-8", "replace")
        body = json.loads(text)
        fmt = body.get("response_format") or {}
        name = (fmt.get("json_schema") or {}).get("name") or "<no json_schema>"
        with self._lock:
            if self.first_llm_body is None:
                self.first_llm_body = body
            self.names.append(name)
        reply = self._reply_for(name, text)
        if reply is None:
            with self._lock:
                self.unknown_names.append(name)
            return JSONResponse({"error": f"no script for {name}"}, status_code=400)
        return JSONResponse(
            {
                "id": f"fake-{len(self.names)}",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", "fake"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )

    async def _embeddings(self, request: Request) -> JSONResponse:
        body = await request.json()
        inputs = body["input"]
        inputs = [inputs] if isinstance(inputs, str) else inputs
        dim = int(body.get("dimensions") or self.dim)
        return JSONResponse(
            {
                "object": "list",
                "model": body.get("model", "fake"),
                "data": [
                    {"object": "embedding", "index": i, "embedding": embed(t, dim)}
                    for i, t in enumerate(inputs)
                ],
                "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
            }
        )

    async def _rerank(self, request: Request) -> JSONResponse:
        body = await request.json()
        query = set(_WORD.findall(body["query"].lower()))
        results = []
        for i, doc in enumerate(body["documents"]):
            words = set(_WORD.findall(doc.lower()))
            results.append(
                {"index": i, "relevance_score": min(1.0, len(query & words) / 10)}
            )
        return JSONResponse({"results": results})


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
