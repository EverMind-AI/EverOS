"""Failure injection over the real server, with deterministic providers.

What the failure-injection and concurrency suite (``tests/integration/
test_faults/``) shares:

* :class:`Upstream` -- a scripted OpenAI-compatible endpoint behind the real
  :class:`~everos.component.llm.OpenAIProvider` (an ``httpx.MockTransport``),
  so a fault goes through the provider's own retries and checks. Every reply
  model has a deterministic answer: an episode repeats the markers (``ZQ...``)
  of the conversation it was shown, a profile update adds one item per fact
  marker (``ZQF...``), the decider keeps the first ten candidates. A fault is
  queued per reply model (:meth:`Upstream.inject`) or held (:attr:`down`,
  :attr:`delay`).
* :class:`FlakyEmbedder` -- the fixed-vector embedder, which can be down
  (``EmbeddingServiceError``, what the real provider raises once its retries
  are spent) or slow.
* :class:`FaultServer` -- the whole application (markdown, ``system.db``,
  LanceDB, cascade, OME, the async queue) over a temporary root, driven
  through its HTTP routes in process, with the reads a fault test asserts on:
  the live markers of every episode (lost or duplicated memory), staging files
  left behind, and markdown-to-index parity.
* :class:`ChildServer` -- the same application in a server process of its
  own (``python -m tests.helpers.fault_server``), for the crash tests: the
  process can stop at a named point and be killed there.
"""

from __future__ import annotations

__all__ = [
    "BASE_ENV",
    "DIM",
    "T0",
    "ChildServer",
    "FaultServer",
    "FlakyEmbedder",
    "Upstream",
    "assert_exactly_once",
    "fault_providers",
    "index_write_fails",
    "line",
    "markdown_write_fails",
    "markers",
    "sqlite_write_fails",
    "wait_until",
]

import asyncio
import contextlib
import json
import os
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
from collections import Counter
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import event

from everos.component.llm import OpenAIProvider
from everos.config import load_settings
from everos.core.errors import EmbeddingServiceError
from everos.entrypoints.api.app import create_app
from everos.service import Application, build_application
from everos.service.bootstrap import Providers
from tests._consistency_assertions import assert_md_lance_strict_consistent
from tests.fakes import FakeEmbeddingProvider
from tests.fakes.storage import StorageView, open_storage_view

from .groups import live_markdown

DIM = 8
"""Vector width of the fixed embedder (and of every index table here)."""

BASE_ENV: Mapping[str, str] = {
    "EVEROS_EMBEDDING__DIM": str(DIM),
    # The test reads the tables the application writes through a second
    # connection; every read must see the latest committed version.
    "EVEROS_LANCEDB__READ_CONSISTENCY_SECONDS": "0",
    # A failed projection is retried on the next sweep: keep sweeps short.
    "EVEROS_CASCADE__SCAN_INTERVAL_SECONDS": "1",
    "EVEROS_DECIDER__RETRY_BACKOFF_SECONDS": "0",
    "EVEROS_MEMORIZE__ASYNC_QUEUE__SCAN_INTERVAL_SECONDS": "0.2",
    "EVEROS_MEMORIZE__ASYNC_QUEUE__RETRY_BACKOFF_SECONDS": "0.1",
    "EVEROS_LOGGING__LEVEL": "WARNING",
}

_MARK = re.compile(r"\bZQ[A-Z0-9]+\b")
_FACT = re.compile(r"\bZQF[A-Z0-9]+\b")
_SEGMENT = re.compile(r'<segment id="(s\d+)">(.*?)</segment>', re.S)
_PICTURE = re.compile(r"\bimg-\d+\b")
_ENTRY = re.compile(r"<!-- entry:(\S+) -->(.*?)<!-- /entry:\1 -->", re.S)
_GP_LINE = re.compile(
    r"^\[([^\]]+)\] \d{4}-\d{2}-\d{2} \d{2}:\d{2} ([^:\n]+): (.*)$", re.MULTILINE
)
_GP_SEGMENT = re.compile(r"^=== SEGMENT (\S+) ===$", re.MULTILINE)
_PROBE = "Reply with OK."

T0 = 1_772_323_200_000
"""2026-03-01 00:00 UTC, the first message time of every conversation here."""

FAULTS = frozenset(
    {"timeout", "500", "502", "503", "429", "malformed", "truncated", "cut"}
    | {"empty", "null"}
)
"""What :meth:`Upstream.inject` takes:

* ``timeout`` -- the request times out in transport;
* ``500`` / ``502`` / ``503`` / ``429`` -- that status, with a 1 ms
  ``retry-after-ms`` so the SDK's own retries do not sleep;
* ``malformed`` -- a reply that is not JSON;
* ``truncated`` -- the reply cut in half, ``finish_reason`` ``length``;
* ``cut`` -- the reply cut in half, ``finish_reason`` ``stop``;
* ``empty`` -- an empty string; ``null`` -- no content at all.
"""


def markers(text: str) -> list[str]:
    """The markers in ``text``, in order."""
    return _MARK.findall(text)


def line(who: str, text: str, at: int, **extra: Any) -> dict[str, Any]:
    """One chat message as ``/memory/add`` takes it (``at``: ms after T0)."""
    return {
        "sender_id": who,
        "role": "assistant" if who == "bot" else "user",
        "timestamp": T0 + at,
        "content": text,
        **extra,
    }


# ── the scripted upstream ───────────────────────────────────────────────────


@dataclass
class Upstream:
    """A deterministic OpenAI-compatible endpoint with scripted faults."""

    calls: list[str] = field(default_factory=list)
    """The reply model of every request received, in order (``probe`` for the
    decider's start-up probe)."""
    down: set[str] = field(default_factory=set)
    """Reply models (``*``: all) answered 503 while listed: a model that is down."""
    delay: dict[str, float] = field(default_factory=dict)
    """Seconds a reply model (``*``: all) waits before answering: a slow model."""
    cut_on_add: bool = False
    """The boundary cuts after the last message of every window, so an add
    extracts at once; off, an add only buffers and a flush cuts."""
    unexpected: list[str] = field(default_factory=list)
    """Reply models no script answers (answered 400)."""
    _plan: dict[str, list[str]] = field(default_factory=dict)

    def inject(self, name: str, *faults: str) -> None:
        """Answer the next calls for reply model ``name`` with ``faults``, in order."""
        unknown = set(faults) - FAULTS
        if unknown:
            raise ValueError(f"unknown faults {sorted(unknown)}")
        self._plan.setdefault(name, []).extend(faults)

    def pending(self, name: str) -> int:
        """Faults still queued for ``name``."""
        return len(self._plan.get(name, []))

    def count(self, name: str) -> int:
        return self.calls.count(name)

    def provider(self, *, decider: bool = False) -> OpenAIProvider:
        """The real provider over this endpoint, as the bootstrap configures it.

        The extraction model: the SDK's default transport retries, one extra
        attempt after an unusable reply. The decider: no SDK retries and no
        response format in the request (``prompt_only``), as ``[decider]``
        ships; the decider's own loop retries.
        """
        host = "decider.faults" if decider else "llm.faults"
        return OpenAIProvider(
            model="scripted",
            api_key="test",
            base_url=f"http://{host}/v1",
            timeout=5.0,
            max_retries=0 if decider else 2,
            response_format_mode="prompt_only" if decider else "json_schema",
            invalid_reply_retries=1,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.handle)),
        )

    async def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        name = _reply_name(request, body)
        self.calls.append(name)
        wait = self.delay.get(name, self.delay.get("*", 0.0))
        if wait:
            await asyncio.sleep(wait)
        queued = self._plan.get(name) or self._plan.get("*")
        fault = queued.pop(0) if queued else None
        if fault is None and (name in self.down or "*" in self.down):
            fault = "503"
        if fault == "timeout":
            raise httpx.ReadTimeout("scripted timeout", request=request)
        if fault in {"500", "502", "503", "429"}:
            return httpx.Response(
                int(fault),
                json={"error": {"message": f"scripted {fault}", "type": "scripted"}},
                headers={"retry-after-ms": "1"},
            )
        text = self._answer(name, body)
        if text is None:
            self.unexpected.append(name)
            return httpx.Response(400, json={"error": {"message": f"no {name}"}})
        finish = "stop"
        reply: str | None = text
        if fault == "malformed":
            reply = "{this is not json"
        elif fault in {"truncated", "cut"}:
            reply = text[: max(1, len(text) // 2)]
            finish = "length" if fault == "truncated" else "stop"
        elif fault == "empty":
            reply = ""
        elif fault == "null":
            reply = None
        return httpx.Response(200, json=_completion(reply, finish))

    def _answer(self, name: str, body: Mapping[str, Any]) -> str | None:
        prompt = _prompt(body)
        said = _conversation(body)
        if name == "probe":
            return "OK"
        if name == "_DecisionReply":
            return json.dumps({"core": list(range(10)), "next_queries": []})
        if name == "_BatchReply":
            # No cut inside the window; with ``cut_on_add`` the whole window is
            # one complete segment (an add extracts it at once).
            return json.dumps(
                {"reasoning": "s", "boundaries": [], "tail_complete": self.cut_on_add}
            )
        if name in {"EpisodeReply", "ImageEpisodeReply"}:
            found = " ".join(dict.fromkeys(markers(said))) or "nothing"
            reply: dict[str, Any] = {
                "title": "Notes",
                "summary": f"Notes on {found}",
                "content": found,
            }
            if name == "ImageEpisodeReply":
                reply["images"] = [
                    {"ref": ref, "description": f"picture {ref} of {found}", "ocr": ""}
                    for ref in dict.fromkeys(_PICTURE.findall(said))
                ]
            return json.dumps(reply)
        if name == "_ProfileUpdateReply":
            return json.dumps({"operations": _profile_operations(prompt)})
        if name == "_ConsolidateReply":
            return json.dumps({"changes": [], "dropped": []})
        if name == "GroupProfileReply":
            return json.dumps({"operations": _group_operations(prompt)})
        if name == "SceneReply":
            found = " ".join(dict.fromkeys(markers(said))) or "nothing"
            return json.dumps({"narrative": found, "summary": found, "title": "Scene"})
        return None


def _reply_name(request: httpx.Request, body: Mapping[str, Any]) -> str:
    if request.url.host == "decider.faults":
        return "probe" if _prompt(body).strip() == _PROBE else "_DecisionReply"
    fmt = body.get("response_format") or {}
    return str((fmt.get("json_schema") or {}).get("name") or "unnamed")


def _texts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    return [
        str(part.get("text", ""))
        for part in content or []
        if isinstance(part, dict) and part.get("type") == "text"
    ]


def _prompt(body: Mapping[str, Any]) -> str:
    return "\n".join(t for m in body.get("messages", []) for t in _texts(m["content"]))


def _conversation(body: Mapping[str, Any]) -> str:
    """The non-system messages: what the conversation shows, without rules."""
    return "\n".join(
        t
        for m in body.get("messages", [])
        if m.get("role") != "system"
        for t in _texts(m["content"])
    )


def _completion(text: str | None, finish: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-scripted",
        "object": "chat.completion",
        "created": 0,
        "model": "scripted",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _profile_operations(prompt: str) -> list[dict[str, Any]]:
    """One item per fact marker of each segment, unless the profile holds it."""
    stored = prompt.split("<new_conversation>")[0]
    ops = [
        {
            "action": "add",
            "type": "explicit_info",
            "data": {
                "evidence": fact,
                "description": f"Holds {fact}",
                "category": "facts",
            },
            "sources": [segment],
            "pictures": [],
        }
        for segment, body in _SEGMENT.findall(prompt)
        for fact in dict.fromkeys(_FACT.findall(body))
        if f"Holds {fact}" not in stored
    ]
    return ops or [{"action": "none"}]


_BLANK_GROUP_OP: Mapping[str, Any] = {
    "target": "",
    "topic_id": "",
    "name": "",
    "summary": "",
    "status": "",
    "statement": "",
    "decided_by": [],
    "agreed_by": [],
    "assignees": [],
    "applies_to": [],
    "held_by": [],
    "about": "",
    "member_id": "",
    "role": "",
    "confidence": "",
}


def _group_operations(prompt: str) -> list[dict[str, Any]]:
    """A decision citing every message of the batch."""
    body = prompt[prompt.find("=== SEGMENT") :]
    segments = _GP_SEGMENT.findall(body)
    texts = _GP_SEGMENT.split(body)[2::2]
    evidence = [
        {"segment": segment, "message": message_id, "quote": said}
        for segment, text in zip(segments, texts, strict=False)
        for message_id, _speaker, said in _GP_LINE.findall(text)
    ]
    if not evidence:
        return []
    quotes = " / ".join(e["quote"] for e in evidence)
    return [
        {
            **_BLANK_GROUP_OP,
            "op": "decision.new",
            "statement": quotes,
            "evidence": evidence,
        }
    ]


# ── the embedder ────────────────────────────────────────────────────────────


class FlakyEmbedder(FakeEmbeddingProvider):
    """The fixed-vector embedder, which can be down or slow."""

    def __init__(self, dim: int = DIM) -> None:
        super().__init__(dim=dim)
        self.down = False
        """Every call fails while set, as the real provider fails once its
        retries are spent."""
        self.delay = 0.0
        """Seconds every call waits first."""
        self.failures = 0

    async def embed(self, text: str) -> list[float]:
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.down:
            self.failures += 1
            raise EmbeddingServiceError("scripted: the embedding endpoint is down")
        return await super().embed(text)


def fault_providers(upstream: Upstream, embedder: FlakyEmbedder) -> Providers:
    """The application's providers over the scripted upstream and embedder."""
    return Providers(
        llm=upstream.provider(),
        decider=upstream.provider(decider=True),
        multimodal_llm=None,
        embedding=embedder,
        vision_embedding=None,
    )


def provision_root(root: Path, ome_toml: str = "") -> None:
    """What ``everos init`` leaves in a root that the server needs: ``ome.toml``.

    ``ome_toml`` is appended to it (strategy overrides, e.g. no OME retries).
    """
    root.mkdir(parents=True, exist_ok=True)
    target = root / "ome.toml"
    if not target.exists():
        shutil.copyfile(str(files("everos.config") / "default_ome.toml"), target)
    if ome_toml:
        with target.open("a", encoding="utf-8") as out:
            out.write("\n" + ome_toml + "\n")


# ── reading back what the server stored ─────────────────────────────────────


def live_episode_markers(root: Path, owner: str | None = None) -> Counter[str]:
    """How many live episodes carry each marker (of ``owner``, or of everyone).

    A marker a conversation sent that is missing is lost memory; one counted
    twice for the same owner is duplicated memory.
    """
    found: Counter[str] = Counter()
    for path, body in live_markdown(root).items():
        parts = path.split("/")
        if "episodes" not in parts:
            continue
        if owner is not None and owner not in parts:
            continue
        for _entry_id, text in _ENTRY.findall(body):
            found.update(set(markers(text)))
    return found


def staging_files(root: Path) -> list[str]:
    """Files a write stages before it renames them into place, left behind.

    The markdown writer stages ``.<name>.tmp.<uuid>``; a write that failed
    must not leave one (nor anything under ``.tmp/``).
    """
    left = [
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and (".tmp." in p.name or p.name.endswith(".tmp"))
    ]
    tmp = root / ".tmp"
    if tmp.is_dir():
        left += [p.relative_to(root).as_posix() for p in tmp.rglob("*") if p.is_file()]
    return sorted(left)


# ── storage faults ──────────────────────────────────────────────────────────


@contextlib.contextmanager
def markdown_write_fails(error: int, *, where: str = "") -> Iterator[list[str]]:
    """Every markdown write whose path holds ``where`` fails with ``error``.

    The write gets as far as half its staging file -- what a disk that fills
    mid-write leaves -- and then raises ``OSError(error)``. Yields the staging
    paths that failed.
    """
    from everos.infra.persistence.markdown._base import writer

    original = writer._write_and_fsync
    failed: list[str] = []

    def failing(tmp: Path, content: str) -> None:
        if where not in str(tmp):
            original(tmp, content)
            return
        failed.append(str(tmp))
        with open(tmp, "w", encoding="utf-8") as out:
            out.write(content[: len(content) // 2])
        raise OSError(error, os.strerror(error), str(tmp))

    writer._write_and_fsync = failing
    try:
        yield failed
    finally:
        writer._write_and_fsync = original


@contextlib.contextmanager
def sqlite_write_fails(
    application: Application,
    *,
    tables: Sequence[str] = (),
    message: str = "database or disk is full",
) -> Iterator[list[str]]:
    """Every write statement on ``tables`` (all: empty) of ``system.db`` fails.

    Raised from the driver, as SQLite reports a full disk
    (``sqlite3.OperationalError``); SQLAlchemy wraps it as it would the real
    one. Yields the statements that failed.
    """
    engine = application._admin._database.engine.sync_engine
    failed: list[str] = []

    def hook(*args: Any) -> None:
        statement = str(args[2])
        head = statement.lstrip().upper()
        if not head.startswith(("INSERT", "UPDATE", "DELETE", "REPLACE")):
            return
        if tables and not any(f" {t.upper()}" in head for t in tables):
            return
        failed.append(" ".join(statement.split()[:4]))
        raise sqlite3.OperationalError(message)

    event.listen(engine, "before_cursor_execute", hook)
    try:
        yield failed
    finally:
        event.remove(engine, "before_cursor_execute", hook)


@contextlib.contextmanager
def index_write_fails(error: BaseException) -> Iterator[list[str]]:
    """Every LanceDB write fails with ``error``, raised inside the table's
    write lock where lance raises its own (so the repository classifies it)."""
    from everos.infra.persistence.lancedb._base.repository import LanceRepoBase

    original = LanceRepoBase._to_arrow
    failed: list[str] = []

    def failing(self: Any, records: Any) -> Any:
        failed.append(self.table_name)
        raise error

    LanceRepoBase._to_arrow = failing  # type: ignore[method-assign]
    try:
        yield failed
    finally:
        LanceRepoBase._to_arrow = original  # type: ignore[method-assign]


# ── the server, in process ──────────────────────────────────────────────────


@dataclass
class FaultServer:
    """The application over ``root``, its HTTP client and a view of its storage."""

    root: Path
    upstream: Upstream
    embedder: FlakyEmbedder
    application: Application
    client: httpx.AsyncClient
    view: StorageView

    @classmethod
    @contextlib.asynccontextmanager
    async def open(
        cls,
        root: Path,
        *,
        env: Mapping[str, str] | None = None,
        upstream: Upstream | None = None,
        embedder: FlakyEmbedder | None = None,
        ome_toml: str = "",
    ) -> AsyncIterator[FaultServer]:
        """Start the application over ``root`` with ``env`` set; stop it on exit.

        ``env`` is set in ``os.environ`` for the server's life and restored
        afterwards (the settings are read from it once, at build);
        ``ome_toml`` is appended to the root's ``ome.toml``.
        """
        provision_root(root, ome_toml)
        settings_env = {**BASE_ENV, "EVEROS_ROOT": str(root), **(env or {})}
        saved = {key: os.environ.get(key) for key in settings_env}
        os.environ.update(settings_env)
        load_settings.cache_clear()
        try:
            settings = load_settings()
            upstream = upstream or Upstream()
            embedder = embedder or FlakyEmbedder()
            application = build_application(
                settings, providers=fault_providers(upstream, embedder)
            )
            app = create_app(settings, application=application)
            view = open_storage_view(root, settings)
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            try:
                async with (
                    app.router.lifespan_context(app),
                    httpx.AsyncClient(
                        transport=transport, base_url="http://faults", timeout=60.0
                    ) as client,
                ):
                    yield cls(
                        root=root,
                        upstream=upstream,
                        embedder=embedder,
                        application=application,
                        client=client,
                        view=view,
                    )
            finally:
                await view.aclose()
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            load_settings.cache_clear()

    # ── the routes ──

    async def add(
        self,
        session: str,
        *messages: Mapping[str, Any],
        group: str | None = None,
        async_mode: bool = False,
        defer: bool = False,
        **fields: Any,
    ) -> httpx.Response:
        body: dict[str, Any] = {
            "session_id": session,
            "messages": list(messages),
            "async_mode": async_mode,
            "defer_extraction": defer,
            **fields,
        }
        if group is not None:
            body["group_id"] = group
        return await self.client.post("/api/v1/memory/add", json=body)

    async def say(
        self,
        session: str,
        user: str,
        tag: str,
        count: int = 3,
        *,
        at: int = 0,
        fact: bool = False,
        **fields: Any,
    ) -> tuple[httpx.Response, list[str]]:
        """``user`` sends ``count`` messages, each with its own marker.

        Markers are ``ZQ<tag><nn>`` (``ZQF<tag><nn>`` with ``fact``: a fact a
        profile update records). Returns the add's answer and the markers.
        """
        prefix = "ZQF" if fact else "ZQ"
        sent = [f"{prefix}{tag}{i:02d}" for i in range(count)]
        lines = [line(user, f"note {m}", at + i * 1000) for i, m in enumerate(sent)]
        return await self.add(session, *lines, **fields), sent

    async def flush(self, session: str, **fields: Any) -> httpx.Response:
        return await self.client.post(
            "/api/v1/memory/flush", json={"session_id": session, **fields}
        )

    async def search(self, query: str, **fields: Any) -> httpx.Response:
        body = {"query": query, "top_k": 10, **fields}
        return await self.client.post("/api/v1/memory/search", json=body)

    async def post(self, path: str, body: Mapping[str, Any]) -> httpx.Response:
        return await self.client.post(f"/api/v1{path}", json=dict(body))

    async def health(self) -> dict[str, Any]:
        return (await self.client.get("/health")).json()

    # ── waiting and reading ──

    async def settle(self, seconds: float = 30.0) -> None:
        """Wait until nothing runs: extractions, the queue, OME and the cascade.

        Three idle samples in a row, since a file written a moment ago may not
        have reached the cascade's ledger yet.
        """
        engine = self.application._engine
        memory = self.application.memory
        streak = 0
        async with asyncio.timeout(seconds):
            while streak < 3:
                idle = await engine.wait_idle(timeout=0.5)
                ledger = await self.view.sqlite.md_change_state.queue_summary()
                queue = await memory.queue_health()
                busy = (
                    not idle
                    or ledger.pending > 0
                    or memory._extractions.in_flight > 0
                    or (queue is not None and queue.pending > 0)
                )
                streak = 0 if busy else streak + 1
                await asyncio.sleep(0.1)

    async def consistent(self) -> None:
        """Markdown and the index agree, row for row (after :meth:`settle`)."""
        await assert_md_lance_strict_consistent(self.root, self.view.repos)

    async def ledger(self) -> Any:
        return await self.view.sqlite.md_change_state.queue_summary()

    async def runs(self, strategy: str) -> list[tuple[str, str | None]]:
        """``(status, error)`` of every OME run of ``strategy``, oldest first."""
        records = await self.application._engine.list_runs(strategy)
        return [(str(r.status.value), r.error) for r in reversed(records)]

    def profile_text(self, user: str) -> str:
        """The user's profile document, frontmatter included (its items live
        there), or ``""`` when there is none."""
        found = [
            p.read_text("utf-8")
            for p in sorted(self.root.rglob("user.md"))
            if p.parent.name == user
        ]
        return "\n".join(found)

    def episodes(self, owner: str | None = None) -> Counter[str]:
        return live_episode_markers(self.root, owner)

    def staging(self) -> list[str]:
        return staging_files(self.root)

    async def searchable(self, user: str, query: str = "notes") -> set[str]:
        """The markers a search of ``user``'s memory returns."""
        resp = await self.search(query, user_id=user, top_k=50)
        assert resp.status_code == 200, resp.text
        return set(markers(json.dumps(resp.json()["data"])))


async def wait_until(
    predicate: Callable[[], Awaitable[bool]], seconds: float = 20.0
) -> None:
    """Poll ``predicate`` until it holds; ``TimeoutError`` after ``seconds``."""
    async with asyncio.timeout(seconds):
        while True:
            if await predicate():
                return
            await asyncio.sleep(0.1)


def assert_exactly_once(found: Counter[str], sent: Iterable[str]) -> None:
    """Every marker sent is in exactly one live episode, and nothing else is."""
    sent = list(sent)
    lost = sorted(set(sent) - set(found))
    doubled = sorted(m for m, n in found.items() if n > 1)
    extra = sorted(set(found) - set(sent))
    assert not lost and not doubled and not extra, (
        f"lost={lost} duplicated={doubled} unexpected={extra}"
    )


# ── the server, in a process of its own ─────────────────────────────────────


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class ChildServer:
    """A server process over ``root`` that the test owns and may kill.

    ``pause_at`` names a point the process stops at (see
    ``tests/helpers/fault_server.py``): it writes ``marker`` there and waits
    to be killed.
    """

    root: Path
    process: subprocess.Popen[bytes]
    port: int
    marker: Path
    log: Path

    @classmethod
    def start(
        cls,
        root: Path,
        *,
        pause_at: str | None = None,
        env: Mapping[str, str] | None = None,
        cut_on_add: bool = False,
    ) -> ChildServer:
        provision_root(root)
        port = _free_port()
        marker = root.parent / f"{root.name}.paused"
        marker.unlink(missing_ok=True)
        log = root.parent / f"{root.name}.server.log"
        args = [sys.executable, "-m", "tests.helpers.fault_server", "--port", str(port)]
        if pause_at:
            args += ["--pause-at", pause_at, "--marker", str(marker)]
        if cut_on_add:
            args.append("--cut-on-add")
        child_env = {
            **os.environ,
            **BASE_ENV,
            "EVEROS_ROOT": str(root),
            "EVEROS_LOGGING__LEVEL": "INFO",
            **(env or {}),
        }
        child_env.pop("PYTEST_CURRENT_TEST", None)
        with log.open("ab") as out:
            process = subprocess.Popen(
                args,
                cwd=Path(__file__).resolve().parents[2],
                env=child_env,
                stdout=out,
                stderr=subprocess.STDOUT,
            )
        return cls(root=root, process=process, port=port, marker=marker, log=log)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.url, timeout=60.0)

    async def ready(self, seconds: float = 60.0) -> None:
        """Wait until ``/health`` answers."""
        async with asyncio.timeout(seconds), self.client() as client:
            while True:
                if self.process.poll() is not None:
                    raise AssertionError(
                        f"server exited {self.process.returncode}: {self.tail()}"
                    )
                with contextlib.suppress(httpx.TransportError):
                    if (await client.get("/health")).status_code == 200:
                        return
                await asyncio.sleep(0.2)

    async def paused(self, seconds: float = 30.0) -> None:
        """Wait until the process reached its pause point."""
        async with asyncio.timeout(seconds):
            while not self.marker.exists():
                if self.process.poll() is not None:
                    raise AssertionError(
                        f"server exited {self.process.returncode}: {self.tail()}"
                    )
                await asyncio.sleep(0.05)

    def kill(self) -> None:
        """SIGKILL: no shutdown hook runs, as in a crash or an OOM kill."""
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGKILL)
        self.process.wait(timeout=30)

    def stop(self) -> None:
        """SIGTERM, and wait for the graceful shutdown."""
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.kill()

    def tail(self, lines: int = 40) -> str:
        text = (
            self.log.read_text("utf-8", errors="replace") if self.log.exists() else ""
        )
        return "\n".join(text.splitlines()[-lines:])


def send_order(*batches: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every marker of ``batches``, in order."""
    return [
        m for batch in batches for msg in batch for m in markers(str(msg["content"]))
    ]
