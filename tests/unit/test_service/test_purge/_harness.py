"""A purge service over real stores on a temp root; only OME is a stand-in."""

from __future__ import annotations

import asyncio
import importlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from everos.config import PurgeSettings
from everos.core import failpoints
from everos.core.purge_marks import MARKS, PurgeMarks
from everos.core.utils.datetime import get_utc_now, today_with_timezone
from everos.infra.cascade import DeleteFence
from everos.infra.persistence import purge_scope
from everos.infra.persistence.markdown import (
    MarkdownOwnerDirStore,
    MarkdownScopePurger,
)
from everos.infra.persistence.markdown._writers import EpisodeWriter
from everos.infra.persistence.records import Episode
from everos.infra.persistence.sqlite import session_scope
from everos.infra.persistence.sqlite._tables import Memcell
from everos.service.purge import PurgeService
from everos.service.purge._runner import PurgeRunner
from tests.fakes import TEST_DIM
from tests.fakes.storage import Storage


@dataclass
class FakeRun:
    run_id: str
    event_id: str
    started_at: datetime
    event_payload: str


@dataclass
class FakeEngine:
    """``running_runs`` / ``redact_payloads`` over an in-memory run list."""

    running: list[FakeRun] = field(default_factory=list)
    failed_payloads: dict[str, str] = field(default_factory=dict)

    async def running_runs(self) -> list[FakeRun]:
        return list(self.running)

    async def redact_payloads(self, match: Callable[[str], bool], marker: str) -> int:
        hits = [k for k, p in self.failed_payloads.items() if p != marker and match(p)]
        for k in hits:
            self.failed_payloads[k] = marker
        return len(hits)


def arm(monkeypatch: Any, tmp_path: Any, spec: str) -> None:
    """Re-read the failpoints with ``spec`` armed (undone by ``disarm``)."""
    monkeypatch.setenv("EVEROS_TEST_FAILPOINTS", spec)
    monkeypatch.setenv("EVEROS_TEST_FAILPOINT_DIR", str(tmp_path / "fp"))
    importlib.reload(failpoints)


def disarm(monkeypatch: Any) -> None:
    monkeypatch.delenv("EVEROS_TEST_FAILPOINTS", raising=False)
    importlib.reload(failpoints)


def release(tmp_path: Any, name: str) -> None:
    (tmp_path / "fp").mkdir(exist_ok=True)
    (tmp_path / "fp" / f"{name}.release").write_text("go")


@dataclass
class Purge:
    service: PurgeService
    storage: Storage
    engine: FakeEngine
    fence: DeleteFence
    extractions: list[datetime]
    """Origins of the extractions the service reports running."""

    @property
    def marks(self) -> PurgeMarks:
        return MARKS.for_root(self.storage.root.root)

    def runner(self, **kw: Any) -> PurgeRunner:
        """A fast runner the service hands its jobs to (not started)."""
        svc = self.service
        runner = PurgeRunner(
            settings=svc.settings,
            marks=self.marks,
            jobs=svc.jobs,
            scope_repo=svc.scope_repo,
            markdown=svc.markdown,
            index=svc.index,
            index_store=svc.index_store,
            engine=svc.engine,
            fence=svc.fence,
            scope_of=svc.scope_of,
            extractions_before=svc.extractions_before,
            **({"backoff": lambda n: 0.01, "poll_seconds": 0.05} | kw),
        )
        svc._runner = runner
        return runner

    async def run_to_end(self, job_id: str, limit_s: float = 10) -> str:
        runner = self.runner()
        runner.start()
        try:
            async with asyncio.timeout(limit_s):
                while True:
                    row = await self.service.jobs.get(job_id)
                    terminal = row is not None and row.status in ("succeeded", "failed")
                    if terminal and self.marks.active_job_id is None:
                        return row.status  # type: ignore[union-attr]
                    await asyncio.sleep(0.02)
        finally:
            await runner.stop()


def make_purge(storage: Storage, **settings: Any) -> Purge:
    root = storage.root
    engine = FakeEngine()
    fence = DeleteFence()
    extractions: list[datetime] = []
    service = PurgeService(
        settings=PurgeSettings(**settings),
        root=root.root,
        scope_of=purge_scope,
        jobs=storage.sqlite.purge_job,
        scope_repo=storage.sqlite.scope_purge,
        markdown=MarkdownScopePurger(root, MarkdownOwnerDirStore(root)),
        index=storage.repos,
        index_store="lancedb",
        engine=engine,  # type: ignore[arg-type]
        fence=fence,
        extractions_before=lambda at, scope: sum(1 for o in extractions if o < at),
    )
    return Purge(service, storage, engine, fence, extractions)


async def seed_scope(storage: Storage, app: str, project: str, n: int = 2) -> None:
    """``n`` episodes in md, ``n`` index rows, ``n`` memcells."""
    root = storage.root
    writer = EpisodeWriter(root)
    for i in range(n):
        await writer.append_entry(
            "u1",
            inline={"owner_id": "u1", "parent_type": "memcell", "parent_id": "mc"},
            sections={"Subject": "s", "Summary": "s", "Content": f"e{i}"},
            date=today_with_timezone(),
            app_id=app,
            project_id=project,
        )
    await storage.repos.episode.upsert(
        [
            Episode(
                id=f"{app}_{project}_ep{i}",
                entry_id=f"ep{i}",
                owner_id="u1",
                owner_type="user",
                app_id=app,
                project_id=project,
                timestamp=datetime(2026, 1, 1, tzinfo=UTC),
                parent_id="mc",
                sender_ids=["u1"],
                episode=f"e{i}",
                episode_tokens=f"e{i}",
                md_path=f"{app}/{project}/users/u1/episodes/x.md",
                content_sha256="a",
                vector=[1.0] + [0.0] * (TEST_DIM - 1),
            )
            for i in range(n)
        ]
    )
    async with session_scope(storage.database.session_factory) as s:
        for i in range(n):
            s.add(
                Memcell(
                    memcell_id=f"mc_{app}_{project}_{i}",
                    app_id=app,
                    project_id=project,
                    session_id="s1",
                    track="user_memory",
                    raw_type="conversation",
                    message_ids_json="[]",
                    sender_ids_json="[]",
                    payload_json="{}",
                    timestamp=get_utc_now(),
                )
            )
        await s.commit()


def payload(app: str, project: str, ts: datetime | None = None) -> str:
    data = {"app_id": app, "project_id": project}
    if ts is not None:
        data["ts"] = ts.isoformat()
    return json.dumps(data)
