"""Model-based test of the whole md → LanceDB cascade: the index follows the files.

Markdown is the source of truth and the LanceDB index is "rebuildable from
md" (docs/architecture.md). The chain that keeps the two aligned is real
here from end to end: entries are written by :class:`EpisodeWriter` into a
temp memory root, the user edits the files by hand, :class:`CascadeScanner`
walks the root and reconciles against ``md_change_state`` (real repository
on a temp SQLite file), :class:`CascadeWorker` claims the rows and runs the
real :class:`EpisodeHandler`, which parses the file, embeds and upserts into
a real LanceDB table in the same temp root, and runs the table's
maintenance (optimize / prune / rebuild) between batches. Only two things
are stood in for: the embedding provider (fixed vectors, with an outage the
machine can switch on) and the tokenizer.

Rules are what happens to a memory root in use: entries appended; a user
deleting one entry, editing one entry's content, or deleting a whole file;
an embedding outage of a few calls; a scanner sweep; a worker drain; a full
sync; a restart with orphan recovery.

The property is the system's one promise: after a sync with the embedding
service healthy, the index equals the files -- one row per entry still on
disk, keyed by ``<owner>_<entry_id>``, carrying that entry's content and
``md_path``, no row for an entry or a file that is gone -- and the
``md_change_state`` table is quiet (nothing pending, processing or failed).
Between syncs the index may lag; that is the design, so the check runs only
in the settled state.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

from hypothesis import HealthCheck, currently_in_test_context, settings
from hypothesis import event as _hypothesis_event
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule
from sqlalchemy import select
from sqlmodel import SQLModel

import everos.component.embedding.accessor as acc
from everos.component.embedding import EmbeddingCapability, EmbeddingProvider
from everos.component.tokenizer import Tokenizer
from everos.config import LanceDBSettings, SqliteSettings
from everos.core.errors import EmbeddingServiceError
from everos.core.persistence import (
    MarkdownReader,
    MemoryRoot,
    create_session_factory,
    create_system_engine,
)
from everos.core.persistence.lancedb import LanceRepoBase, open_lancedb_connection
from everos.core.persistence.markdown import entries as ent
from everos.core.persistence.markdown import frontmatter as fm
from everos.infra.persistence.index import Episode
from everos.infra.persistence.lancedb.repos.episode import _EpisodeRepo
from everos.infra.persistence.markdown import EpisodeWriter
from everos.infra.persistence.predicate import eq
from everos.infra.persistence.sqlite.repos.md_change_state import _MdChangeStateRepo
from everos.infra.persistence.sqlite.tables import MdChangeState
from everos.memory.cascade import scanner as scanner_mod
from everos.memory.cascade import worker as worker_mod
from everos.memory.cascade.handlers import HandlerDeps
from everos.memory.cascade.handlers.episode import EpisodeHandler
from everos.memory.cascade.scanner import CascadeScanner
from everos.memory.cascade.worker import CascadeWorker

settings.register_profile(
    "ci",
    max_examples=15,
    stateful_step_count=12,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "deep",
    max_examples=200,
    stateful_step_count=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))

OWNERS = ("u1", "u2")
DATE = dt.date(2026, 5, 14)
DIM = 1024


def event(label: str) -> None:
    if currently_in_test_context():
        _hypothesis_event(label)


class _Tokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return [t for t in text.split() if t]

    def tokenize_batch(self, texts: Any) -> list[list[str]]:
        return [self.tokenize(t) for t in texts]


class _Embedder(EmbeddingProvider):
    """Fixed vectors; ``outage`` is how many calls still fail."""

    dim = DIM

    def __init__(self) -> None:
        self.outage = 0
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        if self.outage > 0:
            self.outage -= 1
            raise EmbeddingServiceError("embedding 503")
        return [0.1] * DIM

    async def embed_batch(self, texts: Any) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class CascadeProjectionMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.tmp = Path(tempfile.mkdtemp(prefix="cascade_projection_"))
        self.root = MemoryRoot(self.tmp)
        self.root.ensure()
        self.engine = create_system_engine(self.root.system_db, SqliteSettings())
        self.factory = create_session_factory(self.engine)
        self.run(self._create_tables())
        self.state_repo = _MdChangeStateRepo(session_factory=self.factory)
        # The chassis' lock pool is bound to the loop that first used it.
        LanceRepoBase._reset_locks_for_tests()
        self.index: _EpisodeRepo = self.run(self._open_index())
        self.embedder = _Embedder()
        acc._capability = EmbeddingCapability(provider=self.embedder)
        self._patches = [
            mock.patch.object(scanner_mod, "md_change_state_repo", self.state_repo),
            # the scanner's state snapshot opens its own session on the
            # process-wide factory
            mock.patch.object(scanner_mod, "get_session_factory", lambda: self.factory),
            mock.patch.object(worker_mod, "md_change_state_repo", self.state_repo),
            mock.patch.object(EpisodeHandler, "index_repo", self.index),
        ]
        for p in self._patches:
            p.start()
        self.writer = EpisodeWriter(self.root)
        self.scanner = CascadeScanner(self.root, scan_interval_seconds=3600.0)
        self.handler = EpisodeHandler(
            HandlerDeps(memory_root=self.root, tokenizer=_Tokenizer())
        )
        self.worker = self._new_worker()
        # the model: what the files say -> (owner, entry_id) -> content
        self.entries: dict[tuple[str, str], str] = {}
        self.settled = False  # a clean sync happened since the last change
        self.seq = 0
        self.trace: list[str] = []

    def _new_worker(self) -> CascadeWorker:
        return CascadeWorker(
            {"episode": self.handler},
            batch_size=10,
            handler_concurrency=2,
            max_retry=1,
            retry_backoff_seconds=0.0,
            poll_interval_seconds=3600.0,
            optimize_min_interval_seconds=0.0,
            optimize_heartbeat_seconds=3600.0,
            optimize_rebuild_interval_seconds=3600.0,
        )

    async def _create_tables(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    async def _open_index(self) -> _EpisodeRepo:
        conn = await open_lancedb_connection(self.root.lancedb_dir, LanceDBSettings())
        table = await conn.create_table(Episode.TABLE_NAME, schema=Episode)
        return _EpisodeRepo(table=table)

    def run(self, coro: Any) -> Any:
        return self.loop.run_until_complete(coro)

    def teardown(self) -> None:
        for p in self._patches:
            p.stop()
        self.run(self.engine.dispose())
        self.loop.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _rel(self, owner: str) -> str:
        return (
            f"default_app/default_project/users/{owner}/episodes/"
            f"episode-{DATE.isoformat()}.md"
        )

    def _path(self, owner: str) -> Path:
        return self.root.root / self._rel(owner)

    # ── rules: the files ───────────────────────────────────────────────────

    @rule(owner=st.sampled_from(OWNERS), n=st.integers(1, 3))
    def append_entries(self, owner: str, n: int) -> None:
        items = []
        for _ in range(n):
            self.seq += 1
            items.append(
                (
                    {
                        "owner_id": owner,
                        "session_id": "s1",
                        "timestamp": "2026-05-14T10:00:00+00:00",
                        "parent_type": "memcell",
                        "parent_id": f"mc_{self.seq}",
                        "sender_ids": [owner],
                    },
                    {"Subject": "T", "Summary": "S", "Content": f"content {self.seq}"},
                )
            )
        eids = self.run(self.writer.append_entries(owner, items, date=DATE))
        for eid, (_, sections) in zip(eids, items, strict=True):
            self.entries[(owner, eid.format())] = sections["Content"]
        self.settled = False
        self.trace.append(f"append {owner} x{n}")
        event("append")

    @precondition(lambda self: bool(self.entries))
    @rule(which=st.integers(0, 50))
    def user_deletes_an_entry(self, which: int) -> None:
        keys = sorted(self.entries)
        owner, eid = keys[which % len(keys)]
        path = self._path(owner)
        meta, body = fm.parse_frontmatter(path.read_text(encoding="utf-8"))
        span = ent.find_entry(body, eid)
        assert span is not None
        end = span.end + 1 if body[span.end : span.end + 1] == "\n" else span.end
        path.write_text(fm.dump_frontmatter(meta) + body[: span.start] + body[end:])
        del self.entries[(owner, eid)]
        self.settled = False
        self.trace.append(f"user deletes {owner}/{eid}")
        event("user deletes an entry")

    @precondition(lambda self: bool(self.entries))
    @rule(which=st.integers(0, 50))
    def user_edits_an_entry(self, which: int) -> None:
        keys = sorted(self.entries)
        owner, eid = keys[which % len(keys)]
        path = self._path(owner)
        text = path.read_text(encoding="utf-8")
        old = self.entries[(owner, eid)]
        self.seq += 1
        new = f"edited {self.seq}"
        # Match the whole Content line: "content 1" is a prefix of "content 10".
        old_line, new_line = f"\n{old}\n", f"\n{new}\n"
        assert text.count(old_line) == 1, (old, text.count(old_line))
        path.write_text(text.replace(old_line, new_line))
        self.entries[(owner, eid)] = new
        self.settled = False
        self.trace.append(f"user edits {owner}/{eid}")
        event("user edits an entry")

    @precondition(lambda self: bool(self.entries))
    @rule(owner=st.sampled_from(OWNERS))
    def user_deletes_a_file(self, owner: str) -> None:
        path = self._path(owner)
        if not path.exists():
            return
        path.unlink()
        for key in [k for k in self.entries if k[0] == owner]:
            del self.entries[key]
        self.settled = False
        self.trace.append(f"user deletes file {owner}")
        event("user deletes a file")

    # ── rules: the cascade ─────────────────────────────────────────────────

    @rule(calls=st.integers(1, 4))
    def embedding_outage(self, calls: int) -> None:
        self.embedder.outage = calls
        self.settled = False
        self.trace.append(f"embedding outage for {calls} calls")
        event("embedding outage")

    @rule()
    def scan(self) -> None:
        n = len(self.run(self.scanner.scan_once()))
        self.trace.append(f"scan -> {n} decisions")
        event("scan")

    @rule()
    def drain(self) -> None:
        n = self.run(self.worker.drain_until_empty())
        self.trace.append(f"drain -> {n} rows")
        event("drain")

    @rule()
    def restart(self) -> None:
        orphans = self.run(self.state_repo.recover_orphan_processing())
        self.worker = self._new_worker()
        self.trace.append(f"restart (orphans {orphans})")
        event("restart")

    @rule()
    def sync_until_quiet(self) -> None:
        """What an operator's ``cascade sync`` with a healthy embedder does."""
        self.embedder.outage = 0
        for _ in range(6):
            self.run(self.scanner.scan_once())
            self.run(self.worker.drain_until_empty())
            rows = self.run(self._state_rows())
            if all(r.status == "done" for r in rows):
                break
        self.settled = True
        self.trace.append("sync until quiet")
        event("sync")

    # ── invariants ─────────────────────────────────────────────────────────

    async def _state_rows(self) -> list[MdChangeState]:
        async with self.factory() as s:
            return list((await s.execute(select(MdChangeState))).scalars().all())

    @invariant()
    def index_equals_files_once_settled(self) -> None:
        if not self.settled:
            return
        for owner in OWNERS:
            rows = self.run(
                self.index.find_where(eq("md_path", self._rel(owner)), limit=10_000)
            )
            got = {(r.owner_id, r.entry_id): (r.id, r.episode, r.md_path) for r in rows}
            want = {
                k: (f"{k[0]}_{k[1]}", content, self._rel(k[0]))
                for k, content in self.entries.items()
                if k[0] == owner
            }
            assert got == want, (owner, got, want, "\n".join(self.trace[-12:]))
        total = self.run(self.index.count())
        assert total == len(self.entries), (total, len(self.entries))
        state = self.run(self._state_rows())
        assert all(r.status == "done" for r in state), (
            [(r.md_path, r.status, r.error) for r in state if r.status != "done"],
            "\n".join(self.trace[-12:]),
        )

    @invariant()
    def files_still_parse(self) -> None:
        for owner in OWNERS:
            path = self._path(owner)
            if path.exists():
                parsed = self.run(MarkdownReader.read(path))
                ids = {e.id for e in parsed.entries}
                assert ids == {k[1] for k in self.entries if k[0] == owner}


TestCascadeProjectionMachine = CascadeProjectionMachine.TestCase
