"""Model-based test of the LanceDB repository chassis on a real local table.

``LanceRepoBase`` is the one path every derived-index row takes into and out
of LanceDB. Its docstrings promise: ``upsert`` merges on the table's declared
primary key, so two rows with one key cannot come into existence; ``update``
patches columns of the rows a predicate selects; ``delete`` and
``delete_by_md_path`` remove exactly the rows that match; ``count`` is the
row count; ``get_by_id`` fetches one row; ``find_where_paginated`` returns
pages of the predicate's rows sorted by a column plus the predicate's true
total; ``optimize``, ``prune`` and ``rebuild_indexes`` are storage hygiene
and never change what the table holds.

Nothing is faked: a real LanceDB table in a temp directory, through the
real repository. The model is a dict keyed by the primary key.

Rules: upsert a batch (new keys and existing ones, across two scopes that
share ids), update a column on the rows of one md path, delete by id
predicate, delete by md path, optimize / prune / rebuild. Invariants after
every rule: the table's rows equal the model's (key, columns); ``count``
agrees; ``get_by_id`` finds exactly the model's row for an id in the
default scope; the pages of ``find_where_paginated`` over one scope, read
with every page size from 1 to 5, concatenate to that scope's rows sorted
by the sort column with the true total.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, ClassVar

from hypothesis import HealthCheck, currently_in_test_context, settings
from hypothesis import event as _hypothesis_event
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from everos.config import LanceDBSettings
from everos.core.persistence import MemoryRoot
from everos.core.persistence.lancedb import (
    BaseLanceTable,
    LanceRepoBase,
    open_lancedb_connection,
)
from everos.infra.persistence.lancedb import LanceIndexContract

settings.register_profile(
    "ci",
    max_examples=20,
    stateful_step_count=15,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "deep",
    max_examples=300,
    stateful_step_count=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))

IDS = ("a", "b", "c", "d")
SCOPES = (("default", "default"), ("app2", "default"))
MD_PATHS = ("users/u/x.md", "users/u/y.md")


def event(label: str) -> None:
    if currently_in_test_context():
        _hypothesis_event(label)


class _Row(BaseLanceTable):
    TABLE_NAME: ClassVar[str] = "_laws_rows"
    PRIMARY_KEY: ClassVar[tuple[str, ...]] = ("app_id", "project_id", "id")
    id: str
    app_id: str = "default"
    project_id: str = "default"
    md_path: str
    n: int
    text: str


class _Repo(LanceIndexContract[_Row], LanceRepoBase[_Row]):
    schema = _Row


Key = tuple[str, str, str]


def _key(r: _Row) -> Key:
    return (r.app_id, r.project_id, r.id)


class LanceRepoMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.tmp = Path(tempfile.mkdtemp(prefix="lance_repo_stateful_"))
        LanceRepoBase._reset_locks_for_tests()
        self.repo: _Repo = self.run(self._open())
        self.model: dict[Key, _Row] = {}
        self.seq = 0
        self.trace: list[str] = []

    async def _open(self) -> _Repo:
        mr = MemoryRoot(self.tmp)
        mr.ensure()
        conn = await open_lancedb_connection(mr.lancedb_dir, LanceDBSettings())
        table = await conn.create_table(_Row.TABLE_NAME, schema=_Row)
        return _Repo(table=table)

    def run(self, coro: Any) -> Any:
        return self.loop.run_until_complete(coro)

    def teardown(self) -> None:
        self.loop.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ── rules ──────────────────────────────────────────────────────────────

    @rule(
        picks=st.lists(
            st.tuples(
                st.sampled_from(IDS), st.sampled_from(SCOPES), st.sampled_from(MD_PATHS)
            ),
            min_size=1,
            max_size=4,
            unique_by=lambda p: (p[1], p[0]),
        )
    )
    def upsert(self, picks: list[tuple[str, tuple[str, str], str]]) -> None:
        rows = []
        for rid, (app, project), md_path in picks:
            self.seq += 1
            rows.append(
                _Row(
                    id=rid,
                    app_id=app,
                    project_id=project,
                    md_path=md_path,
                    n=self.seq,
                    text=f"v{self.seq}",
                )
            )
        self.trace.append(f"upsert {[(_key(r), r.n) for r in rows]}")
        self.run(self.repo.upsert(rows))
        for r in rows:
            self.model[_key(r)] = r
        event(
            "upsert: "
            + ("update" if any(_key(r) in self.model for r in rows) else "insert")
        )

    @rule(md_path=st.sampled_from(MD_PATHS))
    def update_text_by_md_path(self, md_path: str) -> None:
        self.seq += 1
        text = f"patched{self.seq}"
        self.trace.append(f"update text={text} where md_path={md_path}")
        self.run(self.repo.update({"text": text}, where=f"md_path = '{md_path}'"))
        for k, r in self.model.items():
            if r.md_path == md_path:
                self.model[k] = r.model_copy(update={"text": text})
        event("update")

    @rule(rid=st.sampled_from(IDS))
    def delete_by_id(self, rid: str) -> None:
        self.trace.append(f"delete id={rid}")
        self.run(self.repo.delete(f"id = '{rid}'"))
        for k in [k for k in self.model if k[2] == rid]:
            del self.model[k]
        event("delete by id")

    @rule(md_path=st.sampled_from(MD_PATHS))
    def delete_by_md_path(self, md_path: str) -> None:
        self.trace.append(f"delete_by_md_path {md_path}")
        n = self.run(self.repo.delete_by_md_path(md_path))
        want = [k for k, r in self.model.items() if r.md_path == md_path]
        assert n == len(want), (n, want, "\n".join(self.trace[-10:]))
        for k in want:
            del self.model[k]
        event("delete by md_path")

    @rule(which=st.sampled_from(["optimize", "prune", "rebuild"]))
    def maintenance(self, which: str) -> None:
        self.trace.append(which)
        if which == "optimize":
            self.run(self.repo.optimize())
        elif which == "prune":
            self.run(self.repo.prune(dt.timedelta(seconds=0)))
        else:
            self.run(self.repo.rebuild_indexes())
        event(f"maintenance: {which}")

    # ── invariants ─────────────────────────────────────────────────────────

    async def _all_rows(self) -> list[_Row]:
        rows = await self.repo.find_where("id IS NOT NULL", limit=10_000)
        return [_Row.model_validate(r) for r in rows]

    @invariant()
    def table_matches_model(self) -> None:
        rows = self.run(self._all_rows())
        got = {_key(r): (r.md_path, r.n, r.text) for r in rows}
        assert len(got) == len(rows), ("a key appears twice", [_key(r) for r in rows])
        want = {k: (r.md_path, r.n, r.text) for k, r in self.model.items()}
        assert got == want, "\n".join(self.trace[-10:])
        assert self.run(self.repo.count()) == len(self.model)

    @invariant()
    def get_by_id_finds_the_default_scope_row(self) -> None:
        for rid in IDS:
            got = self.run(self.repo.get_by_id(rid))
            want = [r for k, r in self.model.items() if k[2] == rid]
            if len(want) == 1:
                assert got is not None and (_key(got), got.n) == (
                    _key(want[0]),
                    want[0].n,
                )
            elif not want:
                assert got is None, rid

    @invariant()
    def pages_cover_one_scope_in_order(self) -> None:
        where = "app_id = 'default' AND project_id = 'default'"
        want = sorted(
            (r for k, r in self.model.items() if k[:2] == ("default", "default")),
            key=lambda r: r.n,
            reverse=True,
        )
        for page_size in range(1, 6):
            seen: list[_Row] = []
            page = 1
            while True:
                rows, total = self.run(
                    self.repo.find_where_paginated(
                        where,
                        sort_by="n",
                        descending=True,
                        page=page,
                        page_size=page_size,
                    )
                )
                assert total == len(want), (total, len(want))
                seen.extend(rows)
                if len(rows) < page_size:
                    break
                page += 1
            assert [(_key(r), r.n) for r in seen] == [(_key(r), r.n) for r in want], (
                page_size,
                "\n".join(self.trace[-10:]),
            )


TestLanceRepoMachine = LanceRepoMachine.TestCase
