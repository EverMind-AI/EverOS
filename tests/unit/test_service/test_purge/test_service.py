"""Purge request / confirm / get: the nonce, the mark, the job row."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest

from everos.core.errors import (
    NotFoundError,
    PurgeInProgressError,
    PurgeNonceInvalidError,
    StorageError,
)
from everos.core.purge_marks import MARKS
from everos.core.utils.datetime import get_utc_now
from everos.infra.persistence import purge_scope
from tests.fakes.storage import Storage

from ._harness import arm, disarm, make_purge, release, seed_scope


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    MARKS.reset()
    yield
    MARKS.reset()
    disarm(monkeypatch)


async def _ready(storage: Storage, **settings: float):  # type: ignore[no-untyped-def]
    await storage.create_schema()
    return make_purge(storage, **settings)


async def test_nonce_single_use(storage: Storage) -> None:
    p = await _ready(storage)
    req = await p.service.request("project", "a1", "p1")
    job = await p.service.confirm(req.nonce)
    assert job.status == "queued" and job.scope == "project"
    with pytest.raises(PurgeNonceInvalidError):
        await p.service.confirm(req.nonce)
    with pytest.raises(PurgeNonceInvalidError):
        await p.service.confirm("never-issued-nonce-xyz")


async def test_nonce_expires(storage: Storage) -> None:
    p = await _ready(storage, nonce_ttl_seconds=0.1)
    req = await p.service.request("project", "a1", "p1")
    assert req.expires_at - get_utc_now() <= timedelta(seconds=0.1)
    await asyncio.sleep(0.2)
    with pytest.raises(PurgeNonceInvalidError):
        await p.service.confirm(req.nonce)
    assert p.marks.active_job_id is None


async def test_second_confirm_while_reserved_is_invalid(
    storage: Storage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p = await _ready(storage)
    req = await p.service.request("project", "a1", "p1")
    arm(monkeypatch, tmp_path, "purge.confirm_insert=pause")
    first = asyncio.create_task(p.service.confirm(req.nonce))
    await asyncio.sleep(0.1)
    with pytest.raises(PurgeNonceInvalidError):
        await p.service.confirm(req.nonce)
    release(tmp_path, "purge.confirm_insert")
    job = await asyncio.wait_for(first, 2)
    assert [r.job_id for r in await storage.sqlite.purge_job.all_rows()] == [job.job_id]


async def test_mark_raised_before_insert(
    storage: Storage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p = await _ready(storage)
    req = await p.service.request("project", "a1", "p1")
    arm(monkeypatch, tmp_path, "purge.confirm_insert=pause")
    confirm = asyncio.create_task(p.service.confirm(req.nonce))
    await asyncio.sleep(0.1)
    assert p.marks.active_job_id is not None
    assert p.marks.tombstone_at("a1", "p1") is not None
    assert await storage.sqlite.purge_job.all_rows() == []
    release(tmp_path, "purge.confirm_insert")
    job = await asyncio.wait_for(confirm, 2)
    assert p.marks.active_job_id == job.job_id
    assert p.marks.tombstone_at("a1", "p1") == job.created_at


async def test_insert_failure_keeps_nonce_and_no_mark(
    storage: Storage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p = await _ready(storage)
    req = await p.service.request("project", "a1", "p1")
    arm(monkeypatch, tmp_path, "purge.confirm_insert=raise:sqlite_locked:1")
    with pytest.raises(StorageError):
        await p.service.confirm(req.nonce)
    assert p.marks.active_job_id is None
    assert p.marks.tombstone_at("a1", "p1") is None
    job = await p.service.confirm(req.nonce)  # the same nonce works now
    assert p.marks.active_job_id == job.job_id


async def test_active_job_returned_for_other_scope(storage: Storage) -> None:
    p = await _ready(storage)
    first = await p.service.confirm(
        (await p.service.request("project", "a1", "p1")).nonce
    )
    other = await p.service.request("app", "a2", None)
    again = await p.service.confirm(other.nonce)
    assert (again.job_id, again.scope) == (first.job_id, "project")
    assert len(await storage.sqlite.purge_job.all_rows()) == 1
    with pytest.raises(PurgeNonceInvalidError):  # that nonce was spent on it
        await p.service.confirm(other.nonce)


async def test_confirm_while_the_active_row_is_unwritten_waits_for_it(
    storage: Storage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p = await _ready(storage)
    first_req = await p.service.request("project", "a1", "p1")
    other_req = await p.service.request("app", "a2", None)
    arm(monkeypatch, tmp_path, "purge.confirm_insert=pause")
    first = asyncio.create_task(p.service.confirm(first_req.nonce))
    await asyncio.sleep(0.1)
    other = asyncio.create_task(p.service.confirm(other_req.nonce))
    await asyncio.sleep(0.1)
    assert not other.done()  # the mark is up but the job row is not written
    release(tmp_path, "purge.confirm_insert")
    job = await asyncio.wait_for(first, 2)
    assert (await asyncio.wait_for(other, 2)).job_id == job.job_id
    assert len(await storage.sqlite.purge_job.all_rows()) == 1


async def test_confirm_starts_its_own_job_when_the_pending_one_gives_up(
    storage: Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = await _ready(storage)
    first_req = await p.service.request("project", "a1", "p1")
    other_req = await p.service.request("app", "a2", None)
    go = asyncio.Event()
    real_insert = p.service.jobs.insert
    calls = 0

    async def insert(**fields: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            await go.wait()
            raise OSError("disk full")
        return await real_insert(**fields)

    monkeypatch.setattr(p.service.jobs, "insert", insert)
    first = asyncio.create_task(p.service.confirm(first_req.nonce))
    await asyncio.sleep(0.05)
    other = asyncio.create_task(p.service.confirm(other_req.nonce))
    await asyncio.sleep(0.05)
    go.set()
    with pytest.raises(StorageError):
        await asyncio.wait_for(first, 2)
    job = await asyncio.wait_for(other, 2)
    assert (job.scope, job.app_id) == ("app", "a2")
    assert p.marks.active_job_id == job.job_id


async def test_count_item_timeout(
    storage: Storage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p = await _ready(storage, count_timeout_seconds=0.1)
    await seed_scope(storage, "a1", "p1")
    arm(monkeypatch, tmp_path, "purge.count.index=pause")
    summary = (await p.service.request("project", "a1", "p1")).summary
    assert summary.index_total is None and summary.complete is False
    assert summary.error is not None and "index" in summary.error
    assert summary.markdown_total == 2 and summary.sqlite_total == 2


async def test_summary_counts_entries_not_shells_or_taxonomy(storage: Storage) -> None:
    p = await _ready(storage)
    await seed_scope(storage, "a1", "p1", n=3)
    knowledge = storage.root.knowledge_dir("a1", "p1")
    knowledge.mkdir(parents=True)
    (knowledge / ".taxonomy.md").write_text("---\nkind: knowledge_taxonomy\n---\n")
    summary = (await p.service.request("project", "a1", "p1")).summary
    assert summary.markdown["episode"] == 3 and summary.markdown_total == 3
    assert (summary.sqlite["memcell"], summary.index_total) == (3, 3)
    assert summary.complete and summary.index_store == "lancedb"


async def test_get_unknown_job_is_not_found(storage: Storage) -> None:
    p = await _ready(storage)
    with pytest.raises(NotFoundError):
        await p.service.get("pj_missing")


def test_default_maps_to_reserved_dirs() -> None:
    scope = purge_scope("project", "default", "default")
    assert (scope.app_dir, scope.project_dir) == ("default_app", "default_project")
    assert scope.covers("default", "default")
    with pytest.raises(ValueError):
        purge_scope("app", None, None)


@pytest.mark.parametrize(
    ("scope", "app", "project", "refused"),
    [
        (("project", "a1", "p1"), "a1", "p1", True),
        (("project", "a1", "p1"), "a1", "p2", False),
        (("project", "a1", "p1"), "a2", "p1", False),
        (("app", "a1", None), "a1", "p9", True),
        (("app", "a1", None), "a2", "p1", False),
        (("space", None, None), "a9", "p9", True),
        (("project", "default", "default"), "default", "default", True),
    ],
)
async def test_writes_are_refused_only_in_the_running_purges_range(
    storage: Storage,
    scope: tuple[str, str | None, str | None],
    app: str,
    project: str,
    refused: bool,
) -> None:
    p = await _ready(storage)
    p.service.refuse_writes_to(app, project)  # nothing running
    job = await p.service.confirm((await p.service.request(*scope)).nonce)
    if refused:
        with pytest.raises(PurgeInProgressError, match=job.job_id):
            p.service.refuse_writes_to(app, project)
    else:
        p.service.refuse_writes_to(app, project)
    assert await p.run_to_end(job.job_id) == "succeeded"
    p.service.refuse_writes_to(app, project)  # ended: accepted again


async def test_writes_stay_refused_after_a_restart(storage: Storage) -> None:
    p = await _ready(storage)
    job = await p.service.confirm(
        (await p.service.request("project", "a1", "p1")).nonce
    )
    MARKS.reset()  # a new process: only system.db remembers the job
    again = make_purge(storage)
    again.service.refuse_writes_to("a1", "p1")
    await again.service.restore()
    with pytest.raises(PurgeInProgressError, match=job.job_id):
        again.service.refuse_writes_to("a1", "p1")
