"""The deletion use case (``Deleter``) over real markdown and SQLite stores.

Only the index eraser is a recording double: what reaches the index is the
cascade tests' subject; here it is which entries each scope deletes, what
it refuses, and what it reports.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import pytest
from everalgo.types import ChatMessage, MemCell

from everos.config import MemorizeSettings
from everos.core.errors import DeletionIncompleteError, SessionBusyError
from everos.core.locks import LOCKS
from everos.infra.persistence.factories import build_stores
from everos.infra.persistence.markdown import MarkdownReader
from everos.memory.deps import Stores
from everos.memory.models import (
    AgentCase,
    AtomicFact,
    Episode,
    ImageMemory,
    MemcellRecord,
    Profile,
)
from everos.memory.ports import NO_SOURCE_CHECK, MemoryRef, Scope, SessionKey
from everos.service.memory._delete import Deleter
from tests.fakes.storage import Storage

_TS = 1_747_216_800_000


class _RecordingEraser:
    def __init__(self) -> None:
        self.erased: list[tuple[Scope, dict[str, list[str]]]] = []
        self.owners: list[tuple[Scope, list[str]]] = []

    async def erase(
        self, scope: Scope, removed: Mapping[str, Sequence[MemoryRef]]
    ) -> None:
        self.erased.append((scope, {k: [r.id for r in v] for k, v in removed.items()}))

    async def erase_owner(self, scope: Scope, kinds: Any) -> None:
        self.owners.append((scope, list(kinds)))


@pytest.fixture
async def stores(storage: Storage) -> AsyncIterator[Stores]:
    await storage.create_schema()
    yield build_stores(storage.root, storage.sqlite)


@pytest.fixture
def eraser() -> _RecordingEraser:
    return _RecordingEraser()


def _deleter(stores: Stores, eraser: _RecordingEraser, wait: float = 10.0) -> Deleter:
    return Deleter(
        stores=stores,
        eraser=eraser,  # type: ignore[arg-type]
        settings=MemorizeSettings(delete_wait_seconds=wait),
    )


def _user(owner: str, project: str = "p") -> Scope:
    return Scope(app_id="a", project_id=project, owner_id=owner, owner_type="user")


def _agent(owner: str, project: str = "p") -> Scope:
    return Scope(app_id="a", project_id=project, owner_id=owner, owner_type="agent")


async def _seed(stores: Stores, session: str, *, project: str = "p") -> None:
    """Session ``session``: users u1, u2 and agent a1 took part."""
    key = SessionKey(app_id="a", project_id=project, session_id=session)
    await stores.memcell.insert_many(
        key,
        [
            MemcellRecord(
                memcell_id=f"mc_{project}_{session}",
                cell=MemCell(
                    items=[
                        ChatMessage(
                            id="m",
                            role="user",
                            content="x",
                            timestamp=_TS,
                            sender_id="u1",
                        )
                    ],
                    timestamp=_TS,
                ),
                raw_type="conversation",
                message_ids=["m"],
                sender_ids=["u1", "u2", "a1"],
            )
        ],
    )
    for owner in ("u1", "u2"):
        scope = _user(owner, project)
        (ep,) = await stores.episode.append_many(
            scope,
            [
                Episode(
                    owner_id=owner,
                    episode=f"{owner} in {session}",
                    timestamp=_TS,
                    subject="s",
                    summary="s",
                    session_id=session,
                    parent_id=f"mc_{project}_{session}",
                )
            ],
            only_if=NO_SOURCE_CHECK,
        )
        await stores.atomic_fact.append_many(
            scope,
            [
                AtomicFact(
                    owner_id=owner,
                    content="f",
                    timestamp=_TS,
                    session_id=session,
                    parent_id=ep.entry_id,
                )
            ],
            only_if=NO_SOURCE_CHECK,
        )
    await stores.agent_case.append(
        _agent("a1", project),
        AgentCase(
            owner_id="a1",
            task_intent="t",
            approach="a",
            quality_score=0.5,
            timestamp=_TS,
            parent_id=f"mc_{project}_{session}",
            session_id=session,
        ),
        only_if=NO_SOURCE_CHECK,
    )


async def _entries(
    storage: Storage, owner_type: str, owner: str, project: str = "p"
) -> int:
    base = storage.root.users_dir if owner_type == "user" else storage.root.agents_dir
    folder = base("a", project) / owner
    total = 0
    for md in folder.rglob("*.md"):
        if md.name != "user.md":
            total += len((await MarkdownReader.read(md)).entries)
    return total


async def _enqueue(stores: Stores, session: str = "s1") -> None:
    await stores.memorize_inbox.enqueue(
        SessionKey(app_id="a", project_id="p", session_id=session),
        request_id=f"r_{session}",
        is_final=False,
        defer_extraction=False,
        payload_json="{}",
        message_count=1,
    )


async def _delete(deleter: Deleter, **scope: Any) -> Any:
    fields = {
        "app_id": "a",
        "project_id": "p",
        "user_id": None,
        "agent_id": None,
        "session_id": None,
    }
    return await deleter.delete(**(fields | scope))


async def _snapshot(storage: Storage) -> list[tuple[str, int]]:
    return sorted(
        (str(p.relative_to(storage.root.root)), len(p.read_bytes()))
        for p in storage.root.root.rglob("*.md")
    )


async def test_busy_before_lock_503_nothing_deleted(
    storage: Storage, stores: Stores, eraser: _RecordingEraser
) -> None:
    """A4 (a): queued work for the session -> 503, nothing changed."""
    await _seed(stores, "s1")
    await _enqueue(stores)
    before = await _snapshot(storage)

    with pytest.raises(SessionBusyError, match="nothing was deleted, retry later"):
        await _delete(_deleter(stores, eraser), session_id="s1")

    assert await _snapshot(storage) == before
    assert await stores.memcell.find_by_ids(["mc_p_s1"])
    assert eraser.erased == []


async def test_busy_after_lock_503_nothing_deleted(
    storage: Storage, stores: Stores, eraser: _RecordingEraser
) -> None:
    """A4 (b): a request queued while the deletion waited for the lock."""
    await _seed(stores, "s1")
    before = await _snapshot(storage)

    async with LOCKS.hold("session", "a\x1fp\x1fs1"):  # an add() in progress
        deletion = asyncio.create_task(
            _delete(_deleter(stores, eraser), session_id="s1")
        )
        await asyncio.sleep(0.05)
        await _enqueue(stores)
    with pytest.raises(SessionBusyError):
        await deletion

    assert await _snapshot(storage) == before
    assert await stores.memcell.find_by_ids(["mc_p_s1"])


async def test_lock_wait_timeout_503(stores: Stores, eraser: _RecordingEraser) -> None:
    """A5: past delete_wait_seconds the deletion gives up, deleting nothing."""
    await _seed(stores, "s1")
    async with LOCKS.hold("session", "a\x1fp\x1fs1"), asyncio.timeout(2):
        with pytest.raises(SessionBusyError, match="session is busy"):
            await _delete(_deleter(stores, eraser, wait=0.05), session_id="s1")
    assert await stores.memcell.find_by_ids(["mc_p_s1"])


async def test_filters_and_count_per_scope(
    storage: Storage, stores: Stores, eraser: _RecordingEraser
) -> None:
    """A1 / A6: each scope's filters, and counts equal to what md lost."""
    for session in ("s1", "s2"):
        await _seed(stores, session)
    await stores.profile.write(
        _user("u1"), Profile(owner_id="u1", summary="p", timestamp=_TS)
    )
    d = _deleter(stores, eraser)

    r = await _delete(d, user_id="u1", session_id="s1")
    assert (r.filters, r.count) == (["user_id", "session_id"], 2)
    r = await _delete(d, agent_id="a1", session_id="s1")
    assert (r.filters, r.count) == (["agent_id", "session_id"], 1)
    r = await _delete(d, session_id="s1")  # what is left of s1: u2's two
    assert (r.filters, r.count) == (["session_id"], 2)
    r = await _delete(d, user_id="u1")  # s2's two entries + user.md
    assert (r.filters, r.count) == (["user_id"], 3)
    r = await _delete(d, agent_id="a1")  # s2's case
    assert (r.filters, r.count) == (["agent_id"], 1)
    assert await _entries(storage, "user", "u1") == 0
    assert await _entries(storage, "user", "u2") == 2  # s2 only
    assert await _entries(storage, "agent", "a1") == 0


async def test_owners_from_memcell_senders_users_and_agents(
    storage: Storage, stores: Stores, eraser: _RecordingEraser
) -> None:
    """Every sender of the session's memcells is tried as a user and as an
    agent; the index eraser gets each owner's removed rows."""
    await _seed(stores, "s1")

    result = await _delete(_deleter(stores, eraser), session_id="s1")

    assert result.count == 5  # u1: ep + fact, u2: ep + fact, a1: case
    for owner_type, owner in (("user", "u1"), ("user", "u2"), ("agent", "a1")):
        assert await _entries(storage, owner_type, owner) == 0
    erased = {(s.owner_type, s.owner_id): ids for s, ids in eraser.erased}
    assert len(erased[("user", "u1")]["episode"]) == 1
    assert len(erased[("agent", "a1")]["agent_case"]) == 1
    assert await stores.memcell.find_by_ids(["mc_p_s1"]) == []


def _picture(owner: str, session: str, project: str = "p") -> ImageMemory:
    return ImageMemory(
        owner_id=owner,
        description=f"what {owner} was shown in {session}",
        timestamp=_TS,
        image_id="ab" * 32,  # the same file in every session
        name="IMG_1.jpg",
        session_id=session,
        parent_id=f"mc_{project}_{session}",
    )


async def test_session_delete_removes_its_picture_memories(
    stores: Stores, eraser: _RecordingEraser
) -> None:
    """A deleted session takes its picture memories with it, from the files
    and from the index; another session's memory of the same file stays."""
    for session in ("s1", "s2"):
        await _seed(stores, session)
        await stores.image_memory.append(
            _user("u1"), _picture("u1", session), only_if=NO_SOURCE_CHECK
        )

    result = await _delete(_deleter(stores, eraser), session_id="s1")

    assert result.count == 6  # u1: ep + fact + picture, u2: ep + fact, a1: case
    erased = {(s.owner_type, s.owner_id): ids for s, ids in eraser.erased}
    assert len(erased[("user", "u1")]["image_memory"]) == 1
    assert await stores.image_memory.entries_for_session(_user("u1"), "s1") == []
    assert len(await stores.image_memory.entries_for_session(_user("u1"), "s2")) == 1


async def test_retry_finds_an_owner_left_with_only_pictures(
    storage: Storage, stores: Stores, eraser: _RecordingEraser
) -> None:
    """A retry after the SQLite step: an owner whose only trace of the session
    is a picture memory is still found by the scan of the files."""
    await _seed(stores, "s1")
    await stores.image_memory.append(
        _user("u3"), _picture("u3", "s1"), only_if=NO_SOURCE_CHECK
    )
    await stores.session_purge.purge(
        SessionKey(app_id="a", project_id="p", session_id="s1")
    )

    result = await _delete(_deleter(stores, eraser), session_id="s1")

    assert result.count == 6
    assert await _entries(storage, "user", "u3") == 0


async def test_owners_found_by_scan_when_memcells_are_gone(
    storage: Storage, stores: Stores, eraser: _RecordingEraser
) -> None:
    """A retry after the SQLite step: no memcell left, the slow path finds
    the owners from the files."""
    await _seed(stores, "s1")
    await stores.session_purge.purge(
        SessionKey(app_id="a", project_id="p", session_id="s1")
    )

    result = await _delete(_deleter(stores, eraser), session_id="s1")

    assert result.count == 5


async def test_partial_failure_raises_deletion_incomplete(
    storage: Storage,
    stores: Stores,
    eraser: _RecordingEraser,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R30: a rewrite fails midway -> 500-class error; the same request
    finishes the job and the counts add up."""
    await _seed(stores, "s1")
    original = type(stores.atomic_fact).remove_session
    calls = {"n": 0}

    async def failing(self: Any, scope: Scope, session_id: str) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(28, "No space left on device")
        return await original(self, scope, session_id)

    monkeypatch.setattr(type(stores.atomic_fact), "remove_session", failing)
    d = _deleter(stores, eraser)

    with pytest.raises(DeletionIncompleteError):
        await _delete(d, session_id="s1")
    first_pass = await _entries(storage, "user", "u1")
    again = await _delete(d, session_id="s1")

    assert first_pass > 0
    assert again.count == 5 - 1  # u1's episode went in the failed pass
    for owner_type, owner in (("user", "u1"), ("user", "u2"), ("agent", "a1")):
        assert await _entries(storage, owner_type, owner) == 0


async def test_resend_count_zero(stores: Stores, eraser: _RecordingEraser) -> None:
    await _seed(stores, "s1")
    d = _deleter(stores, eraser)
    assert (await _delete(d, session_id="s1")).count == 5
    assert (await _delete(d, session_id="s1")).count == 0
    assert (await _delete(d, session_id="never")).count == 0
    assert (await _delete(d, user_id="nobody")).count == 0


async def test_session_delete_scoped_to_request_project(
    storage: Storage, stores: Stores, eraser: _RecordingEraser
) -> None:
    """RF2: the same session id in another project is not touched."""
    await _seed(stores, "s1", project="p")
    await _seed(stores, "s1", project="q")

    await _delete(_deleter(stores, eraser), session_id="s1")

    assert await _entries(storage, "user", "u1", project="q") == 2
    assert await stores.memcell.find_by_ids(["mc_q_s1"])


async def test_owner_delete_removes_clusters_reports_and_profile_rows(
    stores: Stores, eraser: _RecordingEraser
) -> None:
    """A14: a whole user takes its clusters, reports and profile index row."""
    await _seed(stores, "s1")
    from everos.memory.models import Cluster, ReflectionReport

    await stores.cluster.upsert(
        _user("u1"),
        "user_memory",
        Cluster(id="cl_1", centroid=(1.0,), count=1, last_ts=1, members=["e"]),
        member_type="episode",
    )
    await stores.reflection_report.create(
        ReflectionReport(
            id="r",
            cluster_id="cl_1",
            owner_id="u1",
            app_id="a",
            project_id="p",
            mode="init",
            source_members=[],
            merged_entry_id="m",
            deprecated_fact_count=0,
        )
    )

    await _delete(_deleter(stores, eraser), user_id="u1")

    assert await stores.cluster.list_for_owner(_user("u1"), "user_memory") == []
    assert await stores.reflection_report.reflected_cluster_ids(_user("u1")) == set()
    assert eraser.owners == [(_user("u1"), ["user_profile"])]
