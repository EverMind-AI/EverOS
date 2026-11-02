"""``POST /memory/delete`` through the app over real md, SQLite and LanceDB.

In-process ASGI (an integration test, not a real-server one): the request
goes through validation, the service and the stores; the assertions read
each layer that owns a fact -- the md files, the memcell table, the index
table -- and the user-facing ``/get``.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import textwrap
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from everalgo.types import ChatMessage, MemCell
from httpx import ASGITransport, AsyncClient

from everos.config import load_settings
from everos.core.utils.datetime import from_timestamp_ms
from everos.entrypoints.api.app import create_app
from everos.infra.persistence.lancedb import Episode as EpisodeRow
from everos.infra.persistence.markdown import MarkdownEpisodeStore, MarkdownReader
from everos.infra.persistence.sqlite import SqliteMemcellStore
from everos.memory.models import Episode, MemcellRecord
from everos.memory.ports import NO_SOURCE_CHECK, MemoryRef, Scope, SessionKey
from everos.service import build_application
from tests.fakes import make_providers
from tests.fakes.storage import StorageView

_TS = 1_747_216_800_000
_REPO = Path(__file__).resolve().parents[2]


def _scope(owner: str) -> Scope:
    return Scope(
        app_id="default", project_id="default", owner_id=owner, owner_type="user"
    )


@pytest.fixture
async def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage_view: StorageView
) -> AsyncIterator[AsyncClient]:
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    load_settings.cache_clear()
    await storage_view.create_schema()
    app = create_app(
        application=build_application(load_settings(), providers=make_providers())
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c
    load_settings.cache_clear()


async def _seed(view: StorageView, owner: str, session: str) -> MemoryRef:
    """One episode of ``owner`` in ``session``: md, memcell and index row."""
    (ref,) = await MarkdownEpisodeStore(view.root).append_many(
        _scope(owner),
        [
            Episode(
                owner_id=owner,
                episode=f"{owner} said something in {session}",
                timestamp=_TS,
                subject="s",
                summary="s",
                session_id=session,
                parent_id=f"mc_{session}",
            )
        ],
        only_if=NO_SOURCE_CHECK,
    )
    await SqliteMemcellStore(view.sqlite.memcell).insert_many(
        SessionKey(app_id="default", project_id="default", session_id=session),
        [
            MemcellRecord(
                memcell_id=f"mc_{session}_{owner}",
                cell=MemCell(
                    items=[
                        ChatMessage(
                            id="m",
                            role="user",
                            content="x",
                            timestamp=_TS,
                            sender_id=owner,
                        )
                    ],
                    timestamp=_TS,
                ),
                raw_type="conversation",
                message_ids=["m"],
                sender_ids=[owner],
            )
        ],
    )
    await view.repos.episode.upsert(
        [
            EpisodeRow(
                id=ref.id,
                entry_id=ref.entry_id,
                owner_id=owner,
                owner_type="user",
                session_id=session,
                timestamp=from_timestamp_ms(_TS),
                parent_id=f"mc_{session}",
                sender_ids=[owner],
                episode="body",
                episode_tokens="body",
                md_path=ref.md_path,
                content_sha256=hashlib.sha256(ref.id.encode()).hexdigest(),
                vector=[0.1] * load_settings().embedding.dim,
            )
        ]
    )
    return ref


async def _md_ids(view: StorageView, ref: MemoryRef) -> set[str]:
    path = view.root.root / ref.md_path
    if not path.exists():
        return set()
    return {e.id for e in (await MarkdownReader.read(path)).entries}


async def _index_ids(view: StorageView) -> set[str]:
    return {row.id for row in await view.repos.episode.scan()}


async def test_session_delete_reaches_md_sqlite_and_index(
    client: AsyncClient, storage_view: StorageView
) -> None:
    gone = await _seed(storage_view, "u1", "s1")
    kept = await _seed(storage_view, "u1", "s2")

    resp = await client.post("/api/v2/memory/delete", json={"session_id": "s1"})

    assert resp.status_code == 200, resp.text
    assert resp.json()["data"] == {"filters": ["session_id"], "count": 1}
    assert await _md_ids(storage_view, gone) == {kept.entry_id}
    assert await _index_ids(storage_view) == {kept.id}
    assert (
        await SqliteMemcellStore(storage_view.sqlite.memcell).find_by_ids(["mc_s1_u1"])
        == []
    )
    listed = await client.post(
        "/api/v2/memory/get", json={"user_id": "u1", "memory_type": "episode"}
    )
    assert [e["id"] for e in listed.json()["data"]["episodes"]] == [kept.id]


async def test_user_delete_empties_the_owner(
    client: AsyncClient, storage_view: StorageView
) -> None:
    await _seed(storage_view, "u1", "s1")
    other = await _seed(storage_view, "u2", "s1")

    resp = await client.post("/api/v1/memory/delete", json={"user_id": "u1"})

    assert resp.status_code == 200
    assert resp.json()["data"] == {"filters": ["user_id"], "count": 1}
    assert await _index_ids(storage_view) == {other.id}


async def test_unknown_scope_answers_zero(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v2/memory/delete", json={"user_id": "nobody", "session_id": "none"}
    )
    assert resp.json()["data"] == {"filters": ["user_id", "session_id"], "count": 0}


@pytest.mark.parametrize("user_id", ["a∕b", "a%2Fb", "..", "a/b"])
async def test_path_like_owner_is_refused_before_anything(
    client: AsyncClient, storage_view: StorageView, user_id: str
) -> None:
    """RF4 / C7: such an id never names a directory (the same rule /add's
    sender_id follows, so no such owner can exist)."""
    ref = await _seed(storage_view, "u1", "s1")

    resp = await client.post("/api/v2/memory/delete", json={"user_id": user_id})

    assert resp.status_code == 422
    assert await _md_ids(storage_view, ref) == {ref.entry_id}


async def test_failpoint_exits_137_and_the_resend_finishes(
    client: AsyncClient, storage_view: StorageView, tmp_path: Path
) -> None:
    """A20 / R15: a crash right after the SQLite step leaves md behind; the
    same request sent again removes it (owners found by scanning)."""
    ref = await _seed(storage_view, "u1", "s1")
    script = textwrap.dedent(
        """
        import asyncio
        from everos.config import load_settings
        from everos.service import build_application
        from tests.fakes import make_providers

        async def main():
            app = build_application(load_settings(), providers=make_providers())
            await app.memory.delete(
                app_id="default", project_id="default",
                user_id=None, agent_id=None, session_id="s1",
            )

        asyncio.run(main())
        """
    )
    env = os.environ | {
        "EVEROS_ROOT": str(tmp_path),
        "EVEROS_TEST_FAILPOINTS": "delete.after_sqlite",
        "PYTHONPATH": str(_REPO),
    }
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        env=env,
        cwd=_REPO,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await asyncio.wait_for(proc.communicate(), 120)

    assert proc.returncode == 137, stderr.decode()[-2000:]
    assert (
        await SqliteMemcellStore(storage_view.sqlite.memcell).find_by_ids(["mc_s1_u1"])
        == []
    )
    assert await _md_ids(storage_view, ref) == {ref.entry_id}  # not reached yet

    resp = await client.post("/api/v2/memory/delete", json={"session_id": "s1"})

    assert resp.json()["data"]["count"] == 1
    assert await _md_ids(storage_view, ref) == set()
    assert await _index_ids(storage_view) == set()
