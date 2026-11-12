"""Update, history, forget and the delete aliases through the app (docs/api.md).

In-process ASGI over real markdown, SQLite and LanceDB (an integration test,
not a real-server one): the request goes through validation, the service and
the stores; the assertions read each layer that owns a fact -- the md files
and their marks, the memcell table, the index table -- and the user-facing
``/get`` and ``/history``. What these pin:

- ``/memory/update`` writes a new entry and marks the old one; ``/history``
  lists both, current first; a stale version is 409 ``VERSION_CONFLICT``;
- ``/memory/forget`` is a dry run by default (nothing changes), a real run
  marks and erases the index rows, and ``/forget/undo`` with its id clears
  the marks;
- ``/memory/delete`` and ``/delete_by_ids`` forget: the entries stay in
  markdown, marked, the memcells stay, the rows leave the index and ``/get``,
  and the answer carries the forgets' ids; an ``agent_id`` is 422: 2.0 has
  no agent memory.
"""

from __future__ import annotations

import hashlib
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
from everos.service import NO_AGENT_MEMORY, build_application
from tests.fakes import make_providers
from tests.fakes.storage import StorageView

_TS = 1_747_216_800_000


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


async def _seed(view: StorageView, owner: str, session: str, text: str) -> MemoryRef:
    """One episode of ``owner`` in ``session``: md, memcell and index row."""
    (ref,) = await MarkdownEpisodeStore(view.root).append_many(
        _scope(owner),
        [
            Episode(
                owner_id=owner,
                episode=text,
                timestamp=_TS,
                subject="s",
                summary="s",
                session_id=session,
                parent_id=f"mc_{session}_{owner}",
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
                            content=text,
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
                parent_id=f"mc_{session}_{owner}",
                sender_ids=[owner],
                episode=text,
                episode_tokens=text,
                md_path=ref.md_path,
                content_sha256=hashlib.sha256(ref.id.encode()).hexdigest(),
                vector=[0.1] * load_settings().embedding.dim,
            )
        ]
    )
    return ref


async def _marks(view: StorageView, ref: MemoryRef) -> dict[str, str]:
    parsed = await MarkdownReader.read(view.root.root / ref.md_path)
    marks = parsed.frontmatter.get("deprecated_entries") or {}
    return {str(k): str(v) for k, v in marks.items()}


async def _entry_ids(view: StorageView, ref: MemoryRef) -> set[str]:
    return {
        e.id for e in (await MarkdownReader.read(view.root.root / ref.md_path)).entries
    }


async def _index_ids(view: StorageView) -> set[str]:
    return {row.id for row in await view.repos.episode.scan()}


async def test_update_writes_a_new_entry_and_history_lists_both(
    client: AsyncClient, storage_view: StorageView
) -> None:
    old = await _seed(storage_view, "u1", "s1", "the meeting is at nine")

    resp = await client.post(
        "/api/v2/memory/update",
        json={"user_id": "u1", "id": old.id, "content": "the meeting is at ten"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert data["supersedes"] == old.id and data["held"] is True
    new_id = data["id"]
    assert (await _marks(storage_view, old))[old.entry_id] == new_id.removeprefix("u1_")

    history = await client.get(
        "/api/v2/memory/history", params={"user_id": "u1", "id": old.id}
    )
    assert history.status_code == 200, history.text
    versions = history.json()["data"]["versions"]
    assert [(v["id"], v["state"]) for v in versions] == [
        (new_id, "current"),
        (old.id, "superseded"),
    ]
    assert versions[0]["content"]["content"] == "the meeting is at ten"

    stale = await client.post(
        "/api/v2/memory/update",
        json={"user_id": "u1", "id": old.id, "content": "the meeting is at eleven"},
    )
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "VERSION_CONFLICT"
    assert new_id in stale.json()["error"]["message"]


async def test_forget_is_a_dry_run_by_default_and_undo_takes_its_id(
    client: AsyncClient, storage_view: StorageView
) -> None:
    ref = await _seed(storage_view, "u1", "s1", "my bank pin is 1234")
    body = {"user_id": "u1", "target": {"kind": "record", "record_ids": [ref.id]}}

    plan = await client.post("/api/v2/memory/forget", json=body)
    assert plan.status_code == 200, plan.text
    assert plan.json()["data"]["dry_run"] is True
    assert plan.json()["data"]["episodes"] == [ref.id]
    assert await _marks(storage_view, ref) == {}
    assert await _index_ids(storage_view) == {ref.id}

    done = await client.post("/api/v2/memory/forget", json={**body, "dry_run": False})
    forget_id = done.json()["data"]["forget_id"]
    assert forget_id.startswith("fg_")
    assert await _marks(storage_view, ref) == {ref.entry_id: "forgotten"}
    assert await _entry_ids(storage_view, ref) == {ref.entry_id}  # still in md
    assert await _index_ids(storage_view) == set()

    undone = await client.post(
        "/api/v2/memory/forget/undo", json={"user_id": "u1", "forget_id": forget_id}
    )
    assert undone.status_code == 200, undone.text
    assert undone.json()["data"]["applied"] is True
    assert await _marks(storage_view, ref) == {}


async def test_a_session_delete_is_a_forget_of_every_user(
    client: AsyncClient, storage_view: StorageView
) -> None:
    gone = await _seed(storage_view, "u1", "s1", "u1 said something in s1")
    also = await _seed(storage_view, "u2", "s1", "u2 said something in s1")
    kept = await _seed(storage_view, "u1", "s2", "u1 said something in s2")

    resp = await client.post("/api/v2/memory/delete", json={"session_id": "s1"})

    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert data["filters"] == ["session_id"] and data["count"] == 2
    assert len(data["forget_ids"]) == 2
    assert await _marks(storage_view, gone) == {gone.entry_id: "forgotten"}
    assert await _marks(storage_view, also) == {also.entry_id: "forgotten"}
    assert await _index_ids(storage_view) == {kept.id}
    # The conversation's segments stay: a forget leaves system.db as it was.
    memcells = SqliteMemcellStore(storage_view.sqlite.memcell)
    assert await memcells.find_by_ids(["mc_s1_u1"]) != []
    listed = await client.post(
        "/api/v2/memory/get", json={"user_id": "u1", "memory_type": "episode"}
    )
    assert [e["id"] for e in listed.json()["data"]["episodes"]] == [kept.id]


async def test_a_user_delete_forgets_the_owner_only(
    client: AsyncClient, storage_view: StorageView
) -> None:
    await _seed(storage_view, "u1", "s1", "u1 in s1")
    other = await _seed(storage_view, "u2", "s1", "u2 in s1")

    resp = await client.post("/api/v1/memory/delete", json={"user_id": "u1"})

    assert resp.status_code == 200
    assert resp.json()["data"]["count"] == 1
    assert await _index_ids(storage_view) == {other.id}


async def test_an_agent_delete_is_refused(client: AsyncClient) -> None:
    resp = await client.post("/api/v2/memory/delete", json={"agent_id": "bot"})
    assert resp.status_code == 422
    assert NO_AGENT_MEMORY in resp.json()["error"]["message"]


@pytest.mark.parametrize("user_id", ["a∕b", "a%2Fb", "..", "a/b"])
async def test_path_like_owner_is_refused_before_anything(
    client: AsyncClient, storage_view: StorageView, user_id: str
) -> None:
    ref = await _seed(storage_view, "u1", "s1", "x")

    resp = await client.post("/api/v2/memory/delete", json={"user_id": user_id})

    assert resp.status_code == 422
    assert await _marks(storage_view, ref) == {}


async def test_delete_by_ids_forgets_and_skips_what_names_nothing(
    client: AsyncClient, storage_view: StorageView
) -> None:
    ref = await _seed(storage_view, "u1", "s1", "x")

    resp = await client.post(
        "/api/v2/memory/delete_by_ids",
        json={
            "memory_type": "episode",
            "memory_ids": [ref.id, "u1_ep_20990101_0001"],
        },
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert data["success"] is True and len(data["forget_ids"]) == 1
    assert await _marks(storage_view, ref) == {ref.entry_id: "forgotten"}
    assert await _index_ids(storage_view) == set()
