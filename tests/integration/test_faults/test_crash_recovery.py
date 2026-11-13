"""A server process killed in the middle of a write, then started again.

Each test runs a real server process of its own (``tests.helpers.
fault_server``) over a temporary root, makes it stop at a named point inside a
write, kills it there with SIGKILL through the test's own process handle (no
shutdown hook runs), starts a fresh process over the same root and checks what
recovery must give: the cascade drains (nothing pending, nothing failed),
markdown and the index agree, and the memory is whole -- each marker live
exactly once, or forgotten where a forget said so.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests._consistency_assertions import assert_md_lance_strict_consistent
from tests.fakes.storage import open_storage_view
from tests.helpers.faults import (
    BASE_ENV,
    ChildServer,
    assert_exactly_once,
    line,
    live_episode_markers,
    markers,
    staging_files,
)

pytestmark = pytest.mark.slow


@pytest.fixture
def servers() -> Iterator[list[ChildServer]]:
    """Every server process a test starts; each is killed at the end."""
    started: list[ChildServer] = []
    yield started
    for server in started:
        server.kill()


async def _start(servers: list[ChildServer], root: Path, **kw: Any) -> ChildServer:
    server = ChildServer.start(root, **kw)
    servers.append(server)
    await server.ready()
    return server


async def _drained(client: httpx.AsyncClient, seconds: float = 30.0) -> dict[str, Any]:
    import asyncio

    async with asyncio.timeout(seconds):
        zero = 0
        while zero < 3:
            cascade = (await client.get("/health")).json()["cascade"]
            zero = zero + 1 if cascade["pending"] == 0 else 0
            await asyncio.sleep(0.3)
    return cascade


async def _consistent(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in BASE_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("EVEROS_ROOT", str(root))
    from everos.config import load_settings

    load_settings.cache_clear()
    view = open_storage_view(root)
    try:
        await assert_md_lance_strict_consistent(root, view.repos)
    finally:
        await view.aclose()


def _batch(
    tag: str, count: int = 3, at: int = 0
) -> tuple[list[dict[str, Any]], list[str]]:
    sent = [f"ZQ{tag}{i:02d}" for i in range(count)]
    return [line("u1", f"note {m}", at + i * 1000) for i, m in enumerate(sent)], sent


async def _add(
    client: httpx.AsyncClient, session: str, batch: list[dict[str, Any]]
) -> None:
    resp = await client.post(
        "/api/v1/memory/add", json={"session_id": session, "messages": batch}
    )
    assert resp.status_code == 200, resp.text


async def test_killed_between_the_markdown_write_and_the_index_write(
    tmp_path: Path, servers: list[ChildServer], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    first = await _start(servers, root, pause_at="index_write")
    batch, sent = _batch("I")
    async with first.client() as client:
        await _add(client, "s1", batch)
        resp = await client.post("/api/v1/memory/flush", json={"session_id": "s1"})
        assert resp.status_code == 200, resp.text
        await first.paused()
    first.kill()
    assert_exactly_once(live_episode_markers(root, "u1"), sent)  # markdown has it

    second = await _start(servers, root)
    async with second.client() as client:
        cascade = await _drained(client)
        assert cascade["failed_permanent"] == cascade["failed_retryable"] == 0
        found = await client.post(
            "/api/v1/memory/search", json={"query": "notes", "user_id": "u1"}
        )
        assert set(sent) <= set(markers(found.text)), found.text
    second.stop()
    await _consistent(root, monkeypatch)
    assert staging_files(root) == []


async def test_killed_between_the_memcell_and_the_episode(
    tmp_path: Path, servers: list[ChildServer]
) -> None:
    root = tmp_path / "root"
    first = await _start(servers, root, pause_at="episode_write")
    batch, sent = _batch("E")
    async with first.client() as client:
        await _add(client, "s1", batch)
        flush = client.post("/api/v1/memory/flush", json={"session_id": "s1"})
        import asyncio

        task = asyncio.ensure_future(flush)
        await first.paused()
        first.kill()
        with pytest.raises(httpx.TransportError):
            await task
    second = await _start(servers, root)
    async with second.client() as client:
        resp = await client.post("/api/v1/memory/flush", json={"session_id": "s1"})
        assert resp.status_code == 200, resp.text
        await _drained(client)
    second.stop()
    assert_exactly_once(live_episode_markers(root, "u1"), sent)


async def _seed_two_sessions(client: httpx.AsyncClient) -> tuple[list[str], list[str]]:
    one, sent_one = _batch("A")
    two, sent_two = _batch("B", at=60_000)
    await _add(client, "s1", one)
    assert (
        await client.post("/api/v1/memory/flush", json={"session_id": "s1"})
    ).status_code == 200
    await _add(client, "s2", two)
    assert (
        await client.post("/api/v1/memory/flush", json={"session_id": "s2"})
    ).status_code == 200
    await _drained(client)
    return sent_one, sent_two


@pytest.mark.xfail(
    strict=True,
    reason=(
        "a forget killed after its marks and before its log entry leaves those "
        "entries forgotten under no forget id: the forget sent again logs them as "
        "nothing to do, and no undo brings them back "
        "(service/memory/_forget.py: the log is written last, no intent record)"
    ),
)
async def test_killed_in_the_middle_of_a_forget(
    tmp_path: Path, servers: list[ChildServer], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A forget that died part-way: the same forget sent again finishes it, and
    its undo brings back everything the two runs took."""
    import asyncio

    root = tmp_path / "root"
    seed = await _start(servers, root)
    async with seed.client() as client:
        sent_one, sent_two = await _seed_two_sessions(client)
    seed.stop()

    body = {
        "user_id": "u1",
        "target": {"kind": "session", "session_id": "s1"},
        "dry_run": False,
    }
    first = await _start(servers, root, pause_at="forget_log")
    async with first.client() as client:
        task = asyncio.ensure_future(client.post("/api/v1/memory/forget", json=body))
        await first.paused()
        first.kill()
        with pytest.raises(httpx.TransportError):
            await task

    second = await _start(servers, root)
    async with second.client() as client:
        await _drained(client)
        again = await client.post("/api/v1/memory/forget", json=body)
        assert again.status_code == 200, again.text
        await _drained(client)
        live = live_episode_markers(root, "u1")
        assert not set(sent_one) & set(live), live
        assert_exactly_once({m: n for m, n in live.items() if m in sent_two}, sent_two)
        undo = await client.post(
            "/api/v1/memory/forget/undo",
            json={"user_id": "u1", "forget_id": again.json()["data"]["forget_id"]},
        )
        assert undo.status_code == 200, undo.text
        await _drained(client)
    second.stop()
    assert_exactly_once(live_episode_markers(root, "u1"), [*sent_one, *sent_two])
    await _consistent(root, monkeypatch)


async def test_killed_in_the_middle_of_an_undo(
    tmp_path: Path, servers: list[ChildServer], monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    root = tmp_path / "root"
    seed = await _start(servers, root)
    async with seed.client() as client:
        sent_one, sent_two = await _seed_two_sessions(client)
        done = await client.post(
            "/api/v1/memory/forget",
            json={
                "user_id": "u1",
                "target": {"kind": "session", "session_id": "s1"},
                "dry_run": False,
            },
        )
        assert done.status_code == 200, done.text
        await _drained(client)
    seed.stop()
    undo = {"user_id": "u1", "forget_id": done.json()["data"]["forget_id"]}

    first = await _start(servers, root, pause_at="undo_requeue")
    async with first.client() as client:
        task = asyncio.ensure_future(
            client.post("/api/v1/memory/forget/undo", json=undo)
        )
        await first.paused()
        first.kill()
        with pytest.raises(httpx.TransportError):
            await task

    second = await _start(servers, root)
    async with second.client() as client:
        await _drained(client)
        again = await client.post("/api/v1/memory/forget/undo", json=undo)
        assert again.status_code == 200, again.text
        await _drained(client)
        found = await client.post(
            "/api/v1/memory/search",
            json={"query": "notes", "user_id": "u1", "top_k": 50},
        )
        assert set(sent_one) <= set(markers(found.text)), found.text
    second.stop()
    assert_exactly_once(live_episode_markers(root, "u1"), [*sent_one, *sent_two])
    await _consistent(root, monkeypatch)
