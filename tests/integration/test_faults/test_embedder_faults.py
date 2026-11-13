"""The embedder down or slow, at add, flush and search.

Markdown is the truth and the index is projected from it, so an embedder that
fails must not cost a write: the add and the flush succeed, the cascade keeps
the projection for later (a retryable failure the health block shows) and
catches up once the embedder answers again. A search that needs a query
vector is answered as an error, not as an empty success, and the next one
works.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from tests.helpers.faults import (
    FaultServer,
    assert_exactly_once,
    markers,
    wait_until,
)


async def test_an_embedder_down_at_flush_costs_no_write_and_the_index_catches_up(
    tmp_path: Path,
) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        srv.embedder.down = True
        resp, sent = await srv.say("s1", "u1", "E")
        assert resp.status_code == 200, resp.text
        flushed = await srv.flush("s1")
        assert flushed.status_code == 200, flushed.text
        assert_exactly_once(srv.episodes("u1"), sent)  # markdown has it all

        async def reported() -> bool:
            cascade = (await srv.health())["cascade"]
            return srv.embedder.failures > 0 and (
                cascade["failed_retryable"] > 0 or cascade["pending"] > 0
            )

        await wait_until(reported)
        cascade = (await srv.health())["cascade"]
        assert cascade["failed_permanent"] == 0, cascade
        srv.embedder.down = False
        await srv.settle()
        assert await srv.searchable("u1") >= set(sent)
        await srv.consistent()
        assert srv.staging() == []


async def test_a_slow_embedder_at_flush(tmp_path: Path) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        srv.embedder.delay = 0.5
        resp, sent = await srv.say("s1", "u1", "E")
        assert resp.status_code == 200, resp.text
        assert (await srv.flush("s1")).status_code == 200
        await srv.settle()
        srv.embedder.delay = 0.0
        assert_exactly_once(srv.episodes("u1"), sent)
        assert await srv.searchable("u1") >= set(sent)
        await srv.consistent()


async def test_an_embedder_down_at_search_is_an_error_and_the_next_search_works(
    tmp_path: Path,
) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        _resp, sent = await srv.say("s1", "u1", "E")
        assert (await srv.flush("s1")).status_code == 200
        await srv.settle()
        srv.embedder.down = True
        failed = await srv.search("notes", user_id="u1")
        assert failed.status_code == 503, failed.text
        assert failed.json()["error"]["code"] == "EXTERNAL_SERVICE_UNAVAILABLE"
        srv.embedder.down = False
        assert await srv.searchable("u1") >= set(sent)


async def test_a_slow_embedder_at_search(tmp_path: Path) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        _resp, sent = await srv.say("s1", "u1", "E")
        assert (await srv.flush("s1")).status_code == 200
        await srv.settle()
        srv.embedder.delay = 1.0
        searches = await asyncio.gather(
            *(srv.search("notes", user_id="u1") for _ in range(4))
        )
        assert [r.status_code for r in searches] == [200] * 4
        assert all(set(sent) <= set(markers(r.text)) for r in searches)
