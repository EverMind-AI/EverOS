"""Upgrade path: Tier 1 -> Tier 2 via ``everos cascade backfill``.

Simulates a real user's journey: start with only an LLM configured, add
memories (written with ``vector=NULL``), then add embedding credentials
and run the vectors backfill phase -- proving the previously-written
rows become semantically searchable without having to re-add them.

Two sequential app lifespans against the *same* ``tmp_path`` /
``EVEROS_ROOT`` stand in for "restart the service": a live process
restart isn't reachable from a test, but exiting the first
``app.router.lifespan_context`` disposes the SQLite engine + LanceDB
connection (its ``shutdown()`` hooks) exactly as a real process exit
would, and re-entering a second one against the same on-disk root is
the closest in-process equivalent to a real restart with new config.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from everos.entrypoints.cli.commands._backfill import run_backfill
from tests.fakes import FakeOmeLock, make_backfill_deps, make_ops

from .conftest import (
    StubEmbedder,
    _tier_client,
    add_payload,
    cascade_progress,
    episode_rows,
)

_N_ITEMS = 4


async def _wait_for_episode_rows(
    owner_id: str, expected: int, *, deadline_seconds: float
) -> list[dict]:
    """Wait for every logical entry in a debounced daily-log update."""
    async with asyncio.timeout(deadline_seconds):
        while True:
            rows = await episode_rows(owner_id)
            if len(rows) >= expected:
                return rows
            await asyncio.sleep(0.1)


async def test_tier1_to_tier2_upgrade_via_backfill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ── Phase A: Tier 1 (LLM only) ──────────────────────────────────────
    async with _tier_client(
        tmp_path, monkeypatch, embed_available=False, rerank_available=False
    ) as client:
        # All N adds are issued before the single wait below (rather than
        # one add-wait-add-wait cycle) so the cascade watcher's debounce
        # window is only paid once for the whole batch instead of once
        # per item -- waiting on every single item independently starved
        # later items past their individual deadlines in practice.
        async with cascade_progress() as wait_drained:
            for i in range(_N_ITEMS):
                session_id = f"s_upgrade_{i}"
                resp = await client.post(
                    "/api/v1/memory/add",
                    json=add_payload(
                        session_id=session_id,
                        content=f"Memory item {i}: Alice went hiking on trail {i}.",
                    ),
                )
                assert resp.status_code == 200, resp.text
                flush_resp = await client.post(
                    "/api/v1/memory/flush", json={"session_id": session_id}
                )
                assert flush_resp.status_code == 200, flush_resp.text
                assert flush_resp.json()["data"]["status"] == "extracted"

            # NB: the cascade queue's unit of work is one *file* change
            # event, not one logical episode -- all N adds land in the
            # same daily md file, so a single drain (min_processed=1)
            # covers the whole batch. The per-episode count is verified
            # below directly against the LanceDB table.
            await wait_drained(deadline_seconds=40.0)

        # The queue drains file-level work, but another append to the same
        # daily log can arrive just after that drain. Wait for the logical
        # entry count as the end-to-end completion condition.
        rows = await _wait_for_episode_rows("u_alice", _N_ITEMS, deadline_seconds=40.0)
        assert len(rows) == _N_ITEMS
        assert all(r["vector"] is None for r in rows), (
            "Tier 1 (no embed) must write every episode with vector=NULL"
        )

        health = await client.get("/health")
        assert health.json()["capabilities"]["embed"] is False

    # Lifespan shutdown above disposed the SQLite engine + LanceDB
    # connection -- the closest in-process stand-in for "stop the
    # service". Phase B reopens both from scratch against the same
    # on-disk root, now with embed configured.

    # ── Phase B: Tier 2 (embed added) ────────────────────────────────────
    async with _tier_client(
        tmp_path, monkeypatch, embed_available=True, rerank_available=False
    ) as client:
        health = await client.get("/health")
        assert health.status_code == 200
        assert health.json()["capabilities"]["embed"] is True, (
            "restarted service must now report embed available"
        )

        pre_episode_rows = await episode_rows("u_alice")
        assert all(r["vector"] is None for r in pre_episode_rows), (
            "rows written before the upgrade must still be NULL pre-backfill"
        )

        # ``run_backfill`` refuses to run against a live OME lock (Phase
        # 1 / 2 / 3 all preflight it — round-4 review J10 aligned Phase
        # 1 with the other two). In-process this test still holds the
        # server lifespan open around the call, so backfill runs over
        # the production wiring with a lock that reads free, simulating
        # the intended "server stopped, then backfill runs" sequence. A
        # real user's flow does stop the server; the probe enforces it.
        ops = make_ops(
            backfill=make_backfill_deps(
                root=tmp_path, embedding=StubEmbedder(), ome_lock=FakeOmeLock()
            )
        )
        exit_code = await run_backfill(ops, phase="vectors", auto_yes=True)
        assert exit_code == 0

        post_episode_rows = await episode_rows("u_alice")
        assert len(post_episode_rows) == _N_ITEMS
        assert all(r["vector"] is not None for r in post_episode_rows), (
            "every pre-upgrade episode must carry a vector after backfill"
        )
        assert all(len(r["vector"]) == 1024 for r in post_episode_rows)

        # The upgraded deployment searches: multi-round retrieval needs the
        # embedder Tier 1 lacked, and recalls the backfilled episodes by
        # BM25 and by the vectors asserted directly above.
        search_resp = await client.post(
            "/api/v1/memory/search",
            json={"user_id": "u_alice", "query": "hiking", "method": "llm_multiround"},
        )
        assert search_resp.status_code == 200, search_resp.text
        episodes = search_resp.json()["data"]["episodes"]
        assert episodes, "search must find the backfilled pre-upgrade episodes"
