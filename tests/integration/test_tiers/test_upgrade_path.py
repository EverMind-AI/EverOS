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

import hashlib
from pathlib import Path

import pytest

from everos.component.utils.datetime import get_utc_now
from everos.entrypoints.cli.commands._backfill_cmd import run_backfill
from everos.infra.persistence.lancedb import (
    AtomicFact,
    Episode,
    atomic_fact_repo,
    get_table,
)
from everos.memory.cascade import _backfill as _backfill_mod

from .conftest import _tier_client, add_payload, cascade_progress

_N_ITEMS = 4


async def _episode_rows(owner_id: str) -> list[dict]:
    table = await get_table(Episode.TABLE_NAME, Episode)
    return await table.query().where(f"owner_id = '{owner_id}'").to_list()


async def _atomic_fact_rows(owner_id: str) -> list[dict]:
    table = await get_table(AtomicFact.TABLE_NAME, AtomicFact)
    return await table.query().where(f"owner_id = '{owner_id}'").to_list()


async def _seed_unvectorized_fact(episode_row: dict) -> None:
    """Seed one ``vector=NULL`` atomic fact for an episode.

    Real Tier 1 usage never populates ``atomic_fact`` (this suite's fake
    LLM only satisfies the boundary/episode JSON contract, not
    ``AtomicFactExtractor``'s -- see ``seed_atomic_fact_for_episode`` in
    conftest.py for the full rationale). Seeding an unvectorized row
    directly exercises Phase 1's *other* re-embedded table and is what
    makes the post-upgrade VECTOR search assertion meaningful: that
    method recalls via atomic_fact MaxSim, not a direct episode scan.
    """
    entry_id = f"af_seed_{episode_row['entry_id']}"
    owner_id = episode_row["owner_id"]
    await atomic_fact_repo.add(
        [
            AtomicFact(
                id=f"{owner_id}_{entry_id}",
                entry_id=entry_id,
                owner_id=owner_id,
                owner_type=episode_row["owner_type"],
                app_id=episode_row["app_id"],
                project_id=episode_row["project_id"],
                session_id=episode_row.get("session_id"),
                timestamp=get_utc_now(),
                parent_id=episode_row["entry_id"],
                sender_ids=episode_row["sender_ids"],
                fact="Alice enjoys hiking in the mountains.",
                fact_tokens="alice enjoys hiking in the mountains",
                md_path=f"users/{owner_id}/.atomic_facts/atomic_fact-seed.md",
                content_sha256=hashlib.sha256(entry_id.encode()).hexdigest(),
                vector=None,
            )
        ]
    )


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

        rows = await _episode_rows("u_alice")
        assert len(rows) == _N_ITEMS
        assert all(r["vector"] is None for r in rows), (
            "Tier 1 (no embed) must write every episode with vector=NULL"
        )

        for row in rows:
            await _seed_unvectorized_fact(row)

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

        pre_episode_rows = await _episode_rows("u_alice")
        assert all(r["vector"] is None for r in pre_episode_rows), (
            "rows written before the upgrade must still be NULL pre-backfill"
        )

        # ``run_backfill`` refuses to run against a live OME lock (Phase
        # 1 / 2 / 3 all preflight it — round-4 review J10 aligned Phase
        # 1 with the other two). In-process this test still holds the
        # server lifespan open around the call, so bypass the probe
        # locally to simulate the intended "server stopped, then
        # backfill runs" sequence. A real user's flow does stop the
        # server; the probe is the mechanism that enforces it.
        monkeypatch.setattr(_backfill_mod, "_probe_ome_lock_available", lambda: True)
        exit_code = await run_backfill(phase="vectors", auto_yes=True)
        assert exit_code == 0

        post_episode_rows = await _episode_rows("u_alice")
        assert len(post_episode_rows) == _N_ITEMS
        assert all(r["vector"] is not None for r in post_episode_rows), (
            "every pre-upgrade episode must carry a vector after backfill"
        )
        assert all(len(r["vector"]) == 1024 for r in post_episode_rows)

        post_fact_rows = await _atomic_fact_rows("u_alice")
        assert len(post_fact_rows) == _N_ITEMS
        assert all(r["vector"] is not None for r in post_fact_rows), (
            "atomic_fact rows are also re-embedded by the vectors phase"
        )

        # Proof the backfilled rows are semantically searchable: VECTOR
        # recalls via atomic_fact MaxSim (see seed_atomic_fact_for_episode
        # in conftest.py), so this only succeeds if the backfill actually
        # embedded the pre-upgrade rows rather than just the schema.
        search_resp = await client.post(
            "/api/v1/memory/search",
            json={"user_id": "u_alice", "query": "hiking", "method": "vector"},
        )
        assert search_resp.status_code == 200, search_resp.text
        episodes = search_resp.json()["data"]["episodes"]
        assert episodes, "VECTOR search must find the backfilled pre-upgrade episodes"
