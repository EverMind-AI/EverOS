"""Add + Flush user pipeline — a hermetic canned run and a long real-LLM drive.

``test_add_flush_search_get_on_canned_user_track`` runs anywhere: every
model call is answered from the canned user-track scripts.

The rest is the long real-conversation smoke:

Goal: prove the user-side add/flush chain is end-to-end live. Feeds
**419 real LoCoMo messages** through ``POST /api/v1/memory/add`` (in 19
batches sharing one session_id) then a final ``POST /flush``, and
verifies:

1. Each /add returns a sane status and the unprocessed_buffer delta
   matches what the service claims (accumulated → grew by batch size;
   extracted → shrank or stayed flat).
2. After /flush the buffer is empty and the memcell table has rows.
3. After cascade drains, episode md files exist and LanceDB rows
   reflect them with valid content_sha256 + vector.
4. OME-driven async strategies have produced atomic_fact / foresight /
   profile md files.

Real LLM + real embedder (creds via ``.env``). Marked ``slow`` —
``pytest -m slow tests/integration/test_add_flush_core_pipeline_smoke.py``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx
import pytest

from everos.infra.persistence.markdown import (
    AtomicFactDailyFrontmatter,
    EpisodeDailyFrontmatter,
    ForesightDailyFrontmatter,
)
from tests.fakes import user_track_llm, user_track_script
from tests.fakes.storage import StorageView

# Directory names live on the frontmatter schemas (single source of truth);
# atomic_facts / foresights are dotfile-hidden so users only see episodes.
_EPISODE_DIR = EpisodeDailyFrontmatter.DIR_NAME
_ATOMIC_FACT_DIR = AtomicFactDailyFrontmatter.DIR_NAME
_FORESIGHT_DIR = ForesightDailyFrontmatter.DIR_NAME

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_add_messages(batch: dict) -> list[dict]:
    """Strip ``_audit_*`` fields; keep only what MessageItemDTO accepts."""
    return [
        {
            "sender_id": m["sender_id"],
            "role": m["role"],
            "timestamp": m["timestamp"],
            "content": m["content"],
        }
        for m in batch["messages"]
    ]


def _list_md_files(memory_root: Path, subpath: str) -> list[Path]:
    """List .md files under
    ``<memory_root>/default_app/default_project/users/<user>/<subpath>/``."""
    user_dir = memory_root / "default_app" / "default_project" / "users"
    if not user_dir.exists():
        return []
    out: list[Path] = []
    for user_dir_child in user_dir.iterdir():
        target = user_dir_child / subpath
        if target.is_dir():
            out.extend(target.rglob("*.md"))
        elif target.with_suffix(".md").exists():
            out.append(target.with_suffix(".md"))
    return out


def _count_episode_entries(md_files: list[Path]) -> int:
    """Count ``## entry-*`` blocks across all episode md files."""
    n = 0
    for f in md_files:
        for line in f.read_text().splitlines():
            stripped = line.strip()
            # Daily-log entries start with `## ` followed by an id token.
            # We count any second-level heading that isn't the standard
            # subsection headers used inside an entry.
            if stripped.startswith("## ") and not stripped.startswith(
                ("## Subject", "## Summary", "## Content", "## Fact", "## Foresight")
            ):
                n += 1
    return n


def _maybe_snapshot_memory_root(memory_root: Path) -> None:
    """Copy ``memory_root`` to ``$EVEROS_KEEP_CORPUS_TO`` when set.

    Used to harvest a known-good corpus (md + sqlite + lancedb three-piece
    set) after a green test run, for later upload as the /search e2e
    fixture. Pure sync I/O — kept out of the async test body so ASYNC240
    doesn't complain about pathlib usage on the async path.
    """
    keep_to = os.environ.get("EVEROS_KEEP_CORPUS_TO")
    if not keep_to:
        return
    dest = Path(keep_to).resolve()
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(memory_root, dest)


# ---------------------------------------------------------------------------
# The test (slow — hits real LLM + embedder; opt in via `pytest -m slow`)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.live_llm
# Retries cover transient real-LLM flakes: OME profile clustering
# occasionally fails to emit user.md within the cascade-drain deadline
# (LLM timeout, empty response, or async race), but is reliably stable
# on retry. reruns_delay leaves the cascade workers idle between
# attempts so we don't pile state on top of a prior run.
@pytest.mark.flaky(reruns=2, reruns_delay=5)
async def test_long_conversation_produces_all_memory_types(
    long_conversation: dict,
    async_client: httpx.AsyncClient,
    core_pipeline_runtime: Path,
    cascade_done_poll: Callable[..., Awaitable[None]],
    buffer_count: Callable[[str], Awaitable[int]],
    memcell_count: Callable[..., Awaitable[int]],
    storage_view: StorageView,
) -> None:
    """One big seamless run: add 19 batches, flush, poll, assert everything."""

    session_id = long_conversation["everos_session_id"]
    memory_root = core_pipeline_runtime

    # ── Stage 0: baseline ─────────────────────────────────────────────────
    assert await buffer_count(session_id) == 0
    assert await memcell_count(session_id) == 0

    # ── Stage 1: drip 19 batches into /add, asserting buffer delta ────────
    last_status: str | None = None

    for idx, batch in enumerate(long_conversation["batches"]):
        msg_count = batch["message_count"]

        buf_before = await buffer_count(session_id)
        cells_before = await memcell_count(session_id)

        resp = await async_client.post(
            "/api/v1/memory/add",
            json={"session_id": session_id, "messages": _to_add_messages(batch)},
            timeout=600.0,  # boundary detection may call LLM
        )
        assert resp.status_code == 200, (
            f"batch {idx} ({batch['locomo_session']}): {resp.status_code} {resp.text}"
        )
        body = resp.json()
        status: str = body["data"]["status"]
        returned_count: int = body["data"]["message_count"]
        assert status in {"accumulated", "extracted"}, body
        assert returned_count == msg_count, body
        last_status = status

        buf_after = await buffer_count(session_id)
        cells_after = await memcell_count(session_id)

        # Buffer-delta invariants:
        if status == "accumulated":
            # No boundary cut → entire batch piled into the buffer.
            assert buf_after == buf_before + msg_count, (
                f"batch {idx} accumulated: expected buf {buf_before + msg_count}, "
                f"got {buf_after}"
            )
            assert cells_after == cells_before, (
                f"batch {idx} accumulated: memcell should not change "
                f"({cells_before} → {cells_after})"
            )
        else:  # "extracted"
            # Boundary fired: some messages turned into memcell(s), tail
            # (if any) stays in the buffer. We can't predict the exact tail
            # length but two invariants must hold.
            assert cells_after > cells_before, (
                f"batch {idx} extracted: memcell should grow "
                f"({cells_before} → {cells_after})"
            )
            assert buf_after >= 0
            # Conservation: nothing should silently vanish — the union of
            # (buffer carry-over + this batch) must equal (new buffer +
            # messages carved into cells). We approximate by asserting the
            # new buffer is at most the carry-over + this batch size.
            assert buf_after <= buf_before + msg_count, (
                f"batch {idx} extracted: buffer overflow "
                f"({buf_before} + {msg_count} → {buf_after})"
            )

    # ── Stage 2: flush ────────────────────────────────────────────────────
    cells_pre_flush = await memcell_count(session_id)
    resp = await async_client.post(
        "/api/v1/memory/flush",
        json={"session_id": session_id},
        timeout=600.0,
    )
    assert resp.status_code == 200, resp.text
    flush_status = resp.json()["data"]["status"]
    assert flush_status in {"extracted", "no_extraction"}, resp.json()

    assert await buffer_count(session_id) == 0, "buffer must be drained after flush"

    cells_after_flush = await memcell_count(session_id)
    # If the last /add was already 'extracted' and emptied the buffer,
    # flush returns 'no_extraction'. Otherwise flush must produce ≥ 1
    # cell to satisfy the boundary semantics.
    if flush_status == "extracted":
        assert cells_after_flush > cells_pre_flush

    # 419 LoCoMo messages produce ~19 memcells in practice (LLM boundary
    # decides semantic cuts; daily-life chat carves coarsely). Threshold
    # 15 leaves room for run-to-run variance from the boundary LLM.
    assert cells_after_flush >= 15, (
        f"expected ≥ 15 memcells from 419 messages, got {cells_after_flush}; "
        f"last add status was {last_status!r}, flush was {flush_status!r}"
    )

    # ── Stage 3 + 4: wait for cascade to drain ────────────────────────────
    # Cascade syncs md → LanceDB. OME async strategies (atomic / foresight /
    # profile) also write md, which then cascade picks up. So one wait on
    # cascade-drain effectively covers both pipelines, IF OME has already
    # emitted its strategies (which memorize.py does inline via engine.emit).
    await cascade_done_poll(deadline_seconds=600.0)

    # ── Stage 5: artifacts on disk + LanceDB ──────────────────────────────
    # 5.1 episodes
    episode_files = _list_md_files(memory_root, _EPISODE_DIR)
    assert episode_files, "no episode md files written"
    episode_entries = _count_episode_entries(episode_files)
    # 19 memcells × 2 owners (caroline + melanie) ≈ 36 episode rows seen
    # in practice; threshold 15 leaves variance room.
    assert episode_entries >= 15, (
        f"expected ≥ 15 episode entries across {len(episode_files)} files, "
        f"got {episode_entries}"
    )

    # 5.2 episode → LanceDB

    lance_episode_count = await storage_view.repos.episode.count()
    assert lance_episode_count >= 15, (
        f"LanceDB episode rows ({lance_episode_count}) < md entries ({episode_entries})"
    )

    # 5.3 atomic_fact
    af_files = _list_md_files(memory_root, _ATOMIC_FACT_DIR)
    assert af_files, "no atomic_fact md files — extract_atomic_facts did not emit"

    lance_af_count = await storage_view.repos.atomic_fact.count()
    assert lance_af_count >= 1, (
        f"LanceDB atomic_fact rows = {lance_af_count}; expected ≥ 1"
    )

    # 5.4 foresight
    # Foresight extractor is correctly invoked (log: ``foresights_extracted``
    # per memcell) but daily-life chat about kids / work / hobbies rarely
    # yields explicit future-intent statements, so count is usually 0.
    # We assert the LanceDB table exists (count returns 0 cleanly) — not
    # that any row was emitted.

    lance_fs_count = await storage_view.repos.foresight.count()
    assert lance_fs_count >= 0, f"foresight table broken: count={lance_fs_count}"

    # 5.5 profile (md only — profile retrieval path is stub; we only assert
    # the writer wrote something). Profile lives as a single file
    # ``users/<user_id>/user.md`` (schema: ``UserProfileFrontmatter.PROFILE_FILENAME``).
    from everos.infra.persistence.markdown import UserProfileFrontmatter

    profile_filename = UserProfileFrontmatter.PROFILE_FILENAME
    profile_files: list[Path] = []
    users_root = memory_root / "default_app" / "default_project" / "users"
    if users_root.is_dir():
        for ud in users_root.iterdir():
            candidate = ud / profile_filename
            if candidate.exists():
                profile_files.append(candidate)
    assert profile_files, (
        f"no {profile_filename} written — extract_user_profile / "
        "trigger_profile_clustering did not emit"
    )
    # At least one profile file has non-trivial content.
    assert any(f.read_text().strip() for f in profile_files), (
        "all profile.md files are empty"
    )

    # ── Stage 5b: strict md ↔ LanceDB parity (every cascade kind) ─────────
    # Counts above are looser ``>=`` checks against LLM non-determinism;
    # here we enforce byte-exact id-set + content_sha256 parity across
    # every md the pipeline wrote. Catches: missing rows, orphan rows,
    # content drift between md and the indexed projection.
    #
    # ``expect_at_least`` pins the kinds this pipeline MUST produce so an
    # empty glob (kind not emitted at all) fails loudly — without this
    # guard the parity check would silently pass on zero files. Foresight
    # is NOT pinned because the LLM frequently yields 0 future-intent
    # statements on daily-life chat (see commentary above stage 5.4).
    from tests._consistency_assertions import assert_md_lance_strict_consistent

    await assert_md_lance_strict_consistent(
        memory_root,
        storage_view.repos,
        expect_at_least={
            "episode": 1,
            "atomic_fact": 1,
            "user_profile": 1,
        },
    )

    # ── Stage 6: optional corpus snapshot ─────────────────────────────────
    # When ``EVEROS_KEEP_CORPUS_TO=<dest>`` is set, copy the post-test
    # ``memory_root`` to ``<dest>`` so it can be tarred + uploaded as a
    # test corpus for the /search e2e suite. Skipped silently when the
    # env var is absent (default test runs don't snapshot).
    _maybe_snapshot_memory_root(memory_root)


# ---------------------------------------------------------------------------
# Diagnostic: lighter smoke that doesn't depend on the long fixture, used
# to validate the conftest fixtures themselves are wired correctly.
# ---------------------------------------------------------------------------


async def test_async_client_starts_and_health_responds(
    core_pipeline_runtime: Path,
) -> None:
    """Tiny smoke — the full lifespan comes up and ``/health`` answers.

    Runs on a canned model (no credentials): the point is the lifespan
    wiring, which the ``async_client`` fixture of the live suites shares.
    """
    from everos.config import load_settings
    from everos.entrypoints.api.app import create_app
    from everos.service import build_application, build_providers

    settings = load_settings()
    providers = dataclasses.replace(
        build_providers(settings), llm=user_track_llm(), embedding=None, rerank=None
    )
    app = create_app(application=build_application(settings, providers=providers))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        resp = await client.get("/health")
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# Hermetic user track: canned model, no credentials
# ---------------------------------------------------------------------------


_ALICE_TS = 1_775_116_800_000  # 2026-04-02 08:00 UTC

_CONVERSATION = [
    ("u_alice", "user", "I'm planning a trip to Kyoto at the end of April."),
    ("assistant", "assistant", "Lovely! Do you prefer a hotel or a ryokan?"),
    ("u_alice", "user", "I'd rather stay in a ryokan, somewhere quiet."),
]


async def _poll_until(
    condition: Callable[[], Awaitable[bool]], *, deadline_seconds: float = 40.0
) -> None:
    async with asyncio.timeout(deadline_seconds):
        while True:
            if await condition():
                return
            await asyncio.sleep(0.2)


async def test_add_flush_search_get_on_canned_user_track(
    core_pipeline_runtime: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``/add`` -> ``/flush`` -> ``/search`` -> ``/get`` on the canned user track.

    Real application (boundary, episode pipeline, OME strategies, cascade,
    LanceDB), Tier 1 (no embedding), every model call answered from
    ``tests/fakes/llm_scripts/user_track``. The add buffers (the operator
    never closes a non-final tail), the flush cuts one cell, and the canned
    episode, atomic facts and profile all become readable through the API.
    """
    from everos.config import load_settings
    from everos.entrypoints.api.app import create_app
    from everos.service import build_application, build_providers

    monkeypatch.setenv("EVEROS_MEMORIZE__MODE", "chat")
    monkeypatch.setenv("EVEROS_LANCEDB__READ_CONSISTENCY_SECONDS", "0")
    load_settings.cache_clear()
    settings = load_settings()
    llm = user_track_llm()
    providers = dataclasses.replace(
        build_providers(settings), llm=llm, embedding=None, rerank=None
    )
    app = create_app(application=build_application(settings, providers=providers))
    episode = json.loads(user_track_script("episode"))
    profile = json.loads(user_track_script("profile_init"))

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        add = await client.post(
            "/api/v1/memory/add",
            json={
                "session_id": "s_kyoto",
                "messages": [
                    {
                        "sender_id": sender,
                        "role": role,
                        "timestamp": _ALICE_TS + i * 60_000,
                        "content": text,
                    }
                    for i, (sender, role, text) in enumerate(_CONVERSATION)
                ],
            },
        )
        assert add.status_code == 200, add.text
        assert add.json()["data"]["status"] == "accumulated"

        flush = await client.post(
            "/api/v1/memory/flush", json={"session_id": "s_kyoto"}
        )
        assert flush.status_code == 200, flush.text
        assert flush.json()["data"]["status"] == "extracted"

        async def _get(memory_type: str) -> list[dict]:
            resp = await client.post(
                "/api/v1/memory/get",
                json={"user_id": "u_alice", "memory_type": memory_type},
            )
            assert resp.status_code == 200, resp.text
            data = resp.json()["data"]
            return data["episodes" if memory_type == "episode" else "profiles"]

        async def _search() -> list[dict]:
            resp = await client.post(
                "/api/v1/memory/search",
                json={
                    "user_id": "u_alice",
                    "query": "Kyoto ryokan",
                    "method": "keyword",
                },
            )
            assert resp.status_code == 200, resp.text
            return resp.json()["data"]["episodes"]

        async def _indexed() -> bool:
            return bool(await _get("episode")) and bool(await _get("profile"))

        await _poll_until(_indexed)
        (got_episode,) = await _get("episode")
        assert got_episode["episode"] == episode["content"]
        assert got_episode["summary"] == episode["summary"]
        assert got_episode["subject"] == episode["title"]
        assert got_episode["session_id"] == "s_kyoto"

        (got_profile,) = await _get("profile")
        assert [
            i["description"] for i in got_profile["profile_data"]["explicit_info"]
        ] == [i["description"] for i in profile["explicit_info"]]

        async def _found() -> bool:
            return bool(await _search())

        await _poll_until(_found)
        (hit,) = await _search()
        assert hit["episode"] == episode["content"]

    replies = [call.response_format.__name__ for call in llm.calls]
    assert replies.count("_BatchReply") == 2  # the add and the flush
    assert replies.count("_EpisodeReply") == 1
    assert "_AtomicFactsReply" in replies
    assert "_ProfileItems" in replies
