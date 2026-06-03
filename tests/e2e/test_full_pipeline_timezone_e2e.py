"""Real full-pipeline timezone e2e — the gold-standard anti-drift test.

Exercises the **complete stack** under a display-tz switch:

    POST /add  →  unprocessed_buffer  →  POST /flush
                                            ↓
                                 boundary detection (memcell)
                                            ↓
                                  markdown writer (episode.md)
                                            ↓
                                 cascade scanner / worker
                                            ↓
                                   LanceDB index (episode row)

then POST /search and POST /get under display tz = Shanghai,
switch display tz to UTC, repeat /search + /get.

Pin: the **UTC instant** of every returned ``timestamp`` field is
identical across all four renders. Only the offset / wall-clock
changes. This is the user-facing contract of the storage-UTC discipline.

Real LLM (boundary detection + episode extraction) + real embedder
(LanceDB vector + FTS) — marked ``@slow`` ``@live_llm``.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Awaitable, Callable

import httpx
import pytest

from everos.component.utils import datetime as dt_module
from everos.component.utils.datetime import from_iso_format
from everos.config import load_settings


async def _switch_display_tz(monkeypatch: pytest.MonkeyPatch, tz: str) -> None:
    """Hot-swap the display tz mid-test + drop both caches.

    The ``_display_tz`` resolver and ``load_settings`` are
    ``functools.cache``-d; missing either ``cache_clear`` would let the
    new env var read silently no-op.
    """
    monkeypatch.setenv("EVEROS_MEMORY__TIMEZONE", tz)
    load_settings.cache_clear()
    dt_module._display_tz.cache_clear()


@pytest.mark.slow
@pytest.mark.live_llm
async def test_full_pipeline_tz_switch_preserves_utc_instant(
    async_client: httpx.AsyncClient,
    pipeline_done_poll: Callable[..., Awaitable[None]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real /add → /flush → cascade → LanceDB → /search /get under tz switch.

    Steps:

    1. Configure ``EVEROS_MEMORY__TIMEZONE=Asia/Shanghai``.
    2. POST /add a single message with a pinned epoch-ms timestamp.
    3. POST /flush — forces boundary detection to carve a memcell out
       of the single-message buffer.
    4. Wait for cascade to drain (md → LanceDB indexed).
    5. POST /search + POST /get: capture episode timestamp strings.
    6. Switch ``EVEROS_MEMORY__TIMEZONE=UTC``.
    7. POST /search + POST /get again: capture episode timestamp strings.
    8. Parse all four timestamp strings back to UTC instants. They must
       all be equal. The offsets and wall-clock numbers will differ
       between Shanghai and UTC renders — that's expected; what must
       NOT differ is the absolute UTC instant.

    Anti-drift contract is end-to-end: writes under one display tz
    must read back under another with zero data drift.
    """
    user_id = "alice_full_tz"
    session_id = "sess_full_tz"
    # 1748498400000 ms = 2026-05-29T06:00:00Z = 2026-05-29T14:00:00+08:00
    pinned_ms = 1748498400000
    expected_instant = dt.datetime.fromtimestamp(pinned_ms / 1000, tz=dt.UTC)

    # ── Step 1+2: configure Shanghai + write via /add ──
    await _switch_display_tz(monkeypatch, "Asia/Shanghai")
    resp = await async_client.post(
        "/api/v1/memory/add",
        json={
            "user_id": user_id,
            "session_id": session_id,
            "messages": [
                {
                    "sender_id": user_id,
                    "role": "user",
                    "timestamp": pinned_ms,
                    "content": "I love climbing in Yosemite every spring.",
                },
            ],
        },
        timeout=60.0,
    )
    assert resp.status_code == 200, resp.text

    # ── Step 3: /flush forces boundary detection on the single-message buffer ──
    resp = await async_client.post(
        "/api/v1/memory/flush",
        json={"user_id": user_id, "session_id": session_id},
        timeout=60.0,
    )
    assert resp.status_code == 200, resp.text

    # ── Step 4: wait for OME strategies + cascade to fully drain ──
    # 10-minute deadline: extract_episode + extract_atomic_facts run under
    # real LLM and the cascade worker only fires after md lands. The
    # `pipeline_done_poll` fixture covers both OME idle and cascade queue
    # empty.
    await pipeline_done_poll(deadline_seconds=600.0)

    # ── Step 5: /search + /get under Shanghai display tz ──
    resp_search_sh = await async_client.post(
        "/api/v1/memory/search",
        json={
            "user_id": user_id,
            "query": "climbing",
            "method": "keyword",  # no embedder cost; FTS index built by cascade
            "filters": {"session_id": session_id},
        },
        timeout=60.0,
    )
    assert resp_search_sh.status_code == 200, resp_search_sh.text
    eps_search_sh = resp_search_sh.json()["data"]["episodes"]
    assert eps_search_sh, (
        f"/search must return an episode after flush+cascade; got {eps_search_sh!r}"
    )
    ts_search_sh = eps_search_sh[0]["timestamp"]
    assert ts_search_sh.endswith("+08:00"), (
        f"Shanghai display tz should render offset +08:00; got {ts_search_sh!r}"
    )

    resp_get_sh = await async_client.post(
        "/api/v1/memory/get",
        json={
            "user_id": user_id,
            "memory_type": "episode",
            "page": 1,
            "page_size": 20,
        },
        timeout=60.0,
    )
    assert resp_get_sh.status_code == 200, resp_get_sh.text
    eps_get_sh = resp_get_sh.json()["data"]["episodes"]
    assert eps_get_sh, "/get must return the same episode /search did"
    ts_get_sh = eps_get_sh[0]["timestamp"]
    assert ts_get_sh.endswith("+08:00"), ts_get_sh

    # ── Step 6: switch to UTC display tz (drops caches) ──
    await _switch_display_tz(monkeypatch, "UTC")

    # ── Step 7: /search + /get again, same on-disk row, new render ──
    resp_search_utc = await async_client.post(
        "/api/v1/memory/search",
        json={
            "user_id": user_id,
            "query": "climbing",
            "method": "keyword",
            "filters": {"session_id": session_id},
        },
        timeout=60.0,
    )
    assert resp_search_utc.status_code == 200, resp_search_utc.text
    eps_search_utc = resp_search_utc.json()["data"]["episodes"]
    assert eps_search_utc
    ts_search_utc = eps_search_utc[0]["timestamp"]
    assert ts_search_utc.endswith("Z") or ts_search_utc.endswith("+00:00"), (
        f"UTC display tz should render Z / +00:00; got {ts_search_utc!r}"
    )

    resp_get_utc = await async_client.post(
        "/api/v1/memory/get",
        json={
            "user_id": user_id,
            "memory_type": "episode",
            "page": 1,
            "page_size": 20,
        },
        timeout=60.0,
    )
    assert resp_get_utc.status_code == 200, resp_get_utc.text
    eps_get_utc = resp_get_utc.json()["data"]["episodes"]
    ts_get_utc = eps_get_utc[0]["timestamp"]
    assert ts_get_utc.endswith("Z") or ts_get_utc.endswith("+00:00"), ts_get_utc

    # ── Step 8: anti-drift assertion — all four UTC instants identical ──
    instants = {
        "search/Shanghai": from_iso_format(ts_search_sh).astimezone(dt.UTC),
        "get/Shanghai": from_iso_format(ts_get_sh).astimezone(dt.UTC),
        "search/UTC": from_iso_format(ts_search_utc).astimezone(dt.UTC),
        "get/UTC": from_iso_format(ts_get_utc).astimezone(dt.UTC),
    }
    distinct = set(instants.values())
    assert len(distinct) == 1, (
        f"display-tz switch must NOT drift the UTC instant. Got distinct "
        f"instants across renders: {instants!r}"
    )
    actual_instant = next(iter(distinct))
    # Episode timestamp inherits from the last message's epoch ms — the
    # pinned input value must round-trip exactly.
    assert actual_instant == expected_instant, (
        f"episode UTC instant must equal the pinned input ms epoch; "
        f"expected {expected_instant.isoformat()}, got {actual_instant.isoformat()}"
    )

    # ── Sanity: across the four renders, identical instant projects to the
    # correct wall-clock under each display tz ──
    # Shanghai: 14:00 wall clock; UTC: 06:00 wall clock.
    assert "T14:00:00" in ts_search_sh, ts_search_sh
    assert "T14:00:00" in ts_get_sh, ts_get_sh
    assert "T06:00:00" in ts_search_utc, ts_search_utc
    assert "T06:00:00" in ts_get_utc, ts_get_utc
