"""Window-segmentation white-box integration tests for boundary stage.

Verifies the **read-merge-boundary-write** semantics of one ``memorize()``
invocation, especially the buffer-as-tail invariant and the **buffer
replacement** behaviour on successive calls:

Invariants under test
---------------------
I1. After one ``add`` with ``boundaries=[k]``:
    - memcell rows: prefix of merged input (first k messages)
    - buffer rows: tail (the remaining messages)
    - every input message_id lands in exactly one of {memcell, buffer}
      (covered ∧ disjoint)

I2. Tail ordering: every buffer row's timestamp ≥ every memcell row's
    timestamp (the tail is the **last** part of the time-ordered slice).

I3. Successive ``add`` consumes prior buffer:
    - Round 2's boundary sees ``prior_buffer + new_batch`` merged.
    - The prior tail (m3 say) ends up in **Round 2's memcell** if the
      boundary cuts past it, NOT in any buffer row.
    - The new buffer is the **fresh** tail, with the old buffer rows
      replaced entirely (semantics of ``_replace_buffer``).

I4. ``flush`` with ``is_final=True`` drains the buffer entirely — every
    remaining message ends up in some memcell.

This is **single-threaded sequential** (the concurrent race is covered
separately in test_memorize_concurrent_session_lock.py). FakeLLM scripts
boundary decisions deterministically so we own exact slicing.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from sqlalchemy import text
from sqlmodel import SQLModel

from everos.core.persistence import MemoryRoot
from everos.service.memorize import memorize

# ---------------------------------------------------------------------------
# FakeLLM with scripted boundary responses (FIFO queue, one pop per call)
# ---------------------------------------------------------------------------


def _boundary_response(boundaries: list[int]) -> str:
    return json.dumps(
        {"reasoning": "test", "boundaries": boundaries, "should_wait": False}
    )


def _episode_response(title: str = "T", content: str = "B") -> str:
    return json.dumps({"title": title, "content": content})


def _make_scripted_llm(
    boundary_responses: list[list[int]],
) -> FakeLLMClient:
    """Boundary calls FIFO-pop from ``boundary_responses``.

    Episode calls (for downstream pipeline) get a canned response.
    """
    queue: list[list[int]] = list(boundary_responses)

    def handler(messages: list[LLMChatMessage], **_: Any) -> ChatResponse:
        prompt = messages[0].content
        if "boundaries" in prompt.lower() or "memcell" in prompt.lower():
            cuts = queue.pop(0) if queue else []
            return ChatResponse(content=_boundary_response(cuts), model="fake")
        return ChatResponse(content=_episode_response(), model="fake")

    return FakeLLMClient(handler=handler)


# ---------------------------------------------------------------------------
# Fixture — mirrors the locked-env fixture in the concurrent test
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def memorize_env_scripted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[Callable[..., AsyncMock]]:
    monkeypatch.setattr(
        MemoryRoot, "default", classmethod(lambda cls: MemoryRoot(root=tmp_path))
    )
    (tmp_path / ".index" / "sqlite").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ome.toml").write_text("# test\n")

    svc = importlib.import_module("everos.service.memorize")
    af_mod = importlib.import_module("everos.memory.strategies.extract_atomic_facts")
    fs_mod = importlib.import_module("everos.memory.strategies.extract_foresight")
    client_mod = importlib.import_module("everos.component.llm.client")
    lock_mod = importlib.import_module("everos.service._session_lock")

    for attr in (
        "_episode_writer",
        "_prompt_loader",
        "_user_pipeline",
        "_agent_pipeline",
        "_ome_engine",
    ):
        monkeypatch.setattr(svc, attr, None, raising=False)
    monkeypatch.setattr(client_mod, "_llm_client", None, raising=False)
    monkeypatch.setattr(af_mod, "_writer", None, raising=False)
    monkeypatch.setattr(fs_mod, "_writer", None, raising=False)
    lock_mod._reset_for_tests()

    started: dict[str, Any] = {"engine": None}

    async def _setup(*, fake_llm: FakeLLMClient) -> None:
        monkeypatch.setenv("EVEROS_MEMORIZE__MODE", "chat")
        monkeypatch.setenv("EVEROS_LLM__API_KEY", "fake-key")
        monkeypatch.setenv("EVEROS_LLM__BASE_URL", "https://fake.example.com")
        from everos.config import load_settings

        load_settings.cache_clear()

        monkeypatch.setattr(client_mod, "_llm_client", fake_llm)

        from everos.infra.persistence.sqlite import get_engine

        db_engine = get_engine()
        async with db_engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        # Silence OME strategies — orthogonal to boundary segmentation.
        mock_af = AsyncMock(return_value=[])
        mock_fs = AsyncMock(return_value=[])
        monkeypatch.setattr(
            af_mod,
            "AtomicFactExtractor",
            lambda *a, **k: type("M", (), {"aextract": mock_af})(),
        )
        monkeypatch.setattr(
            fs_mod,
            "ForesightExtractor",
            lambda *a, **k: type("M", (), {"aextract": mock_fs})(),
        )

        engine = svc._get_engine()
        await engine.start()
        started["engine"] = engine

    yield _setup

    if started["engine"] is not None:
        await started["engine"].stop()
    from everos.infra.persistence.sqlite import dispose_engine

    await dispose_engine()


# ---------------------------------------------------------------------------
# Helpers — message factory + state inspectors
# ---------------------------------------------------------------------------


_BASE_TS = 1_700_000_000_000  # 2023-11-14, plenty of headroom


def _msg(idx: int, sender: str = "alice") -> dict[str, Any]:
    """Build one canonical /add message with monotonically increasing ts."""
    return {
        "sender_id": sender,
        "role": "user",
        "timestamp": _BASE_TS + idx * 1000,
        "content": f"msg-{idx}",
    }


async def _buffer_rows(session_id: str) -> list[tuple[str, int]]:
    """Return ``[(message_id, timestamp_ms)]`` for buffer rows, time-ordered."""
    from everos.component.utils.datetime import from_iso_format, to_timestamp_ms
    from everos.infra.persistence.sqlite import get_engine

    eng = get_engine()
    async with eng.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT message_id, timestamp FROM unprocessed_buffer "
                "WHERE session_id = :s ORDER BY timestamp"
            ),
            {"s": session_id},
        )
        rows: list[tuple[str, int]] = []
        for mid, ts in result.fetchall():
            # sqlite stores DateTime as ISO 8601 string via SQLAlchemy.
            ts_ms = to_timestamp_ms(from_iso_format(ts))
            rows.append((mid, ts_ms))
        return rows


async def _memcell_rows(session_id: str) -> list[tuple[str, list[str]]]:
    """Return ``[(memcell_id, message_ids[])]`` in insertion order."""
    from everos.infra.persistence.sqlite import get_engine

    eng = get_engine()
    async with eng.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT memcell_id, message_ids_json FROM memcell "
                "WHERE session_id = :s ORDER BY created_at"
            ),
            {"s": session_id},
        )
        return [(mid, json.loads(raw)) for mid, raw in result.fetchall()]


# ---------------------------------------------------------------------------
# I1 + I2: single add with boundaries=[k] — prefix→memcell, suffix→buffer
# ---------------------------------------------------------------------------


async def test_single_add_no_cut_accumulates_full_batch_in_buffer(
    memorize_env_scripted: Callable[..., AsyncMock],
) -> None:
    """boundaries=[] → no memcell, entire batch sits in buffer."""
    await memorize_env_scripted(fake_llm=_make_scripted_llm([[]]))

    session = "s_no_cut"
    inputs = [_msg(i) for i in range(3)]
    await memorize({"session_id": session, "messages": inputs})

    cells = await _memcell_rows(session)
    buffer = await _buffer_rows(session)

    assert cells == [], f"expected no memcell, got {cells}"
    assert len(buffer) == 3, f"expected 3 buffer rows, got {len(buffer)}"
    # buffer holds all 3 input message_ids, time-ordered
    buffer_ts = [ts for _, ts in buffer]
    assert buffer_ts == sorted(buffer_ts)


async def test_single_add_with_cut_splits_prefix_to_cell_suffix_to_buffer(
    memorize_env_scripted: Callable[..., AsyncMock],
) -> None:
    """boundaries=[2] on a 3-msg batch → cell=[m0,m1], buffer=[m2]."""
    await memorize_env_scripted(fake_llm=_make_scripted_llm([[2]]))

    session = "s_cut"
    inputs = [_msg(i) for i in range(3)]
    await memorize({"session_id": session, "messages": inputs})

    cells = await _memcell_rows(session)
    buffer = await _buffer_rows(session)

    # Exactly one memcell carved.
    assert len(cells) == 1, cells
    cell_msg_ids = set(cells[0][1])
    assert len(cell_msg_ids) == 2

    # Buffer holds the remaining one message.
    assert len(buffer) == 1
    buf_msg_id = buffer[0][0]

    # Disjoint: buffer message NOT in the memcell.
    assert buf_msg_id not in cell_msg_ids, (
        "buffer row leaked into memcell — buffer should be the tail only"
    )

    # I2 — tail comes AFTER prefix in time.
    cell_max_ts = max(_BASE_TS + i * 1000 for i in (0, 1))
    buf_ts = buffer[0][1]
    assert buf_ts >= cell_max_ts, (
        f"tail ts ({buf_ts}) must be >= max cell ts ({cell_max_ts})"
    )


# ---------------------------------------------------------------------------
# I3: successive add — prior buffer feeds into next memcell, then is REPLACED
# ---------------------------------------------------------------------------


async def test_second_add_consumes_prior_buffer_and_replaces_tail(
    memorize_env_scripted: Callable[..., AsyncMock],
) -> None:
    """Core test: prior tail must end up in next memcell, NOT remain in buffer."""
    # Round 1: cut after 2 of 3 → cell=[m0,m1], buffer=[m2]
    # Round 2: merged input = [m2,m3,m4,m5]; cut after 3 → cell=[m2,m3,m4],
    #          buffer=[m5]
    await memorize_env_scripted(
        fake_llm=_make_scripted_llm([[2], [3]]),
    )

    session = "s_replace"

    # Round 1
    r1_inputs = [_msg(i) for i in range(3)]
    await memorize({"session_id": session, "messages": r1_inputs})

    r1_cells = await _memcell_rows(session)
    r1_buffer = await _buffer_rows(session)
    assert len(r1_cells) == 1
    assert len(r1_buffer) == 1
    prior_tail_msg_id = r1_buffer[0][0]

    # Round 2 — fresh messages m3, m4, m5
    r2_inputs = [_msg(i) for i in range(3, 6)]
    await memorize({"session_id": session, "messages": r2_inputs})

    r2_cells = await _memcell_rows(session)
    r2_buffer = await _buffer_rows(session)

    # Two memcells total: one from round 1, one from round 2.
    assert len(r2_cells) == 2, r2_cells
    round1_cell_msgs = set(r2_cells[0][1])
    round2_cell_msgs = set(r2_cells[1][1])

    # ★ KEY ASSERTION ★ — prior buffer's message landed in round 2 cell.
    assert prior_tail_msg_id in round2_cell_msgs, (
        f"prior buffer msg {prior_tail_msg_id} should have been consumed "
        f"into round 2's memcell, but it's missing from {round2_cell_msgs}"
    )
    # Round 2 cell should have exactly 3 messages (prior tail + first 2 of new).
    assert len(round2_cell_msgs) == 3

    # Round 1 cell unchanged.
    assert len(round1_cell_msgs) == 2
    assert prior_tail_msg_id not in round1_cell_msgs

    # Buffer is the NEW tail — exactly 1 fresh row.
    assert len(r2_buffer) == 1
    new_tail_id = r2_buffer[0][0]

    # ★ KEY ASSERTION ★ — the OLD buffer entry is gone (replaced, not appended).
    assert new_tail_id != prior_tail_msg_id, (
        "old buffer entry survived into round 2's buffer — "
        "_replace_buffer is supposed to wipe + reinsert, not append"
    )

    # Buffer ∩ all memcells = ∅
    all_cell_msgs = round1_cell_msgs | round2_cell_msgs
    assert new_tail_id not in all_cell_msgs

    # Conservation: 6 distinct message ids covered across cells + buffer.
    # (We avoid hard-coding id format here — gen_message_id encodes the
    # per-batch index, not a global one.)
    covered = all_cell_msgs | {new_tail_id}
    assert len(covered) == 6, (
        f"expected 6 distinct ids covered, got {len(covered)}: {covered}"
    )


# ---------------------------------------------------------------------------
# I4: flush drains buffer entirely (is_final=True path)
# ---------------------------------------------------------------------------


async def test_flush_after_accumulation_drains_buffer_into_memcell(
    memorize_env_scripted: Callable[..., AsyncMock],
) -> None:
    """add(boundaries=[]) → buffer accumulates → flush → cell=all, buffer=[]."""
    # Round 1 add: boundaries=[] → no cut, all into buffer.
    # Flush: is_final=True passes empty boundaries → algo closes tail into cell.
    await memorize_env_scripted(
        fake_llm=_make_scripted_llm([[], []]),
    )

    session = "s_flush"
    inputs = [_msg(i) for i in range(3)]
    await memorize({"session_id": session, "messages": inputs})

    # Post-add: nothing in memcell yet.
    cells = await _memcell_rows(session)
    buffer = await _buffer_rows(session)
    assert cells == []
    assert len(buffer) == 3

    # Flush
    await memorize({"session_id": session, "messages": []}, is_final=True)

    cells = await _memcell_rows(session)
    buffer = await _buffer_rows(session)

    assert len(cells) == 1, cells
    assert len(cells[0][1]) == 3
    assert buffer == []


# ---------------------------------------------------------------------------
# Sanity: empty boundaries + multiple sequential adds keep conservation
# ---------------------------------------------------------------------------


async def test_three_sequential_adds_conservation_no_loss(
    memorize_env_scripted: Callable[..., AsyncMock],
) -> None:
    """3 sequential adds with mixed cuts: every input id covered exactly once."""
    # add 1: 3 msgs, no cut → buffer holds [m0,m1,m2]
    # add 2: 3 msgs, cut after 4 of merged [m0..m5] → cell=[m0..m3], buffer=[m4,m5]
    # add 3: 3 msgs, cut after 3 of merged [m4..m8] → cell=[m4,m5,m6], buffer=[m7,m8]
    await memorize_env_scripted(
        fake_llm=_make_scripted_llm([[], [4], [3]]),
    )

    session = "s_seq"
    total_inputs = 0
    for batch_start in (0, 3, 6):
        await memorize(
            {
                "session_id": session,
                "messages": [_msg(i) for i in range(batch_start, batch_start + 3)],
            }
        )
        total_inputs += 3

    cells = await _memcell_rows(session)
    buffer = await _buffer_rows(session)

    in_cells: set[str] = set()
    for _, msg_ids in cells:
        in_cells.update(msg_ids)
    in_buffer = {mid for mid, _ in buffer}

    covered = in_cells | in_buffer
    assert len(covered) == total_inputs, (
        f"expected {total_inputs} ids covered, got {len(covered)}"
    )
    # Disjoint
    assert not (in_cells & in_buffer)
