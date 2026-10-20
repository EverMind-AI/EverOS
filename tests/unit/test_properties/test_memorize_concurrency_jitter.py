"""Random concurrent memorize() calls against the real session lock and timeout.

``memorize()`` serialises one session's calls on an asyncio lock and wraps
lock plus work in ``asyncio.timeout(session_lock_timeout_seconds)``; the
``_session_lock`` module docstring says why: two concurrent ``/add`` calls
on one session would otherwise lose a tail. ``_boundary`` promises that every
cell gets exactly one memcell row, and the memorize docs that the pipelines
run for every cell. This test does not script what should happen; it lands
random calls -- adds, deferred adds, flushes, on two sessions, in overlapping
tasks, with a boundary call that takes a random time and sometimes fails --
under a random lock timeout, and checks only what the documentation promises
once everything has settled:

1. a message the service accepted (the call returned) is in the buffer or
   in exactly one memcell, never both, never nowhere; a message of a call
   that timed out or failed may be present or absent, but never in two
   places;
2. the buffer holds each id once, per session in ``(timestamp, id)`` order;
3. every memcell row the table holds was handed to the pipeline exactly
   once, and the pipeline never saw an id that is not in the table;
4. a poison message -- one the boundary stand-in can never segment, drawn
   as "the k-th message created" -- costs a session at most
   ``_POISON_CALLS`` consecutive failed calls before its buffer is set
   aside on the ``quarantine`` track (bounded cost), after which the
   session's later healthy messages are extracted again (exit), and other
   sessions never notice (isolation). The quarantine track counts as a
   place in law 1.

What is real: ``memorize()`` end to end (ingest, lock, timeout, boundary
stage, the three SQLite repositories on a temp file). What is replaced: the
boundary LLM call, the extraction pipeline (records what it was handed),
the LLM client sentinel, the prompt loader, and the settings (chat mode, the
drawn timeout).
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import random
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from everalgo.boundary.chat import DetectionResult
from everalgo.types import ChatMessage, MemCell
from hypothesis import HealthCheck, currently_in_test_context, given, settings
from hypothesis import event as _hypothesis_event
from hypothesis import strategies as st
from sqlalchemy import select
from sqlmodel import SQLModel

from everos.config import SqliteSettings, load_settings
from everos.core.errors import SessionBufferQuarantinedError
from everos.core.persistence import (
    MemoryRoot,
    create_session_factory,
    create_system_engine,
)
from everos.infra.persistence.sqlite.repos.conversation_status import (
    _ConversationStatusRepo,
)
from everos.infra.persistence.sqlite.repos.memcell import _MemcellRepo
from everos.infra.persistence.sqlite.repos.unprocessed_buffer import (
    _UnprocessedBufferRepo,
)
from everos.infra.persistence.sqlite.tables import Memcell
from everos.service import _boundary as boundary_mod
from everos.service import _session_lock as session_lock_mod

memorize_mod = importlib.import_module(
    "everos.service.memorize"
)  # the package re-exports the function

settings.register_profile(
    "ci", max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile(
    "deep",
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))

_DEEP = os.environ.get("HYPOTHESIS_PROFILE") == "deep"
SESSIONS = ("s1", "s2")
BASE_TS = 1_700_000_000_000


def event(label: str) -> None:
    if currently_in_test_context():
        _hypothesis_event(label)


class _Run:
    def __init__(
        self, seed: int, slow_max: float, fail_rate: float, poison: frozenset[int]
    ) -> None:
        self.rng = random.Random(seed)
        self.slow_max = slow_max
        self.fail_rate = fail_rate
        self.poison_positions = poison  # the k-th message created is poison
        self.poison_ids: set[str] = set()
        self.consecutive_failures: dict[str, int] = defaultdict(int)
        self.worst_streak = 0
        self.pipeline_log: list[tuple[str, tuple[str, ...]]] = []
        self.unexpected: list[tuple[int, str, str]] = []
        self.pipelines_active = 0
        self.trace: list[str] = []
        self.t0 = asyncio.get_event_loop().time()

    def note(self, what: str) -> None:
        self.trace.append(
            f"{(asyncio.get_event_loop().time() - self.t0) * 1000:7.1f}ms {what}"
        )

    async def detect(
        self,
        messages: list[ChatMessage],
        *,
        llm: Any,
        prompt: Any = None,
        is_final: bool = False,
        hard_token_limit: int = 0,
        hard_msg_limit: int = 0,
    ) -> DetectionResult:
        await asyncio.sleep(self.rng.random() * self.slow_max)
        if any(m.id in self.poison_ids for m in messages):
            self.note("  boundary: ValueError (poison in the input)")
            raise ValueError("cannot segment this")
        if self.rng.random() < self.fail_rate:
            self.note("  boundary: ValueError")
            raise ValueError("bad reply")
        k = len(messages) if is_final else self.rng.randint(0, len(messages))
        n_cells = self.rng.randint(1, 2) if k else 0
        cells = []
        if k:
            # The algo contract: the cells and the tail partition the input.
            # (A first version let the last chunk run past ``k`` into the
            # tail, and the deep run promptly reported a message in a cell
            # and in the buffer -- the harness's bug, not the stage's.)
            cut = k // n_cells or 1
            chunks = [messages[i : min(i + cut, k)] for i in range(0, k, cut)]
            cells = [
                MemCell(items=list(c), timestamp=c[-1].timestamp) for c in chunks if c
            ]
        self.note(
            f"  boundary: L={len(messages)} final={is_final} k={k} cells={len(cells)}"
        )
        return DetectionResult(cells=cells, tail=list(messages[k:]))

    async def pipeline_run(
        self, ingested: Any, cells: Any, memcell_ids: Any, per_cell_all_senders: Any
    ) -> Any:
        self.pipelines_active += 1
        try:
            await asyncio.sleep(self.rng.random() * 0.01)
        finally:
            self.pipelines_active -= 1
        self.pipeline_log.append((ingested.session_id, tuple(memcell_ids)))
        self.note(f"  pipeline: {ingested.session_id} {list(memcell_ids)}")
        return SimpleNamespace(
            status="extracted", track="user", message_count=len(cells)
        )


async def _jittered(
    plan: list[tuple[str, str, int, float]],
    seed: int,
    lock_timeout: float,
    slow_max: float,
    fail_rate: float,
    poison: frozenset[int] = frozenset(),
) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="memorize_jitter_"))
    run = _Run(seed, slow_max, fail_rate, poison)
    boundary_mod._unusable_replies.clear()
    # The per-session locks are process-wide and bound to the loop that
    # first used them; every example runs on a fresh loop.
    session_lock_mod._session_locks.clear()
    root = MemoryRoot(tmp)
    root.ensure()
    engine = create_system_engine(root.system_db, SqliteSettings())
    factory = create_session_factory(engine)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    buffer_repo = _UnprocessedBufferRepo(session_factory=factory)
    memcell_repo = _MemcellRepo(session_factory=factory)
    status_repo = _ConversationStatusRepo(session_factory=factory)
    base = load_settings()
    cfg = base.model_copy(
        update={
            "memorize": base.memorize.model_copy(
                update={"mode": "chat", "session_lock_timeout_seconds": lock_timeout}
            )
        }
    )
    pipeline = SimpleNamespace(run=run.pipeline_run)
    patches = [
        mock.patch.object(boundary_mod, "unprocessed_buffer_repo", buffer_repo),
        mock.patch.object(boundary_mod, "memcell_repo", memcell_repo),
        mock.patch.object(boundary_mod, "conversation_status_repo", status_repo),
        mock.patch.object(boundary_mod, "detect_boundaries", run.detect),
        mock.patch.object(memorize_mod, "load_settings", lambda: cfg),
        mock.patch.object(memorize_mod, "get_llm_client", lambda: object()),
        mock.patch.object(
            memorize_mod,
            "_get_prompt_loader",
            lambda: SimpleNamespace(load=lambda name: "p"),
        ),
        mock.patch.object(memorize_mod, "_get_user_pipeline", lambda: pipeline),
    ]
    for p in patches:
        p.start()
    next_ts = [0]
    accepted: dict[str, set[str]] = defaultdict(
        set
    )  # session -> ids of calls that returned
    unknown: dict[str, set[str]] = defaultdict(
        set
    )  # ids of calls that timed out / failed
    try:

        async def call(k: int, session: str, kind: str, n: int, pause: float) -> None:
            await asyncio.sleep(pause)
            msgs = []
            ids = []
            for idx in range(n if kind != "flush" else 0):
                next_ts[0] += 1
                ts = BASE_TS + next_ts[0]
                msgs.append(
                    {
                        "content": f"m{next_ts[0]}",
                        "timestamp": ts,
                        "sender_id": "u",
                        "role": "user",
                    }
                )
                mid = f"m_{session}_{ts}_{idx:03d}"
                ids.append(mid)
                if next_ts[0] in run.poison_positions:
                    run.poison_ids.add(mid)
            payload = {"session_id": session, "messages": msgs}
            run.note(f"call#{k} {kind} {session} n={len(msgs)}")
            try:
                result = await memorize_mod.memorize(
                    payload,
                    is_final=(kind == "flush"),
                    defer_extraction=(kind == "defer"),
                )
            except TimeoutError:
                run.note(f"call#{k} -> timeout")
                unknown[session].update(ids)
                event("call: lock timeout")
                return
            except SessionBufferQuarantinedError:
                run.note(f"call#{k} -> session buffer quarantined")
                accepted[session].update(ids)  # persisted, on the quarantine track
                run.consecutive_failures[session] = 0
                event("call: session buffer quarantined")
                return
            except ValueError:
                run.note(f"call#{k} -> boundary failed")
                unknown[session].update(ids)
                run.consecutive_failures[session] += 1
                run.worst_streak = max(
                    run.worst_streak, run.consecutive_failures[session]
                )
                event("call: boundary failed")
                return
            except Exception as exc:  # anything else is a finding in itself
                run.note(f"call#{k} -> {type(exc).__name__}: {exc}")
                unknown[session].update(ids)
                run.unexpected.append((k, kind, f"{type(exc).__name__}: {exc}"))
                event(f"call: unexpected {type(exc).__name__}")
                return
            run.note(f"call#{k} -> {result.status}")
            accepted[session].update(ids)
            if kind != "defer":
                run.consecutive_failures[session] = 0
            event(f"call: {kind} -> {result.status}")

        await asyncio.gather(*(call(k, *item) for k, item in enumerate(plan)))
        # Extraction may outlive a timed-out call; give it the same bound.
        for _ in range(500):
            if run.pipelines_active == 0:
                break
            await asyncio.sleep(0.01)
        assert run.pipelines_active == 0, "\n".join(run.trace)
        assert not run.unexpected, (
            "a call raised something other than the lock timeout or the boundary error",
            run.unexpected,
            "\n".join(run.trace),
        )

        async with factory() as s:
            memcells = list((await s.execute(select(Memcell))).scalars().all())
        # Bounded cost: no session fails more than _POISON_CALLS calls in a row
        # (the next one is the quarantine, which the model does not count).
        assert run.worst_streak < boundary_mod._POISON_CALLS + 1, (
            run.worst_streak,
            "\n".join(run.trace),
        )
        for session in SESSIONS:
            rows = await buffer_repo.list_for_track(session, "memorize")
            buffer_ids = [r.message_id for r in rows]
            quarantine = {
                r.message_id
                for r in await buffer_repo.list_for_track(session, "quarantine")
            }
            if quarantine:
                event("quarantine track holds messages")
            assert len(set(buffer_ids)) == len(buffer_ids), buffer_ids
            keys = [(r.timestamp, r.message_id) for r in rows]
            assert keys == sorted(keys), "\n".join(run.trace)
            cell_ids = [
                mid
                for m in memcells
                if m.session_id == session
                for mid in json.loads(m.message_ids_json)
            ]
            dup = sorted(i for i in set(cell_ids) if cell_ids.count(i) > 1)
            cells_of = {
                m.memcell_id: json.loads(m.message_ids_json)
                for m in memcells
                if m.session_id == session
            }
            assert not dup, (
                "a message in two memcells",
                dup,
                cells_of,
                "\n".join(run.trace),
            )
            placed = set(buffer_ids) | set(cell_ids) | quarantine
            assert not (quarantine & set(cell_ids)) and not (
                quarantine & set(buffer_ids)
            ), ("quarantined and elsewhere", "\n".join(run.trace))
            both = sorted(set(buffer_ids) & set(cell_ids))
            assert not both, (
                "buffer and memcell",
                both,
                buffer_ids,
                cells_of,
                "\n".join(run.trace),
            )
            assert accepted[session] <= placed, (
                "accepted message lost",
                accepted[session] - placed,
                "\n".join(run.trace),
            )
            assert placed <= accepted[session] | unknown[session], (
                "unknown id",
                "\n".join(run.trace),
            )
        handed = [mid for _, mids in run.pipeline_log for mid in mids]
        assert len(set(handed)) == len(handed), (
            "a memcell handed to the pipeline twice",
            "\n".join(run.trace),
        )
        table_ids = {m.memcell_id for m in memcells}
        assert set(handed) <= table_ids, (
            "pipeline saw a memcell not in the table",
            "\n".join(run.trace),
        )
        missing = table_ids - set(handed)
        assert not missing, (
            "memcells never handed to the pipeline",
            missing,
            "\n".join(run.trace),
        )
    finally:
        for p in patches:
            p.stop()
        await engine.dispose()
        shutil.rmtree(tmp, ignore_errors=True)


@given(
    plan=st.lists(
        st.tuples(
            st.sampled_from(SESSIONS),
            st.sampled_from(("add", "add", "defer", "flush")),
            st.integers(1, 3),
            st.sampled_from((0.0, 0.005, 0.02, 0.05)),
        ),
        min_size=1,
        max_size=14 if _DEEP else 8,
    ),
    seed=st.integers(0, 10_000),
    lock_timeout=st.sampled_from((0.03, 0.1, 5.0)),
    slow_max=st.sampled_from((0.0, 0.02, 0.08)),
    fail_rate=st.sampled_from((0.0, 0.3)),
    poison=st.frozensets(st.integers(1, 12), max_size=2),
)
def test_concurrent_memorize_keeps_every_message_in_one_place(
    plan: list[tuple[str, str, int, float]],
    seed: int,
    lock_timeout: float,
    slow_max: float,
    fail_rate: float,
    poison: frozenset[int],
) -> None:
    asyncio.run(_jittered(plan, seed, lock_timeout, slow_max, fail_rate, poison))
