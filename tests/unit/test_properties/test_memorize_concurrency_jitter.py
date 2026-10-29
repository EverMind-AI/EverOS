"""Random concurrent memorize() calls against the real session lock and timeout.

``memorize()`` serialises one session's calls on an asyncio lock and wraps
lock plus work in ``asyncio.timeout(session_lock_timeout_seconds)``; the
``_session_lock`` module docstring says why: two concurrent ``/add`` calls
on one session would otherwise lose a tail. Boundary detection promises that every
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
   ``POISON_CALLS`` consecutive failed calls before its buffer is set
   aside on the ``quarantine`` track (bounded cost), after which the
   session's later healthy messages are extracted again (exit), and other
   sessions never notice (isolation). The quarantine track counts as a
   place in law 1.

What is real: ``memorize()`` end to end (ingest, lock, timeout,
:class:`BoundaryDetection` and everalgo's boundary operator, the three SQLite
repositories on a temp file). What is replaced: the model's replies (random
delay, random cuts, unusable on poison or at random), the extraction pipeline
(records what it was handed), and the settings (chat mode, the drawn
timeout).
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from everalgo.llm import LLMOutputError
from everalgo.testing import FakeLLMClient
from hypothesis import HealthCheck, currently_in_test_context, given, settings
from hypothesis import event as _hypothesis_event
from hypothesis import strategies as st
from sqlalchemy import select
from sqlmodel import SQLModel

from everos.config import SqliteSettings, load_settings
from everos.core.errors import SessionBufferQuarantinedError
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.sqlite import (
    SqliteMemcellStore,
    SqliteSessionBufferStore,
    create_session_factory,
    create_system_engine,
)
from everos.infra.persistence.sqlite.repos.conversation_status import (
    ConversationStatusRepo,
)
from everos.infra.persistence.sqlite.repos.memcell import MemcellRepo
from everos.infra.persistence.sqlite.repos.unprocessed_buffer import (
    UnprocessedBufferRepo,
)
from everos.infra.persistence.sqlite.tables import Memcell
from everos.memory.extract.boundary import BoundaryDetection
from everos.memory.extract.boundary.quarantine import POISON_CALLS
from everos.memory.extract.pipeline import CellOutcome
from everos.service import _session_lock as session_lock_mod
from tests.fakes import (
    UNUSABLE_REPLY,
    boundary_reply,
    bundled_instructions,
    make_memory_service,
)
from tests.fakes.deps import make_stores

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
# One numbered line per message in the boundary prompt: ``[N] [time] u: m7``.
_PROMPT_LINE = re.compile(r"^\[\d+\] \[[^\]]*\] [^:]*: (m\d+)$", re.MULTILINE)


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
        self.poison_texts: set[str] = set()
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

    async def reply(self, messages: Any, **_kwargs: Any) -> str:
        """The model: a random delay, then unusable or a random set of cuts."""
        await asyncio.sleep(self.rng.random() * self.slow_max)
        window = _PROMPT_LINE.findall(messages[-1].content)
        if self.poison_texts.intersection(window):
            self.note("  boundary: unusable (poison in the input)")
            return UNUSABLE_REPLY
        if self.rng.random() < self.fail_rate:
            self.note("  boundary: unusable")
            return UNUSABLE_REPLY
        # The operator closes the trailing run itself on a final call; any
        # cut it is given falls strictly inside the window.
        candidates = range(1, len(window))
        cuts = sorted(self.rng.sample(candidates, min(len(candidates), 2)))
        cuts = cuts[: self.rng.randint(0, len(cuts))]
        self.note(f"  boundary: L={len(window)} cuts={cuts}")
        return boundary_reply(cuts)

    async def pipeline_run_cell(self, ingested: Any, prepared: Any) -> Any:
        self.pipelines_active += 1
        try:
            await asyncio.sleep(self.rng.random() * 0.01)
        finally:
            self.pipelines_active -= 1
        self.pipeline_log.append((ingested.session_id, (prepared.memcell_id,)))
        self.note(f"  pipeline: {ingested.session_id} {prepared.memcell_id}")
        return CellOutcome(status="extracted", message_count=1)


class _NullBus:
    async def publish(self, events: Any) -> None:
        return None


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
    # The per-session locks are process-wide and bound to the loop that
    # first used them; every example runs on a fresh loop.
    session_lock_mod._session_locks.clear()
    root = MemoryRoot(tmp)
    root.ensure()
    engine = create_system_engine(root.system_db, SqliteSettings())
    factory = create_session_factory(engine)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    buffer_repo = UnprocessedBufferRepo(session_factory=factory)
    memcell_repo = MemcellRepo(session_factory=factory)
    status_repo = ConversationStatusRepo(session_factory=factory)
    base = load_settings()
    cfg = base.model_copy(
        update={
            "memorize": base.memorize.model_copy(
                update={"mode": "chat", "session_lock_timeout_seconds": lock_timeout}
            )
        }
    )
    pipeline = SimpleNamespace(run_cell=run.pipeline_run_cell)
    stores = make_stores(
        session_buffer=SqliteSessionBufferStore(buffer_repo, status_repo),
        memcell=SqliteMemcellStore(memcell_repo),
    )
    memorize_mod = make_memory_service(
        settings=cfg,
        boundary=BoundaryDetection(
            llm=FakeLLMClient(handler=run.reply),
            instructions=bundled_instructions(),
            session_buffer=stores.session_buffer,
            memcells=stores.memcell,
            limits=cfg.boundary_detection,
            attempt_timeout=lock_timeout,
        ),
        stores=stores,
        user_pipeline=pipeline,
        bus=_NullBus(),
    )
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
                    run.poison_texts.add(f"m{next_ts[0]}")
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
            except LLMOutputError:
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
        # Bounded cost: no session fails more than POISON_CALLS calls in a row
        # (the next one is the quarantine, which the model does not count).
        assert run.worst_streak < POISON_CALLS + 1, (
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
