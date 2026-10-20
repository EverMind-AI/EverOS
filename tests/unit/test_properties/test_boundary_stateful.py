"""Model-based test of the boundary stage: every message lands in exactly one place.

The boundary stage (``everos.service._boundary``) owns the session buffer
(``unprocessed_buffer``), the boundary call that cuts the buffer into cells,
and the ``memcell`` ledger that records which messages each cell took. The
property that matters is conservation: a message the service accepted is in
the buffer or in exactly one memcell, never both and never nowhere.

What is real: ``ingest.process`` (ids, canonical rows), ``buffer_messages``
and ``prepare_cells``, and the three repositories on a temp SQLite file.
What is scripted: ``everalgo.boundary.detect_boundaries`` (the LLM call),
which the machine tells how many of the merged messages to cut into how
many cells, and whether to fail first (the stage retries a ``ValueError``
three times); the prompt loader; the LLM client, which is only checked for
``None``.

Rules: ``add`` (one ``/add`` on a session, deferred or not, with the
boundary's reply scripted), ``flush`` (``is_final``: the boundary must cut
everything). Two sessions, so scoping is exercised.

Invariants after every rule, from the module docstring and the repos:

1. conservation: the ids in the buffer and in all memcells together are
   exactly the accepted ids, with no id in two places;
2. the buffer table holds the model's rows in ``(timestamp, id)`` order,
   per session;
3. the memcell table holds one row per cell with that cell's ids and
   senders, in order;
4. ``conversation_status.last_memcell_ts`` is the latest cell timestamp,
   ``last_message_ts`` the latest message the buffer paths saw;
5. a boundary call that keeps failing leaves everything untouched, and the
   stage returns the status the docstring assigns to each path;
6. the third consecutive call on one session whose attempts all return
   unusable replies moves the session's buffer and that call's messages to
   the ``quarantine`` track (nothing dropped, the session starts over), and
   the quarantine track holds exactly those ids.

Timestamps are unique per message, so ids are unique. ``gen_message_id`` is
deterministic in ``(session, timestamp_ms, index)`` by design (retries of
one payload dedupe through the buffer's primary key); two *different*
messages sent in separate calls with the same millisecond and index share
an id and the second is dropped as a duplicate -- a property of the id
scheme, outside this machine's domain and recorded separately.

A mid-call crash between the memcell insert and the buffer rewrite (the
at-least-once case of #516) is not scripted either: the stage is not
atomic there, and that is the issue's subject.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from everalgo.boundary.chat import DetectionResult
from everalgo.types import ChatMessage, MemCell
from hypothesis import HealthCheck, currently_in_test_context, settings
from hypothesis import event as _hypothesis_event
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule
from sqlalchemy import select
from sqlmodel import SQLModel

from everos.config import SqliteSettings
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
from everos.infra.persistence.sqlite.tables import (
    ConversationStatus,
    Memcell,
    UnprocessedBuffer,
)
from everos.memory.extract.ingest import service as ingest
from everos.service import _boundary as boundary_mod

settings.register_profile(
    "ci",
    max_examples=40,
    stateful_step_count=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "deep",
    max_examples=1000,
    stateful_step_count=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))

SESSIONS = ("s1", "s2")
BASE_TS = 1_700_000_000_000  # above the seconds/ms threshold
ATTEMPTS = boundary_mod._BOUNDARY_MAX_ATTEMPTS
POISON_CALLS = boundary_mod._POISON_CALLS


def event(label: str) -> None:
    if currently_in_test_context():
        _hypothesis_event(label)


class _Prompts:
    def load(self, name: str) -> str | None:
        return "boundary prompt"


class BoundaryMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.tmp = Path(tempfile.mkdtemp(prefix="boundary_stateful_"))
        root = MemoryRoot(self.tmp)
        root.ensure()
        self.engine = create_system_engine(root.system_db, SqliteSettings())
        self.factory = create_session_factory(self.engine)
        self.run(self._create_tables())
        self.buffer_repo = _UnprocessedBufferRepo(session_factory=self.factory)
        self.memcell_repo = _MemcellRepo(session_factory=self.factory)
        self.status_repo = _ConversationStatusRepo(session_factory=self.factory)
        self._patches = [
            mock.patch.object(
                boundary_mod, "unprocessed_buffer_repo", self.buffer_repo
            ),
            mock.patch.object(boundary_mod, "memcell_repo", self.memcell_repo),
            mock.patch.object(
                boundary_mod, "conversation_status_repo", self.status_repo
            ),
            mock.patch.object(boundary_mod, "detect_boundaries", self._detect),
        ]
        for p in self._patches:
            p.start()
        boundary_mod._unusable_replies.clear()
        # the model
        self.accepted: dict[str, dict[str, Any]] = {}  # id -> message
        self.buffer: dict[str, list[str]] = defaultdict(list)  # session -> ids
        self.cells: dict[str, list[list[str]]] = defaultdict(list)
        self.quarantined: dict[str, list[str]] = defaultdict(list)
        self.unusable_calls: dict[str, int] = defaultdict(int)
        self.last_message_ts: dict[str, int] = {}
        self.last_memcell_ts: dict[str, int] = {}
        self.next_ts = 0
        self.detect_calls = 0
        self.expected_detect_calls = 0
        # the script for the next boundary call
        self.script: tuple[float, int, int] = (1.0, 1, 0)
        self.trace: list[str] = []

    async def _create_tables(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    def run(self, coro: Any) -> Any:
        return self.loop.run_until_complete(coro)

    def teardown(self) -> None:
        for p in self._patches:
            p.stop()
        self.run(self.engine.dispose())
        self.loop.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ── the scripted boundary ───────────────────────────────────────────────

    async def _detect(
        self,
        messages: list[ChatMessage],
        *,
        llm: Any,
        prompt: str | None = None,
        is_final: bool = False,
        hard_token_limit: int = 0,
        hard_msg_limit: int = 0,
    ) -> DetectionResult:
        self.detect_calls += 1
        cut, n_cells, fails = self.script
        if fails > 0:
            self.script = (cut, n_cells, fails - 1)
            raise ValueError("boundary reply was not JSON")
        k = len(messages) if is_final else round(cut * len(messages))
        cells = [
            MemCell(items=list(chunk), timestamp=chunk[-1].timestamp)
            for chunk in _chunks(messages[:k], n_cells)
        ]
        self.trace.append(
            f"  detect L={len(messages)} final={is_final} -> k={k} cells={len(cells)}"
        )
        return DetectionResult(cells=cells, tail=list(messages[k:]))

    # ── the model ──────────────────────────────────────────────────────────

    def _merged(self, session: str, fresh: list[dict[str, Any]]) -> list[str]:
        ids = {*self.buffer[session], *(m["id"] for m in fresh)}
        return sorted(ids, key=lambda i: (self.accepted[i]["ts"], i))

    def _apply(
        self,
        session: str,
        fresh: list[dict[str, Any]],
        *,
        defer: bool,
        final: bool,
        script: tuple[float, int, int],
    ) -> tuple[str, bool]:
        """What the stage must do; returns (status, raises)."""
        for m in fresh:
            self.accepted[m["id"]] = m
        merged = self._merged(session, fresh)
        if not fresh and not final:
            return "skipped", False
        if not merged:
            return "accumulated", False
        if defer or (
            not final and not any(self.accepted[i]["role"] == "user" for i in merged)
        ):
            self.buffer[session] = merged
            self.last_message_ts[session] = max(self.accepted[i]["ts"] for i in merged)
            return "accumulated", False
        cut, n_cells, fails = script
        if fails >= ATTEMPTS:
            self.expected_detect_calls += ATTEMPTS
            self.unusable_calls[session] += 1
            if self.unusable_calls[session] < POISON_CALLS:
                for m in fresh:  # the call failed: the service never kept them
                    del self.accepted[m["id"]]
                return "", True
            # The session's buffer and this call's messages are set aside and
            # the session starts over (spec: SessionBufferQuarantinedError).
            self.unusable_calls[session] = 0
            self.quarantined[session].extend(self.buffer[session])
            self.quarantined[session].extend(m["id"] for m in fresh)
            self.buffer[session] = []
            return "", "quarantined"
        self.unusable_calls[session] = 0
        self.expected_detect_calls += fails + 1
        k = len(merged) if final else round(cut * len(merged))
        if k == 0:
            self.buffer[session] = merged
            self.last_message_ts[session] = max(self.accepted[i]["ts"] for i in merged)
            return "accumulated", False
        cells = [list(c) for c in _chunks(merged[:k], n_cells)]
        self.cells[session].extend(cells)
        self.buffer[session] = merged[k:]
        self.last_memcell_ts[session] = max(self.accepted[c[-1]]["ts"] for c in cells)
        return "extracted", False

    def _fresh(self, session: str, msgs: list[tuple[str, str]]) -> list[dict[str, Any]]:
        out = []
        for idx, (role, text) in enumerate(msgs):
            self.next_ts += 1
            ts = BASE_TS + self.next_ts
            out.append(
                {
                    "id": ingest.gen_message_id(session, ts, idx),
                    "ts": ts,
                    "role": role,
                    "text": text,
                    "sender": "u" if role == "user" else "a",
                }
            )
        return out

    async def _call(
        self, session: str, fresh: list[dict[str, Any]], *, defer: bool, final: bool
    ) -> str:
        payload = {
            "session_id": session,
            "messages": [
                {
                    "content": m["text"],
                    "timestamp": m["ts"],
                    "sender_id": m["sender"],
                    "role": m["role"],
                }
                for m in fresh
            ],
        }
        ingested = await ingest.process(payload)
        assert [m.message_id for m in ingested.messages] == [m["id"] for m in fresh]
        if defer:
            outcome = await boundary_mod.buffer_messages(ingested, mode="chat")
        else:
            outcome = await boundary_mod.prepare_cells(
                ingested,
                mode="chat",
                is_final=final,
                llm_client=object(),
                prompt_loader=_Prompts(),  # type: ignore[arg-type]
                hard_token_limit=10_000,
                hard_msg_limit=100,
            )
        return outcome.status

    # ── rules ──────────────────────────────────────────────────────────────

    @rule(
        session=st.sampled_from(SESSIONS),
        msgs=st.lists(
            st.tuples(st.sampled_from(["user", "assistant"]), st.text(max_size=8)),
            min_size=1,
            max_size=4,
        ),
        defer=st.booleans(),
        script=st.tuples(
            st.sampled_from([0.0, 0.5, 1.0]),
            st.integers(1, 3),
            st.integers(0, ATTEMPTS),
        ),
    )
    def add(
        self,
        session: str,
        msgs: list[tuple[str, str]],
        defer: bool,
        script: tuple[float, int, int],
    ) -> None:
        fresh = self._fresh(session, msgs)
        self.script = script
        self.trace.append(f"add {session} {msgs} defer={defer} script={script}")
        want, raises = self._apply(
            session, fresh, defer=defer, final=False, script=script
        )
        try:
            got = self.run(self._call(session, fresh, defer=defer, final=False))
        except SessionBufferQuarantinedError:
            assert raises == "quarantined", "quarantined before the third failing call"
            event("add: session buffer quarantined")
            return
        except ValueError:
            assert raises is True, (
                "boundary raised but the script did not exhaust its retries"
            )
            event("add: boundary kept failing, nothing kept")
            return
        assert not raises, "the boundary should have exhausted its retries"
        assert got == want, (got, want, "\n".join(self.trace[-12:]))
        event(f"add: {'deferred' if defer else want}")

    @rule(session=st.sampled_from(SESSIONS))
    def model_cannot_segment_this_session(self, session: str) -> None:
        """Enough consecutive unusable replies to reach the quarantine."""
        for _ in range(POISON_CALLS):
            fresh = self._fresh(session, [("user", "poison")])
            self.script = (1.0, 1, ATTEMPTS)
            self.trace.append(f"add {session} poison script={self.script}")
            _, raises = self._apply(
                session, fresh, defer=False, final=False, script=self.script
            )
            try:
                self.run(self._call(session, fresh, defer=False, final=False))
            except SessionBufferQuarantinedError:
                assert raises == "quarantined", "quarantined before the third call"
                event("add: session buffer quarantined")
                return
            except ValueError:
                assert raises is True
                continue
            raise AssertionError("an unusable reply must raise")
        raise AssertionError("the third consecutive failing call must quarantine")

    @rule(
        session=st.sampled_from(SESSIONS),
        script=st.tuples(st.just(1.0), st.integers(1, 3), st.integers(0, ATTEMPTS)),
    )
    def flush(self, session: str, script: tuple[float, int, int]) -> None:
        self.script = script
        self.trace.append(f"flush {session} script={script}")
        want, raises = self._apply(session, [], defer=False, final=True, script=script)
        try:
            got = self.run(self._call(session, [], defer=False, final=True))
        except SessionBufferQuarantinedError:
            assert raises == "quarantined"
            event("flush: session buffer quarantined")
            return
        except ValueError:
            assert raises is True
            event("flush: boundary kept failing, nothing kept")
            return
        assert not raises
        assert got == want, (got, want, "\n".join(self.trace[-12:]))
        event(f"flush: {want}")

    # ── invariants ─────────────────────────────────────────────────────────

    async def _rows(self, model: Any) -> list[Any]:
        async with self.factory() as s:
            return list((await s.execute(select(model))).scalars().all())

    @invariant()
    def every_accepted_message_is_in_exactly_one_place(self) -> None:
        placed: list[str] = []
        for session in SESSIONS:
            placed.extend(self.buffer[session])
            placed.extend(self.quarantined[session])
            for cell in self.cells[session]:
                placed.extend(cell)
        assert len(placed) == len(set(placed)), "an id is in two places"
        assert set(placed) == set(self.accepted), "\n".join(self.trace[-12:])

    @invariant()
    def quarantine_track_matches_model(self) -> None:
        for session in SESSIONS:
            rows = self.run(self.buffer_repo.list_for_track(session, "quarantine"))
            assert sorted(r.message_id for r in rows) == sorted(
                self.quarantined[session]
            ), (session, "\n".join(self.trace[-12:]))

    @invariant()
    def buffer_table_matches_model(self) -> None:
        for session in SESSIONS:
            rows = self.run(self.buffer_repo.list_for_track(session, "memorize"))
            assert [r.message_id for r in rows] == self.buffer[session], (
                session,
                "\n".join(self.trace[-12:]),
            )
            for r in rows:
                m = self.accepted[r.message_id]
                assert (r.text, r.role, r.sender_id) == (
                    m["text"],
                    m["role"],
                    m["sender"],
                )
        all_rows = self.run(self._rows(UnprocessedBuffer))
        assert len(all_rows) == sum(
            len(self.buffer[s]) + len(self.quarantined[s]) for s in SESSIONS
        )

    @invariant()
    def memcell_table_matches_model(self) -> None:
        rows = self.run(self._rows(Memcell))
        by_session: dict[str, list[Memcell]] = defaultdict(list)
        for r in rows:
            by_session[r.session_id].append(r)
        for session in SESSIONS:
            got = sorted(
                (
                    json.loads(r.message_ids_json),
                    json.loads(r.sender_ids_json),
                    r.raw_type,
                )
                for r in by_session[session]
            )
            raw_type = boundary_mod._RAW_TYPE_BY_MODE["chat"]
            want = sorted(
                (cell, _senders(cell, self.accepted), raw_type)
                for cell in self.cells[session]
            )
            assert got == want, (session, "\n".join(self.trace[-12:]))
        assert len({r.memcell_id for r in rows}) == len(rows)

    @invariant()
    def status_row_matches_model(self) -> None:
        rows = {r.session_id: r for r in self.run(self._rows(ConversationStatus))}
        for session in SESSIONS:
            r = rows.get(session)
            want_msg = self.last_message_ts.get(session)
            want_cell = self.last_memcell_ts.get(session)
            if want_msg is None and want_cell is None:
                assert r is None, session
                continue
            assert r is not None, session
            got_msg = None if r.last_message_ts is None else _ms(r.last_message_ts)
            got_cell = None if r.last_memcell_ts is None else _ms(r.last_memcell_ts)
            assert (got_msg, got_cell) == (want_msg, want_cell), (
                session,
                "\n".join(self.trace[-12:]),
            )

    @invariant()
    def boundary_was_called_as_often_as_the_retry_rule_says(self) -> None:
        assert self.detect_calls == self.expected_detect_calls


def _chunks(items: list[Any], n: int) -> list[list[Any]]:
    """Split ``items`` into at most ``n`` non-empty runs."""
    if not items:
        return []
    n = max(1, min(n, len(items)))
    size, extra = divmod(len(items), n)
    out, pos = [], 0
    for i in range(n):
        end = pos + size + (1 if i < extra else 0)
        out.append(items[pos:end])
        pos = end
    return out


def _senders(cell: list[str], accepted: dict[str, dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for i in cell:
        s = accepted[i]["sender"]
        if s not in out:
            out.append(s)
    return out


def _ms(d: Any) -> int:
    from everos.component.utils.datetime import to_timestamp_ms

    return to_timestamp_ms(d)


TestBoundaryMachine = BoundaryMachine.TestCase


def test_machine_catches_a_lost_tail_message() -> None:
    """Self-check: drop the last tail message before the buffer rewrite and
    the conservation invariant must trip."""
    machine = BoundaryMachine()
    try:
        real = boundary_mod._slice_tail

        def lossy(merged: Any, tail: Any) -> Any:
            return real(merged, tail)[:-1]

        with mock.patch.object(boundary_mod, "_slice_tail", lossy):
            machine.add("s1", [("user", "a"), ("user", "b")], False, (0.5, 1, 0))
        with pytest.raises(AssertionError):
            machine.buffer_table_matches_model()
    finally:
        machine.teardown()
