"""Model-based test of the daily-log writer: entry ids stay unique and ordered.

A daily-log file (``users/<id>/episodes/episode-<date>.md`` and its three
siblings) is append-only through :class:`BaseDailyWriter`: the writer reads
the file's ``entry_count``, allocates the next ids from it, renders the
entries and rewrites the file atomically, all under one per-path lock. The
file is also the user's: they delete entries by hand and may edit the
frontmatter. Everything downstream (the cascade handlers, the LanceDB rows)
is keyed by the entry id, so the property that matters is that the writer
never hands out an id the file already holds, whatever happened to the file
in between.

What is real: :class:`EpisodeWriter`, :class:`MarkdownWriter`,
:class:`MarkdownReader`, the frontmatter and entry parsers, a temp memory
root. Nothing is faked.

Rules: append a few entries; append concurrently from several tasks; the
user deletes an entry; the user edits ``entry_count`` (lower, zero, higher,
not a number); an official ``patch_frontmatter`` merge.

Invariants after every rule:

1. the file parses; its entries are the model's, in order, with unique
   ids whose sequence numbers strictly increase down the file and whose
   ``## <id>`` header matches the marker;
2. ``entry_count`` equals the last sequence number the writer allocated,
   until the user edits it;
3. an append never returns an id the file already contains, and after an
   append ``entry_count`` is again the highest sequence in the file --
   also after the user lowered it or replaced it with text.

Before this test the writers trusted ``entry_count`` alone: a lowered count
made the next append reuse an id (two entries with one id; the index keeps
one), and a non-numeric count raised a ``TypeError`` from the id allocator.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from hypothesis import HealthCheck, currently_in_test_context, settings
from hypothesis import event as _hypothesis_event
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from everos.core.persistence import MarkdownReader, MemoryRoot
from everos.core.persistence.markdown import entries as ent
from everos.core.persistence.markdown import frontmatter as fm
from everos.core.persistence.markdown.writer import MarkdownWriter
from everos.infra.persistence.markdown.writers.episode_writer import EpisodeWriter

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

DATE = dt.date(2026, 1, 15)
SCOPES = ("u1", "u2")
COUNT_EDITS = ("lower", "zero", "higher", "text")


def event(label: str) -> None:
    if currently_in_test_context():
        _hypothesis_event(label)


class DailyLogMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.tmp = Path(tempfile.mkdtemp(prefix="daily_log_stateful_"))
        root = MemoryRoot(self.tmp)
        root.ensure()
        self.writer = EpisodeWriter(root)
        self.ids: dict[str, list[str]] = {s: [] for s in SCOPES}  # file order
        self.count: dict[str, Any] = {}  # frontmatter entry_count, if the file exists
        self.extra_fm: dict[str, dict[str, Any]] = {s: {} for s in SCOPES}
        self.trace: list[str] = []

    def run(self, coro: Any) -> Any:
        return self.loop.run_until_complete(coro)

    def teardown(self) -> None:
        self.loop.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _path(self, scope: str) -> Path:
        return self.writer._resolve_path(scope, DATE, "default", "default")

    def _highest_seq(self, scope: str) -> int:
        return max((ent.EntryId.parse(i).seq for i in self.ids[scope]), default=0)

    def _item(self, n: int) -> tuple[dict[str, object], dict[str, str]]:
        return {"n": n}, {"Summary": f"entry {n}"}

    # ── rules ──────────────────────────────────────────────────────────────

    @rule(scope=st.sampled_from(SCOPES), n=st.integers(1, 3))
    def append(self, scope: str, n: int) -> None:
        self.trace.append(f"append {scope} x{n} count={self.count.get(scope)!r}")
        eids = self.run(
            self.writer.append_entries(
                scope, [self._item(i) for i in range(n)], date=DATE
            )
        )
        got = [e.format() for e in eids]
        assert not set(got) & set(self.ids[scope]), (
            f"an id was handed out twice: {got} vs file {self.ids[scope]}\n"
            + "\n".join(self.trace[-10:])
        )
        self.ids[scope].extend(got)
        self.count[scope] = eids[-1].seq
        assert self.count[scope] == self._highest_seq(scope)
        event("append")

    @rule(scope=st.sampled_from(SCOPES), k=st.integers(2, 4))
    def append_concurrently(self, scope: str, k: int) -> None:
        self.trace.append(f"append_concurrently {scope} x{k}")

        async def many() -> list[ent.EntryId]:
            return list(
                await asyncio.gather(
                    *(
                        self.writer.append_entry(
                            scope, inline={"n": i}, sections={"S": "x"}, date=DATE
                        )
                        for i in range(k)
                    )
                )
            )

        eids = self.run(many())
        got = {e.format() for e in eids}
        assert len(got) == k and not got & set(self.ids[scope]), (
            got,
            self.ids[scope],
            "\n".join(self.trace[-10:]),
        )
        parsed = self.run(MarkdownReader.read(self._path(scope)))
        self.ids[scope] = [e.id for e in parsed.entries]
        self.count[scope] = max(e.seq for e in eids)
        assert self.count[scope] == self._highest_seq(scope)
        event("append concurrently")

    @precondition(lambda self: any(self.ids[s] for s in SCOPES))
    @rule(scope=st.sampled_from(SCOPES), which=st.integers(0, 10))
    def user_deletes_an_entry(self, scope: str, which: int) -> None:
        if not self.ids[scope]:
            return
        victim = self.ids[scope][which % len(self.ids[scope])]
        self.trace.append(f"user deletes {victim} from {scope}")
        path = self._path(scope)
        meta, body = fm.parse_frontmatter(path.read_text(encoding="utf-8"))
        span = ent.find_entry(body, victim)
        assert span is not None
        end = span.end + 1 if body[span.end : span.end + 1] == "\n" else span.end
        path.write_text(fm.dump_frontmatter(meta) + body[: span.start] + body[end:])
        self.ids[scope].remove(victim)
        event("user deletes an entry")

    @precondition(lambda self: bool(self.count))
    @rule(scope=st.sampled_from(SCOPES), how=st.sampled_from(COUNT_EDITS))
    def user_edits_entry_count(self, scope: str, how: str) -> None:
        if scope not in self.count:
            return
        path = self._path(scope)
        meta, body = fm.parse_frontmatter(path.read_text(encoding="utf-8"))
        current = meta["entry_count"]
        new: Any = {
            "lower": max(0, (current if isinstance(current, int) else 0) - 2),
            "zero": 0,
            "higher": (current if isinstance(current, int) else 0) + 3,
            "text": "seven",
        }[how]
        self.trace.append(f"user sets entry_count of {scope} to {new!r}")
        meta["entry_count"] = new
        path.write_text(fm.dump_frontmatter(meta) + body)
        self.count[scope] = new
        event(f"user edits entry_count: {how}")

    @precondition(lambda self: bool(self.count))
    @rule(scope=st.sampled_from(SCOPES), key=st.sampled_from(["note", "flag"]))
    def official_frontmatter_patch(self, scope: str, key: str) -> None:
        if scope not in self.count:
            return
        self.trace.append(f"patch_frontmatter {scope} {key}")
        self.run(self.writer.patch_frontmatter(self._path(scope), {key: "v"}))
        self.extra_fm[scope][key] = "v"
        event("patch_frontmatter")

    # ── invariants ─────────────────────────────────────────────────────────

    @invariant()
    def file_matches_model(self) -> None:
        for scope in SCOPES:
            path = self._path(scope)
            if scope not in self.count:
                assert not path.exists(), scope
                continue
            parsed = self.run(MarkdownReader.read(path))
            got = [e.id for e in parsed.entries]
            assert got == self.ids[scope], (scope, got, "\n".join(self.trace[-10:]))
            assert len(set(got)) == len(got)
            seqs = [ent.EntryId.parse(i).seq for i in got]
            assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), seqs
            for e in parsed.entries:
                assert e.as_structured().header == e.id
            assert parsed.frontmatter["entry_count"] == self.count[scope]
            assert parsed.frontmatter["user_id"] == scope
            assert parsed.frontmatter["date"] == DATE.isoformat()
            for k, v in self.extra_fm[scope].items():
                assert parsed.frontmatter[k] == v


TestDailyLogMachine = DailyLogMachine.TestCase


def test_machine_catches_a_lost_update_without_the_lock() -> None:
    """Self-check: take the per-path lock away and concurrent appends race
    on read-modify-write; the machine must see the duplicate or lost id."""
    machine = DailyLogMachine()
    try:
        with (
            mock.patch.object(
                MarkdownWriter, "lock_for", lambda self, path: contextlib.nullcontext()
            ),
            pytest.raises(AssertionError),
        ):
            for _ in range(5):
                machine.append_concurrently("u1", 4)
                machine.file_matches_model()
    finally:
        machine.teardown()
