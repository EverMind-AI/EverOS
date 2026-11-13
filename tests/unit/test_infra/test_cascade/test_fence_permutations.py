"""P2: every interleaving of one cascade run with one deletion of its entry.

Drives the real :class:`EpisodeHandler` and the real
:class:`FencedIndexEraser` over one md file holding entries ``e`` (deleted)
and ``k`` (kept). The cascade run is cut at the points where production code
suspends -- after reading the file (embedding), before the upsert writes,
after it wrote -- and the deletion at its two effects -- rewriting the md
file, then fencing and deleting the index row. Every order of those steps is
run, for every combination of: the index already holds ``e`` or not, the
upsert succeeds / lands then times out / times out without landing, and the
index delete succeeds or fails.

Checked, for every interleaving:

* when the upsert and the index delete both succeeded: once the run and the
  deletion have both finished, ``e`` is not in the index -- the fence closed
  the window with no other run's help (I1b);
* in every case: after the cascade's follow-up runs (the retry of a failed
  run, the run of a file the eraser queued again), ``e`` is not in the index
  and the fence is empty (A19).
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from everos.core.errors import VectorStoreBusyError
from everos.infra.cascade import DeleteFence, FencedIndexEraser
from everos.infra.cascade._projections.episode import EpisodeHandler
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown import (
    EpisodeWriter,
    MarkdownReader,
    MarkdownWriter,
)
from everos.memory.ports import All, AnyOf, Comparison, Filter, In, MemoryRef, Scope
from tests.fakes.storage import handler_deps, index_repositories

_CASCADE = ("c_read", "c_filter", "c_write", "c_end")
_DELETE = ("d_rewrite", "d_erase")
_SCOPE = Scope(app_id="default", project_id="default", owner_id="u1", owner_type="user")


def _matches(where: Filter, row: Any) -> bool:
    if isinstance(where, All):
        return all(_matches(c, row) for c in where.children)
    if isinstance(where, AnyOf):
        return any(_matches(c, row) for c in where.children)
    if isinstance(where, In):
        return getattr(row, where.field) in where.values
    if isinstance(where, Comparison) and where.operator == "eq":
        return getattr(row, where.field) == where.value
    raise AssertionError(f"unexpected predicate {where!r}")


class _Gates:
    """Lets the driver run one actor up to its next named stop at a time."""

    def __init__(self) -> None:
        self.arrived: dict[str, asyncio.Event] = {}
        self.go: dict[str, asyncio.Event] = {}
        self.enabled = True

    def _pair(self, name: str) -> tuple[asyncio.Event, asyncio.Event]:
        return (
            self.arrived.setdefault(name, asyncio.Event()),
            self.go.setdefault(name, asyncio.Event()),
        )

    async def stop(self, name: str) -> None:
        if not self.enabled:
            return
        arrived, go = self._pair(name)
        arrived.set()
        await go.wait()

    async def release(self, name: str) -> None:
        arrived, go = self._pair(name)
        await asyncio.wait_for(arrived.wait(), 2)
        go.set()


class _Index:
    """The episode table: rows by id, with the one write mode under test."""

    def __init__(self, gates: _Gates, upsert_mode: str) -> None:
        self.rows: dict[str, Any] = {}
        self._gates = gates
        self._upsert_mode = upsert_mode

    async def find_where(
        self, where: Filter, *, limit: int = 100, skip_vectors: bool = False
    ) -> list[Any]:
        return [r for r in self.rows.values() if _matches(where, r)]

    async def upsert(self, rows: Sequence[Any]) -> None:
        await self._gates.stop("c_write")
        if self._upsert_mode != "dropped":
            self.rows.update((r.id, r) for r in rows)
        await self._gates.stop("c_end")
        if self._upsert_mode != "ok":
            raise VectorStoreBusyError("write-lock deadline")  # C14: landed or not

    async def delete(self, where: Filter) -> None:
        for key in [k for k, r in self.rows.items() if _matches(where, r)]:
            del self.rows[key]

    async def delete_by_md_path(self, md_path: str) -> int:
        raise AssertionError("the file is never removed here")


class _Failing:
    async def delete(self, where: Filter) -> None:
        raise VectorStoreBusyError("backend unavailable")


class _Ledger:
    def __init__(self) -> None:
        self.requeued: list[str] = []

    async def force_enqueue(
        self, md_path: str, kind: str, *, mtime: float = 0.0
    ) -> int:
        self.requeued.append(md_path)
        return 1


class _GatedEmbedder:
    dim = 4

    def __init__(self, gates: _Gates) -> None:
        self._gates = gates

    async def embed(self, text: str) -> list[float]:
        return [0.1] * self.dim

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        await self._gates.stop("c_filter")  # read is done; embedding in flight
        return [[0.1] * self.dim for _ in texts]


async def _setup(root: MemoryRoot) -> tuple[str, str, str]:
    writer = EpisodeWriter(root)
    ids = []
    for body in ("deleted", "kept"):
        eid = await writer.append_entry(
            "u1",
            inline={
                "owner_id": "u1",
                "session_id": "s",
                "timestamp": "2026-05-14T10:00:00+00:00",
                "parent_type": "memcell",
                "parent_id": "mc",
            },
            sections={"Subject": "s", "Summary": "s", "Content": body},
        )
        ids.append(eid.format())
    md_path = writer.path_for("u1").relative_to(root.root).as_posix()
    return md_path, ids[0], ids[1]


async def _rewrite_without(root: MemoryRoot, md_path: str, entry_id: str) -> None:
    """What the deletion's md step leaves: the file without the entry."""
    path = root.root / md_path
    raw = path.read_text(encoding="utf-8")
    start = raw.index(f"<!-- entry:{entry_id} -->")
    end_marker = f"<!-- /entry:{entry_id} -->\n"
    end = raw.index(end_marker) + len(end_marker)
    await MarkdownWriter(root).write(path, raw[:start] + raw[end:])


def _interleavings() -> list[tuple[str, ...]]:
    out = []
    for slots in itertools.combinations(range(6), 2):
        cascade, delete = iter(_CASCADE), iter(_DELETE)
        out.append(
            tuple(next(delete) if i in slots else next(cascade) for i in range(6))
        )
    return out


_CASES = [
    pytest.param(
        order, prior, mode, erase_ok, id=f"{'-'.join(order)}|{prior}|{mode}|{erase_ok}"
    )
    for order in _interleavings()
    for prior in ("indexed", "new")
    for mode in ("ok", "landed", "dropped")
    for erase_ok in (True, False)
]


@pytest.mark.parametrize(("order", "prior", "mode", "erase_ok"), _CASES)
async def test_deleted_entry_never_stays_in_the_index(
    tmp_path: Path, order: tuple[str, ...], prior: str, mode: str, erase_ok: bool
) -> None:
    root = MemoryRoot(tmp_path)
    root.ensure()
    md_path, deleted, _kept = await _setup(root)
    gates = _Gates()
    index = _Index(gates, mode)
    fence = DeleteFence()
    handler = EpisodeHandler(
        handler_deps(
            memory_root=root,
            tokenizer=_Tokenizer(),
            index=index_repositories(episode=index),
            embedding=_GatedEmbedder(gates),
            fence=fence,
        )
    )
    if prior == "indexed":  # an earlier run projected the file already
        gates.enabled, index._upsert_mode = False, "ok"
        await handler.handle_added_or_modified(md_path)
        gates.enabled, index._upsert_mode = True, mode
        # This run then has something new to write: the kept entry changed.
        index.rows = {k: v for k, v in index.rows.items() if deleted in k}
    ledger = _Ledger()
    eraser = FencedIndexEraser(
        index_repositories(episode=index if erase_ok else _Failing()),
        fence,
        ledger,  # type: ignore[arg-type]
    )

    async def run() -> None:
        await gates.stop("c_read")
        # Registered as the cascade worker registers every run it dispatches.
        token = fence.register_run()
        try:
            with contextlib.suppress(VectorStoreBusyError):  # the worker retries it
                await handler.handle_added_or_modified(md_path)
        finally:
            fence.end_run(token)

    async def delete() -> None:
        await gates.stop("d_rewrite")
        await _rewrite_without(root, md_path, deleted)
        await gates.stop("d_erase")
        ref = MemoryRef(id=f"u1_{deleted}", entry_id=deleted, md_path=md_path)
        await eraser.erase(_SCOPE, {"episode": [ref]})

    tasks = [asyncio.create_task(run()), asyncio.create_task(delete())]
    for step in order:
        try:
            await gates.release(step)
        except TimeoutError:
            for t in tasks:  # an actor that died never reaches its next stop
                if t.done() and t.exception():
                    raise t.exception() from None  # type: ignore[misc]
            raise AssertionError(f"{step} never reached") from None
    await asyncio.wait_for(asyncio.gather(*tasks), 2)

    gone = f"u1_{deleted}" not in index.rows
    if mode == "ok" and erase_ok:
        assert gone, "the fence let the run write the deleted entry back"

    # Follow-up runs: a failed run is retried; a file the eraser queued again
    # is run again. Both read the rewritten file.
    gates.enabled = False
    if mode != "ok" or ledger.requeued:
        index._upsert_mode = "ok"
        await handler.handle_added_or_modified(md_path)
    assert f"u1_{deleted}" not in index.rows
    assert not fence.blocks(_Row(f"u1_{deleted}"))
    parsed = await MarkdownReader.read(root.root / md_path)
    assert deleted not in {e.id for e in parsed.entries}


class _Row:
    app_id = "default"
    project_id = "default"

    def __init__(self, row_id: str) -> None:
        self.id = row_id


class _Tokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def tokenize_batch(self, texts: Sequence[str]) -> list[list[str]]:
        return [t.split() for t in texts]
