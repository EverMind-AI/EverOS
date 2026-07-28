"""Round-2 finding #3: worker drain must honor ``kinds`` scope.

Round-1 plumbed ``kinds={"agent_skill"}`` through
``Orchestrator.sync_once → Scanner.scan_once`` but stopped there.
The worker's ``drain_until_empty`` remained kind-agnostic, so a
scoped Phase-3 sync would still claim any pending row of any kind —
including a queued ``knowledge_topic`` whose handler this process
doesn't have registered — and mark it ``failed(retryable=False)``.

Round-2 threads ``kinds`` through ``drain_once`` /
``drain_until_empty`` and down to ``claim_pending_batch`` (SQL
parameter binding, not string interpolation — SQL injection safe).
This file pins the drain-side contract with a recording repo that
captures the kwargs handed to ``claim_pending_batch``.
"""

from __future__ import annotations

from dataclasses import dataclass

from everos.memory.cascade.handlers import Handler
from everos.memory.cascade.types import HandlerOutcome
from everos.memory.cascade.worker import CascadeWorker


@dataclass
class _Row:
    md_path: str
    kind: str = "episode"
    change_type: str = "added"
    retry_count: int = 0


class _RecordingRepo:
    """Records the ``kinds`` arg on every ``claim_pending_batch`` call
    so tests can assert the drain plumbing."""

    def __init__(self, batch_by_kind: dict[str | None, list[_Row]]) -> None:
        self.batch_by_kind = batch_by_kind
        self.claim_calls: list[frozenset[str] | None] = []
        self.done: list[str] = []
        self.failed: list[tuple[str, bool, str, int]] = []

    async def claim_pending_batch(
        self, limit: int, *, kinds: set[str] | None = None
    ) -> list[_Row]:
        # Normalise ``kinds`` to a frozenset for equality assertions —
        # ``set`` isn't hashable so we can't put it in a list-of-sets
        # comparison.
        self.claim_calls.append(frozenset(kinds) if kinds is not None else None)
        if kinds is None:
            # Merge every seeded kind bucket for the unscoped sweep.
            merged: list[_Row] = []
            for k, rows in self.batch_by_kind.items():
                if k is not None:
                    merged.extend(rows)
            # Drain the buckets so a follow-up call returns empty and
            # ``drain_until_empty`` terminates.
            for k in list(self.batch_by_kind):
                if k is not None:
                    self.batch_by_kind[k] = []
            return merged
        # Scoped sweep: return only rows whose kind is in the filter.
        picked: list[_Row] = []
        for k in list(kinds):
            picked.extend(self.batch_by_kind.get(k, []))
            self.batch_by_kind[k] = []
        return picked

    async def mark_done(self, md_path: str) -> None:
        self.done.append(md_path)

    async def mark_failed(
        self,
        md_path: str,
        *,
        retryable: bool,
        error: str,
        new_retry_count: int,
    ) -> None:
        self.failed.append((md_path, retryable, error, new_retry_count))


class _OkHandler(Handler):
    def __init__(self) -> None:
        # Bypass the base Handler init — this test only exercises the
        # worker's dispatch shape, not the deps-carrying protocol.
        pass

    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        return HandlerOutcome(
            md_path=md_path, kind="episode", upserted=1, deleted=0, skipped=0
        )

    async def handle_deleted(self, md_path: str) -> HandlerOutcome:
        return HandlerOutcome(
            md_path=md_path, kind="episode", upserted=0, deleted=1, skipped=0
        )


async def test_drain_once_default_drains_all_kinds(monkeypatch) -> None:
    """Round-1 regression guard: ``drain_once()`` without ``kinds``
    keeps its full-registry behaviour so the background worker loop
    and CLI ``cascade sync`` keep draining everything."""
    from everos.memory.cascade import worker as worker_mod

    repo = _RecordingRepo(
        batch_by_kind={
            "episode": [_Row(md_path="ep.md", kind="episode")],
            "agent_skill": [_Row(md_path="s.md", kind="agent_skill")],
        }
    )
    monkeypatch.setattr(worker_mod, "md_change_state_repo", repo)

    w = CascadeWorker(
        {"episode": _OkHandler(), "agent_skill": _OkHandler()},
        retry_backoff_seconds=0,
    )
    processed = await w.drain_once()
    assert processed == 2
    assert repo.claim_calls == [None]


async def test_drain_once_with_kinds_only_pulls_matching(monkeypatch) -> None:
    """Round-2 fix: ``drain_once(kinds={"agent_skill"})`` claims only
    agent_skill rows; other kinds stay pending in the repo (proxied by
    remaining in the recording bucket)."""
    from everos.memory.cascade import worker as worker_mod

    repo = _RecordingRepo(
        batch_by_kind={
            "episode": [_Row(md_path="ep.md", kind="episode")],
            "agent_skill": [_Row(md_path="s.md", kind="agent_skill")],
            "knowledge_topic": [_Row(md_path="kt.md", kind="knowledge_topic")],
        }
    )
    monkeypatch.setattr(worker_mod, "md_change_state_repo", repo)

    # Only register a handler for agent_skill so an unscoped drain
    # would flip the knowledge_topic row to permanently failed —
    # exactly the bug round-2 closes.
    w = CascadeWorker({"agent_skill": _OkHandler()}, retry_backoff_seconds=0)
    processed = await w.drain_once(kinds={"agent_skill"})

    assert processed == 1
    assert repo.claim_calls == [frozenset({"agent_skill"})]
    # Other kinds untouched by claim → the queue rows remain pending.
    assert repo.batch_by_kind["episode"] == [_Row(md_path="ep.md", kind="episode")]
    assert repo.batch_by_kind["knowledge_topic"] == [
        _Row(md_path="kt.md", kind="knowledge_topic")
    ]
    # And crucially — no permanent-failure write against a non-scoped row.
    assert repo.failed == []


async def test_drain_until_empty_forwards_kinds_on_every_pass(monkeypatch) -> None:
    """``drain_until_empty`` iterates ``drain_once`` up to
    ``max_passes``; each call must carry the same ``kinds`` filter so
    scoping doesn't decay across passes."""
    from everos.memory.cascade import worker as worker_mod

    repo = _RecordingRepo(
        batch_by_kind={
            "agent_skill": [
                _Row(md_path="s1.md", kind="agent_skill"),
                _Row(md_path="s2.md", kind="agent_skill"),
            ],
        }
    )
    monkeypatch.setattr(worker_mod, "md_change_state_repo", repo)

    w = CascadeWorker({"agent_skill": _OkHandler()}, retry_backoff_seconds=0)
    total = await w.drain_until_empty(kinds={"agent_skill"})

    assert total == 2
    # First call drains everything, second call sees empty and stops.
    assert repo.claim_calls[0] == frozenset({"agent_skill"})
    for call in repo.claim_calls:
        assert call == frozenset({"agent_skill"})
