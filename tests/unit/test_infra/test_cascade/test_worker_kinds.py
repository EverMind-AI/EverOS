"""``drain_once`` claims the pending rows of every kind, unscoped.

The kind-scoped drain (``kinds``) served the backfill's skills phase, which
EverOS 2.0 removed with agent memory. The worker claims across every kind --
the shape the background loop and the CLI ``cascade sync`` always used -- and
this pins it with a repo that records each claim.
"""

from __future__ import annotations

from dataclasses import dataclass

from everos.infra.cascade._projections import Handler
from everos.infra.cascade._worker import CascadeWorker
from everos.infra.cascade.models import HandlerOutcome


@dataclass
class _Row:
    md_path: str
    kind: str = "episode"
    change_type: str = "added"
    retry_count: int = 0
    lsn: int = 1
    mtime: float = 0.0
    status: str = "processing"


class _RecordingRepo:
    """Hands out every seeded row on the first claim, nothing after it."""

    def __init__(self, rows: list[_Row]) -> None:
        self.pending = list(rows)
        self.rows = {row.md_path: row for row in rows}
        self.claim_limits: list[int] = []
        self.done: list[str] = []
        self.failed: list[tuple[str, bool, str, int]] = []

    async def read_one(self, md_path: str) -> _Row | None:
        return self.rows.get(md_path) or _Row(md_path)

    async def claim_pending_batch(self, limit: int) -> list[_Row]:
        self.claim_limits.append(limit)
        claimed, self.pending = self.pending[:limit], self.pending[limit:]
        return claimed

    async def mark_done(self, md_path: str, **_: object) -> None:
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


async def test_drain_once_drains_every_kind_in_one_claim() -> None:
    repo = _RecordingRepo(
        [
            _Row(md_path="ep.md", kind="episode"),
            _Row(md_path="im.md", kind="image_memory"),
        ]
    )
    w = CascadeWorker(
        {"episode": _OkHandler(), "image_memory": _OkHandler()},
        retry_backoff_seconds=0,
        ledger=repo,
    )

    processed = await w.drain_once()

    assert processed == 2
    assert len(repo.claim_limits) == 1
    assert sorted(repo.done) == ["ep.md", "im.md"]
    assert repo.failed == []


async def test_drain_until_empty_claims_until_a_pass_finds_nothing() -> None:
    repo = _RecordingRepo(
        [
            _Row(md_path="ep1.md", kind="episode"),
            _Row(md_path="ep2.md", kind="episode"),
            _Row(md_path="im.md", kind="image_memory"),
        ]
    )
    w = CascadeWorker(
        {"episode": _OkHandler(), "image_memory": _OkHandler()},
        retry_backoff_seconds=0,
        ledger=repo,
    )

    total = await w.drain_until_empty()

    assert total == 3
    # The pass that claimed everything, then the one that found nothing.
    assert len(repo.claim_limits) == 2
    assert sorted(repo.done) == ["ep1.md", "ep2.md", "im.md"]
