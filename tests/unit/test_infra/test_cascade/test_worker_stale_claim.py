"""A claim that was re-enqueued while its handler ran must not mark done."""

from __future__ import annotations

from typing import Any

from everos.infra.cascade._worker._worker import CascadeWorker
from everos.infra.cascade.models import HandlerOutcome
from everos.infra.persistence.sqlite import MdChangeState


class _Handler:
    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        return HandlerOutcome(
            md_path=md_path, kind="episode", upserted=1, deleted=0, skipped=0
        )

    async def handle_deleted(self, md_path: str) -> HandlerOutcome:
        return HandlerOutcome(
            md_path=md_path, kind="episode", upserted=0, deleted=1, skipped=0
        )


class _Ledger:
    def __init__(self, current: MdChangeState | None) -> None:
        self.current = current
        self.done: list[tuple[str, dict[str, Any]]] = []
        self.failed: list[dict[str, Any]] = []
        self.attempts: list[dict[str, Any]] = []
        self.upserts: list[dict[str, Any]] = []

    async def read_one(self, md_path: str) -> MdChangeState | None:
        return self.current

    async def mark_done(self, md_path: str, **values: Any) -> None:
        self.done.append((md_path, values))

    async def mark_failed(self, md_path: str, **values: Any) -> None:
        self.failed.append({"md_path": md_path, **values})

    async def record_attempt(self, md_path: str, **values: Any) -> None:
        self.attempts.append({"md_path": md_path, **values})

    async def upsert(
        self,
        md_path: str,
        *,
        kind: str,
        change_type: str,
        mtime: float,
    ) -> int:
        self.upserts.append(
            {
                "md_path": md_path,
                "kind": kind,
                "change_type": change_type,
                "mtime": mtime,
            }
        )
        return 2


def _row(*, lsn: int, status: str = "processing", mtime: float = 1.0) -> MdChangeState:
    return MdChangeState(
        md_path="users/u1/episodes/episode.md",
        kind="episode",
        change_type="modified",
        mtime=mtime,
        lsn=lsn,
        status=status,
        retry_count=0,
    )


async def test_stale_claim_is_re_enqueued_instead_of_marked_done() -> None:
    current = _row(lsn=2, status="done", mtime=2.0)
    ledger = _Ledger(current)
    worker = CascadeWorker({"episode": _Handler()}, ledger)  # type: ignore[arg-type]

    result = await worker._process_one(_row(lsn=1))

    assert result is None
    assert ledger.done == []
    assert ledger.upserts == [
        {
            "md_path": current.md_path,
            "kind": "episode",
            "change_type": "modified",
            "mtime": 2.0,
        }
    ]


async def test_own_claim_is_marked_done() -> None:
    current = _row(lsn=1)
    ledger = _Ledger(current)
    worker = CascadeWorker({"episode": _Handler()}, ledger)  # type: ignore[arg-type]

    result = await worker._process_one(_row(lsn=1))

    assert result == "episode"
    assert ledger.upserts == []
    assert ledger.done == [
        ("users/u1/episodes/episode.md", {"change_type": "modified"})
    ]
