"""The cascade worker registers every projection run with the delete fence.

A purge waits for the projections in flight when it moved the scope's files
(its step 7); a projection that never registers -- profile, skill and
knowledge ones used to register nothing -- could land rows after the purge
cleared the ledger.
"""

from __future__ import annotations

from everos.infra.cascade import DeleteFence
from everos.infra.cascade._worker import CascadeWorker
from tests.unit.test_infra.test_cascade.test_worker_kinds import (
    _OkHandler,
    _RecordingRepo,
    _Row,
)


class _RecordingFence(DeleteFence):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, int]] = []

    def register_run(self) -> int:
        token = super().register_run()
        self.events.append(("start", token))
        return token

    def end_run(self, token: int) -> None:
        self.events.append(("end", token))
        super().end_run(token)


async def test_every_handler_registers() -> None:
    repo = _RecordingRepo(
        batch_by_kind={
            "episode": [_Row(md_path="a/p/users/u1/episodes/e.md", kind="episode")],
            "user_profile": [_Row(md_path="a/p/users/u1/user.md", kind="user_profile")],
        }
    )
    fence = _RecordingFence()
    worker = CascadeWorker(
        {"episode": _OkHandler(), "user_profile": _OkHandler()},
        repo,  # type: ignore[arg-type]
        retry_backoff_seconds=0,
        fence=fence,
    )
    assert await worker.drain_once() == 2
    starts = [t for kind, t in fence.events if kind == "start"]
    ends = [t for kind, t in fence.events if kind == "end"]
    assert len(starts) == 2 and sorted(starts) == sorted(ends)
    assert fence.in_flight() == frozenset()
