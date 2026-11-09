"""``MemoryService.flush`` speaks the flush vocabulary, not memorize's."""

from __future__ import annotations

from typing import Any, Literal

import pytest

from everos.service import MemorizeResult
from tests.fakes import make_memory_service, make_stores


class _EmptyInbox:
    """An ``async_mode`` inbox with no backlog, so ``flush`` runs inline."""

    async def count_pending(self, session: Any = None) -> int:
        return 0


@pytest.mark.parametrize(
    ("memorized", "flushed", "permanent"),
    [
        ("accumulated", "no_extraction", 0),
        ("extracted", "extracted", 0),
        ("degraded", "degraded", 2),
    ],
)
async def test_flush_maps_memorize_outcome(
    monkeypatch: pytest.MonkeyPatch,
    memorized: Literal["accumulated", "extracted", "degraded"],
    flushed: str,
    permanent: int,
) -> None:
    service = make_memory_service(stores=make_stores(memorize_inbox=_EmptyInbox()))
    calls: list[tuple[dict[str, Any], bool]] = []

    async def fake_memorize(
        payload: dict[str, Any], *, is_final: bool = False, **_: Any
    ) -> MemorizeResult:
        calls.append((payload, is_final))
        return MemorizeResult(
            message_count=0,
            status=memorized,
            failed_permanent=permanent,
            failed_retryable=1 if memorized == "degraded" else 0,
        )

    monkeypatch.setattr(service, "_memorize", fake_memorize)

    result = await service.flush(session_id="s1", request_id="r1")

    assert result.status == flushed
    assert result.failed_permanent == (permanent if flushed == "degraded" else 0)
    assert result.failed_retryable == (1 if flushed == "degraded" else 0)
    assert calls == [
        (
            {
                "session_id": "s1",
                "app_id": "default",
                "project_id": "default",
                "messages": [],
            },
            True,
        )
    ]
