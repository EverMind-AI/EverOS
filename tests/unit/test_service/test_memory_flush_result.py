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
    ("memorized", "flushed"),
    [("accumulated", "no_extraction"), ("extracted", "extracted")],
)
async def test_flush_maps_memorize_outcome(
    monkeypatch: pytest.MonkeyPatch,
    memorized: Literal["accumulated", "extracted"],
    flushed: str,
) -> None:
    service = make_memory_service(stores=make_stores(memorize_inbox=_EmptyInbox()))
    calls: list[tuple[dict[str, Any], bool]] = []

    async def fake_memorize(
        payload: dict[str, Any], *, is_final: bool = False, **_: Any
    ) -> MemorizeResult:
        calls.append((payload, is_final))
        return MemorizeResult(message_count=0, status=memorized)

    monkeypatch.setattr(service, "_memorize", fake_memorize)

    result = await service.flush(session_id="s1", request_id="r1")

    assert result.status == flushed
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


async def test_deferred_add_triggers_auto_flush_at_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a deferred add pushes the session buffer to >=12 messages, a supervised
    background flush is spawned automatically."""
    import asyncio
    from everos.memory.models import CanonicalMessage
    from everos.core.utils.datetime import get_utc_now

    class _ThresholdBuffer:
        async def list(self, session: Any) -> list[CanonicalMessage]:
            return [
                CanonicalMessage(
                    message_id=f"m_{i}",
                    session_id="s1",
                    sender_id="u1",
                    role="user",
                    timestamp=get_utc_now(),
                    text=f"msg {i}",
                )
                for i in range(12)
            ]

    service = make_memory_service(
        stores=make_stores(
            memorize_inbox=_EmptyInbox(),
            session_buffer=_ThresholdBuffer(),
        )
    )
    calls: list[tuple[dict[str, Any], bool, bool]] = []

    async def fake_memorize(
        payload: dict[str, Any],
        *,
        is_final: bool = False,
        defer_extraction: bool = False,
    ) -> MemorizeResult:
        calls.append((payload, is_final, defer_extraction))
        return MemorizeResult(message_count=len(payload.get("messages", [])), status="accumulated")

    monkeypatch.setattr(service, "_memorize", fake_memorize)
    await service.add(
        {"session_id": "s1", "messages": [{"content": "hi"}]},
        request_id="r1",
        async_mode=False,
        defer_extraction=True,
    )
    await asyncio.sleep(0.05)
    assert len(calls) == 2
    assert calls[0][1:] == (False, True)
    assert calls[1][1:] == (True, False)
