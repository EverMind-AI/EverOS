"""Flush and add do not say ``extracted`` when the files they wrote failed to index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from everos.infra.cascade.projection import ProjectionVerdict
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown import MarkdownWriter, collect_md_writes
from everos.service import MemorizeResult
from tests.fakes import make_memory_service, make_stores

_EPISODE = "default/default/users/u/episodes/episode-2026-11-07.md"


class _EmptyInbox:
    async def count_pending(self, session: Any = None) -> int:
        return 0


class _Projection:
    def __init__(self, verdict: ProjectionVerdict) -> None:
        self.verdict = verdict
        self.calls: list[set[str]] = []

    async def await_written(self, paths: set[str]) -> ProjectionVerdict:
        self.calls.append(set(paths))
        return self.verdict


def _service(projection: _Projection | None) -> Any:
    return make_memory_service(
        stores=make_stores(memorize_inbox=_EmptyInbox()),
        index_projection=projection,
        collect_writes=collect_md_writes,
    )


async def _write(tmp_path: Path) -> None:
    """Write one episode while a collection is active, so the call records it."""
    await MarkdownWriter(MemoryRoot(tmp_path)).write(tmp_path / _EPISODE, "episode\n")


async def test_flush_is_degraded_when_the_written_file_fails_to_index(
    tmp_path: Path, monkeypatch: Any
) -> None:
    projection = _Projection(
        ProjectionVerdict(observed=True, failed_permanent=1, failed_retryable=0)
    )
    service = _service(projection)

    async def fake_run(*_a: Any, **_k: Any) -> MemorizeResult:
        await _write(tmp_path)
        return MemorizeResult(message_count=0, status="extracted")

    monkeypatch.setattr(service._memorizer, "run", fake_run)

    result = await service.flush(session_id="s1", request_id="r1")

    assert result.status == "degraded"
    assert result.status != "extracted"
    assert result.failed_permanent == 1
    assert result.failed_retryable == 0
    assert projection.calls == [{_EPISODE}]


async def test_add_is_degraded_when_the_index_write_is_retryable(
    tmp_path: Path, monkeypatch: Any
) -> None:
    projection = _Projection(
        ProjectionVerdict(observed=True, failed_permanent=0, failed_retryable=2)
    )
    service = _service(projection)

    async def fake_run(*_a: Any, **_k: Any) -> MemorizeResult:
        await _write(tmp_path)
        return MemorizeResult(message_count=1, status="extracted")

    monkeypatch.setattr(service._memorizer, "run", fake_run)

    result = await service.add(
        {
            "session_id": "s1",
            "app_id": "default",
            "project_id": "default",
            "messages": [{"content": "x"}],
        },
        request_id="r1",
        async_mode=False,
        defer_extraction=False,
    )

    assert result.status == "degraded"
    assert result.failed_retryable == 2
    assert result.failed_permanent == 0


async def test_extracted_stands_when_the_projection_has_not_failed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Lag is not a failure: a row still pending keeps ``extracted``."""
    projection = _Projection(ProjectionVerdict(observed=True, pending=1))
    service = _service(projection)

    async def fake_run(*_a: Any, **_k: Any) -> MemorizeResult:
        await _write(tmp_path)
        return MemorizeResult(message_count=0, status="extracted")

    monkeypatch.setattr(service._memorizer, "run", fake_run)

    result = await service.flush(session_id="s1", request_id="r1")

    assert result.status == "extracted"
    assert result.failed_permanent == 0


async def test_no_markdown_write_does_not_ask_the_projection(monkeypatch: Any) -> None:
    projection = _Projection(ProjectionVerdict(observed=True, failed_permanent=9))
    service = _service(projection)

    async def fake_run(*_a: Any, **_k: Any) -> MemorizeResult:
        return MemorizeResult(message_count=0, status="extracted")

    monkeypatch.setattr(service._memorizer, "run", fake_run)

    result = await service.flush(session_id="s1", request_id="r1")

    assert result.status == "extracted"
    assert projection.calls == []
