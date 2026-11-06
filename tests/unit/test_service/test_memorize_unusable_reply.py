"""A cell whose extraction never got a usable LLM reply fails as an LLM error.

An operator raises ``LLMOutputError`` when it rejects the model's reply, and
``retry_llm_output`` re-raises the last one. Leaving it as is turned
``/memory/flush`` into a 500 (#564); ``memorize`` re-raises it as an
``LLMServiceError``, which the API maps to a retryable 503. Any other
pipeline failure propagates unchanged.
"""

from __future__ import annotations

from typing import Any

import pytest
from everalgo.llm import LLMOutputError

from everos.core.errors import LLMServiceError
from everos.memory import IngestResult
from everos.memory.extract.boundary import BoundaryOutcome
from everos.memory.extract.pipeline import PreparedCell
from tests.fakes import make_memory_service


class _OneCellBoundary:
    """A boundary stage that cuts one memcell on every call."""

    async def detect(self, ingested: IngestResult, **_: Any) -> BoundaryOutcome:
        return BoundaryOutcome(
            cells=[object()],  # type: ignore[list-item]
            memcell_ids=["mc_1"],
            per_cell_message_ids=[["m1"]],
            per_cell_all_senders=[["u1"]],
            status="extracted",
            message_count=1,
        )


class _FailingPipeline:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def run_cell(self, ingested: IngestResult, prepared: PreparedCell) -> Any:
        raise self._exc


async def _flush(exc: Exception) -> None:
    pipeline = _FailingPipeline(exc)
    service = make_memory_service(
        boundary=_OneCellBoundary(), user_pipeline=pipeline, agent_pipeline=pipeline
    )
    await service._memorize({"session_id": "s1", "messages": []}, is_final=True)


async def test_unusable_reply_raises_llm_service_error() -> None:
    reply_error = LLMOutputError("episode reply is not a valid _EpisodeReply")
    with pytest.raises(LLMServiceError) as info:
        await _flush(reply_error)
    assert info.value.__cause__ is reply_error
    assert "episode reply is not a valid _EpisodeReply" in str(info.value)


async def test_other_pipeline_failure_propagates_unchanged() -> None:
    with pytest.raises(RuntimeError, match="disk full"):
        await _flush(RuntimeError("disk full"))
