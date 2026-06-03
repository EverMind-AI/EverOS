"""Pipeline contract — every concrete pipeline implements ``Pipeline``."""

from __future__ import annotations

from typing import Protocol

from everos.memory import IngestResult, PipelineOutcome


class Pipeline(Protocol):
    """Asynchronous extraction pipeline.

    Each implementation owns its own buffer slice (one ``track``) and is
    invoked concurrently with siblings by ``service.memorize()``.
    """

    async def run(
        self,
        ingested: IngestResult,
        *,
        is_final: bool = False,
    ) -> PipelineOutcome: ...
