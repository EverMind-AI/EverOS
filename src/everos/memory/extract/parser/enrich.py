"""Parse non-text content items via component.parser, backfilling in place.

Delegates actual parsing to :func:`everos.component.parser.aparse_file` which
handles LLM injection and error mapping. This module owns the batch
concurrency and per-item degradation logic specific to the add/ingest path.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from everos.core.errors import LLMServiceError, UnsupportedModalityError
from everos.core.observability.logging import get_logger

from .availability import needs_parser

if TYPE_CHECKING:
    from everalgo.llm import LLMClient

    from everos.component.parser import RawFileBuilder

logger = get_logger(__name__)


async def enrich_content_items(
    items: list[dict[str, Any]],
    *,
    llm: LLMClient | None,
    raw_file_builder: RawFileBuilder,
    max_concurrency: int = 4,
) -> None:
    """Parse each non-text item and backfill ``parsed_content`` in place.

    Synchronous to the request; items parse concurrently under a bounded
    semaphore. Deterministic failures (unsupported modality, missing system
    dependency) propagate and abort the batch; a failed parser model call
    degrades per item (``parse_status="failed"``) without dropping the rest --
    counted by ingest as a failed item. ``aparse_file`` reports that call as
    :class:`~everos.core.errors.LLMServiceError` (the provider's
    ``LLMCallError`` is its subclass), so that is what is caught: catching only
    the subclass let every failed parse fail the whole request. A picture
    ingest refused is skipped (:func:`needs_parser`).

    Args:
        items: ContentItem dicts (mutated in place).
        llm: The multimodal LLM the parser reads with; ``None`` when
            ``[multimodal]`` is not configured, which fails the batch.
        raw_file_builder: Hydrates parser input using host file-access policy.
        max_concurrency: Upper bound on concurrent parse calls.
    """
    from everos.component.parser import aparse_file  # Deferred: optional dep

    targets = [item for item in items if needs_parser(item)]
    if not targets:
        return

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _parse_one(item: dict[str, Any]) -> None:
        async with semaphore:
            try:
                raw = await raw_file_builder(item)
            except ValueError as exc:
                raise UnsupportedModalityError(str(exc)) from exc
            try:
                parsed = await aparse_file(raw, llm=llm)
            except LLMServiceError as exc:
                item["parse_status"] = "failed"
                item["parse_error"] = "LLMError"
                logger.warning(
                    "multimodal_parse_failed",
                    content_type=item.get("type"),
                    error=f"{type(exc).__name__}: {exc}"[:200],
                )
                return
            item["parsed_content"] = parsed.text
            item["parse_status"] = "success"

    await asyncio.gather(*(_parse_one(item) for item in targets))
