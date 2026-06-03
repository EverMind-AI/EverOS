"""Parse non-text content items via everalgo.parser, backfilling in place.

The ``everalgo.parser`` import is deferred to call time so importing this
module never requires the optional ``everos[multimodal]`` extra. The ingest
stage calls :func:`require_multimodal` first, so a missing extra surfaces the
guided install error before this runs.
"""

from __future__ import annotations

import asyncio
from typing import Any

from everalgo.llm import LLMError
from everalgo.llm.protocols import LLMClient

from everos.core.errors import MultimodalNotEnabledError, UnsupportedModalityError
from everos.core.observability.logging import get_logger

from .mapping import build_raw_file

logger = get_logger(__name__)


async def enrich_content_items(
    items: list[dict[str, Any]], *, llm: LLMClient, max_concurrency: int = 4
) -> None:
    """Parse each non-text item and backfill ``parsed_content`` in place.

    Synchronous to the request; items parse concurrently under a bounded
    semaphore. Deterministic failures (unsupported modality, missing system
    dependency) raise a :class:`~everos.core.errors.MultimodalError` subclass
    and abort the batch; transient failures (LLM errors) degrade per item
    (``parse_status="failed"``) without dropping the rest.

    Args:
        items: ContentItem dicts (mutated in place).
        llm: Multimodal LLM client passed to ``everalgo.parser.aparse``.
        max_concurrency: Upper bound on concurrent parse calls.
    """
    from everalgo.parser import aparse  # optional dependency, imported lazily

    targets = [
        item
        for item in items
        if item.get("type") != "text" and "parsed_content" not in item
    ]
    if not targets:
        return

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _parse_one(item: dict[str, Any]) -> None:
        async with semaphore:
            try:
                parsed = await aparse(await build_raw_file(item), llm=llm)
            except NotImplementedError as exc:
                raise UnsupportedModalityError(
                    f"modality not supported: {item.get('type')!r}"
                ) from exc
            except LLMError as exc:
                # Transient: degrade this item, keep the rest of the batch.
                item["parse_status"] = "failed"
                item["parse_error"] = type(exc).__name__
                logger.warning(
                    "multimodal_parse_failed",
                    extra={"content_type": item.get("type")},
                )
                return
            except ValueError as exc:
                # everalgo dispatch / mapping rejected the input.
                raise UnsupportedModalityError(str(exc)) from exc
            except RuntimeError as exc:
                # e.g. LibreOffice missing for Office documents.
                raise MultimodalNotEnabledError(str(exc)) from exc
            item["parsed_content"] = parsed.text
            item["parse_status"] = "success"

    await asyncio.gather(*(_parse_one(item) for item in targets))
