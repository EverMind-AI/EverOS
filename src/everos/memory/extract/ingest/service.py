"""Ingest pipeline entry — normalise external input into canonical form.

Input shape (received from the service layer, decoupled from any
specific DTO module):

    {
      "session_id": "...",
      "messages": [
        {
          "sender_id": "...",
          "sender_name": "...",        # optional
          "role": "user" | "assistant" | "tool",
          "timestamp": 1740564000000,  # unix ms
          "content": "..." | [ContentItem dicts],
          "tool_calls": [...] | None,  # OpenAI shape
          "tool_call_id": "..." | None,
        },
        ...
      ]
    }

Output: :class:`everos.memory.IngestResult`.

Pictures are handled first, before anything renders text: each one is stored
by content hash and the item is annotated with its id, so the derived ``text``
carries ``[IMAGE: <id>]`` where the picture was and the extraction model can be
handed the picture itself beside that reference. A picture replaces the
alternative of parsing it into prose at ingest — the same information, read by
the model that has to remember it rather than by a captioner that does not know
what will be asked.

What was dropped is counted, each item once: ``rejected_count`` for a picture
refused at ingest (over a limit, not a picture, unreadable -- never handed to
a model), ``failed_count`` for a non-text item that could not be turned into
text (a parser call that failed). ``/add`` returns both. One bad upload never
fails the request. With ``[extraction] vision = false`` pictures are parser
input, not pictures to store: they are neither stored nor refused.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from everos.component.images import ImageLimits
from everos.config import load_settings
from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import from_timestamp
from everos.memory import CanonicalMessage, IngestResult, ToolCall
from everos.memory.extract.parser import (
    enrich_content_items,
    has_unparsed_multimodal,
    require_multimodal,
)

from .id_gen import gen_message_id
from .multimodal import (
    IMAGE_TYPE,
    PictureBudget,
    coerce_items,
    derive_text,
    store_images,
)

if TYPE_CHECKING:
    from everalgo.llm import LLMClient

    from everos.component.parser import RawFileBuilder
    from everos.memory.ports.images import ImageStore

logger = get_logger(__name__)


async def process(
    payload: dict[str, Any],
    *,
    multimodal_llm: LLMClient | None,
    raw_file_builder: RawFileBuilder,
    image_store: ImageStore | None = None,
    image_limits: ImageLimits | None = None,
    vision: bool = True,
) -> IngestResult:
    """Normalise the raw add payload into an :class:`IngestResult`.

    Args:
        payload: The raw add payload.
        multimodal_llm: Parses non-text content items; ``None`` when
            ``[multimodal]`` is not configured.
        raw_file_builder: Host input hydration, passed to each unparsed item.
        image_store: Where pictures are stored; ``None`` disables the picture
            path and every picture is reported as rejected.
        image_limits: ``[images]`` bounds; read from settings when omitted.
        vision: Whether the extraction model reads images
            (``[extraction] vision``). False routes pictures through the
            parser into text instead of storing them.

    Returns:
        The canonical messages, with ``rejected_count`` (pictures refused)
        and ``failed_count`` (non-text items the parser could not read).
    """
    session_id: str = payload["session_id"]
    app_id: str = payload.get("app_id") or "default"
    project_id: str = payload.get("project_id") or "default"
    group_id: str | None = payload.get("group_id") or None
    raw_messages: list[dict[str, Any]] = payload["messages"]
    limits = image_limits or _image_limits()

    # One budget for the whole request, not one per message: the bound exists
    # because every picture costs the extraction model visual tokens, and that
    # cost is per request.
    request_budget = PictureBudget(
        max_bytes=limits.max_request_bytes,
        max_pictures=limits.max_per_request,
    )

    canonical: list[CanonicalMessage] = []
    failed = 0
    rejected = 0
    for idx, m in enumerate(raw_messages):
        content_items = coerce_items(m["content"])
        # With vision off a picture is parser input, like a pdf: not stored,
        # not refused.
        if vision and any(item.get("type") == IMAGE_TYPE for item in content_items):
            await store_images(
                content_items,
                store=image_store,
                limits=limits,
                raw_file_builder=raw_file_builder,
                budget=request_budget,
            )
        if has_unparsed_multimodal(content_items):
            require_multimodal()
            await enrich_content_items(
                content_items,
                llm=multimodal_llm,
                raw_file_builder=raw_file_builder,
                max_concurrency=load_settings().multimodal.max_concurrency,
            )
        text, non_text = derive_text(content_items)
        refused = sum(1 for item in content_items if item.get("status") == "rejected")
        rejected += refused
        failed += non_text - refused

        ts_ms: int = int(m["timestamp"])
        message_id = gen_message_id(session_id, ts_ms, idx)
        ts = from_timestamp(ts_ms)

        canonical.append(
            CanonicalMessage(
                message_id=message_id,
                session_id=session_id,
                sender_id=m["sender_id"],
                sender_name=m.get("sender_name"),
                role=m["role"],
                timestamp=ts,
                content_items=content_items,
                text=text,
                tool_calls=_coerce_tool_calls(m.get("tool_calls")),
                tool_call_id=m.get("tool_call_id"),
                # The caller's id is kept for a group batch only: the
                # one-to-one path stays exactly as it was.
                client_message_id=m.get("message_id") if group_id else None,
            )
        )

    if rejected or failed:
        logger.warning(
            "memorize.ingest_dropped",
            session_id=session_id,
            rejected=rejected,
            failed=failed,
        )
    return IngestResult(
        session_id=session_id,
        app_id=app_id,
        project_id=project_id,
        messages=canonical,
        unparsed_non_text_count=rejected + failed,
        rejected_count=rejected,
        failed_count=failed,
        group_id=group_id,
    )


def _image_limits() -> ImageLimits:
    """``[images]`` as the store's bounds."""
    settings = load_settings().images
    return ImageLimits(
        max_bytes=settings.max_bytes,
        max_per_message=settings.max_per_message,
        max_request_bytes=settings.max_request_bytes,
        max_per_request=settings.max_per_request,
        formats=tuple(settings.formats),
    )


def _coerce_tool_calls(
    raw: list[dict[str, Any]] | list[Any] | None,
) -> list[ToolCall] | None:
    """Convert raw tool-call dicts to ToolCall models."""
    if not raw:
        return None
    out: list[ToolCall] = []
    for tc in raw:
        if isinstance(tc, ToolCall):
            out.append(tc)
        elif hasattr(tc, "model_dump"):
            out.append(ToolCall.model_validate(tc.model_dump()))
        else:
            out.append(ToolCall.model_validate(tc))
    return out
