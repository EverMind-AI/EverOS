"""Multimodal parser output contract."""

from __future__ import annotations

__all__ = ["ParsedContent"]

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from everos.algo.types.modality import Modality


class ParsedContent(BaseModel):
    """Normalized output of any parser submodule (image / audio / pdf / ...).

    The shape is intentionally flat and provider-agnostic so callers can treat
    every modality uniformly: pull ``text`` for indexing, ``modality`` /
    ``mime`` for routing, ``metadata`` for modality-specific extras (page
    count, audio duration, image dimensions, ...).

    Examples:
    --------
    >>> from everos.algo.types import Modality, ParsedContent
    >>> ParsedContent(
    ...     text="extracted text...",
    ...     modality=Modality.PDF,
    ...     mime="application/pdf",
    ...     metadata={"page_count": 12},
    ... )
    """

    text: str = Field(default="", description="Extracted text content (main payload).")
    modality: Modality = Field(
        default=Modality.UNKNOWN,
        description="The modality classification the parser handled.",
    )
    mime: str = Field(
        default="", description="Source MIME type, e.g. ``application/pdf``."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Modality-specific extras the parser chose to surface "
            "(e.g. page_count / duration_seconds / dimensions). "
            "Schema is open by design — parsers populate only what they know."
        ),
    )

    model_config = ConfigDict(extra="ignore")
