"""Multimodal parser — top-level dispatch by modality.

Public surface (re-exports for ergonomic ``from everos.component.parser.backend import ...``):

- Functions: ``aparse`` — the dispatch entry point.
- Types: ``ParsedContent`` / ``Modality`` / ``RawFile`` (re-exported from
  ``everos.algo.types`` for one-stop import).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from PIL import Image

from everos.algo.types import (
    Modality,
    ParsedContent,
    RawFile,
    get_extension_from_mime,
    get_modality,
    get_modality_from_mime,
)
from everos.algo.utils import InvalidInputError
from everos.component.parser.backend import audio, document, image, url

if TYPE_CHECKING:
    from everos.algo.llm import LLMClient

__all__ = [
    "Modality",
    "ParsedContent",
    "RawFile",
    "aparse",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Explicit decompression-bomb cap (Pillow default raises ``DecompressionBombError``
# only at ~2x this threshold). 100M pixels = 10000x10000, ample for any realistic
# screenshot / scan while keeping a hostile caller from exhausting memory via a
# crafted PNG/TIFF. Applications can override by re-assigning
# ``PIL.Image.MAX_IMAGE_PIXELS`` after importing the parser.
Image.MAX_IMAGE_PIXELS = 100_000_000


async def aparse(raw_file: RawFile, *, llm: LLMClient) -> ParsedContent:
    """Top-level dispatch: classify the input and route to a submodule.

    Dispatch precedence:

    1. ``content`` empty + ``uri`` is ``http``/``https`` → ``url.aparse``
       (the URL parser fetches and re-dispatches by the response Content-Type).
    2. ``content`` non-empty → resolve a ``Modality`` from
       ``extension`` first, falling back to ``mime`` when extension is
       missing or ``UNKNOWN``.
    3. Submodule routing by resolved ``Modality``:
       - ``IMAGE`` → ``image.aparse``
       - ``AUDIO`` → ``audio.aparse``
       - ``PDF`` / ``DOCUMENT`` / ``HTML`` / ``EMAIL`` → ``document.aparse``
       - ``DIRECT`` → no LLM; bytes are UTF-8-decoded inline.
       - Anything else → ``InvalidInputError``.

    When dispatch resolves a modality via ``mime`` but the submodule
    internally requires ``extension`` (image / audio / document all do for
    their ``_MIME_MAP`` lookups), this function fills ``extension`` in
    from the canonical extension table before forwarding.

    Parameters
    ----------
    raw_file : RawFile
        Hydrated ``content`` or ``uri`` (at least one is required).
    llm : LLMClient
        Per-call LLM client. Required for all LLM-backed modalities (IMAGE, AUDIO, PDF, etc.).

    Returns:
    -------
    ParsedContent
        Normalized parser output.
    """
    # URL path: only when caller did not pre-hydrate content.
    if not raw_file.content and raw_file.uri and _is_http_uri(raw_file.uri):
        return await url.aparse(raw_file, llm=llm)

    if not raw_file.content:
        raise InvalidInputError(
            "RawFile has neither content nor an http(s) uri, so there is nothing to parse: "
            "pass the file bytes in content, or an http(s) uri for the URL parser to fetch"
        )

    # Resolve modality: extension wins; fall back to mime.
    modality = (
        get_modality(raw_file.extension) if raw_file.extension else Modality.UNKNOWN
    )
    if modality is Modality.UNKNOWN and raw_file.mime:
        modality = get_modality_from_mime(raw_file.mime)
    if modality is Modality.UNKNOWN:
        raise InvalidInputError(
            f"RawFile extension={raw_file.extension!r} mime={raw_file.mime!r} map to no known modality: "
            "set a supported extension or a matching mime before calling"
        )

    # Ensure ``extension`` is populated for downstream submodules that key
    # their internal MIME tables off it (image / audio / document do).
    routed = _ensure_extension(raw_file)

    if modality is Modality.IMAGE:
        return await image.aparse(routed, llm=llm)
    if modality is Modality.AUDIO:
        return await audio.aparse(routed, llm=llm)
    if modality in (Modality.PDF, Modality.DOCUMENT, Modality.HTML, Modality.EMAIL):
        return await document.aparse(routed, llm=llm)
    if modality is Modality.DIRECT:
        return ParsedContent(
            text=routed.content.decode("utf-8", errors="replace"),
            modality=Modality.DIRECT,
            mime=routed.mime,
        )
    raise InvalidInputError(
        f"RawFile modality {modality.value!r} (extension={raw_file.extension!r}, mime={raw_file.mime!r}) "
        "is not parseable: this parser handles image, audio, document and plain-text files only"
    )


# ---- helpers ----


def _is_http_uri(uri: str) -> bool:
    """Return True for ``http`` / ``https`` schemes only."""
    return urlparse(uri).scheme in ("http", "https")


def _ensure_extension(raw_file: RawFile) -> RawFile:
    """Return a ``RawFile`` whose ``extension`` is populated.

    If ``extension`` is missing but ``mime`` resolves via ``MIME_TO_EXTENSION``,
    we hand the submodule a copy carrying the canonical extension. Returns
    the original when ``extension`` is already set or no mime fallback exists.
    """
    if raw_file.extension:
        return raw_file
    derived = get_extension_from_mime(raw_file.mime)
    if not derived:
        return raw_file
    return raw_file.model_copy(update={"extension": derived})
