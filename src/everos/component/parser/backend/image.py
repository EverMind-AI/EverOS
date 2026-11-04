"""Image parser — multimodal LLM OCR with tall-image splitting.

- Normal-ratio images go through a single LLM call.
- Tall screenshots (height/width > 10) are split into overlapping slices
  via ``_utils.split_image_with_overlap``; each slice is OCR'd independently
  and the per-slice results are merged with a second LLM call using
  ``PROMPT_FOR_MERGE`` to dedupe the overlap regions.
- Every call sends the rules as the system message and the image (or the
  tagged OCR texts) as the user message, and reads the reply's ``text``.
- BMP / TIFF are transcoded to PNG via Pillow (Gemini does not accept those
  MIMEs natively).
- SVG is rasterised to PNG via ``cairosvg`` (the ``everos[multimodal]`` extra).
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

from PIL import Image

from everos.algo.llm import image_url_part_from_bytes
from everos.algo.types import Modality, ParsedContent
from everos.algo.utils import InvalidInputError, render_prompt, tagged
from everos.component.parser.backend._replies import _call_llm_for_text
from everos.component.parser.backend._utils import (
    check_aspect_ratio,
    split_image_with_overlap,
)
from everos.component.parser.backend.prompts.en import (
    MERGE_USER_PROMPT,
    PROMPT_FOR_MERGE,
    PROMPT_FOR_PICTURE,
)

if TYPE_CHECKING:
    from everos.algo.llm import LLMClient
    from everos.algo.types import RawFile

__all__ = ["aparse"]

logger = logging.getLogger(__name__)


_MIME_MAP: dict[str, str] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "svg": "image/svg+xml",
    "tif": "image/tiff",
    "tiff": "image/tiff",
}

# Formats Gemini multimodal does not accept natively. We transcode to PNG via
# Pillow before sending.
_TRANSCODE_TO_PNG: frozenset[str] = frozenset({"bmp", "tif", "tiff"})

# SVG is vector; rasterise via cairosvg before sending.
_SVG_EXTENSIONS: frozenset[str] = frozenset({"svg"})

_TALL_IMAGE_RATIO_THRESHOLD = 10.0

# Token budgets. Per-slice OCR caps at 8K; the merge step gets 4x because it
# joins multiple OCR'd chunks and the merged text is longer than any individual slice.
_MAX_OCR_OUTPUT_TOKENS = 8_000
_MAX_MERGE_OUTPUT_TOKENS = 32_000


async def aparse(raw_file: RawFile, *, llm: LLMClient) -> ParsedContent:
    """Parse an image ``RawFile`` into ``ParsedContent`` via multimodal LLM.

    Tall images (height/width > 10) are split into overlapping vertical
    slices and OCR'd in turn; per-slice results are then merged with a
    second LLM call. Normal-ratio images go through a single OCR call.

    Raises:
        InvalidInputError: If ``raw_file`` has no content or an unsupported extension.
        LLMOutputError: If an OCR reply is not a valid ``{"text": "..."}`` object.
    """
    if not raw_file.content:
        raise InvalidInputError(
            "RawFile.content is empty, there is nothing to read as an image: pass the image bytes in content"
        )

    extension = raw_file.extension.lower().lstrip(".")
    mime = _MIME_MAP.get(extension)
    if mime is None:
        raise InvalidInputError(
            f"RawFile.extension {raw_file.extension!r} is not a supported image extension: "
            f"set it to one of {sorted(_MIME_MAP)}"
        )

    payload_bytes = raw_file.content
    payload_mime = mime
    if extension in _TRANSCODE_TO_PNG:
        payload_bytes = _transcode_to_png(raw_file.content)
        payload_mime = "image/png"
    elif extension in _SVG_EXTENSIONS:
        payload_bytes = _rasterise_svg_to_png(raw_file.content)
        payload_mime = "image/png"

    needs_split, ratio = check_aspect_ratio(
        payload_bytes, max_ratio=_TALL_IMAGE_RATIO_THRESHOLD
    )
    if needs_split:
        return await _aparse_tall(payload_bytes, payload_mime, raw_file, ratio, llm)
    return await _aparse_single(payload_bytes, payload_mime, raw_file, llm)


# ---- internals ----


async def _aparse_single(
    payload_bytes: bytes,
    payload_mime: str,
    raw_file: RawFile,
    client: LLMClient,
) -> ParsedContent:
    """Single-shot OCR call for normal-ratio images.

    Raises:
        LLMOutputError: If the OCR reply is not a valid ``{"text": "..."}`` object.
    """
    logger.debug(
        "parsing image (single) bytes=%d mime=%s", len(payload_bytes), payload_mime
    )
    text = await _ocr(client, payload_bytes, payload_mime)
    return ParsedContent(
        text=text,
        modality=Modality.IMAGE,
        mime=raw_file.mime or payload_mime,
        metadata={"tall_image_parts": 1},
    )


async def _aparse_tall(
    payload_bytes: bytes,
    payload_mime: str,
    raw_file: RawFile,
    ratio: float,
    client: LLMClient,
) -> ParsedContent:
    r"""Split-then-merge OCR for tall screenshots.

    1. Split into N overlapping vertical slices.
    2. OCR each slice; **drop empty results** so the merge step sees only
       real text (an all-empty slice from a blank PDF region would otherwise
       confuse the merge LLM).
    3. ≤ 1 non-empty result → return that single text (or empty string when
       all slices came back blank).
    4. Otherwise call the merge LLM with ``temperature=0.0`` and a 4x token
       cap; on merge failure (the call raises, or its reply is invalid or
       empty) fall back to a deterministic ``\n\n`` join so the caller at
       least gets the raw OCR rather than an exception.

    Raises:
        LLMOutputError: If a slice's OCR reply is not a valid ``{"text": "..."}`` object.
    """
    parts = split_image_with_overlap(
        payload_bytes, max_ratio=_TALL_IMAGE_RATIO_THRESHOLD
    )
    logger.info(
        "tall image detected: ratio=%.2f, splitting into %d parts", ratio, len(parts)
    )

    ocr_results: list[str] = []
    for i, part_bytes in enumerate(parts, 1):
        logger.debug(
            "OCR tall image part %d/%d, bytes=%d", i, len(parts), len(part_bytes)
        )
        # Drop empty per-slice OCR so merge prompt isn't padded with blank parts.
        text = (await _ocr(client, part_bytes, payload_mime)).strip()
        if text:
            ocr_results.append(text)

    metadata: dict[str, object] = {
        "tall_image_parts": len(parts),
        "aspect_ratio": round(ratio, 2),
    }

    # All slices came back empty — return an empty ParsedContent rather than
    # call the merge LLM with no content (matches original ``parse()`` bailout).
    if not ocr_results:
        metadata["warning"] = "all OCR slices returned empty"
        return ParsedContent(
            text="",
            modality=Modality.IMAGE,
            mime=raw_file.mime or payload_mime,
            metadata=metadata,
        )

    # Single non-empty result — skip the merge LLM, return as-is (saves 1 call).
    if len(ocr_results) == 1:
        return ParsedContent(
            text=ocr_results[0],
            modality=Modality.IMAGE,
            mime=raw_file.mime or payload_mime,
            metadata=metadata,
        )

    merged_text = await _merge(client, ocr_results)
    if merged_text is None:
        merged_text = "\n\n".join(ocr_results)
        metadata["merge_fallback"] = "deterministic_join"
    return ParsedContent(
        text=merged_text,
        modality=Modality.IMAGE,
        mime=raw_file.mime or payload_mime,
        metadata=metadata,
    )


async def _ocr(client: LLMClient, image_bytes: bytes, mime: str) -> str:
    """OCR one image: the rules as the system message, the image alone as the user message.

    Raises:
        LLMOutputError: If the reply is not a valid ``{"text": "..."}`` object.
    """
    return await _call_llm_for_text(
        client,
        system=PROMPT_FOR_PICTURE,
        user=[image_url_part_from_bytes(image_bytes, mime)],
        operator="parser image",
        max_tokens=_MAX_OCR_OUTPUT_TOKENS,
    )


async def _merge(client: LLMClient, ocr_results: list[str]) -> str | None:
    """Merge the slices' OCR texts with the LLM; None when the merge fails or comes back empty.

    Algorithm-layer fallback (not an engineering retry): a merge
    that raises — the call itself, or an invalid reply — or returns no text leaves
    the caller to join the raw OCR deterministically, so it never loses it.
    """
    logger.debug("merging %d OCR parts", len(ocr_results))
    try:
        merged = await _call_llm_for_text(
            client,
            system=PROMPT_FOR_MERGE,
            user=_render_ocr_parts(ocr_results),
            operator="parser image merge",
            temperature=0.0,
            max_tokens=_MAX_MERGE_OUTPUT_TOKENS,
        )
    except Exception as exc:
        logger.warning(
            "LLM merge of tall-image OCR parts failed (%s); falling back to deterministic join",
            exc,
            exc_info=True,
        )
        return None
    return merged or None


def _render_ocr_parts(ocr_results: list[str]) -> str:
    """The merge call's user message: each OCR text in a numbered ``<part>``, all in ``<ocr_parts count="N">``."""
    parts = "\n".join(
        tagged("part", text, number=str(i)) for i, text in enumerate(ocr_results, 1)
    )
    return render_prompt(MERGE_USER_PROMPT, None, count=len(ocr_results), parts=parts)


def _transcode_to_png(data: bytes) -> bytes:
    """Decode arbitrary Pillow-readable bytes and re-encode as PNG."""
    with Image.open(io.BytesIO(data)) as src:
        img: Image.Image = (
            src.convert("RGBA" if "A" in src.mode else "RGB")
            if src.mode not in ("RGB", "RGBA")
            else src.copy()
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def _rasterise_svg_to_png(
    data: bytes,
) -> bytes:  # pragma: no cover  (cairosvg native, not in CI)
    """Rasterise SVG bytes to PNG via cairosvg (optional extra)."""
    try:
        import cairosvg  # type: ignore[import-not-found]  # pragma: no cover
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(  # pragma: no cover
            "SVG support requires the optional `cairosvg` dependency. Install via: pip install 'everos[multimodal]'"
        ) from exc
    return cairosvg.svg2png(bytestring=data)  # type: ignore[no-any-return]  # pragma: no cover
