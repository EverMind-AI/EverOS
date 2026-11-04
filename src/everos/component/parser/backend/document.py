"""Document parser — handles PDF / HTML / EMAIL / Office modalities.

The
``document`` submodule covers every non-image / non-audio file shape:

- ``Modality.PDF`` — sent as a single ``application/pdf`` data URI to the
  multimodal LLM.
- ``Modality.HTML`` — bs4-cleaned (``_utils.clean_html_for_llm``) before
  sending the structured HTML to the LLM with the HTML prompt.
- ``Modality.EMAIL`` — pure-python ``email`` stdlib parsing; when inline
  images are attached and an ``LLMClient`` is available, each image is
  OCR'd via ``image.aparse`` and the result substitutes the original
  ``<img src="cid:...">`` references (now ``[image:N]`` placeholders).
- ``Modality.DOCUMENT`` — Office / iWork / ODF; LibreOffice subprocess to
  convert to PDF, then reuse the PDF path. Requires ``soffice`` on PATH.

Every LLM call sends the rules as the system message and the file (or the
tagged HTML) as the user message, and reads the reply's ``text``.

This module is stateless: callers hydrate ``RawFile.content`` upstream and
pass it in. Retry, rate-limit, fallback are caller / deployment concerns.
"""

from __future__ import annotations

import asyncio
import email
import email.policy
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from everos.algo.llm import image_url_part_from_bytes
from everos.algo.types import Modality, ParsedContent, RawFile, get_modality
from everos.algo.utils import InvalidInputError, decode_bytes, render_prompt
from everos.component.parser.backend._replies import _call_llm_for_text
from everos.component.parser.backend._utils import (
    clean_html_content,
    clean_html_for_llm,
)
from everos.component.parser.backend.prompts.en import (
    HTML_USER_PROMPT,
    PROMPT_FOR_FILE,
    PROMPT_FOR_HTML,
    PROMPT_FOR_PICTURE,
)

if TYPE_CHECKING:
    from email.message import EmailMessage

    from everos.algo.llm import LLMClient

__all__ = ["aparse"]

logger = logging.getLogger(__name__)


_PDF_MIME = "application/pdf"
_HTML_MAX_INPUT_CHARS = 1_000_000
_MACOS_SOFFICE = "/Applications/LibreOffice.app/Contents/MacOS/soffice"

# Inline image OCR caps at 8K tokens (single-image scope, no need for the
# 32K merge budget used by the long-image stitcher).
_INLINE_OCR_MAX_TOKENS = 8_000
_INLINE_IMAGE_MIMETYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/webp",
        "image/bmp",
        "image/tiff",
        "image/svg+xml",
    }
)
_MIME_TO_EXTENSION = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
    "image/svg+xml": "svg",
}


async def aparse(raw_file: RawFile, *, llm: LLMClient) -> ParsedContent:
    """Parse a document-shaped ``RawFile`` into ``ParsedContent``.

    Raises:
        InvalidInputError: If ``raw_file`` has no content or is not a document modality.
        LLMOutputError: If a PDF, HTML or Office reply is not a valid ``{"text": "..."}`` object. An email's
            inline-image OCR never raises it: a failed image gets a placeholder instead.
    """
    if not raw_file.content:
        raise InvalidInputError(
            "RawFile.content is empty, there is nothing to parse as a document: pass the file bytes in content"
        )

    modality = get_modality(raw_file.extension)
    if modality is Modality.PDF:
        return await _aparse_pdf(raw_file, llm=llm)
    if modality is Modality.HTML:
        return await _aparse_html(raw_file, llm=llm)
    if modality is Modality.EMAIL:
        return await _aparse_email(raw_file, llm=llm)
    if modality is Modality.DOCUMENT:
        return await _aparse_office(raw_file, llm=llm)
    raise InvalidInputError(
        f"document.aparse got modality {modality.value!r} (extension={raw_file.extension!r}), which is not a document: "
        "route images, audio and URLs through the parser facade or their own module"
    )


# ---- per-modality handlers (private) ----


async def _aparse_pdf(raw_file: RawFile, *, llm: LLMClient) -> ParsedContent:
    """PDF → ``ParsedContent`` via a single multimodal LLM call."""
    logger.debug("parsing PDF, bytes=%d", len(raw_file.content))
    text = await _read_pdf(llm, raw_file.content, operator="parser pdf")
    return ParsedContent(
        text=text, modality=Modality.PDF, mime=raw_file.mime or _PDF_MIME
    )


async def _aparse_html(raw_file: RawFile, *, llm: LLMClient) -> ParsedContent:
    """HTML → ``ParsedContent`` via bs4 cleanup + LLM extraction."""
    html_text = decode_bytes(raw_file.content)
    cleaned = clean_html_for_llm(html_text)
    if not cleaned:
        logger.warning("HTML content is empty after cleaning")
        return ParsedContent(
            text="",
            modality=Modality.HTML,
            mime=raw_file.mime or "text/html",
            metadata={"warning": "empty after html cleanup"},
        )
    if len(cleaned) > _HTML_MAX_INPUT_CHARS:
        logger.warning(
            "HTML truncated from %d to %d chars", len(cleaned), _HTML_MAX_INPUT_CHARS
        )
        cleaned = cleaned[:_HTML_MAX_INPUT_CHARS]

    logger.debug(
        "parsing HTML, raw_chars=%d cleaned_chars=%d", len(html_text), len(cleaned)
    )
    text = await _call_llm_for_text(
        llm,
        system=PROMPT_FOR_HTML,
        user=render_prompt(HTML_USER_PROMPT, None, html=cleaned),
        operator="parser html",
    )
    return ParsedContent(
        text=text,
        modality=Modality.HTML,
        mime=raw_file.mime or "text/html",
        metadata={
            "raw_chars": len(html_text),
            "cleaned_chars": len(cleaned),
        },
    )


async def _aparse_email(raw_file: RawFile, *, llm: LLMClient) -> ParsedContent:
    """EML → ``ParsedContent`` via stdlib ``email`` + inline-image OCR.

    1. Extract headers.
    2. Collect inline images (Content-ID → bytes).
    3. Extract body: prefer ``text/plain`` when there are no cid images
       (cheaper); prefer ``text/html`` when cid images are present so we can
       substitute placeholders at their original positions.
    4. If inline images exist, OCR each one via the LLM and substitute the placeholders.
       Images without a cid get appended at end.
    """
    msg = email.message_from_bytes(raw_file.content, policy=email.policy.default)
    parts: list[str] = []

    headers = _extract_email_headers(msg)
    if headers:
        parts.append(headers)

    inline_images = _collect_inline_images(msg)
    has_cid = any(cid for cid, _, _, _ in inline_images.values())
    body = _extract_email_body(msg, inline_images, prefer_html=has_cid)
    if body:
        parts.append(body)

    metadata: dict[str, object] = {
        "from": msg.get("From", ""),
        "subject": msg.get("Subject", ""),
        "inline_image_count": len(inline_images),
    }

    if not inline_images:
        return ParsedContent(
            text="\n\n".join(parts),
            modality=Modality.EMAIL,
            mime=raw_file.mime or "message/rfc822",
            metadata=metadata,
        )

    ocr_results = await _ocr_inline_images(inline_images, client=llm)
    result = "\n\n".join(parts)
    appended: list[str] = []
    for idx, ocr_text in ocr_results.items():
        placeholder = f"[image:{idx}]"
        if placeholder in result:
            result = result.replace(placeholder, ocr_text)
        else:
            appended.append(ocr_text)
    if appended:
        result = result + "\n\n" + "\n\n".join(appended)

    metadata["inline_image_ocr"] = "ok"
    return ParsedContent(
        text=result,
        modality=Modality.EMAIL,
        mime=raw_file.mime or "message/rfc822",
        metadata=metadata,
    )


async def _aparse_office(
    raw_file: RawFile, *, llm: LLMClient
) -> ParsedContent:  # pragma: no cover  (libreoffice binary, not in CI)
    """Office docs (docx / xlsx / pptx / ...) → convert to PDF → reuse PDF path."""
    soffice = _find_soffice()  # pragma: no cover
    pdf_bytes = await _convert_to_pdf_via_soffice(
        raw_file.content, raw_file.extension, soffice
    )  # pragma: no cover
    if not pdf_bytes:  # pragma: no cover
        return ParsedContent(  # pragma: no cover
            text="",
            modality=Modality.DOCUMENT,
            mime=raw_file.mime,
            metadata={"warning": "LibreOffice conversion returned empty bytes"},
        )

    logger.debug(  # pragma: no cover
        "parsing Office doc extension=%s converted_pdf_bytes=%d",
        raw_file.extension,
        len(pdf_bytes),
    )
    text = await _read_pdf(llm, pdf_bytes, operator="parser office")  # pragma: no cover
    return ParsedContent(  # pragma: no cover
        text=text,
        modality=Modality.DOCUMENT,
        mime=raw_file.mime,
        metadata={"intermediate_pdf_bytes": len(pdf_bytes)},
    )


async def _read_pdf(llm: LLMClient, pdf_bytes: bytes, *, operator: str) -> str:
    """Read one PDF: the rules as the system message, the PDF alone as the user message.

    Raises:
        LLMOutputError: If the reply is not a valid ``{"text": "..."}`` object.
    """
    return await _call_llm_for_text(
        llm,
        system=PROMPT_FOR_FILE,
        user=[image_url_part_from_bytes(pdf_bytes, _PDF_MIME)],
        operator=operator,
    )


# ---- email helpers ----


def _extract_email_headers(msg: EmailMessage) -> str:
    lines: list[str] = []
    for field in ("Subject", "From", "To", "Cc", "Date"):
        value = msg.get(field)
        if value:
            lines.append(f"{field}: {value}")
    return "\n".join(lines)


def _collect_inline_images(msg: EmailMessage) -> dict[int, tuple[str, bytes, str, str]]:
    """Return ``{idx: (cid, image_bytes, mime, filename)}`` for inline images."""
    images: dict[int, tuple[str, bytes, str, str]] = {}
    idx = 1
    for part in msg.walk():
        content_type = part.get_content_type()
        if content_type not in _INLINE_IMAGE_MIMETYPES:
            continue
        image_bytes = part.get_content()
        if not isinstance(image_bytes, bytes) or not image_bytes:
            continue
        cid = part.get("Content-ID", "")
        if cid:
            cid = cid.strip("<>")
        filename = part.get_filename() or f"image_{idx}"
        images[idx] = (cid, image_bytes, content_type, filename)
        idx += 1
    return images


def _extract_email_body(
    msg: EmailMessage,
    inline_images: dict[int, tuple[str, bytes, str, str]],
    *,
    prefer_html: bool = False,
) -> str:
    """Extract body text, optionally replacing ``<img cid:>`` with placeholders."""
    if not prefer_html:
        body = msg.get_body(preferencelist=("plain",))
        if body is not None:
            content = body.get_content()
            if isinstance(content, str) and content.strip():
                return content.strip()

    body = msg.get_body(preferencelist=("html",))
    if body is not None:
        content = body.get_content()
        if isinstance(content, str) and content.strip():
            cid_to_placeholder = {
                cid: f"[image:{idx}]"
                for idx, (cid, _, _, _) in inline_images.items()
                if cid
            }
            html_with_placeholders = _replace_cid_images(content, cid_to_placeholder)
            cleaned = clean_html_content(html_with_placeholders)
            if cleaned:
                return cleaned

    if prefer_html:
        # Fallback: prefer-html mode but HTML was empty/missing.
        body = msg.get_body(preferencelist=("plain",))
        if body is not None:
            content = body.get_content()
            if isinstance(content, str) and content.strip():
                return content.strip()

    return ""


def _replace_cid_images(html: str, cid_to_placeholder: dict[str, str]) -> str:
    """Substitute ``<img src="cid:xxx">`` references with ``[image:N]`` markers."""
    if not cid_to_placeholder:
        return html

    def _replace_match(match: re.Match[str]) -> str:
        cid = match.group(1)
        return cid_to_placeholder.get(cid, match.group(0))

    return re.sub(
        r'<img\b[^>]*\bsrc=["\']cid:([^"\']+)["\'][^>]*/?>',
        _replace_match,
        html,
        flags=re.IGNORECASE,
    )


async def _ocr_inline_images(
    inline_images: dict[int, tuple[str, bytes, str, str]],
    *,
    client: LLMClient,
) -> dict[int, str]:
    """OCR every inline image with the same client; wrap results in markers."""
    results: dict[int, str] = {}
    for idx, (_, image_bytes, mime_type, filename) in inline_images.items():
        logger.debug("OCR inline email image: %s (%s)", filename, mime_type)
        try:
            text = await _call_llm_for_text(
                client,
                system=PROMPT_FOR_PICTURE,
                user=[image_url_part_from_bytes(image_bytes, mime_type)],
                operator="parser email image",
                max_tokens=_INLINE_OCR_MAX_TOKENS,
            )
            text = text.strip()
            if text:
                results[idx] = (
                    f"[image: {filename}, content follows]\n{text}\n[end of image]"
                )
            else:
                results[idx] = f"[image: {filename}, no content recognised]"
        except Exception as exc:
            # Best-effort per-image: a single OCR failure (the call raising, or an
            # invalid reply) should not abort the whole email parse — log and
            # continue with a placeholder marker.
            logger.warning(
                "OCR failed for inline image %s: %s", filename, exc, exc_info=True
            )
            results[idx] = f"[image: {filename}, OCR failed]"
    return results


# ---- generic helpers ----


def _find_soffice() -> str:  # pragma: no cover  (libreoffice binary, not in CI)
    """Locate the LibreOffice ``soffice`` binary."""
    path = shutil.which("soffice")  # pragma: no cover
    if path:  # pragma: no cover
        return path  # pragma: no cover
    if Path(_MACOS_SOFFICE).exists():  # pragma: no cover
        return _MACOS_SOFFICE  # pragma: no cover
    raise RuntimeError(  # pragma: no cover
        "LibreOffice not found on PATH. Install it: "
        "macOS: `brew install --cask libreoffice`, "
        "Linux: `apt install libreoffice`."
    )


async def _convert_to_pdf_via_soffice(
    content: bytes, extension: str, soffice: str
) -> bytes:  # pragma: no cover  (libreoffice binary, not in CI)
    """Run LibreOffice headless to convert bytes → PDF bytes."""
    # Reject path-traversal / directory separators in `extension` so a hostile
    # caller cannot escape the tempdir via e.g. ``extension="../../etc/passwd"``.
    # `extension` is expected to be a short ASCII suffix (docx / xlsx / ...).
    if not extension or not extension.isalnum() or len(extension) > 8:
        raise InvalidInputError(
            f"extension {extension!r} is not a short alphanumeric suffix, refusing the soffice conversion: "
            "set RawFile.extension to the file's real suffix, for example docx"
        )
    with tempfile.TemporaryDirectory() as tmpdir:  # pragma: no cover
        in_path = Path(tmpdir) / f"input.{extension}"  # pragma: no cover
        in_path.write_bytes(content)  # pragma: no cover
        proc = await asyncio.create_subprocess_exec(  # pragma: no cover
            soffice,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            tmpdir,
            str(in_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()  # pragma: no cover
        if proc.returncode != 0:  # pragma: no cover
            raise RuntimeError(  # pragma: no cover
                f"LibreOffice conversion failed (code={proc.returncode}): {stderr.decode(errors='replace')[:500]}"
            )
        pdf_path = Path(tmpdir) / "input.pdf"  # pragma: no cover
        if not pdf_path.exists():  # pragma: no cover
            return b""  # pragma: no cover
        return pdf_path.read_bytes()  # pragma: no cover
