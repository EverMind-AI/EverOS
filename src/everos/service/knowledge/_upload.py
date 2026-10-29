"""Turning an uploaded knowledge file into parsed text.

Plain-text uploads are decoded as UTF-8 without the parser, which needs the
``[multimodal]`` extra and configuration that markdown / plaintext do not.
Everything else goes through the parser when it is installed.
"""

from __future__ import annotations

__all__ = [
    "KnowledgeUpload",
    "decode_utf8",
    "extension_of",
    "looks_like_utf8_text",
    "require_content",
]

from typing import Protocol

from everalgo.types import ParsedContent

from everos.core.errors import InvalidInputError, UnsupportedModalityError

_PLAIN_TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {"md", "txt", "rst", "markdown", "text"}
)
# An explicit allowlist, not a ``text/*`` prefix: ``text/html`` must reach the
# parser, whose HTML cleaner strips scripts, styles and comments and caps the
# output before anything is handed to the extraction LLM.
_PLAIN_TEXT_MIMES: frozenset[str] = frozenset(
    {"text/plain", "text/markdown", "text/x-rst", "text/x-markdown"}
)


class KnowledgeUpload(Protocol):
    """An uploaded file as the HTTP layer receives it (Starlette's ``UploadFile``).

    ``size`` is the size the client declared, known before the body is read.
    """

    @property
    def filename(self) -> str | None: ...

    @property
    def content_type(self) -> str | None: ...

    @property
    def size(self) -> int | None: ...

    async def read(self) -> bytes: ...


def extension_of(filename: str | None) -> str:
    """The lowercase extension of ``filename``, ``""`` when it has none."""
    if filename and "." in filename:
        return filename.rsplit(".", 1)[-1].lower()
    return ""


def looks_like_utf8_text(upload: KnowledgeUpload) -> bool:
    """Whether to skip the parser and decode the upload as UTF-8.

    Plain text when the mime type is on the allowlist (browsers send
    ``text/markdown`` / ``text/plain`` for ``.md`` / ``.txt``), or when the
    mime type is missing or ``application/octet-stream`` (``curl -F
    file=@x.md`` without a ``type=``) and the extension is on the allowlist.
    ``text/html`` is deliberately excluded (see :data:`_PLAIN_TEXT_MIMES`).
    """
    mime = (upload.content_type or "").lower()
    if mime in _PLAIN_TEXT_MIMES:
        return True
    if mime and mime != "application/octet-stream":
        return False
    return extension_of(upload.filename) in _PLAIN_TEXT_EXTENSIONS


def decode_utf8(raw: bytes) -> ParsedContent:
    """``raw`` decoded as UTF-8 text.

    Raises:
        UnsupportedModalityError: ``raw`` is not UTF-8.
        InvalidInputError: The text is empty or whitespace only.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UnsupportedModalityError(
            "File is not UTF-8 text. "
            "Install everos[multimodal] for PDF/HTML/DOCX support."
        ) from exc
    parsed = ParsedContent(text=text)
    require_content(parsed)
    return parsed


def require_content(parsed: ParsedContent) -> None:
    """Raise when parsing produced no text.

    Raises:
        InvalidInputError: The parsed text is empty or whitespace only.
    """
    if not parsed.text or not parsed.text.strip():
        raise InvalidInputError("Uploaded file has no valid content.")
