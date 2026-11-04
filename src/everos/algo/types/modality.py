"""Modality classification for multimodal parser routing.

A ``Modality`` is the kind of content a raw file holds, independent of its MIME type or extension. Parsers dispatch
by modality; ``EXTENSION_TO_MODALITY`` and ``MIME_TO_MODALITY`` cover the well-known cases and ``UNKNOWN`` absorbs the
rest. It lives with the shared types so a caller can classify a file before deciding whether to invoke
the parser at all (``DIRECT`` content needs no parser).
"""

from __future__ import annotations

__all__ = [
    "EXTENSION_TO_MODALITY",
    "MIME_TO_EXTENSION",
    "MIME_TO_MODALITY",
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_MIMES",
    "Modality",
    "get_extension_from_mime",
    "get_modality",
    "get_modality_from_mime",
]

from enum import StrEnum


class Modality(StrEnum):
    """Top-level classification for parser dispatch.

    - ``IMAGE`` — bitmap / vector images (png / jpg / svg / ...).
    - ``PDF`` — PDF documents (handled separately from ``DOCUMENT`` because the
      parsing path differs: PDFs go straight to the multimodal LLM, while
      ``DOCUMENT`` formats are first converted to PDF via LibreOffice).
    - ``URL`` — remote ``http``/``https`` resource the parser fetches before
      dispatching by the fetched ``Content-Type``. The fetched bytes typically
      become ``HTML`` content, but the URL modality is logged separately so
      callers can tell "loaded from network" vs "passed in as bytes".
    - ``AUDIO`` — speech / sound (mp3 / wav / m4a / ...).
    - ``DOCUMENT`` — Office / iWork / ODF documents (docx / xlsx / pptx /
      pages / odt / rtf / ...). Requires conversion before LLM ingest.
    - ``HTML`` — HTML files (html / htm).
    - ``EMAIL`` — eml mailbox files.
    - ``DIRECT`` — plain-text formats that need no parsing (txt / md / csv /
      tsv / vtt). Callers should read the bytes directly and skip the parser.
    - ``UNKNOWN`` — anything else. Parser raises rather than guessing.
    """

    IMAGE = "image"
    PDF = "pdf"
    AUDIO = "audio"
    DOCUMENT = "document"
    HTML = "html"
    EMAIL = "email"
    URL = "url"
    DIRECT = "direct"
    UNKNOWN = "unknown"


_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {"png", "jpg", "jpeg", "webp", "tiff", "tif", "bmp", "svg"}
)
_PDF_EXTENSIONS: frozenset[str] = frozenset({"pdf"})
_AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {"mp3", "wav", "m4a", "amr", "aiff", "aac", "ogg", "flac"}
)
_DOCUMENT_EXTENSIONS: frozenset[str] = frozenset(
    {
        # Office (modern)
        "docx",
        "pptx",
        "xlsx",
        # Office (legacy)
        "doc",
        "ppt",
        "xls",
        # iWork
        "pages",
        "key",
        "numbers",
        # OpenDocument
        "odt",
        "ods",
        "odp",
        # Rich text
        "rtf",
    }
)
_HTML_EXTENSIONS: frozenset[str] = frozenset({"html", "htm"})
_EMAIL_EXTENSIONS: frozenset[str] = frozenset({"eml"})
_DIRECT_EXTENSIONS: frozenset[str] = frozenset({"txt", "md", "vtt", "csv", "tsv"})


def _build_extension_to_modality() -> dict[str, Modality]:
    mapping: dict[str, Modality] = {}
    for ext in _IMAGE_EXTENSIONS:
        mapping[ext] = Modality.IMAGE
    for ext in _PDF_EXTENSIONS:
        mapping[ext] = Modality.PDF
    for ext in _AUDIO_EXTENSIONS:
        mapping[ext] = Modality.AUDIO
    for ext in _DOCUMENT_EXTENSIONS:
        mapping[ext] = Modality.DOCUMENT
    for ext in _HTML_EXTENSIONS:
        mapping[ext] = Modality.HTML
    for ext in _EMAIL_EXTENSIONS:
        mapping[ext] = Modality.EMAIL
    for ext in _DIRECT_EXTENSIONS:
        mapping[ext] = Modality.DIRECT
    return mapping


EXTENSION_TO_MODALITY: dict[str, Modality] = _build_extension_to_modality()
"""Frozen mapping: lowercase extension (no leading dot) → ``Modality``."""

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(EXTENSION_TO_MODALITY.keys())
"""All known extensions across every modality."""


def get_modality(extension: str) -> Modality:
    """Classify a file extension into a ``Modality``.

    Parameters
    ----------
    extension : str
        File extension, with or without a leading dot, any case.

    Returns:
    -------
    Modality
        The matching modality, or ``Modality.UNKNOWN`` if the extension is not
        in ``EXTENSION_TO_MODALITY``.
    """
    key = extension.lower().lstrip(".")
    return EXTENSION_TO_MODALITY.get(key, Modality.UNKNOWN)


# ---- MIME-based dispatch (parallel to extension-based) ----


_IMAGE_MIMES: frozenset[str] = frozenset(
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
_PDF_MIMES: frozenset[str] = frozenset({"application/pdf"})
_AUDIO_MIMES: frozenset[str] = frozenset(
    {
        "audio/mpeg",
        "audio/mp3",
        "audio/wav",
        "audio/x-wav",
        "audio/mp4",
        "audio/m4a",
        "audio/amr",
        "audio/aiff",
        "audio/x-aiff",
        "audio/aac",
        "audio/ogg",
        "audio/flac",
        "audio/x-flac",
    }
)
_DOCUMENT_MIMES: frozenset[str] = frozenset(
    {
        # Office (modern OOXML)
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # docx
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # xlsx
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # pptx
        # Office (legacy)
        "application/msword",  # doc
        "application/vnd.ms-excel",  # xls
        "application/vnd.ms-powerpoint",  # ppt
        # iWork
        "application/x-iwork-pages-sffpages",
        "application/x-iwork-keynote-sffkey",
        "application/x-iwork-numbers-sffnumbers",
        # OpenDocument
        "application/vnd.oasis.opendocument.text",
        "application/vnd.oasis.opendocument.spreadsheet",
        "application/vnd.oasis.opendocument.presentation",
        # Rich Text
        "application/rtf",
        "text/rtf",
    }
)
_HTML_MIMES: frozenset[str] = frozenset({"text/html", "application/xhtml+xml"})
_EMAIL_MIMES: frozenset[str] = frozenset({"message/rfc822"})
_DIRECT_MIMES: frozenset[str] = frozenset(
    {
        "text/plain",
        "text/markdown",
        "text/x-markdown",
        "text/csv",
        "text/tab-separated-values",
        "text/vtt",
    }
)


def _build_mime_to_modality() -> dict[str, Modality]:
    mapping: dict[str, Modality] = {}
    for mime in _IMAGE_MIMES:
        mapping[mime] = Modality.IMAGE
    for mime in _PDF_MIMES:
        mapping[mime] = Modality.PDF
    for mime in _AUDIO_MIMES:
        mapping[mime] = Modality.AUDIO
    for mime in _DOCUMENT_MIMES:
        mapping[mime] = Modality.DOCUMENT
    for mime in _HTML_MIMES:
        mapping[mime] = Modality.HTML
    for mime in _EMAIL_MIMES:
        mapping[mime] = Modality.EMAIL
    for mime in _DIRECT_MIMES:
        mapping[mime] = Modality.DIRECT
    return mapping


MIME_TO_MODALITY: dict[str, Modality] = _build_mime_to_modality()
"""Frozen mapping: lowercase MIME (without parameters) → ``Modality``."""

SUPPORTED_MIMES: frozenset[str] = frozenset(MIME_TO_MODALITY.keys())
"""All known MIME types across every modality."""


# Default extension per MIME — used when the URL parser fetches over HTTP and
# needs to hand the bytes off to a submodule that dispatches by extension.
# Picks the canonical extension for each MIME (the one the parser's ``_MIME_MAP``
# tables use as their primary key).
_MIME_TO_EXTENSION: dict[str, str] = {
    # image
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
    "image/svg+xml": "svg",
    # pdf
    "application/pdf": "pdf",
    # audio
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mp4": "m4a",
    "audio/m4a": "m4a",
    "audio/amr": "amr",
    "audio/aiff": "aiff",
    "audio/x-aiff": "aiff",
    "audio/aac": "aac",
    "audio/ogg": "ogg",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    # office
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/msword": "doc",
    "application/vnd.ms-excel": "xls",
    "application/vnd.ms-powerpoint": "ppt",
    "application/x-iwork-pages-sffpages": "pages",
    "application/x-iwork-keynote-sffkey": "key",
    "application/x-iwork-numbers-sffnumbers": "numbers",
    "application/vnd.oasis.opendocument.text": "odt",
    "application/vnd.oasis.opendocument.spreadsheet": "ods",
    "application/vnd.oasis.opendocument.presentation": "odp",
    "application/rtf": "rtf",
    "text/rtf": "rtf",
    # html
    "text/html": "html",
    "application/xhtml+xml": "html",
    # email
    "message/rfc822": "eml",
    # direct
    "text/plain": "txt",
    "text/markdown": "md",
    "text/x-markdown": "md",
    "text/csv": "csv",
    "text/tab-separated-values": "tsv",
    "text/vtt": "vtt",
}

MIME_TO_EXTENSION: dict[str, str] = dict(_MIME_TO_EXTENSION)
"""Frozen mapping: lowercase MIME (without parameters) → canonical extension."""


def _normalise_mime(mime: str) -> str:
    """Strip parameters and lowercase: ``"text/html; charset=utf-8"`` → ``"text/html"``."""
    if not mime:
        return ""
    return mime.lower().split(";", 1)[0].strip()


def get_modality_from_mime(mime: str) -> Modality:
    """Classify a MIME type into a ``Modality``.

    Parameters
    ----------
    mime : str
        MIME type, optionally with ``;`` parameters (charset / boundary).
        Case-insensitive.

    Returns:
    -------
    Modality
        The matching modality, or ``Modality.UNKNOWN`` if not in
        ``MIME_TO_MODALITY``.
    """
    return MIME_TO_MODALITY.get(_normalise_mime(mime), Modality.UNKNOWN)


def get_extension_from_mime(mime: str) -> str:
    """Look up the canonical lower-case extension for a MIME type.

    Parameters
    ----------
    mime : str
        MIME type, optionally with ``;`` parameters. Case-insensitive.

    Returns:
    -------
    str
        Canonical extension (no leading dot) — e.g. ``application/pdf`` →
        ``"pdf"``, ``image/jpeg`` → ``"jpg"``. Empty string when MIME is
        unknown.
    """
    return MIME_TO_EXTENSION.get(_normalise_mime(mime), "")
