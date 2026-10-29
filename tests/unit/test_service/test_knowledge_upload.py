"""Pins how :class:`KnowledgeService` turns an uploaded file into parsed text.

Contracts: an upload whose declared size is over ``max_upload_bytes`` is
refused before its body is read; plain-text uploads (by mime type, or by
extension when the mime type says nothing) are decoded as UTF-8 even when
the optional parser is installed; everything else goes to the parser when it
is installed and is refused as unsupported otherwise; empty text is refused.
"""

from __future__ import annotations

from typing import Any

import pytest
from everalgo.types import ParsedContent, RawFile

from everos.core.errors import InvalidInputError, UnsupportedModalityError
from everos.service._upload import looks_like_utf8_text
from tests.fakes import make_knowledge_service


class _Upload:
    """An ``UploadFile``-shaped upload that records whether it was read."""

    def __init__(
        self,
        content: bytes = b"# Hello",
        *,
        filename: str | None = "note.md",
        content_type: str | None = "text/markdown",
        size: int | None = None,
    ) -> None:
        self.filename = filename
        self.content_type = content_type
        self.size = len(content) if size is None else size
        self._content = content
        self.read_calls = 0

    async def read(self) -> bytes:
        self.read_calls += 1
        return self._content


class _ExtractionReachedError(Exception):
    """Raised by the stub extractor: parsing finished and extraction began."""


class _CapturingExtractor:
    def __init__(self) -> None:
        self.parsed: ParsedContent | None = None

    async def aextract(self, parsed: ParsedContent, **_: Any) -> list[Any]:
        self.parsed = parsed
        raise _ExtractionReachedError


class _NoDocuments:
    """The document rows the id minting asks about: there are none."""

    async def doc_id_exists(self, doc_id: str) -> bool:
        return False


def _service(**overrides: Any) -> tuple[Any, _CapturingExtractor]:
    extractor = _CapturingExtractor()
    service = make_knowledge_service(
        extractor=extractor, documents=_NoDocuments(), **overrides
    )
    return service, extractor


@pytest.mark.parametrize(
    ("mime", "filename", "expected"),
    [
        ("text/markdown", "note.md", True),
        ("text/plain", "note.txt", True),
        ("TEXT/MARKDOWN", "note.md", True),
        ("text/x-rst", "readme.rst", True),
        # HTML must reach the parser, whose cleaner strips active markup.
        ("text/html", "page.html", False),
        ("TEXT/HTML", "page.html", False),
        ("application/octet-stream", "note.md", True),
        (None, "note.md", True),
        ("", "notes.rst", True),
        ("application/octet-stream", "note.bin", False),
        (None, "note.bin", False),
        (None, None, False),
        ("application/pdf", "doc.pdf", False),
        ("image/png", "img.png", False),
        ("application/json", "data.json", False),
        ("text/xml", "data.xml", False),
    ],
)
def test_plain_text_detection(
    mime: str | None, filename: str | None, expected: bool
) -> None:
    upload = _Upload(filename=filename, content_type=mime)
    assert looks_like_utf8_text(upload) is expected


async def test_an_oversized_upload_is_refused_before_it_is_read() -> None:
    service, _ = _service(max_upload_bytes=10)
    upload = _Upload(size=11)

    with pytest.raises(InvalidInputError, match="exceeds"):
        await service.create_from_upload(upload, title="t")

    assert upload.read_calls == 0


@pytest.mark.parametrize("size", [10, None])
async def test_an_upload_within_the_limit_or_of_unknown_size_is_read(
    size: int | None,
) -> None:
    service, extractor = _service(max_upload_bytes=10)
    upload = _Upload(b"# Hi", size=size)
    upload.size = size

    with pytest.raises(_ExtractionReachedError):
        await service.create_from_upload(upload, title="t")

    assert extractor.parsed == ParsedContent(text="# Hi")


async def test_plain_text_skips_an_installed_parser() -> None:
    async def parser_must_not_run(raw: RawFile) -> ParsedContent:
        raise AssertionError("the parser must not run for plain text")

    service, extractor = _service(parser_available=True, parse_file=parser_must_not_run)
    with pytest.raises(_ExtractionReachedError):
        await service.create_from_upload(_Upload(b"plain"), title="t")
    assert extractor.parsed == ParsedContent(text="plain")


async def test_a_binary_upload_goes_to_the_installed_parser() -> None:
    seen: list[RawFile] = []

    async def parse(raw: RawFile) -> ParsedContent:
        seen.append(raw)
        return ParsedContent(text="parsed pdf")

    service, extractor = _service(parser_available=True, parse_file=parse)
    upload = _Upload(b"%PDF", filename="Doc.PDF", content_type="application/pdf")
    with pytest.raises(_ExtractionReachedError):
        await service.create_from_upload(upload, title="t")

    assert seen == [RawFile(content=b"%PDF", mime="application/pdf", extension="pdf")]
    assert extractor.parsed == ParsedContent(text="parsed pdf")


async def test_a_binary_upload_without_the_parser_is_unsupported() -> None:
    service, _ = _service()
    upload = _Upload(b"\x80\x81", filename="x.bin", content_type=None)
    with pytest.raises(UnsupportedModalityError):
        await service.create_from_upload(upload, title="t")


@pytest.mark.parametrize("parser_available", [False, True])
async def test_empty_text_is_refused(parser_available: bool) -> None:
    async def parse(raw: RawFile) -> ParsedContent:
        return ParsedContent(text="  ")

    service, _ = _service(parser_available=parser_available, parse_file=parse)
    upload = (
        _Upload(b"  ")
        if not parser_available
        else _Upload(b"%PDF", filename="x.pdf", content_type="application/pdf")
    )
    with pytest.raises(InvalidInputError, match="no valid content"):
        await service.create_from_upload(upload, title="t")
