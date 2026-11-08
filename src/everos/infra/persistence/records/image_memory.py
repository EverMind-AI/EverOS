"""Derived-index ``image_memory`` record — one row per picture occurrence.

The picture *file* is content-addressed and stored once (``figures/``); this
row is the memory written about one appearance of it. The same picture used
in two sessions therefore has one file on disk and two rows here, each with
its own time, provenance and status — because "when was this picture sent"
is the question retrieval asks, and one file answers it twice.

Two text columns carry the keyword channel: ``description`` (what the
extraction model saw) and ``ocr`` (the text visible in the picture). Pictures
have no keyword surface of their own — a visual bag-of-words was superseded
by vectors, and a perceptual hash only finds near-duplicates — so BM25 over
these two fields is the only lexical route, and it is what answers "which
sign said EST. 2019".

``vector`` is the *original file's* embedding: the design puts pictures and
text in one vector space, so a text query recalls a picture directly. It is
``None`` whenever the configured embedder is text-only, which leaves the row
keyword-searchable rather than dropping it.
"""

from __future__ import annotations

import datetime as _dt
from typing import ClassVar

from everos.infra.persistence.records.base import IndexRecord

from ._parent_type import ParentType
from .vector import DenseVector


class ImageMemory(IndexRecord):
    """One appearance of one picture in the derived index."""

    TABLE_NAME: ClassVar[str] = "image_memory"
    BM25_FIELDS: ClassVar[list[str]] = ["description_tokens", "ocr_tokens"]
    PRIMARY_KEY: ClassVar[tuple[str, ...]] = ("app_id", "project_id", "id")

    id: str
    """``<owner_id>_<entry_id>``; unique within ``(app_id, project_id)``."""

    entry_id: str
    """md-side seq id ``im_<YYYYMMDD>_<NNNN>`` (cascade reverse-lookup)."""

    owner_id: str
    owner_type: str
    app_id: str = "default"
    project_id: str = "default"
    session_id: str | None = None
    timestamp: _dt.datetime

    parent_type: str = ParentType.MEMCELL.value
    """Source pointer — the cell this picture arrived in."""

    parent_id: str
    """Source memcell id. With ``image_id`` it answers "which segment did this
    picture hang in", which is what forget uses to find what to re-extract."""

    image_id: str
    """Content hash (SHA-256 hex) of the picture file.

    Stable across every occurrence: this is the id ``forget`` takes and the
    id an episode body references.
    """

    original_path: str
    """Memory-root-relative path of the original, ``figures/<aa>/<hash>.<ext>``."""

    name: str = ""
    """What the client called the picture on this arrival (``IMG_2231.jpg``).

    Not indexed and not part of the content digest: a renamed picture is the
    same picture, so a name edit must not re-embed the row. Present so a
    search result can hand the caller the file name it sent -- the only handle
    an evaluation harness has for matching a memory to its dataset entry.
    """

    media_type: str
    """Format the file is stored as (``png`` / ``jpeg`` / ``webp`` / ``gif``)."""

    width: int
    height: int

    description: str
    """What the picture shows, as the extraction model wrote it. The text half
    of the picture's memory and the BM25 source beside ``ocr``."""

    description_tokens: str
    """App-layer pre-tokenised ``description`` (whitespace-tokenised BM25)."""

    ocr: str = ""
    """Text visible in the picture (signs, labels, screenshots). Empty when
    there is none — which is the common case, not an error."""

    ocr_tokens: str = ""

    md_path: str
    content_sha256: str
    """SHA-256 over the content-bearing fields of the md entry; a matching
    digest on reconcile skips the row (see ``Episode`` for the rationale)."""

    deprecated_by: str | None = None
    """Set when a later entry supersedes this one (a forget-triggered
    re-extraction, or a description edit). ``NULL`` means current."""

    vector: DenseVector | None = None
