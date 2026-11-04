"""Raw input data contracts — pre-boundary and parser inputs to the algorithms.

``RawFile`` accepts caller-hydrated bytes or an HTTP(S) URL. It never points
the parser at a caller-selected local path; parser implementations may fetch
the URL or use an internal managed temporary directory during conversion.
"""

from __future__ import annotations

__all__ = ["RawData", "RawFile"]

from pydantic import BaseModel, ConfigDict, Field


class RawData(BaseModel):
    """Generic raw structured payload (Jira / Email / Confluence / Agent trace ...).

    Stub — schema fields TBD (T1).
    """

    id: str = Field(default="", description="TBD (T1 review)")
    source_type: str = Field(default="", description="TBD (T1 review)")


class RawFile(BaseModel):
    """Multimodal raw file payload — input to the multimodal parser.

    The parser needs *one of* ``content`` (hydrated bytes) or ``uri`` (an
    ``http``/``https`` URL the parser may fetch). ``mime`` / ``extension``
    are advisory hints used for parser dispatch and reporting.

    Parser hydration rules (the parser side, not this type):
    - ``content`` present → use as-is, no network I/O.
    - ``content`` empty + ``uri`` set + scheme is ``http``/``https`` → the
      parser fetches via HTTP and fills ``content`` internally.
    - ``content`` empty + ``uri`` empty → ``InvalidInputError`` at parse time.
    - ``file://`` and other local-filesystem URIs are *rejected* so callers
      cannot make the parser read an arbitrary local path.

    Parser dispatch precedence (cheapest to most reliable):
    1. ``extension`` (when present) — maps to ``Modality`` via
       ``everos.algo.types.get_modality``.
    2. Otherwise: a ``uri`` with ``http``/``https`` scheme routes to the
       URL parser, which performs the fetch and re-dispatches by the
       fetched ``Content-Type``.

    Examples:
    --------
    >>> from everos.algo.types import RawFile
    >>> # bytes-in form (caller already hydrated)
    >>> RawFile(content=b"%PDF-1.4 ...", mime="application/pdf", extension="pdf")
    >>> # url-in form (parser fetches)
    >>> RawFile(uri="https://example.com/article")
    """

    content: bytes = Field(
        default=b"",
        description="Raw file bytes. Optional when ``uri`` is provided.",
    )
    mime: str = Field(
        default="", description="MIME type hint, e.g. ``application/pdf``."
    )
    extension: str = Field(
        default="",
        description="Lowercase extension without leading dot (``pdf``, ``png``, ...). Used for ``Modality`` dispatch.",
    )
    uri: str = Field(
        default="",
        description="Origin URI. ``http``/``https`` is fetched by the URL parser; ``file://`` is rejected.",
    )

    model_config = ConfigDict(extra="ignore")
