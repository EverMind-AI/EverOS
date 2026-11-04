"""URL parser — HTTP fetch then dispatch by the fetched ``Content-Type``.

Two-stage:

1. **Fetch** ``raw_file.uri`` via ``_http.fetch_uri``. ``http`` / ``https``
   only; ``file://`` and other local-filesystem schemes are rejected so the
   library stays stateless re: the host filesystem.
2. **Dispatch by the fetched Content-Type**:
   - HTML response → bs4 cleanup + LLM extraction, plus OG / Twitter /
     ``<meta>`` tag extraction into ``ParsedContent.metadata``.
   - PDF / image / audio / Office / EML / direct response → delegate to
     the matching submodule (``document.aparse`` / ``image.aparse`` /
     ``audio.aparse``).
   - Unknown Content-Type → fall back to the HTML handler (most URLs are
     HTML; the LLM tolerates degraded input).

The returned ``ParsedContent.modality`` is always ``Modality.URL`` so
callers can distinguish "loaded from network" from "passed in as bytes";
the inner modality is recorded in ``metadata["inner_modality"]``.

Schema source for the metadata extractor: an upstream internal URL
extractor reference implementation (``_extract_og_tags`` /
``_extract_twitter_tags`` / ``_extract_meta_tags`` / ``_extract_title`` /
``_extract_favicon``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from everos.algo.types import (
    Modality,
    ParsedContent,
    RawFile,
    get_extension_from_mime,
    get_modality_from_mime,
)
from everos.algo.utils import InvalidInputError, decode_bytes

from ._http import fetch_uri

if TYPE_CHECKING:
    from everos.algo.llm import LLMClient

__all__ = ["aparse"]

logger = logging.getLogger(__name__)


async def aparse(raw_file: RawFile, *, llm: LLMClient) -> ParsedContent:
    """Fetch ``raw_file.uri`` (or use pre-hydrated content) and parse.

    Parameters
    ----------
    raw_file : RawFile
        ``raw_file.uri`` must be an ``http`` / ``https`` URL when
        ``content`` is empty. When ``content`` is non-empty it is used
        as-is (network skipped); ``mime`` is taken from ``raw_file.mime``,
        defaulting to ``text/html`` when missing.
    llm : LLMClient
        LLM client. Required for HTML / PDF / image / audio / Office / EML responses.

    Returns:
    -------
    ParsedContent
        ``modality = URL``; ``text`` carries whatever the inner handler
        extracted; ``metadata`` carries the inner handler's metadata plus
        ``fetched_uri`` / ``fetched_mime`` / ``inner_modality``, and (for
        HTML responses) the OG / Twitter / meta card.
    """
    if raw_file.content:
        body = raw_file.content
        fetched_mime = raw_file.mime
    else:
        if not raw_file.uri:
            raise InvalidInputError(
                "RawFile has neither content nor uri, so url.aparse has nothing to read: "
                "pass the page bytes in content, or the uri to fetch"
            )
        body, fetched_mime = await fetch_uri(raw_file.uri)

    if not body:
        raise InvalidInputError(
            f"uri {_locate(raw_file.uri)!r} returned an empty body, there is nothing to parse: "
            "check that the uri points at a real document, or pass its bytes in content"
        )

    inner_modality = get_modality_from_mime(fetched_mime)
    if inner_modality is Modality.UNKNOWN:
        # Most URLs are HTML; fall back so we don't reject ambiguous responses.
        logger.warning(
            "url.aparse: unknown Content-Type %r from %s; falling back to HTML handler",
            fetched_mime,
            _locate(raw_file.uri),
        )
        inner_modality = Modality.HTML

    inner = await _dispatch_inner(
        inner_modality, body, fetched_mime, raw_file.uri, llm=llm
    )

    merged_metadata: dict[str, object] = dict(inner.metadata)
    if inner_modality is Modality.HTML:
        html_text = decode_bytes(body)
        merged_metadata.update(_extract_metadata(html_text, base_url=raw_file.uri))
    merged_metadata["fetched_uri"] = raw_file.uri
    merged_metadata["fetched_mime"] = fetched_mime
    merged_metadata["inner_modality"] = inner_modality.value

    return ParsedContent(
        text=inner.text,
        modality=Modality.URL,
        mime=fetched_mime or inner.mime,
        metadata=merged_metadata,
    )


# ---- inner dispatch ----


async def _dispatch_inner(
    modality: Modality,
    body: bytes,
    fetched_mime: str,
    uri: str,
    *,
    llm: LLMClient,
) -> ParsedContent:
    """Route the fetched body to the appropriate submodule.

    The local imports break a circular module dependency (the parser
    facade imports ``url``; ``url`` imports submodules only at call time).
    """
    extension = get_extension_from_mime(fetched_mime)
    inner_rf = RawFile(content=body, mime=fetched_mime, extension=extension, uri=uri)

    if modality is Modality.IMAGE:
        from everos.component.parser.backend import image

        return await image.aparse(inner_rf, llm=llm)
    if modality is Modality.AUDIO:
        from everos.component.parser.backend import audio

        return await audio.aparse(inner_rf, llm=llm)
    if modality in (Modality.PDF, Modality.DOCUMENT, Modality.HTML, Modality.EMAIL):
        from everos.component.parser.backend import document

        # ``document.aparse`` keys off ``extension`` for its 4-way internal
        # switch (PDF / HTML / EML / Office). The fallback to ``html`` keeps
        # the legacy unknown-content-type path working for HTTP responses
        # whose only useful hint was ``text/html``.
        if not inner_rf.extension:
            fallback_ext = {
                Modality.PDF: "pdf",
                Modality.DOCUMENT: "docx",
                Modality.HTML: "html",
                Modality.EMAIL: "eml",
            }[modality]
            inner_rf = inner_rf.model_copy(update={"extension": fallback_ext})
        return await document.aparse(inner_rf, llm=llm)
    if modality is Modality.DIRECT:
        return ParsedContent(
            text=body.decode("utf-8", errors="replace"),
            modality=Modality.DIRECT,
            mime=fetched_mime,
        )
    raise InvalidInputError(
        f"uri {_locate(uri)!r} served modality {modality.value!r} (content type {fetched_mime!r}), which the parser cannot read: "
        "point the uri at an image, audio, document or web page"
    )


# ---- metadata extraction (port from upstream internal URL extractor) ----


def _locate(uri: str) -> str:
    """``scheme://host[:port]/path`` of ``uri``, for an error message: a signed URL's query, fragment and userinfo stay out of logs."""
    parts = urlsplit(uri)
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{host}{port}{parts.path}"


def _extract_metadata(html_text: str, *, base_url: str) -> dict[str, object]:
    """Pull title / description / image / OG / Twitter tags out of HTML."""
    if not html_text or not html_text.strip():
        return {}
    soup = BeautifulSoup(html_text, "html.parser")
    og = _extract_og_tags(soup)
    tw = _extract_twitter_tags(soup)
    meta = _extract_meta_tags(soup)

    title = (
        _safe(og.get("title"))
        or _safe(tw.get("title"))
        or _safe(_extract_title(soup))
        or _safe(meta.get("title"))
    )
    description = (
        _safe(og.get("description"))
        or _safe(tw.get("description"))
        or _safe(meta.get("description"))
    )
    image = _safe(og.get("image")) or _safe(tw.get("image"))
    site_name = _safe(og.get("site_name"))
    og_type = _safe(og.get("type"))
    favicon = _extract_favicon(soup, base_url=base_url)

    result: dict[str, object] = {
        "title": title,
        "description": description,
        "image": image,
        "site_name": site_name,
        "og_type": og_type,
        "favicon": favicon,
        "og_tags": og,
        "twitter_tags": tw,
        "meta_tags": meta,
    }
    return {k: v for k, v in result.items() if v not in (None, "", {})}


def _extract_og_tags(soup: BeautifulSoup) -> dict[str, str]:
    """``<meta property="og:*" content="...">`` → ``{name_without_og_prefix: content}``."""
    tags: dict[str, str] = {}
    for tag in soup.find_all(
        "meta", property=lambda x: bool(x) and x.startswith("og:")
    ):
        prop = tag["property"]
        if not isinstance(prop, str):
            continue
        content = tag.get("content")
        if not isinstance(content, str):
            continue
        tags[prop[3:]] = content.strip()
    return tags


def _extract_twitter_tags(soup: BeautifulSoup) -> dict[str, str]:
    """``<meta name="twitter:*" content="...">`` → ``{name_without_twitter_prefix: content}``."""
    tags: dict[str, str] = {}
    for tag in soup.find_all(
        "meta", attrs={"name": lambda x: x is not None and x.startswith("twitter:")}
    ):
        name = tag["name"]
        if not isinstance(name, str):
            continue
        content = tag.get("content")
        if not isinstance(content, str):
            continue
        tags[name[8:]] = content.strip()
    return tags


def _extract_meta_tags(soup: BeautifulSoup) -> dict[str, str]:
    """Plain ``<meta name="title|description|keywords|author">`` extraction."""
    tags: dict[str, str] = {}
    for name in ("title", "description", "keywords", "author"):
        tag = soup.find("meta", attrs={"name": name})
        if tag:
            content = tag.get("content")
            if isinstance(content, str) and content:
                tags[name] = content.strip()
    return tags


def _extract_title(soup: BeautifulSoup) -> str | None:
    """``<title>...</title>`` text, stripped."""
    tag = soup.find("title")
    if tag and tag.get_text():
        return tag.get_text().strip()
    return None


def _extract_favicon(soup: BeautifulSoup, *, base_url: str) -> str | None:
    """``<link rel="icon|shortcut icon">`` href, resolved against ``base_url``."""
    for rel in ("icon", "shortcut icon"):
        tag = soup.find("link", rel=rel)
        if tag:
            href = tag.get("href")
            if isinstance(href, str) and href:
                return urljoin(base_url, href.strip()) if base_url else href.strip()
    return None


def _safe(value: object) -> str | None:
    """Reject template-variable leftovers (``{{ ... }}``, ``${...}``) that some CMSes emit."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if "{{" in stripped or "${" in stripped:
        return None
    return stripped
