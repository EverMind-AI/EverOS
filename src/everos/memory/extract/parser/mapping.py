"""ContentItem dict -> everalgo ``RawFile``.

``RawFile`` ships in everalgo-core (always installed), so this module is safe
to import without the ``everos[multimodal]`` extra.

everalgo deliberately never reads the host filesystem (``http(s)`` fetch or
caller-hydrated bytes only). EverOS — the trust boundary that legitimately
touches local files — owns ``file://`` support: :func:`build_raw_file` reads
the file locally (with guardrails) and hands everalgo a *hydrated* RawFile
(``content`` bytes), so the library stays filesystem-stateless.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import anyio
from everalgo.types import RawFile

from everos.config import load_settings
from everos.core.observability.logging import get_logger

logger = get_logger(__name__)


def to_raw_file(item: dict[str, Any]) -> RawFile:
    """Build an :class:`everalgo.types.RawFile` from a ContentItem dict.

    ``uri``-backed items are handed to everalgo as-is (it fetches ``http(s)``
    and dispatches by Content-Type). ``base64``-backed items are decoded and
    keyed by the item's ``ext`` (everalgo dispatches by extension).

    Note: ``file://`` uris are **not** resolved here — they are hydrated
    upstream by :func:`build_raw_file`. A ``file://`` item reaching this
    function is returned as a uri RawFile (which everalgo then rejects).

    Raises:
        ValueError: When the item carries neither ``uri`` nor ``base64``.
    """
    uri = item.get("uri")
    if uri:
        return RawFile(uri=uri)
    encoded = item.get("base64")
    if encoded:
        return RawFile(
            content=base64.b64decode(encoded),
            extension=(item.get("ext") or "").lstrip(".").lower(),
            mime=item.get("mime") or "",
        )
    raise ValueError(
        f"content item has neither uri nor base64 (type={item.get('type')!r})"
    )


def _is_file_uri(uri: str) -> bool:
    return urlparse(uri).scheme == "file"


def _resolve_file_uri(uri: str) -> Path:
    """Parse a ``file://`` uri into a canonical local path (symlinks resolved).

    Raises ``ValueError`` for a remote host component or a path that does not
    exist (``resolve(strict=True)``).
    """
    parsed = urlparse(uri)
    if parsed.netloc and parsed.netloc not in ("", "localhost"):
        raise ValueError(f"file uri with remote host not supported: {parsed.netloc!r}")
    try:
        return Path(unquote(parsed.path)).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"cannot resolve file uri {uri!r}: {exc}") from exc


def _validated_file_path(uri: str) -> Path:
    """Resolve + guardrail-check a ``file://`` uri (synchronous fs ops).

    This is the host-filesystem access everalgo refuses by design, so the
    guardrails live here (the application trust boundary):

    * canonical, symlink-resolved path that must be an existing regular file;
    * size-capped by ``settings.multimodal.file_uri_max_bytes``;
    * confined to ``settings.multimodal.file_uri_allow_dirs`` when that
      allowlist is non-empty (empty = allow any readable file, the
      local-first default).

    Raises:
        ValueError: missing / non-regular / oversized / out-of-allowlist file.
    """
    cfg = load_settings().multimodal
    path = _resolve_file_uri(uri)
    if not path.is_file():
        raise ValueError(f"file uri target is not a regular file: {path}")

    allow = [Path(d).expanduser().resolve() for d in cfg.file_uri_allow_dirs]
    if allow and not any(path.is_relative_to(r) for r in allow):
        raise ValueError(
            f"file uri {path} is outside the allowed roots "
            "(set EVEROS_MULTIMODAL__FILE_URI_ALLOW_DIRS to permit it)"
        )

    size = path.stat().st_size
    if size > cfg.file_uri_max_bytes:
        raise ValueError(
            f"file uri target too large: {size} bytes "
            f"(cap {cfg.file_uri_max_bytes}; raise "
            "EVEROS_MULTIMODAL__FILE_URI_MAX_BYTES)"
        )
    return path


async def read_file_uri(uri: str, *, ext_hint: str | None = None) -> tuple[bytes, str]:
    """Read a guardrail-checked ``file://`` asset into ``(bytes, extension)``.

    Raises:
        ValueError: missing / non-regular / oversized / out-of-allowlist file.
    """
    path = _validated_file_path(uri)
    content = await anyio.Path(path).read_bytes()
    ext = (ext_hint or path.suffix).lstrip(".").lower()
    logger.debug("file_uri_hydrated", path=str(path), ext=ext)
    return content, ext


async def build_raw_file(item: dict[str, Any]) -> RawFile:
    """Build a RawFile for the parser, hydrating ``file://`` uris locally.

    ``http(s)`` uris and ``base64`` payloads route through the synchronous
    :func:`to_raw_file`. A ``file://`` uri is read here into a hydrated
    RawFile (``content`` bytes + ``extension``) so everalgo never touches the
    filesystem. The original item dict is left unchanged (the buffer keeps the
    lightweight ``file://`` uri, not the inlined bytes).
    """
    uri = item.get("uri")
    if uri and _is_file_uri(uri):
        content, ext = await read_file_uri(uri, ext_hint=item.get("ext"))
        return RawFile(content=content, extension=ext, mime=item.get("mime") or "")
    return to_raw_file(item)
