"""Content-addressed picture storage — originals, derived copies, metadata.

A picture arrives as a content item, possibly many times: the same file in two
sessions, or one file forwarded twice in one conversation. It is stored **once**,
under its SHA-256, and each arrival only adds a reference. That is what keeps a
store whose originals are never re-encoded and never deleted from growing with
*conversation volume* rather than with *distinct pixels*.

Layout (all relative to the memory root):

    figures/<first two hex>/<sha256>.<ext>   the original, byte-for-byte as sent
    figures/.cache/<sha256>-<edge>.<ext>     a derived copy, rebuilt on demand

Two invariants carry the design:

- **The original is the truth and is never rewritten.** A re-encode would make
  the bytes differ from what the hash names, so a store could no longer prove
  which picture it holds. Every derived copy is a pure function of it and can
  be dropped at any time.
- **Writes are atomic and hash-checked.** Bytes land in a temporary file under
  ``.cache/``, are read back and hashed, and only then renamed into place; a
  crash leaves a temporary file, never a truncated original under a name that
  claims a hash it does not have.

The metadata (format, bytes, dimensions, first-seen time, reference count)
lives in SQLite, not here: this module is a filesystem store and knows nothing
about memories. ``ImageStore`` is the interface the memory layer uses, so the
filesystem details stay on this side of the boundary.
"""

from __future__ import annotations

__all__ = [
    "ImageFormat",
    "ImageLimits",
    "ImageRejectedError",
    "StoredImage",
    "StoredImageInfo",
    "decode_image",
    "dedupe",
    "derived_relpath",
    "hash_bytes",
    "original_relpath",
    "scaled_copy",
]

import hashlib
import io
from collections.abc import Iterable
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

#: Pillow format name → the extension the original is stored under. The list
#: is closed on purpose: accepting a format means the models downstream accept
#: it, and every added format is a promise about an endpoint we cannot test here.
_EXTENSIONS: dict[str, str] = {
    "PNG": "png",
    "JPEG": "jpeg",
    "WEBP": "webp",
    "GIF": "gif",
}

#: Pillow reports the JPEG format as ``"JPEG"``; the config spells the same
#: set in lowercase, so this maps one onto the other without a second list.
_CONFIG_NAMES: dict[str, str] = {
    "png": "PNG",
    "jpeg": "JPEG",
    "jpg": "JPEG",
    "webp": "WEBP",
    "gif": "GIF",
}


class ImageRejectedError(ValueError):
    """A content item that claims to be a picture but cannot be stored.

    Raised for bytes Pillow cannot identify, a format outside the accepted
    set, and a picture over the configured size. The ingest layer counts
    these and reports the count; it never fails the whole request for one
    bad picture, because the text around it is still worth remembering.
    """


@dataclass(frozen=True)
class ImageLimits:
    """Ingress bounds, read from ``[images]`` once per request.

    The two scale fields ride here rather than in a second type because every
    reader of a bound reads both: ingest accepts a picture, and the extraction
    loop immediately renders the copy it hands the model at ``model_edge``.

    Args:
        max_bytes: Largest accepted picture after decoding its content item.
        max_per_message: Pictures one message may contribute; the rest are
            rejected. Zero disables pictures entirely.
        max_request_bytes: Decoded picture bytes one add request may carry.
        max_per_request: Pictures one add request may carry in total.
        formats: Accepted formats, by config spelling.
        model_edge: Long-side cap of the copy the extraction and answering
            models are shown.
        cache_quality: JPEG quality of both derived copies.
    """

    max_bytes: int
    max_per_message: int
    max_request_bytes: int
    max_per_request: int
    formats: tuple[str, ...]
    model_edge: int = 1024
    cache_quality: int = 85

    def accepts(self, format_name: str) -> bool:
        """Whether ``format_name`` (Pillow's spelling, e.g. ``"JPEG"``) is enabled."""
        wanted = {
            _CONFIG_NAMES.get(f.strip().lower(), f.strip().upper())
            for f in self.formats
        }
        return format_name.upper() in wanted


@dataclass(frozen=True)
class StoredImage:
    """What an arrived picture is, once decoded and hashed.

    Args:
        image_id: The SHA-256 hex digest of the original bytes — the picture's
            identity everywhere: its storage name, the id an episode body
            references, and the handle ``forget`` takes.
        data: The original bytes, unchanged.
        format: Pillow's format name (``"JPEG"``).
        media_type: The lowercase extension the original is stored under.
        width: Pixel width of the original.
        height: Pixel height of the original.
    """

    image_id: str
    data: bytes
    format: str
    media_type: str
    width: int
    height: int

    @property
    def extension(self) -> str:
        """File extension without the dot, as used on disk."""
        return _EXTENSIONS.get(self.format, self.media_type)


ImageFormat = str
"""Pillow's format name for a stored picture (``"PNG"`` / ``"JPEG"`` / ...)."""


@dataclass(frozen=True)
class StoredImageInfo:
    """What the store knows about a picture it already holds.

    Deliberately not the bytes: a caller asking "how big is this picture" is
    rendering a description or building a download route, and loading the file
    to answer would turn a metadata query into a decode.

    Args:
        image_id: SHA-256 hex digest.
        media_type: Stored extension.
        byte_size: Size of the original in bytes.
        width: Pixel width.
        height: Pixel height.
        original_path: Memory-root-relative path.
        ref_count: Image-memory entries pointing at this file.
    """

    image_id: str
    media_type: str
    byte_size: int
    width: int
    height: int
    original_path: str
    ref_count: int


def hash_bytes(data: bytes) -> str:
    """The picture's identity: SHA-256 of the original bytes, lowercase hex."""
    return hashlib.sha256(data).hexdigest()


def original_relpath(image_id: str, extension: str) -> str:
    """``figures/<first two hex>/<image_id>.<extension>``, root-relative.

    Two levels, not one: a flat directory would hold every picture of every
    user in one inode list, and directory lookup on a network mount degrades
    well before the hash space runs out.
    """
    return f"figures/{image_id[:2]}/{image_id}.{extension}"


def derived_relpath(image_id: str, edge: int, extension: str = "jpeg") -> str:
    """``figures/.cache/<image_id>-<edge>.<extension>``, root-relative.

    One cache for every picture rather than a cache per hash bucket: the
    cache is disposable, so keeping it in one place makes "delete the derived
    copies" a single directory removal, and it never grows a second level of
    small directories alongside the originals.
    """
    return f"figures/.cache/{image_id}-{edge}.{extension}"


def decode_image(data: bytes, *, limits: ImageLimits) -> StoredImage:
    """Identify and validate one picture from its bytes.

    Reads only enough of the stream to learn the format and dimensions, then
    hands back the bytes untouched: the original is stored as sent, so that
    the file on disk hashes to the id it is filed under.

    Raises:
        ImageRejectedError: Over ``limits.max_bytes``, not an identifiable
            image, or a format outside ``limits.formats``.
    """
    if len(data) > limits.max_bytes:
        raise ImageRejectedError(
            f"picture is {len(data)} bytes, over the {limits.max_bytes} limit"
        )
    if not data:
        raise ImageRejectedError("picture is empty")
    try:
        with Image.open(io.BytesIO(data)) as image:
            format_name = (image.format or "").upper()
            width, height = image.size
    except UnidentifiedImageError as exc:
        raise ImageRejectedError("bytes are not an identifiable image") from exc
    except OSError as exc:
        raise ImageRejectedError(f"cannot read image: {exc}") from exc
    if format_name == "MPO":
        # A Multi-Picture Object (CIPA DC-007) is a baseline JPEG with more
        # frames appended, which is how many phones and stereo cameras save a
        # ``.jpg``. Every JPEG decoder shows its first frame, and so does
        # :func:`scaled_copy`, so it is filed and served as the JPEG it starts
        # with. The bytes are kept as sent, so the hash still names the file.
        format_name = "JPEG"
    if format_name not in _EXTENSIONS:
        raise ImageRejectedError(f"unsupported image format {format_name!r}")
    if not limits.accepts(format_name):
        raise ImageRejectedError(f"image format {format_name!r} is not enabled")
    return StoredImage(
        image_id=hash_bytes(data),
        data=data,
        format=format_name,
        media_type=_EXTENSIONS[format_name],
        width=width,
        height=height,
    )


def scaled_copy(data: bytes, *, edge: int, quality: int) -> tuple[bytes, str]:
    """Render one derived copy whose long side is at most ``edge`` pixels.

    Never enlarges: a picture already inside the bound is re-encoded as it is,
    which only normalises the container. An animated GIF collapses to its first
    frame — a decider or an extraction model reads one image per part, and
    keeping the animation would multiply payload for nothing.

    Also flattens transparency onto white -- an alpha channel (a PNG's or a
    WEBP's), a palette's transparent entry (GIF, PNG) or a transparent colour
    key -- because the wire formats a model accepts do not all carry an alpha
    channel and a transparent region reaching them as black is a silent
    misreading of the picture.

    The copy's format follows the picture and never ``edge``: PNG for a
    greyscale picture, JPEG for any other, so every copy of one picture has the
    same extension.

    Returns:
        ``(bytes, extension)`` of the derived copy.
    """
    with Image.open(io.BytesIO(data)) as image:
        image.seek(0)
        frame = _opaque(image.copy())
    width, height = frame.size
    longest = max(width, height)
    if longest > edge:
        scale = edge / longest
        frame = frame.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
    # JPEG is the wire format every endpoint accepts and the smallest of the
    # two for photographs; a picture with no colour (a screenshot of text)
    # stays PNG, where it is both sharper and smaller.
    out_format = "PNG" if frame.mode == "L" else "JPEG"
    buffer = io.BytesIO()
    frame.save(buffer, format=out_format, quality=quality, optimize=True)
    return buffer.getvalue(), _EXTENSIONS[out_format]


def _opaque(frame: Image.Image) -> Image.Image:
    """``frame`` as a copy is written: ``L`` when it is greyscale, else ``RGB``.

    A pixel that is not fully opaque is blended onto white by its alpha. A frame
    with no such pixel takes the plain conversion, so its copy is exactly what it
    was before transparency was handled; and the mode, hence the copy's format,
    never depends on transparency.
    """
    target = "L" if frame.mode == "L" else "RGB"
    if frame.has_transparency_data:
        with_alpha = frame.convert("LA" if target == "L" else "RGBA")
        alpha = with_alpha.getchannel("A")
        if alpha.getextrema()[0] < 255:
            background = Image.new(target, frame.size, "white")
            background.paste(with_alpha.convert(target), mask=alpha)
            return background
    return frame if frame.mode == target else frame.convert(target)


def dedupe(images: Iterable[StoredImage]) -> list[StoredImage]:
    """Drop repeats within one set of arrivals, keeping first-seen order.

    A message that carries the same picture twice still writes one memory for
    it: the second arrival adds a reference to the same file, and there is
    nothing new to describe.
    """
    seen: set[str] = set()
    kept: list[StoredImage] = []
    for image in images:
        if image.image_id in seen:
            continue
        seen.add(image.image_id)
        kept.append(image)
    return kept
