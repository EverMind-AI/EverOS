"""The derived copy: transparency on white, and one format per picture.

What these pin:

- Every kind of transparency is flattened onto white -- an alpha channel (PNG,
  WEBP, greyscale with alpha), a palette's transparent entry (GIF, PNG) and a
  colour key -- whatever the original's format. A transparent region that
  reached a model as black (the colour hidden under it) is a misreading of the
  picture: the decider, the extraction model and the answering model all read
  these copies.
- A picture with no transparent pixel renders exactly as a plain conversion
  would: an opaque picture's copy does not change.
- The copy's format follows the picture and never the edge (PNG for a greyscale
  picture, JPEG for any other), so every copy of one picture has the same
  extension and the search can name the thumbnail after the model copy.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from everos.component.images import scaled_copy

_HIDDEN = (0, 0, 0)  # what a transparent pixel stores under its alpha
_RED = (220, 20, 30)


def _encode(image: Image.Image, fmt: str, **params: object) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **params)
    return buffer.getvalue()


def _half_transparent_rgba(size: int = 64) -> Image.Image:
    """Left half fully transparent over black, right half opaque red."""
    image = Image.new("RGBA", (size, size), (*_HIDDEN, 0))
    image.paste((*_RED, 255), (size // 2, 0, size, size))
    return image


def _copy(data: bytes, edge: int = 1024) -> Image.Image:
    rendered, _ = scaled_copy(data, edge=edge, quality=95)
    return Image.open(io.BytesIO(rendered)).convert("RGB")


def _near(pixel: tuple[int, ...], expected: tuple[int, ...], slack: int = 12) -> bool:
    return all(abs(a - b) <= slack for a, b in zip(pixel, expected, strict=True))


@pytest.mark.parametrize("fmt", ["PNG", "WEBP"])
def test_an_alpha_channel_is_flattened_onto_white(fmt: str) -> None:
    params = {"lossless": True} if fmt == "WEBP" else {}
    copy = _copy(_encode(_half_transparent_rgba(), fmt, **params))

    assert _near(copy.getpixel((8, 32)), (255, 255, 255))
    assert _near(copy.getpixel((56, 32)), _RED)


def test_a_half_transparent_pixel_is_blended_onto_white() -> None:
    image = Image.new("RGBA", (32, 32), (*_HIDDEN, 128))

    copy = _copy(_encode(image, "PNG"))

    assert _near(copy.getpixel((16, 16)), (127, 127, 127))


def test_a_gif_s_transparent_entry_is_white() -> None:
    image = Image.new("P", (32, 32), 0)
    image.putpalette([*_HIDDEN, *_RED] + [0] * (254 * 3))
    image.paste(1, (16, 0, 32, 32))

    copy = _copy(_encode(image, "GIF", transparency=0))

    assert _near(copy.getpixel((4, 16)), (255, 255, 255))
    assert _near(copy.getpixel((28, 16)), _RED)


def test_a_palette_png_s_transparent_entry_is_white() -> None:
    image = Image.new("P", (32, 32), 0)
    image.putpalette([*_HIDDEN, *_RED] + [0] * (254 * 3))
    image.paste(1, (16, 0, 32, 32))

    copy = _copy(_encode(image, "PNG", transparency=0))

    assert _near(copy.getpixel((4, 16)), (255, 255, 255))
    assert _near(copy.getpixel((28, 16)), _RED)


def test_greyscale_with_alpha_is_flattened_onto_white() -> None:
    image = Image.new("LA", (32, 32), (0, 0))
    image.paste((40, 255), (16, 0, 32, 32))

    copy = _copy(_encode(image, "PNG"))

    assert _near(copy.getpixel((4, 16)), (255, 255, 255))
    assert _near(copy.getpixel((28, 16)), (40, 40, 40))


def test_a_greyscale_colour_key_is_white_and_stays_png() -> None:
    image = Image.new("L", (32, 32), 0)
    image.paste(90, (16, 0, 32, 32))

    rendered, extension = scaled_copy(
        _encode(image, "PNG", transparency=0), edge=1024, quality=85
    )

    assert extension == "png"
    with Image.open(io.BytesIO(rendered)) as copy:
        assert copy.mode == "L"
        assert copy.getpixel((4, 16)) == 255
        assert copy.getpixel((28, 16)) == 90


def test_an_opaque_picture_renders_as_a_plain_conversion() -> None:
    """An alpha channel with nothing transparent changes nothing in the copy."""
    opaque = Image.new("RGBA", (300, 200), (*_RED, 255))
    opaque.paste((10, 200, 90, 255), (0, 0, 150, 100))

    with_alpha = scaled_copy(_encode(opaque, "PNG"), edge=128, quality=85)
    without = scaled_copy(_encode(opaque.convert("RGB"), "PNG"), edge=128, quality=85)

    assert with_alpha == without


@pytest.mark.parametrize(
    ("mode", "fmt", "extension"),
    [
        ("L", "PNG", "png"),
        ("L", "JPEG", "png"),
        ("RGB", "JPEG", "jpeg"),
        ("RGBA", "PNG", "jpeg"),
        ("LA", "PNG", "jpeg"),
        ("P", "GIF", "jpeg"),
    ],
)
def test_the_format_follows_the_picture_at_every_edge(
    mode: str, fmt: str, extension: str
) -> None:
    data = _encode(Image.new(mode, (900, 600)), fmt)

    assert {scaled_copy(data, edge=e, quality=85)[1] for e in (64, 448, 1024)} == {
        extension
    }
