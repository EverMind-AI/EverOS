"""Text measurement and decoding that holds across languages and unknown charsets."""

from __future__ import annotations

__all__ = ["ascii_width", "decode_bytes"]

import unicodedata


def ascii_width(text: str) -> int:
    """Return the length of ``text`` in ASCII-equivalent units: East Asian wide characters count 2, the rest 1.

    One yardstick for every language: 200 units read as one or two short sentences in English and in CJK alike.
    """
    return sum(
        2 if unicodedata.east_asian_width(char) in ("W", "F") else 1 for char in text
    )


def decode_bytes(content: bytes) -> str:
    """Decode bytes of unknown charset as UTF-8, else GBK, else Latin-1.

    Latin-1 maps every byte to a character, so decoding never fails; GBK covers GB2312.
    """
    for encoding in ("utf-8", "gbk"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("latin-1")
