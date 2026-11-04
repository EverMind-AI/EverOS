"""The ``<reply_format>`` block every parser system prompt ends with, shared by the English and Chinese sets."""

from __future__ import annotations

__all__ = ["TEXT_REPLY_FORMAT"]

from everos.algo.utils import tagged

TEXT_REPLY_FORMAT = tagged(
    "reply_format",
    """\
Reply with one JSON object, nothing else.
{"text": "..."}""",
)
"""Asks for ``parser._replies._TextReply``: the output the rules above describe goes in ``text``."""
