"""component.parser — shared multimodal file parsing via everalgo-parser.

External usage:
    from everos.component.parser import aparse_file, parser_available
"""

from __future__ import annotations

from ._core import aparse_file as aparse_file
from ._core import parser_available as parser_available
from ._core import require_parser as require_parser
from .input import RawFileBuilder as RawFileBuilder
from .input import build_raw_file as build_raw_file

__all__ = [
    "RawFileBuilder",
    "aparse_file",
    "build_raw_file",
    "parser_available",
    "require_parser",
]
