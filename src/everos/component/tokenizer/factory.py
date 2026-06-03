"""Factory for the cascade-time tokenizer.

Single implementation today (``JiebaTokenizer``). Lifting this into a
factory keeps callers (cascade handler) decoupled from the concrete
choice, so swapping to char-bigram / hf tokenizer later is a one-file
change — see ``17_lancedb_tables_design.md`` §2.4.1.
"""

from __future__ import annotations

from .jieba_provider import JiebaTokenizer
from .protocol import Tokenizer


def build_tokenizer() -> Tokenizer:
    """Build the default tokenizer (``JiebaTokenizer``)."""
    return JiebaTokenizer()
