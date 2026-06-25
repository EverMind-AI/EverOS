"""Factory for the cascade-time tokenizer.

Single implementation today (``JiebaTokenizer``). Lifting this into a
factory keeps callers (cascade handler) decoupled from the concrete
choice, so swapping to char-bigram / hf tokenizer later is a one-file
change — see ``17_lancedb_tables_design.md`` §2.4.1.
"""

from __future__ import annotations

from .protocol import Tokenizer


def build_tokenizer() -> Tokenizer:
    """Build the default tokenizer (``JiebaTokenizer``)."""
    # Deferred: jieba contains invalid escape sequences that raise
    # SyntaxError on Python 3.12+; defer so the cost is paid only when
    # tokenization is actually needed (not at import time).
    from .jieba_provider import JiebaTokenizer

    return JiebaTokenizer()
