"""Tokenizer provider — sync app-layer tokenisation for BM25 indexing.

Public surface:

- :class:`Tokenizer` — Protocol every provider satisfies.
- :class:`JiebaTokenizer` — default jieba-backed implementation.
- :func:`build_tokenizer` — factory returning the configured tokenizer.

External usage::

    from everos.component.tokenizer import build_tokenizer
    tk = build_tokenizer()
    tokens = tk.tokenize("hello 世界")  # ['hello', '世界']
"""

from .factory import build_tokenizer as build_tokenizer
from .jieba_provider import JiebaTokenizer as JiebaTokenizer
from .protocol import Tokenizer as Tokenizer

__all__ = [
    "JiebaTokenizer",
    "Tokenizer",
    "build_tokenizer",
]
