"""Tokenizer provider — sync app-layer tokenisation for BM25 indexing.

Public surface:

- :class:`Tokenizer` — Protocol every provider satisfies.
- :func:`build_tokenizer` — factory returning the tokenizer ``[tokenizer]``
  selects: ``jieba`` (default, mixed Chinese / English) or ``nltk`` (the
  EverAlgo LoCoMo harness pipeline, English only).

External usage::

    from everos.component.tokenizer import build_tokenizer
    tk = build_tokenizer(settings.tokenizer)
    tokens = tk.tokenize("hello 世界")  # jieba: ['hello', '世界']
"""

from .factory import build_tokenizer as build_tokenizer
from .protocol import Tokenizer as Tokenizer

__all__ = [
    "Tokenizer",
    "build_tokenizer",
]
