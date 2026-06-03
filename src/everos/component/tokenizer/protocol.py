"""Tokenizer protocol.

App-layer tokenisation gates every BM25-indexed field in LanceDB
(``17_lancedb_tables_design.md`` §2.4.1): the source surface form lives
in ``<field>`` while the space-joined token stream lives in
``<field>_tokens``, and the FTS index reads only the latter using a
whitespace tokenizer. Keeping the tokenizer decision in the app layer
means it can swap (jieba → unigram → hf) without re-indexing or
touching LanceDB schemas.

The protocol is sync — every concrete tokenizer in scope today (jieba,
char-bigram, regex word-split) is CPU-bound with no IO, so an async
wrapper would just shuffle work onto the event loop. If a future GPU
or remote tokenizer needs IO it should add an async method explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Sync tokeniser contract used by the cascade handler."""

    def tokenize(self, text: str) -> list[str]:
        """Return the ordered token list for ``text``.

        Implementations must drop empty / whitespace-only tokens so the
        resulting space-joined string never carries adjacent spaces.
        """
        ...

    def tokenize_batch(self, texts: Sequence[str]) -> list[list[str]]:
        """Tokenise many strings, preserving input order."""
        ...
