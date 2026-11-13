"""NLTK tokenizer — the keyword pipeline of the EverAlgo LoCoMo harness.

Reproduces ``benchmarks/common/stages/_tokenize.py`` of EverAlgo ``d9fe22e``
token for token: lowercase, NLTK ``word_tokenize``, keep alphabetic tokens of
at least two characters that are not NLTK English stopwords, Porter stem
(NLTK's default mode). A store indexed with it scores keyword recall the way
that harness does, which is what the LoCoMo benchmark selects it for.

Load-bearing rules:

* English only. Digits never become tokens, and CJK text is not segmented: an
  unspaced run is at most one token, dropped whenever it carries punctuation.
  ``jieba`` stays the default.
* The ``punkt_tab`` and ``stopwords`` NLTK data must already be installed.
  Nothing is downloaded at runtime: a missing resource fails at construction.
* The token stream depends on the NLTK release, so the ``[nltk]`` extra pins
  the one the harness used.
"""

from __future__ import annotations

__all__ = ["NltkTokenizer"]

from collections.abc import Sequence
from typing import Final

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from everos.core.errors import CapabilityError

_REQUIRED_DATA: Final[dict[str, str]] = {
    "punkt_tab": "tokenizers/punkt_tab/english/",
    "stopwords": "corpora/stopwords",
}
_MIN_TOKEN_LENGTH: Final[int] = 2


class NltkTokenizer:
    """English keyword tokenizer matching the EverAlgo LoCoMo harness.

    Raises:
        CapabilityError: When the ``punkt_tab`` or ``stopwords`` NLTK data is
            not installed.
    """

    def __init__(self) -> None:
        _require_data()
        self._stopwords = frozenset(stopwords.words("english"))
        self._stem = PorterStemmer().stem

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        return [
            self._stem(token)
            for token in word_tokenize(text.lower())
            if self._is_kept(token)
        ]

    def tokenize_batch(self, texts: Sequence[str]) -> list[list[str]]:
        return [self.tokenize(text) for text in texts]

    def _is_kept(self, token: str) -> bool:
        return (
            token.isalpha()
            and len(token) >= _MIN_TOKEN_LENGTH
            and token not in self._stopwords
        )


def _require_data() -> None:
    """Fail at construction when NLTK data the pipeline reads is missing.

    Raises:
        CapabilityError: Naming every missing resource and the install command.
    """
    missing = [name for name, path in _REQUIRED_DATA.items() if not _has(path)]
    if missing:
        raise CapabilityError(
            f"The nltk tokenizer needs NLTK data that is not installed: "
            f"{', '.join(missing)}. Install with: "
            f"python -m nltk.downloader {' '.join(missing)}"
        )


def _has(resource: str) -> bool:
    try:
        nltk.data.find(resource)
    except LookupError:
        return False
    return True
