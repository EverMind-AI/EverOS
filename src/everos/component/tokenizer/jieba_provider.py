"""Jieba-based tokenizer — covers CJK + English mixed content.

Uses ``jieba.cut_for_search`` (search-mode segmentation: yields both the
greedy max-match segment and its finer sub-segments for compound CJK
words). Same mode as the legacy enterprise keyword-search path uses on
the query side — keeping cascade write and search query symmetric is
the hard contract for BM25 recall to work.

After segmentation we drop:

* whitespace / empty tokens (so the join-on-space output stays clean),
* tokens shorter than ``min_token_length`` (default 2 — same threshold
  enterprise's ``filter_stopwords(min_length=2)`` uses; single-char
  fragments mostly hurt BM25 precision),
* tokens in a small bilingual stopword set (Chinese function words +
  English articles / prepositions / aux verbs).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import jieba

# Small bilingual stopword set. Intentionally tight (not a full
# Chinese stopword list) so the behaviour is predictable; callers
# tuning recall can subclass / extend.
_DEFAULT_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        # English — articles / prepositions / aux verbs that dominate BM25
        # idf-noise but add no recall value.
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        # Chinese — function words / particles. ``cut_for_search`` emits
        # these as single-char tokens anyway, and the min_length=2 floor
        # would drop most; listing them explicitly makes the intent clear
        # and is a no-op when min_length filtering also kicks in.
        "的",
        "了",
        "和",
        "是",
        "在",
        "我",
        "你",
        "他",
        "她",
        "它",
        "也",
        "都",
        "就",
        "还",
        "或",
        "及",
        "与",
        "对",
        "把",
        "被",
        "有",
        "没",
        "不",
        "啊",
        "吗",
        "呢",
        "吧",
        "哦",
    }
)

_DEFAULT_MIN_TOKEN_LENGTH: Final[int] = 2


class JiebaTokenizer:
    """Tokenizer that calls into ``jieba.cut_for_search`` and filters."""

    def __init__(
        self,
        *,
        min_token_length: int = _DEFAULT_MIN_TOKEN_LENGTH,
        extra_stopwords: frozenset[str] | None = None,
    ) -> None:
        # Touching ``jieba.initialize()`` here would force eager dict load
        # at import time and balloon test-collection latency. ``jieba.cut*``
        # lazy-loads on first call instead.
        self._min_len = min_token_length
        self._stopwords = (
            _DEFAULT_STOPWORDS | extra_stopwords
            if extra_stopwords
            else _DEFAULT_STOPWORDS
        )

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        out: list[str] = []
        for raw in jieba.cut_for_search(text):
            tok = raw.strip().lower()
            if not tok or tok.isspace():
                continue
            if len(tok) < self._min_len:
                continue
            if tok in self._stopwords:
                continue
            out.append(tok)
        return out

    def tokenize_batch(self, texts: Sequence[str]) -> list[list[str]]:
        return [self.tokenize(t) for t in texts]
