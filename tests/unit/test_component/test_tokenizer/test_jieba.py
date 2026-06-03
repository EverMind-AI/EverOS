"""Unit tests for :class:`JiebaTokenizer`.

Verify the contract that callers downstream depend on:

* clean token list (no whitespace, no empty strings),
* CJK + ASCII pass-through under ``cut_for_search`` segmentation,
* default stopword + ``min_length=2`` filter applied,
* batch preserves order.

The tokenizer is symmetric — cascade write side and search query side
both go through this code path, so changes here change BM25 recall on
both ends.
"""

from __future__ import annotations

from everos.component.tokenizer import JiebaTokenizer, build_tokenizer


def test_tokenize_returns_list_for_english() -> None:
    tokens = JiebaTokenizer().tokenize("hello world")
    assert tokens == ["hello", "world"]


def test_tokenize_drops_pure_whitespace() -> None:
    """Whitespace-only tokens never reach the BM25 column."""
    tokens = JiebaTokenizer().tokenize("foo   bar")
    assert all(t.strip() for t in tokens)


def test_tokenize_empty_input() -> None:
    assert JiebaTokenizer().tokenize("") == []


def test_tokenize_cjk_keeps_multichar_words() -> None:
    """``cut_for_search`` keeps multi-character compounds usable by BM25."""
    tokens = JiebaTokenizer().tokenize("我爱北京天安门")
    # Single-char tokens (我 / 爱) are filtered by min_length=2 (and 我
    # is also in the default stopword set). Multi-char compounds survive.
    assert "我" not in tokens
    assert "爱" not in tokens
    assert "北京" in tokens
    assert any(t in {"天安门", "天安"} for t in tokens)


def test_tokenize_drops_default_english_stopwords() -> None:
    tokens = JiebaTokenizer().tokenize("the quick brown fox")
    assert "the" not in tokens
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens


def test_tokenize_drops_short_tokens_below_min_length() -> None:
    """Single-char ASCII tokens are dropped by the default ``min_length=2``."""
    tokens = JiebaTokenizer().tokenize("a quick b run")
    assert "a" not in tokens
    assert "b" not in tokens
    assert "quick" in tokens
    assert "run" in tokens


def test_tokenize_is_case_insensitive() -> None:
    """Lowercasing is part of the symmetric contract."""
    tokens = JiebaTokenizer().tokenize("HELLO World")
    assert tokens == ["hello", "world"]


def test_extra_stopwords_extend_defaults() -> None:
    tk = JiebaTokenizer(extra_stopwords=frozenset({"hello"}))
    tokens = tk.tokenize("hello world")
    assert "hello" not in tokens
    assert "world" in tokens


def test_custom_min_token_length_relaxes_filter() -> None:
    """Lower ``min_length`` lets shorter tokens through.

    Stopword filter still applies — even at ``min_length=1`` the English
    article ``"a"`` stays filtered because it's in the default stopwords.
    """
    tokens = JiebaTokenizer(min_token_length=1).tokenize("a quick b")
    # 'a' is in the default English stopword set even at min_length=1.
    assert "a" not in tokens
    assert "b" in tokens
    assert "quick" in tokens


def test_tokenize_batch_preserves_order() -> None:
    tk = JiebaTokenizer()
    out = tk.tokenize_batch(["foo bar", "baz", ""])
    assert len(out) == 3
    assert out[2] == []


def test_build_tokenizer_returns_jieba_default() -> None:
    """Factory exposes the same JiebaTokenizer the cascade handler uses."""
    assert isinstance(build_tokenizer(), JiebaTokenizer)
