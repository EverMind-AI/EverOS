"""Unit tests for :class:`NltkTokenizer` and its selection through the factory.

Pins the contract the LoCoMo benchmark relies on: the tokens equal what the
EverAlgo harness (``benchmarks/common/stages/_tokenize.py`` at ``d9fe22e``)
produces for the same text. Every expected list below was produced by running
that harness function on the input. Missing NLTK data or a missing extra must
stop construction with an actionable error instead of downloading or
returning an empty token stream.
"""

from __future__ import annotations

import sys
import warnings

import nltk
import pytest

from everos.component.tokenizer import build_tokenizer
from everos.component.tokenizer.nltk_provider import NltkTokenizer
from everos.config import TokenizerSettings
from everos.core.errors import CapabilityError


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Caroline's paintings were shown at the LGBTQ center; she painted sunsets.",
            ["carolin", "paint", "shown", "lgbtq", "center", "paint", "sunset"],
        ),
        (
            "On May 18, 2023 at 13:47 UTC, Melanie said she'd go hiking in 2024!",
            ["may", "utc", "melani", "said", "go", "hike"],
        ),
        (
            "I went to a café in Los Angeles - it's wonderful.",
            ["went", "café", "lo", "angel", "wonder"],
        ),
        (
            "The kids' swimming lessons don't start until next week.",
            ["kid", "swim", "lesson", "start", "next", "week"],
        ),
        ("What did Melanie paint?", ["melani", "paint"]),
        ("我上周去北京参加了机器学习研讨会。", []),
        ("a I x", []),
        ("", []),
    ],
)
def test_tokens_match_the_everalgo_harness(text: str, expected: list[str]) -> None:
    assert NltkTokenizer().tokenize(text) == expected


def test_tokenize_batch_preserves_order() -> None:
    texts = ["What did Melanie paint?", "", "swimming lessons"]
    assert NltkTokenizer().tokenize_batch(texts) == [
        ["melani", "paint"],
        [],
        ["swim", "lesson"],
    ]


def test_missing_nltk_data_fails_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every missing resource is named, with the command that installs it."""

    def _missing(resource: str) -> None:
        raise LookupError(resource)

    monkeypatch.setattr(nltk.data, "find", _missing)
    with pytest.raises(CapabilityError) as err:
        NltkTokenizer()
    assert "python -m nltk.downloader punkt_tab stopwords" in str(err.value)


def test_the_factory_builds_nltk_when_selected() -> None:
    tokenizer = build_tokenizer(TokenizerSettings(name="nltk"))
    assert isinstance(tokenizer, NltkTokenizer)


def test_the_factory_defaults_to_jieba() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid escape sequence", DeprecationWarning)
        from everos.component.tokenizer.jieba_provider import JiebaTokenizer

    assert isinstance(build_tokenizer(TokenizerSettings()), JiebaTokenizer)


def test_selecting_nltk_without_the_extra_names_the_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``None`` in ``sys.modules`` makes the deferred import raise ImportError."""
    monkeypatch.setitem(sys.modules, "everos.component.tokenizer.nltk_provider", None)
    with pytest.raises(CapabilityError) as err:
        build_tokenizer(TokenizerSettings(name="nltk"))
    assert "pip install 'everos[nltk]'" in str(err.value)
