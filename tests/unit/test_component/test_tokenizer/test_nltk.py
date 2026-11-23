"""Unit tests for :class:`NltkTokenizer` and its selection through the factory.

Pins the contract the LoCoMo benchmark relies on: the tokens equal what the
EverAlgo harness (``benchmarks/common/stages/_tokenize.py`` at ``d9fe22e``)
produces for the same text. Every expected list below was produced by running
that harness function on the input. Missing NLTK data or a missing extra must
stop construction with an actionable error instead of downloading or
returning an empty token stream.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import nltk
import pytest

from everos.component.tokenizer import build_tokenizer, nltk_provider
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


def test_a_word_after_an_opening_quote_is_kept() -> None:
    """NLTK 3.10 splits a leading quote off a word; 3.9.4 dropped the word.

    The one token difference between the two releases on LoCoMo + LongMemEval
    text (449 of 83,484 texts). The harness above ran 3.9.4, so it has no
    such case; this pins the pinned release's behaviour.
    """
    text = "Camping with my fam, 'cause nature helps."
    assert NltkTokenizer().tokenize(text) == ["camp", "fam", "caus", "natur", "help"]


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
    monkeypatch.setattr(nltk.data, "path", [])  # nothing on disk either
    with pytest.raises(CapabilityError) as err:
        NltkTokenizer()
    assert "python -m nltk.downloader punkt_tab stopwords" in str(err.value)


@pytest.mark.skipif(os.name != "posix", reason="NLTK checks ownership on POSIX only")
def test_data_another_account_could_write_is_named_with_the_fix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """NLTK >= 3.10 skips group- or world-writable data, so it reads as missing.

    The error must say why and how to fix it, not ask for a download.
    """
    root = tmp_path / "nltk_data"
    for resource in ("tokenizers/punkt_tab/english", "corpora/stopwords"):
        (root / resource).mkdir(parents=True)
    monkeypatch.setattr(nltk.data, "path", [str(root)])

    for d in [root, *root.rglob("*")]:
        d.chmod(0o775)
    with (
        pytest.warns(RuntimeWarning, match="NLTK will not read"),
        pytest.raises(CapabilityError) as err,
    ):
        nltk_provider._require_data()
    assert f"chmod -R go-w {root}" in str(err.value)
    assert "nltk.downloader" not in str(err.value)

    for d in [root, *root.rglob("*")]:
        d.chmod(0o755)
    nltk_provider._require_data()  # the same data, private: accepted


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
