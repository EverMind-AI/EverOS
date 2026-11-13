"""Factory for the keyword tokenizer shared by indexing and search.

``[tokenizer] name`` picks the implementation. The composition root builds one
instance and hands it to both the cascade (write side) and the recallers
(query side); building two would let them disagree.
"""

from __future__ import annotations

from everos.config import TokenizerSettings
from everos.core.errors import CapabilityError

from .protocol import Tokenizer


def build_tokenizer(settings: TokenizerSettings | None = None) -> Tokenizer:
    """Build the tokenizer ``settings`` selects.

    Args:
        settings: The ``[tokenizer]`` section; ``None`` means its defaults
            (``jieba``).

    Returns:
        The selected :class:`Tokenizer`.

    Raises:
        CapabilityError: ``nltk`` is selected without the ``[nltk]`` extra or
            without its NLTK data.
    """
    name = (settings or TokenizerSettings()).name
    if name == "nltk":
        return _build_nltk()
    # Deferred: jieba ships invalid escape sequences that warn on Python 3.12+;
    # importing at call time keeps that cost off module import.
    from .jieba_provider import JiebaTokenizer

    return JiebaTokenizer()


def _build_nltk() -> Tokenizer:
    try:
        # Deferred: nltk is the optional ``everos[nltk]`` extra.
        from .nltk_provider import NltkTokenizer
    except ImportError as exc:
        raise CapabilityError(
            "[tokenizer] name = 'nltk' needs the nltk extra. "
            "Install with: pip install 'everos[nltk]'"
        ) from exc
    return NltkTokenizer()
