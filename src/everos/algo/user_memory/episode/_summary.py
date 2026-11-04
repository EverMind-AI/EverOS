"""The width guard on the stored ``summary``: an over-cap one is cut at a sentence boundary, never failed.

The summary is a display preview, and the caller's failure model retries the whole extraction on an exception and
drops the episode after the last attempt; failing the main product over a cosmetic field is the one outcome this guard
exists to prevent.
"""

from __future__ import annotations

import logging

from everos.algo.utils import ascii_width

# The package's name rather than this module's: hosts collect the log contract under it.
logger = logging.getLogger("everos.algo.user_memory.episode")

# Hard cap on the stored ``summary``, in ASCII-equivalent units. The 400-unit ceiling accommodates the prompt's
# ~50-English-word target with headroom while still catching pathological outputs. Production
# 2026-08: a qwen3-4b finetune restated the whole ``content`` as ``summary`` on ~22% of
# episodes (max 9038 chars) — the prompt alone does not hold the line, so the extractor does.
_SUMMARY_WIDTH_CAP = 400

_ELLIPSIS = "\u2026"
_SENTENCE_TERMINATORS = ".\u3002\uff0e\uff01\uff1f!?"


def _cap_summary(summary: str) -> str:
    """Return ``summary`` unchanged when within ``_SUMMARY_WIDTH_CAP``, else its sentence-boundary truncation.

    The truncated preview ends with an ellipsis and loses coverage, never correctness: every kept sentence is one the
    model wrote.
    """
    original_width = ascii_width(summary)
    if original_width <= _SUMMARY_WIDTH_CAP:
        return summary
    logger.warning(
        "episode summary over cap (%d > %d units)", original_width, _SUMMARY_WIDTH_CAP
    )
    truncated = _truncate_at_sentence_boundary(summary, _SUMMARY_WIDTH_CAP)
    logger.info(
        "episode summary truncated: %d -> %d units",
        original_width,
        ascii_width(truncated),
    )
    return truncated


def _truncate_at_sentence_boundary(text: str, cap: int) -> str:
    """Longest prefix of whole sentences within ``cap`` width, ellipsis-terminated.

    The ellipsis REPLACES the final sentence terminator (never sits after it), so the
    cut reads as one mark, not two. Falls back to a bare width cut only when even the first sentence exceeds the budget
    (a sentence-less run); mid-word is still better than over-cap there, and the ellipsis
    marks the cut either way.
    """
    budget = max(cap - ascii_width(_ELLIPSIS), 1)
    width = 0
    last_sentence_end = 0
    hard_cut = len(text)
    for i, ch in enumerate(text):
        width += ascii_width(ch)
        if width > budget:
            hard_cut = i
            break
        if ch in _SENTENCE_TERMINATORS:
            last_sentence_end = i + 1
    if width <= budget:
        return text
    cut = last_sentence_end or hard_cut
    return text[:cut].rstrip().rstrip(_SENTENCE_TERMINATORS) + _ELLIPSIS
