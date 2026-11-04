"""Write one Episode — a titled third-person narrative with a short summary — from a MemCell, or just its summary.

Modules, each importing only those below it: ``_extractor`` (:class:`EpisodeExtractor`) and ``_summarizer``
(:class:`EpisodeSummarizer`, the summary of an episode text already stored) each wire ``_render`` (the system and user
messages the model is shown), ``_replies`` (the reply models and the LLM callsites held to them) and ``_summary`` (the
width guard on the stored summary, the same on both paths).

Log lines, all on ``everos.algo.user_memory.episode`` whichever module writes them; none carries a prompt or model text:

    ``extracting episode: <n> items, <n> rendered chars, output_language=<value>`` (INFO)
        Once per :class:`EpisodeExtractor` call, before the model is asked. ``rendered chars`` is the user message's
        length: the conversation as rendered in its ``<conversation>`` element. ``output_language`` is the caller's
        argument as given.
    ``episode extracted: content <n> units, summary <n> units`` (INFO)
        Once per :class:`EpisodeExtractor` call that returns an Episode, widths in ASCII-equivalent units after the
        summary guard.
    ``episode summarized: text <n> units, summary <n> units, truncated=<bool>`` (INFO)
        Once per :class:`EpisodeSummarizer` call that returns a summary: the input text's width, the summary's width
        after the guard, and whether the guard cut it.
    ``episode summary over cap (<n> > <cap> units)`` (WARNING)
        The model wrote a summary wider than the cap, on either path; its rate is the production model's violation
        rate.
    ``episode summary truncated: <n> -> <n> units`` (INFO)
        Follows every over-cap line: the stored summary's width after the sentence-boundary cut.
"""

from __future__ import annotations

__all__ = ["EpisodeExtractor", "EpisodeSummarizer"]

from everos.algo.user_memory.episode._extractor import EpisodeExtractor
from everos.algo.user_memory.episode._summarizer import EpisodeSummarizer
