"""Synthesize a user Profile from chronological MemCells or dated Episodes, keeping the profile's timeline.

Modules, each importing only those
below it: ``_extractor`` (:class:`ProfileExtractor`) wires ``_stages`` (prompt templates per stage, over
``_replies``: reply models and LLM callsites), ``_render`` (what the model is shown) and ``_merge`` (UPDATE ops,
dedupe, the stored profile); ``_render`` and ``_merge`` share only ``_items``.

Log lines:
    The lines below are a contract: each starts with ``profile <event>:`` followed by space-separated
    ``key=value`` pairs, and none carries a prompt, a model output or an item's content — only counts, dates
    (epoch ms), lengths and enum values. ``reason`` is the one value with spaces; it runs to the next
    `` key=`` or the end of the line. Every line is logged on ``everos.algo.user_memory.profile``, whichever
    module writes it.

    ``profile update applied: pass= observed_at_ms= add= update= delete= none= rejected= rewrote_description=
    history_appended= grounding_only= canonicalised= profile_timestamp_ms=<old>-><new> explicit= implicit=`` (INFO)
        Once per UPDATE pass. ``pass`` is ``historical`` or ``current`` for the two halves of an Episode batch
        split around the stored timestamp, ``single`` for the MemCell path, which never splits.
        ``observed_at_ms`` is the date the pass's ops are judged against. ``add`` / ``update`` / ``delete`` /
        ``none`` count the reply's ops by action; ``rejected`` counts the ops refused, where the updates to an
        index the same reply deletes count once. Of the updates that land — several on one index merge into one —
        ``rewrote_description`` restate the fact, ``history_appended`` come from a pass older than their item and
        add earlier history in a parenthesis after its description (the item keeps its ``observed_at``), and
        ``grounding_only`` only add evidence or basis.
        ``canonicalised`` counts adds whose label was respelled to one already on file. ``explicit`` /
        ``implicit`` are the merged bucket sizes, before any COMPACT or REGROUP.
    ``profile update op rejected: action= type= index= reason=`` (WARNING)
        One per refused op. A stale rewrite or delete appends its criterion:
        ``reason=stale evidence item_observed_at_ms= batch_observed_at_ms=``.
    ``profile add label canonicalised: bucket= from_len= to_len=`` (DEBUG)
        One per respelled add label; lengths only, the label being model-written text.
    ``profile duplicates dropped: source= count=`` (WARNING)
        One per deduplicated list that lost items; ``source`` is ``init``, ``episode-init``, ``update``,
        ``compact`` or ``regroup``.

    The Episode-path entry line, ``extracting profile from Episodes: ...``, carries ``categories=``, the size of
    the caller's category snapshot after normalisation. Both entry lines log ``output_language`` once it is
    validated, as the :class:`OutputLanguage` member's value or ``None``. The REGROUP lines name the group by
    ``bucket`` and ``label_len``, never by the label's text.
"""

from __future__ import annotations

__all__ = ["ProfileExtractor"]

from everos.algo.user_memory.profile._extractor import ProfileExtractor
