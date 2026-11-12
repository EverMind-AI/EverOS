"""Episode daily-log writer — md is the SoT for Episode memories.

Stays in the chassis style: caller hands in pre-built ``inline`` and
``sections`` dicts plus the scope id (``owner_id``). Domain →
structured-entry shaping lives in the calling pipeline (cf. architecture
rule: ``infra`` may not import ``memory``).

One writer for a user's episodes (``users/<id>/episodes/``, ``ep_`` ids) and
a group's (``groups/<id>/episodes/``, ``gep_`` ids); an instance writes one
owner type's (:class:`~.base.OwnedDailyWriter`).

This milestone assumes well-behaved callers (no retransmit dedupe needed).
The writer just appends; the chassis manages the in-file ``entry_id``
sequence, which is the single source of identity for an md entry.
"""

from __future__ import annotations

from .._mds import EpisodeDailyFrontmatter, GroupEpisodeDailyFrontmatter
from .base import OwnedDailyWriter


class EpisodeWriter(OwnedDailyWriter):
    """Daily-log writer for the Episode schema (md = SoT).

    ``append_entry`` / ``append_entries`` come from
    :class:`BaseDailyWriter`; the ``entry_id`` (``ep_<YYYYMMDD>_<NNNN>``, a
    group's ``gep_``) is the in-file identity allocated under the per-path
    lock. Callers can derive a globally-unique id from
    ``(owner_id, entry_id)`` without persisting any algo-side uuid.
    """

    schemas = {"user": EpisodeDailyFrontmatter, "group": GroupEpisodeDailyFrontmatter}
    log_name = "episode_log"
