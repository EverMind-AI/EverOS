"""Cascade handlers — one per kind, sharing the :class:`Handler` chassis.

Three daily-log handlers (episode / image_memory / group_profile) inherit
:class:`BaseDailyLogHandler` for the shared read / diff / embed / upsert /
delete loop; the per-kind subclass only declares its repo binding, the texts
to embed (``_embed_texts``, none for group_profile) and the ``_build_row``
mapping. The profiles (``user_profile`` and ``group_member_profile``, one
:class:`ProfileHandler`) and ``group_roster`` stand alone — they're
single-file kinds (no entries, no per-entry diff), so they implement
:class:`Handler` directly and own their reconcile loop.
"""

from .base import Handler as Handler
from .base import HandlerDeps as HandlerDeps
from .episode import EpisodeHandler as EpisodeHandler
from .group_profile import GroupProfileHandler as GroupProfileHandler
from .group_roster import GroupRosterHandler as GroupRosterHandler
from .image_memory import ImageMemoryHandler as ImageMemoryHandler
from .profile import ProfileHandler as ProfileHandler

__all__ = [
    "EpisodeHandler",
    "GroupProfileHandler",
    "GroupRosterHandler",
    "Handler",
    "HandlerDeps",
    "ImageMemoryHandler",
    "ProfileHandler",
]
