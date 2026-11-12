"""Scene daily-log writer -- md is the SoT for scene versions.

Same chassis as :class:`EpisodeWriter`: the caller hands in pre-built
``inline`` and ``sections`` dicts plus the owner id, and this class only binds
the schemas, a user's ``users/<id>/scenes/`` and a group's
``groups/<id>/scenes/`` (docs/scenes.md).
"""

from __future__ import annotations

from .._mds import GroupSceneDailyFrontmatter, SceneDailyFrontmatter
from .base import OwnedDailyWriter


class SceneWriter(OwnedDailyWriter):
    """Daily-log writer for scene versions (``sc_<YYYYMMDD>_<seq>``, a group's
    ``gsc_``)."""

    schemas = {"user": SceneDailyFrontmatter, "group": GroupSceneDailyFrontmatter}
    log_name = "scene_log"
