"""Image-memory daily-log writer — md is the SoT for picture memories.

Same chassis as :class:`EpisodeWriter`: the caller hands in pre-built
``inline`` and ``sections`` dicts plus the owner id, and this class only binds
the schemas (a user's ``users/<id>/images/``, a group's
``groups/<id>/images/``). Domain → structured-entry shaping lives in the
store (``infra`` may not import ``memory``).
"""

from __future__ import annotations

from .._mds import GroupImageMemoryDailyFrontmatter, ImageMemoryDailyFrontmatter
from .base import OwnedDailyWriter


class ImageMemoryWriter(OwnedDailyWriter):
    """Daily-log writer for the image-memory schema (md = SoT).

    ``entry_id`` (``im_<YYYYMMDD>_<NNNN>``, a group's ``gim_``) is the in-file
    identity allocated under the per-path lock; the globally-unique row key
    is ``<owner_id>_<entry_id>``, derived by the store and the cascade alike.
    """

    schemas = {
        "user": ImageMemoryDailyFrontmatter,
        "group": GroupImageMemoryDailyFrontmatter,
    }
    log_name = "image_log"
