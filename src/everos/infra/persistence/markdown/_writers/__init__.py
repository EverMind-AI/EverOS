"""Business markdown writers.

Each storage strategy from the EverOS Markdown First spec gets a class
here:

    * :class:`BaseDailyWriter` — daily-log append (episode / image
      memory / scene). Subclass and bind ``schema``;
      :class:`OwnedDailyWriter` binds a user's and a group's schema of one
      kind (episode / image memory / scene).

A person's profile (``user.md`` or a member's ``members/<m>.md``) is one
whole-file rewrite, done by its store
(:class:`~everos.infra.persistence.markdown.stores.MarkdownProfileStore`).

External usage::

    from everos.infra.persistence.markdown._writers import (
        BaseDailyWriter,
        EpisodeWriter,
    )
"""

from .base import BaseDailyWriter as BaseDailyWriter
from .base import OwnedDailyWriter as OwnedDailyWriter
from .episode_writer import EpisodeWriter as EpisodeWriter
from .group_writer import GroupProfileWriter as GroupProfileWriter
from .group_writer import GroupWriter as GroupWriter
from .image_memory_writer import ImageMemoryWriter as ImageMemoryWriter
from .scene_writer import SceneWriter as SceneWriter

__all__ = [
    "BaseDailyWriter",
    "EpisodeWriter",
    "GroupProfileWriter",
    "GroupWriter",
    "ImageMemoryWriter",
    "OwnedDailyWriter",
    "SceneWriter",
]
