"""Business markdown writers.

Each storage strategy from the EverOS Markdown First spec gets a class
here:

    * :class:`BaseDailyWriter` — daily-log append (episode / image
      memory / agent case). Subclass and bind ``schema``.
    * :class:`AgentSkillWriter` — directory + progressive disclosure
      (``skills/skill_<name>/{SKILL.md, references/, scripts/}``).
      Single class, no subclassing.

A person's profile (``user.md`` or a member's ``members/<m>.md``) is one
whole-file rewrite, done by its store
(:class:`~everos.infra.persistence.markdown.stores.MarkdownProfileStore`).

External usage::

    from everos.infra.persistence.markdown._writers import (
        BaseDailyWriter,
        EpisodeWriter,
        AgentSkillWriter,
    )
"""

from .agent_case_writer import AgentCaseWriter as AgentCaseWriter
from .agent_skill_writer import AgentSkillWriter as AgentSkillWriter
from .base import BaseDailyWriter as BaseDailyWriter
from .episode_writer import EpisodeWriter as EpisodeWriter
from .group_writer import GroupEpisodeWriter as GroupEpisodeWriter
from .group_writer import GroupImageMemoryWriter as GroupImageMemoryWriter
from .group_writer import GroupProfileWriter as GroupProfileWriter
from .group_writer import GroupSceneWriter as GroupSceneWriter
from .group_writer import GroupWriter as GroupWriter
from .image_memory_writer import ImageMemoryWriter as ImageMemoryWriter
from .knowledge_writer import KnowledgeWriter as KnowledgeWriter
from .scene_writer import SceneWriter as SceneWriter

__all__ = [
    "AgentCaseWriter",
    "AgentSkillWriter",
    "BaseDailyWriter",
    "EpisodeWriter",
    "GroupEpisodeWriter",
    "GroupImageMemoryWriter",
    "GroupProfileWriter",
    "GroupSceneWriter",
    "GroupWriter",
    "ImageMemoryWriter",
    "KnowledgeWriter",
    "SceneWriter",
]
