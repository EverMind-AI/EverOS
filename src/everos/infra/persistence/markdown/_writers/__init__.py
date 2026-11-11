"""Business markdown writers.

Each storage strategy from the EverOS Markdown First spec gets a class
here:

    * :class:`BaseDailyWriter` — daily-log append (episode / image
      memory / agent case). Subclass and bind ``schema``.
    * :class:`AgentSkillWriter` — directory + progressive disclosure
      (``skills/skill_<name>/{SKILL.md, references/, scripts/}``).
      Single class, no subclassing.
    * :class:`ProfileWriter` — single-file rewrite at a fixed name
      (``user.md`` / ``behaviors.md`` / ``agent.md`` / ``soul.md`` /
      ``tools.md``). Single class, no subclassing — caller hands in a
      frontmatter instance whose ``PROFILE_FILENAME`` ClassVar pins
      the filename.

External usage::

    from everos.infra.persistence.markdown._writers import (
        BaseDailyWriter,
        EpisodeWriter,
        AgentSkillWriter,
        ProfileWriter,
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
from .profile_writer import ProfileWriter as ProfileWriter

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
    "ProfileWriter",
]
