"""Business markdown writers.

Each storage strategy from the EverOS Markdown First spec gets a class
here:

    * :class:`BaseDailyWriter` — daily-log append (episode / atomic
      fact / foresight / agent case). Subclass and bind ``schema``.
    * :class:`AgentSkillWriter` — directory + progressive disclosure
      (``skills/skill_<name>/{SKILL.md, references/, scripts/}``).
      Single class, no subclassing.
    * :class:`ProfileWriter` — single-file rewrite at a fixed name
      (``user.md`` / ``behaviors.md`` / ``agent.md`` / ``soul.md`` /
      ``tools.md``). Single class, no subclassing — caller hands in a
      frontmatter instance whose ``PROFILE_FILENAME`` ClassVar pins
      the filename.

External usage::

    from everos.infra.persistence.markdown.writers import (
        BaseDailyWriter,
        EpisodeWriter,
        AgentSkillWriter,
        ProfileWriter,
    )
"""

from .agent_case_writer import AgentCaseWriter as AgentCaseWriter
from .agent_skill_writer import AgentSkillWriter as AgentSkillWriter
from .atomic_fact_writer import AtomicFactWriter as AtomicFactWriter
from .base import BaseDailyWriter as BaseDailyWriter
from .episode_writer import EpisodeWriter as EpisodeWriter
from .foresight_writer import ForesightWriter as ForesightWriter
from .knowledge_writer import KnowledgeWriter as KnowledgeWriter
from .profile_writer import ProfileWriter as ProfileWriter

__all__ = [
    "AgentCaseWriter",
    "AgentSkillWriter",
    "AtomicFactWriter",
    "BaseDailyWriter",
    "EpisodeWriter",
    "ForesightWriter",
    "KnowledgeWriter",
    "ProfileWriter",
]
