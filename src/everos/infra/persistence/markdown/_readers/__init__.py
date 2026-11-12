"""Business markdown readers — symmetric with the writers.

Daily-log markdown is parsed via :class:`MarkdownReader` from ``core``
(the base reader returns frontmatter dict + body + entry markers, all
schema-agnostic). Reader classes here add the **business-aware
locator** layer:

    * :class:`BaseDailyReader` + subclasses — bind a daily-log schema,
      resolve ``(scope_id, date)`` to a file, locate entries by id,
      and optionally upgrade to :class:`StructuredEntry`. Symmetric
      with :class:`BaseDailyWriter`.
    * :class:`AgentSkillReader` — reads ``SKILL.md`` and parses the
      frontmatter into the caller-supplied ``AgentSkillFrontmatter``
      subclass; also reads individual reference / script files.

By design, no batch / list APIs live here: bulk enumeration for
prompt-budget or cross-record queries goes through sqlite/lancedb
(see the cascade daemon's index sync), not a markdown directory walk.

External usage::

    from everos.infra.persistence.markdown._readers import (
        BaseDailyReader,
        EpisodeReader,
        AgentSkillReader,
    )
"""

from .agent_case_reader import AgentCaseReader as AgentCaseReader
from .agent_skill_reader import AgentSkillReader as AgentSkillReader
from .base import BaseDailyReader as BaseDailyReader
from .episode_reader import EpisodeReader as EpisodeReader
from .taxonomy_reader import ensure_taxonomy as ensure_taxonomy
from .taxonomy_reader import parse_taxonomy as parse_taxonomy

__all__ = [
    "AgentCaseReader",
    "AgentSkillReader",
    "BaseDailyReader",
    "EpisodeReader",
    "ensure_taxonomy",
    "parse_taxonomy",
]
