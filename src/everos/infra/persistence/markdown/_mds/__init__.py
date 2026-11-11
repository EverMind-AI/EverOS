"""Business markdown frontmatter schemas (mds = "markdown schemas").

Each business record kind that is stored as markdown gets a concrete
frontmatter class here, subclassing one of the chassis classes from
:mod:`everos.infra.persistence.markdown._base`:

    * :class:`UserScopedFrontmatter` for user-track records
    * :class:`AgentScopedFrontmatter` for agent-track records
    * :class:`BaseFrontmatter` for scope-agnostic records (rare)

Schemas drive path resolution via ClassVars; each storage strategy has
its own conventions:

- **Daily-log** schemas declare ``ENTRY_ID_PREFIX`` (token in
  ``<prefix>_<date>_<seq>``), ``DIR_NAME`` (sub-directory under
  ``<scope>/<id>/``) and ``FILE_PREFIX`` (leading token of the daily
  filename joined with ``-<YYYY-MM-DD>.md``).
- **Skill** schemas (:class:`AgentSkillFrontmatter`) pin the directory
  layout via five ``SKILL_*`` ClassVars (container / dir prefix /
  main filename / references / scripts).
- **Profile** schemas declare ``PROFILE_FILENAME`` (``"user.md"`` /
  ``"agent.md"`` / …) and inherit ``SCOPE_DIR`` from a scope mixin; no
  profile base class — the writer/reader pair is duck-typed.

The group profile's module (:mod:`.group_profile`) also owns the format of
its entries: :func:`render_group_profile_entry` and
:func:`group_profile_item_from_entry` are the one writer and the one reader
of it, shared by the store and the cascade.
"""

from .agent_case import AgentCaseDailyFrontmatter as AgentCaseDailyFrontmatter
from .agent_skill import AgentSkillFrontmatter as AgentSkillFrontmatter
from .atomic_fact import AtomicFactDailyFrontmatter as AtomicFactDailyFrontmatter
from .episode import EpisodeDailyFrontmatter as EpisodeDailyFrontmatter
from .foresight import ForesightDailyFrontmatter as ForesightDailyFrontmatter
from .group import GroupEpisodeDailyFrontmatter as GroupEpisodeDailyFrontmatter
from .group import GroupFrontmatter as GroupFrontmatter
from .group import (
    GroupImageMemoryDailyFrontmatter as GroupImageMemoryDailyFrontmatter,
)
from .group import GroupMemberProfileFrontmatter as GroupMemberProfileFrontmatter
from .group import GroupSceneDailyFrontmatter as GroupSceneDailyFrontmatter
from .group_profile import (
    GroupProfileDailyFrontmatter as GroupProfileDailyFrontmatter,
)
from .group_profile import (
    GroupProfilePendingFrontmatter as GroupProfilePendingFrontmatter,
)
from .group_profile import (
    group_profile_item_from_entry as group_profile_item_from_entry,
)
from .group_profile import (
    render_group_profile_entry as render_group_profile_entry,
)
from .image_memory import ImageMemoryDailyFrontmatter as ImageMemoryDailyFrontmatter
from .knowledge_document import (
    KnowledgeDocumentFrontmatter as KnowledgeDocumentFrontmatter,
)
from .knowledge_topic import KnowledgeTopicFrontmatter as KnowledgeTopicFrontmatter
from .profile import UserProfileFrontmatter as UserProfileFrontmatter

__all__ = [
    "AgentCaseDailyFrontmatter",
    "AgentSkillFrontmatter",
    "AtomicFactDailyFrontmatter",
    "EpisodeDailyFrontmatter",
    "ForesightDailyFrontmatter",
    "GroupEpisodeDailyFrontmatter",
    "GroupFrontmatter",
    "GroupImageMemoryDailyFrontmatter",
    "GroupMemberProfileFrontmatter",
    "GroupProfileDailyFrontmatter",
    "GroupProfilePendingFrontmatter",
    "GroupSceneDailyFrontmatter",
    "ImageMemoryDailyFrontmatter",
    "KnowledgeDocumentFrontmatter",
    "KnowledgeTopicFrontmatter",
    "UserProfileFrontmatter",
    "group_profile_item_from_entry",
    "render_group_profile_entry",
]
