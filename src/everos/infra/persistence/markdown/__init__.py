"""Markdown business persistence layer.

Sits on top of :mod:`everos.core.persistence.markdown` (atomic write +
parse + frontmatter chassis) and provides:

    * concrete frontmatter schemas under :mod:`.mds`
    * concrete business writers under :mod:`.writers`
      (``BaseDailyWriter`` + subclasses, ``AgentSkillWriter``,
      ``ProfileWriter``)
    * concrete business readers under :mod:`.readers`
      (``BaseDailyReader`` + subclasses, ``AgentSkillReader``,
      ``ProfileReader``)

External usage::

    from everos.infra.persistence.markdown import (
        BaseDailyWriter, BaseDailyReader,
        EpisodeWriter, EpisodeReader, EpisodeDailyFrontmatter,
        AtomicFactDailyFrontmatter,
        DecisionDailyFrontmatter, DecisionWriter, DecisionReader,
        ForesightDailyFrontmatter,
        AgentCaseDailyFrontmatter,
        AgentSkillFrontmatter, AgentSkillWriter, AgentSkillReader,
        ProfileWriter, ProfileReader,
    )

Outer layers MUST go through this top-level package because
``infra.persistence.markdown.**`` (sub-packages) are forbidden to outer
layers by import-linter.
"""

from .mds import AgentCaseDailyFrontmatter as AgentCaseDailyFrontmatter
from .mds import AgentSkillFrontmatter as AgentSkillFrontmatter
from .mds import AtomicFactDailyFrontmatter as AtomicFactDailyFrontmatter
from .mds import DecisionDailyFrontmatter as DecisionDailyFrontmatter
from .mds import EpisodeDailyFrontmatter as EpisodeDailyFrontmatter
from .mds import ForesightDailyFrontmatter as ForesightDailyFrontmatter
from .mds import KnowledgeDocumentFrontmatter as KnowledgeDocumentFrontmatter
from .mds import KnowledgeTopicFrontmatter as KnowledgeTopicFrontmatter
from .mds import PrincipleFrontmatter as PrincipleFrontmatter
from .mds import PrincipleItem as PrincipleItem
from .mds import UserProfileFrontmatter as UserProfileFrontmatter
from .mds import mint_principle_id as mint_principle_id
from .mds import render_principles_body as render_principles_body
from .readers import AgentCaseReader as AgentCaseReader
from .readers import AgentSkillReader as AgentSkillReader
from .readers import AtomicFactReader as AtomicFactReader
from .readers import BaseDailyReader as BaseDailyReader
from .readers import DecisionReader as DecisionReader
from .readers import EpisodeReader as EpisodeReader
from .readers import ForesightReader as ForesightReader
from .readers import ProfileReader as ProfileReader
from .readers import ensure_taxonomy as ensure_taxonomy
from .readers import parse_taxonomy as parse_taxonomy
from .writers import AgentCaseWriter as AgentCaseWriter
from .writers import AgentSkillWriter as AgentSkillWriter
from .writers import AtomicFactWriter as AtomicFactWriter
from .writers import BaseDailyWriter as BaseDailyWriter
from .writers import DecisionWriter as DecisionWriter
from .writers import EpisodeWriter as EpisodeWriter
from .writers import ForesightWriter as ForesightWriter
from .writers import KnowledgeWriter as KnowledgeWriter
from .writers import ProfileWriter as ProfileWriter

__all__ = [
    "AgentCaseDailyFrontmatter",
    "AgentCaseReader",
    "AgentCaseWriter",
    "AgentSkillFrontmatter",
    "AgentSkillReader",
    "AgentSkillWriter",
    "AtomicFactDailyFrontmatter",
    "AtomicFactReader",
    "AtomicFactWriter",
    "BaseDailyReader",
    "BaseDailyWriter",
    "DecisionDailyFrontmatter",
    "DecisionReader",
    "DecisionWriter",
    "EpisodeDailyFrontmatter",
    "EpisodeReader",
    "EpisodeWriter",
    "ForesightDailyFrontmatter",
    "ForesightReader",
    "ForesightWriter",
    "KnowledgeDocumentFrontmatter",
    "KnowledgeTopicFrontmatter",
    "KnowledgeWriter",
    "PrincipleFrontmatter",
    "PrincipleItem",
    "ProfileReader",
    "ProfileWriter",
    "UserProfileFrontmatter",
    "ensure_taxonomy",
    "mint_principle_id",
    "parse_taxonomy",
    "render_principles_body",
]
