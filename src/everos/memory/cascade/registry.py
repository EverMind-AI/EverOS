"""Kind registry — single source of truth mapping ``kind name`` → (schema,
repo, handler factory).

Adding a new business kind to cascade = adding a :class:`KindSpec` here.
The watcher / scanner / worker / CLI all read off this tuple, so neither
the path-glob patterns nor the handler dispatch table appear anywhere
else in the codebase. Order matters only when two specs would match the
same path — :func:`match_kind` returns the first match.

Path matching uses :class:`pathlib.PurePosixPath.match` (not bare
``fnmatch``) so that ``*`` matches a single path component, never the
``/`` separator — see ``17_lancedb_tables_design.md`` §2.4.2 and
``12_cascade_design.md`` §5.1 (path filter is a single whitelist layer).
"""

from __future__ import annotations

import dataclasses
from pathlib import PurePosixPath

from everos.core.persistence.markdown import BaseFrontmatter
from everos.infra.persistence.lancedb import (
    AgentCase,
    AgentSkill,
    AtomicFact,
    Episode,
    Foresight,
    KnowledgeTopic,
    UserProfile,
    agent_case_repo,
    agent_skill_repo,
    atomic_fact_repo,
    episode_repo,
    foresight_repo,
    knowledge_topic_repo,
    user_profile_repo,
)
from everos.infra.persistence.markdown import (
    AgentCaseDailyFrontmatter,
    AgentSkillFrontmatter,
    AtomicFactDailyFrontmatter,
    EpisodeDailyFrontmatter,
    ForesightDailyFrontmatter,
    KnowledgeDocumentFrontmatter,
    KnowledgeTopicFrontmatter,
    UserProfileFrontmatter,
)

from .handlers import (
    AgentCaseHandler,
    AgentSkillHandler,
    AtomicFactHandler,
    EpisodeHandler,
    ForesightHandler,
    Handler,
    HandlerDeps,
    KnowledgeDocumentHandler,
    KnowledgeTopicHandler,
    UserProfileHandler,
)


@dataclasses.dataclass(frozen=True)
class KindSpec:
    """One cascade kind — md schema + LanceDB binding + handler factory.

    ``frontmatter_schema`` carries the ``path_glob()`` classmethod the
    scanner uses to enumerate eligible files; the same schema is also
    the contract the reader / writer share at the markdown layer.
    ``lance_schema`` + ``lance_repo`` describe the destination side.
    ``handler_factory`` is a callable that receives the shared
    :class:`HandlerDeps` bundle and returns the kind's :class:`Handler`.
    """

    name: str
    frontmatter_schema: type[BaseFrontmatter]
    handler_factory: type[Handler]
    lance_schema: type | None = None
    lance_repo: object | None = None

    def path_glob(self) -> str:
        """Glob (relative to memory root) for every md this kind covers."""
        return self.frontmatter_schema.path_glob()

    def matches(self, rel_md_path: str) -> bool:
        """Whether ``rel_md_path`` (relative to memory root) is in scope.

        Uses POSIX-style component-aware glob matching: ``*`` matches a
        single path component, not ``/``. See module docstring for why
        :class:`pathlib.PurePosixPath.match` is preferred over bare
        :func:`fnmatch.fnmatch`.
        """
        return PurePosixPath(rel_md_path).match(self.path_glob())


KIND_REGISTRY: tuple[KindSpec, ...] = (
    KindSpec(
        name="episode",
        frontmatter_schema=EpisodeDailyFrontmatter,
        lance_schema=Episode,
        lance_repo=episode_repo,
        handler_factory=EpisodeHandler,
    ),
    KindSpec(
        name="atomic_fact",
        frontmatter_schema=AtomicFactDailyFrontmatter,
        lance_schema=AtomicFact,
        lance_repo=atomic_fact_repo,
        handler_factory=AtomicFactHandler,
    ),
    KindSpec(
        name="foresight",
        frontmatter_schema=ForesightDailyFrontmatter,
        lance_schema=Foresight,
        lance_repo=foresight_repo,
        handler_factory=ForesightHandler,
    ),
    KindSpec(
        name="agent_case",
        frontmatter_schema=AgentCaseDailyFrontmatter,
        lance_schema=AgentCase,
        lance_repo=agent_case_repo,
        handler_factory=AgentCaseHandler,
    ),
    KindSpec(
        name="agent_skill",
        frontmatter_schema=AgentSkillFrontmatter,
        lance_schema=AgentSkill,
        lance_repo=agent_skill_repo,
        handler_factory=AgentSkillHandler,
    ),
    KindSpec(
        name="user_profile",
        frontmatter_schema=UserProfileFrontmatter,
        lance_schema=UserProfile,
        lance_repo=user_profile_repo,
        handler_factory=UserProfileHandler,
    ),
    KindSpec(
        name="knowledge_document",
        frontmatter_schema=KnowledgeDocumentFrontmatter,
        handler_factory=KnowledgeDocumentHandler,
        lance_schema=None,
        lance_repo=None,
    ),
    KindSpec(
        name="knowledge_topic",
        frontmatter_schema=KnowledgeTopicFrontmatter,
        handler_factory=KnowledgeTopicHandler,
        lance_schema=KnowledgeTopic,
        lance_repo=knowledge_topic_repo,
    ),
)
"""Every cascade kind, evaluated in declaration order by :func:`match_kind`."""


def match_kind(rel_md_path: str) -> KindSpec | None:
    """Return the first :class:`KindSpec` matching ``rel_md_path``, or ``None``.

    First-match semantics (DD-7): registry order is the precedence order.
    Today's globs are disjoint by directory name so order is academic; if
    overlap is ever introduced the registry order resolves it.
    """
    for spec in KIND_REGISTRY:
        if spec.matches(rel_md_path):
            return spec
    return None


def build_handlers(deps: HandlerDeps) -> dict[str, Handler]:
    """Instantiate every registered handler bound to the shared deps.

    Returns a ``{kind_name: Handler}`` map used by the worker for
    dispatch. Constructing once at orchestrator startup keeps the
    per-row hot path free of factory churn.
    """
    return {spec.name: spec.handler_factory(deps) for spec in KIND_REGISTRY}
