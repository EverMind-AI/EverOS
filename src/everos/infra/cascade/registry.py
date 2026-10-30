"""Kind registry — single source of truth mapping ``kind name`` → (schema,
handler factory).

Adding a new business kind to cascade = adding a :class:`KindSpec` here.
The watcher / scanner / worker / CLI all read off this tuple, so neither
the path-glob patterns nor the handler dispatch table appear anywhere
else in the codebase. Order matters only when two specs would match the
same path — :func:`match_kind` returns the first match.

Path matching uses :class:`pathlib.PurePosixPath.match` (not bare
``fnmatch``) so that ``*`` matches a single path component, never the
``/`` separator (the path filter is a single whitelist layer).
"""

from __future__ import annotations

import dataclasses
from pathlib import PurePosixPath

from everos.infra.persistence.markdown import (
    AgentCaseDailyFrontmatter,
    AgentSkillFrontmatter,
    AtomicFactDailyFrontmatter,
    BaseFrontmatter,
    EpisodeDailyFrontmatter,
    ForesightDailyFrontmatter,
    KnowledgeDocumentFrontmatter,
    KnowledgeTopicFrontmatter,
    UserProfileFrontmatter,
)
from everos.infra.persistence.records import (
    AgentCase,
    AgentSkill,
    AtomicFact,
    Episode,
    Foresight,
    KnowledgeTopic,
    UserProfile,
)

from .projections import (
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
    ``index_schema`` names the destination table; the repository that
    writes it is the handler's, injected through :class:`HandlerDeps`, so
    this table stays plain data.
    ``handler_factory`` is a callable that receives the shared
    :class:`HandlerDeps` bundle and returns the kind's :class:`Handler`.
    """

    name: str
    frontmatter_schema: type[BaseFrontmatter]
    handler_factory: type[Handler]
    index_schema: type | None = None

    def path_glob(self) -> str:
        """Glob (relative to memory root) for every md this kind covers."""
        return self.frontmatter_schema.path_glob()

    def matches(self, rel_md_path: str) -> bool:
        """Whether ``rel_md_path`` (relative to memory root) is in scope.

        Uses POSIX-style component-aware glob matching: ``*`` matches a
        single path component, not ``/``. See module docstring for why
        :class:`pathlib.PurePosixPath.match` is preferred over bare
        :func:`fnmatch.fnmatch`.

        ``PurePosixPath.match`` is anchored on the right, so on its own it
        also accepted a path with extra leading components -- one the
        scanner's root-anchored ``glob`` would never visit, so the watcher
        and the scanner disagreed on it. Requiring the same number of
        components makes the match a full one.
        """
        path = PurePosixPath(rel_md_path)
        pattern = PurePosixPath(self.path_glob())
        return len(path.parts) == len(pattern.parts) and path.match(str(pattern))


KIND_REGISTRY: tuple[KindSpec, ...] = (
    KindSpec(
        name="episode",
        frontmatter_schema=EpisodeDailyFrontmatter,
        index_schema=Episode,
        handler_factory=EpisodeHandler,
    ),
    KindSpec(
        name="atomic_fact",
        frontmatter_schema=AtomicFactDailyFrontmatter,
        index_schema=AtomicFact,
        handler_factory=AtomicFactHandler,
    ),
    KindSpec(
        name="foresight",
        frontmatter_schema=ForesightDailyFrontmatter,
        index_schema=Foresight,
        handler_factory=ForesightHandler,
    ),
    KindSpec(
        name="agent_case",
        frontmatter_schema=AgentCaseDailyFrontmatter,
        index_schema=AgentCase,
        handler_factory=AgentCaseHandler,
    ),
    KindSpec(
        name="agent_skill",
        frontmatter_schema=AgentSkillFrontmatter,
        index_schema=AgentSkill,
        handler_factory=AgentSkillHandler,
    ),
    KindSpec(
        name="user_profile",
        frontmatter_schema=UserProfileFrontmatter,
        index_schema=UserProfile,
        handler_factory=UserProfileHandler,
    ),
    KindSpec(
        name="knowledge_document",
        frontmatter_schema=KnowledgeDocumentFrontmatter,
        handler_factory=KnowledgeDocumentHandler,
        index_schema=None,
    ),
    KindSpec(
        name="knowledge_topic",
        frontmatter_schema=KnowledgeTopicFrontmatter,
        handler_factory=KnowledgeTopicHandler,
        index_schema=KnowledgeTopic,
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

    Every handler registers unconditionally. Capability-dependent
    steps live inside the handlers themselves:

    - :class:`KnowledgeDocumentHandler` is SQLite-only; no capability
      dependency.
    - :class:`KnowledgeTopicHandler.handle_added_or_modified` writes
      ``vector=None`` when no embedding provider is injected — the column has been
      nullable since the embedding-soft-dependency migration.
      ``handle_deleted`` is pure SQL/LanceDB delete.

    Prior versions gated the two knowledge handlers off as an atomic
    pair when embed OR rerank was unavailable — that gate broke the
    delete path (worker marks the row failed with "no handler
    registered", stranding SQLite / LanceDB rows after
    ``shutil.rmtree`` cleared the md). Search-side gating is
    enforced separately at the route level, so cascade does not need
    to gate writes.
    """
    return {spec.name: spec.handler_factory(deps) for spec in KIND_REGISTRY}
