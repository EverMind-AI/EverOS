"""Application assembly: the one place EverOS's objects are built and wired.

:func:`build_application` turns :class:`~everos.config.Settings` into an
:class:`Application` -- three facades (:class:`MemoryService`,
:class:`KnowledgeService`, :class:`OpsService`) plus the snapshot of which
optional providers were configured. Everything below the facades is built
here and nowhere else: providers, storage adapters, the OME engine with every
strategy registered, the event bus, the cascade orchestrator, the domain
pipelines and managers. Entry points (HTTP, CLI) hold one ``Application`` and
call facade methods.

Building has no side effect beyond configuring the process-wide tracing and
metrics backends: no file is written and no connection opened. Those happen in
:meth:`Application.start`, which the server runs and the CLI does not --
a CLI command opens storage per command through
:meth:`OpsService.index_session`. Either way the caller ends with
:meth:`Application.aclose`, which undoes exactly what was started and shuts
tracing down.

An unconfigured optional provider becomes ``None``; whoever needs it decides
what that means (search degrades, knowledge uploads refuse, the server's
:meth:`~Application.start` refuses to run without an extraction LLM).
"""

from __future__ import annotations

__all__ = ["Application", "Providers", "build_application", "build_providers"]

import contextlib
import dataclasses
import functools
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

from everalgo.llm import LLMClient

from everos.component.embedding import EmbeddingProvider, build_embedding_provider
from everos.component.llm import (
    LLMClientConfig,
    LLMNotConfiguredError,
    build_llm_provider,
    observer_for,
    resolve_decider_config,
    resolve_llm_config,
    resolve_multimodal_config,
)
from everos.component.parser import aparse_file, parser_available
from everos.component.rerank import RerankProvider, build_rerank_provider
from everos.component.tokenizer import Tokenizer, build_tokenizer
from everos.config import Settings, resolve_root
from everos.core.observability.logging import get_logger
from everos.core.observability.metrics import configure_metrics
from everos.core.observability.tracing import (
    configure_tracing,
    init_score_sink,
    shutdown_score_sink,
    shutdown_tracing,
)
from everos.infra.cascade import CascadeAdmin, CascadeOrchestrator
from everos.infra.ome import OfflineEngine, OMEConfig, OmeEventBus, to_ome_strategy
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.factories import (
    build_derived_index,
    build_indexes,
    build_stores,
)
from everos.infra.persistence.index import EmbeddingProvenance
from everos.infra.persistence.sqlite import SqliteDatabase, SqliteRepos
from everos.memory.custom_instructions import CustomInstructions
from everos.memory.deps import Indexes, MemoryDeps
from everos.memory.extract.boundary import BoundaryDetection
from everos.memory.extract.pipeline import AgentMemoryPipeline, UserMemoryPipeline
from everos.memory.get import GetManager
from everos.memory.ports import EventBus
from everos.memory.search.manager import SearchManager
from everos.memory.search.recall import (
    AgentCaseRecaller,
    AgentSkillRecaller,
    AtomicFactRecaller,
    EpisodeRecaller,
    KnowledgeTopicRecaller,
    ProfileRecaller,
    RecallerDeps,
)
from everos.memory.strategies import STRATEGY_SPECS
from everos.service.backfill import BackfillContext
from everos.service.knowledge import KnowledgeExtractor, KnowledgeService
from everos.service.memory import MemoryService
from everos.service.ops import Capabilities, OpsService

logger = get_logger(__name__)

_CONFIG_ROOT = Path(__file__).resolve().parent.parent / "config"
"""Bundled custom instructions (``src/everos/config/``), shipped in the wheel."""

_TRUE_VALUES = ("1", "true", "yes")


@dataclasses.dataclass(frozen=True)
class Providers:
    """The swappable providers an application is built over; ``None`` = absent.

    Built from settings by :func:`build_providers`; a test that needs a fake
    passes its own (``dataclasses.replace(build_providers(s), llm=fake)``).
    """

    llm: LLMClient | None
    decider: LLMClient | None
    multimodal_llm: LLMClient | None
    embedding: EmbeddingProvider | None
    rerank: RerankProvider | None


@dataclasses.dataclass(frozen=True)
class Application:
    """What an entry point holds: three facades plus the provider snapshot.

    ``start`` brings up what a long-running server needs -- system DB,
    index, cascade, OME, the ``async_mode`` queue -- in that order;
    ``aclose`` stops whatever was started in reverse, then shuts tracing
    down. ``aclose`` is safe after a failed or absent ``start``.
    """

    memory: MemoryService
    knowledge: KnowledgeService
    ops: OpsService
    capabilities: Capabilities
    _settings: Settings = dataclasses.field(repr=False)
    _root: MemoryRoot = dataclasses.field(repr=False)
    _admin: CascadeAdmin = dataclasses.field(repr=False)
    _engine: OfflineEngine = dataclasses.field(repr=False)
    _stack: contextlib.AsyncExitStack = dataclasses.field(
        default_factory=contextlib.AsyncExitStack, repr=False
    )

    async def start(self) -> None:
        """Bring the application up for serving.

        The order is load-bearing: the OME engine starts before the queue
        worker (the worker's deliveries publish into the engine), and the
        stack unwinds in reverse, so the worker stops -- and running
        extractions are waited for -- before the engine they publish to.

        Raises:
            LLMNotConfiguredError: No extraction LLM is configured; memory
                extraction is the server's purpose, so this fails startup
                rather than every request.
        """
        if not self.capabilities.llm:
            resolve_llm_config(self._settings)  # raises with the toml hint
            raise LLMNotConfiguredError("no extraction LLM was provided")
        scores = await init_score_sink(self._settings.observability)
        self._stack.push_async_callback(_shielded("score_sink", shutdown_score_sink))
        logger.info("tracing_score_sink_ready", enabled=scores)

        self._root.ensure()
        tables = await self._admin.create_schema()
        self._stack.push_async_callback(_shielded("sqlite", self._admin.dispose))
        logger.info("sqlite_schema_ready", tables=tables)

        await self._admin.start_index()
        self._stack.push_async_callback(_shielded("index", self._admin.close_index))
        if self._admin.backend == "lancedb":
            await _log_unbackfilled_hint(self._admin)
        logger.info("derived_index_ready", backend=self._admin.backend)

        await self.ops.start()
        self._stack.push_async_callback(_shielded("cascade", self.ops.aclose))

        self._root.ome_db.parent.mkdir(parents=True, exist_ok=True)
        await self._engine.start()
        self._stack.push_async_callback(_shielded("ome", self._engine.stop))
        logger.info("ome_engine_started")

        await self.memory.start()
        self._stack.push_async_callback(_shielded("memory", self.memory.aclose))
        logger.info("application_started")

    async def aclose(self) -> None:
        """Stop what :meth:`start` started, in reverse; then shut tracing down."""
        await self._stack.aclose()
        shutdown_tracing()
        logger.info("application_closed")


def build_providers(settings: Settings) -> Providers:
    """Build every optional provider ``settings`` configures; ``None`` otherwise."""
    return Providers(
        llm=_llm_or_none(settings, "llm", resolve_llm_config),
        decider=_llm_or_none(settings, "decider", resolve_decider_config),
        multimodal_llm=_llm_or_none(
            settings, "multimodal_llm", resolve_multimodal_config
        ),
        embedding=_embedding_or_none(settings),
        rerank=_rerank_or_none(settings),
    )


def build_application(
    settings: Settings, *, providers: Providers | None = None
) -> Application:
    """Assemble the application ``settings`` describes.

    Args:
        settings: Application settings.
        providers: Providers to build over; ``None`` builds them from
            ``settings``.
    """
    configure_tracing(settings.observability)
    configure_metrics()
    providers = providers or build_providers(settings)
    tokenizer = build_tokenizer()
    root = MemoryRoot(resolve_root())
    database = SqliteDatabase(root, settings.sqlite)
    sqlite = SqliteRepos.over(database)
    index = build_derived_index(settings, root)
    stores = build_stores(root, sqlite)
    indexes = build_indexes(index)
    deps = MemoryDeps(
        llm=providers.llm,
        embedding=providers.embedding,
        stores=stores,
        indexes=indexes,
        instructions=CustomInstructions(_CONFIG_ROOT),
    )
    engine = _build_engine(root, deps)
    capabilities = _capabilities(settings, providers)
    admin = CascadeAdmin(
        database=database,
        ledger=sqlite.md_change_state,
        index=index,
        provenance=EmbeddingProvenance(
            settings.embedding, configured=providers.embedding is not None
        ),
    )
    return Application(
        memory=_build_memory_service(
            settings, providers, deps, tokenizer, OmeEventBus(engine)
        ),
        knowledge=_build_knowledge_service(
            settings, root, providers, indexes, sqlite, tokenizer
        ),
        ops=OpsService(
            settings=settings,
            root=root,
            admin=admin,
            cascade=CascadeOrchestrator(
                memory_root=root,
                tokenizer=tokenizer,
                embedding=providers.embedding,
                index=index,
                sqlite=sqlite,
            ),
            run_cascade=not _env_flag("EVEROS_DISABLE_CASCADE"),
            engine=engine,
            inbox=stores.memorize_inbox,
            backfill=BackfillContext(
                root=root,
                embedding=providers.embedding,
                deps=deps,
                index=index,
                database=database,
                sqlite=sqlite,
                tokenizer=tokenizer,
            ),
            capabilities=capabilities,
        ),
        capabilities=capabilities,
        _settings=settings,
        _root=root,
        _admin=admin,
        _engine=engine,
    )


# ── providers ──────────────────────────────────────────────────────────────


def _llm_or_none(
    settings: Settings, role: str, resolve: Callable[[Settings], LLMClientConfig]
) -> LLMClient | None:
    try:
        config = resolve(settings)
    except LLMNotConfiguredError as exc:
        logger.warning("llm_not_configured", role=role, reason=str(exc))
        return None
    return build_llm_provider(config, observer=observer_for(settings))


def _embedding_or_none(settings: Settings) -> EmbeddingProvider | None:
    try:
        return build_embedding_provider(settings.embedding)
    except ValueError as exc:
        # Unconfigured and misconfigured both surface as the same 422 later;
        # this line is the only place an operator can tell them apart.
        logger.warning("embedding_capability_build_failed", reason=str(exc))
        return None


def _rerank_or_none(settings: Settings) -> RerankProvider | None:
    try:
        return build_rerank_provider(settings.rerank)
    except ValueError as exc:
        logger.warning("rerank_capability_build_failed", reason=str(exc))
        return None


def _capabilities(settings: Settings, providers: Providers) -> Capabilities:
    return Capabilities(
        llm=providers.llm is not None,
        decider=providers.decider is not None,
        embedding=providers.embedding is not None,
        rerank=providers.rerank is not None,
        multimodal_llm=providers.multimodal_llm is not None,
        # Resolving it once here also pays the optional parser's heavy import
        # at startup rather than inside a request.
        parser=parser_available(),
        index_backend=settings.index.backend,
    )


# ── engine and facades ─────────────────────────────────────────────────────


def _build_engine(root: MemoryRoot, deps: MemoryDeps) -> OfflineEngine:
    """The OME engine with every strategy registered over ``deps``.

    Strategies that need an embedding provider return no events when
    ``deps.embedding`` is ``None``; providers are resolved once, so a tier
    change needs a restart.
    """
    engine = OfflineEngine(
        config=OMEConfig(jobstore_path=root.ome_db, config_path=root.ome_config)
    )
    for spec in STRATEGY_SPECS:
        engine.register(to_ome_strategy(spec, deps))
    return engine


def _build_memory_service(
    settings: Settings,
    providers: Providers,
    deps: MemoryDeps,
    tokenizer: Tokenizer,
    bus: EventBus,
) -> MemoryService:
    stores, indexes = deps.stores, deps.indexes
    recaller_deps = RecallerDeps(tokenizer=tokenizer)
    return MemoryService(
        settings=settings,
        boundary=BoundaryDetection(
            llm=providers.llm,
            instructions=deps.instructions,
            session_buffer=stores.session_buffer,
            memcells=stores.memcell,
            limits=settings.boundary_detection,
            attempt_timeout=settings.memorize.session_lock_timeout_seconds,
        ),
        multimodal_llm=providers.multimodal_llm,
        stores=stores,
        user_pipeline=UserMemoryPipeline(
            episode_store=stores.episode,
            instructions=deps.instructions,
            llm_client=providers.llm,
        ),
        agent_pipeline=AgentMemoryPipeline(),
        bus=bus,
        search_manager=SearchManager(
            episode_recaller=EpisodeRecaller(recaller_deps, indexes.episode),
            atomic_fact_recaller=AtomicFactRecaller(recaller_deps, indexes.atomic_fact),
            agent_case_recaller=AgentCaseRecaller(recaller_deps, indexes.agent_case),
            agent_skill_recaller=AgentSkillRecaller(recaller_deps, indexes.agent_skill),
            profile_recaller=ProfileRecaller(indexes.profile),
            session_buffer=stores.session_buffer,
            cluster_store=stores.cluster,
            embedding=providers.embedding,
            reranker=providers.rerank,
            llm_client=providers.llm,
            decider_client=providers.decider,
        ),
        get_manager=GetManager(
            episode_index=indexes.episode,
            agent_case_index=indexes.agent_case,
            agent_skill_index=indexes.agent_skill,
            profile_index=indexes.profile,
        ),
    )


def _build_knowledge_service(
    settings: Settings,
    root: MemoryRoot,
    providers: Providers,
    indexes: Indexes,
    sqlite: SqliteRepos,
    tokenizer: Tokenizer,
) -> KnowledgeService:
    return KnowledgeService(
        root=root,
        extractor=_knowledge_extractor(providers.llm),
        embedding=providers.embedding,
        reranker=providers.rerank,
        recaller=KnowledgeTopicRecaller(
            RecallerDeps(tokenizer=tokenizer), indexes.knowledge_topic
        ),
        search_config=settings.knowledge.search,
        parse_file=functools.partial(aparse_file, llm=providers.multimodal_llm),
        documents=sqlite.knowledge_document,
        topics=sqlite.knowledge_topic,
    )


def _knowledge_extractor(llm: LLMClient | None) -> KnowledgeExtractor | None:
    if llm is None:
        return None
    # Deferred: everalgo.knowledge is heavy and only knowledge uploads need it.
    from everalgo.knowledge import KnowledgeExtractor as AlgoKnowledgeExtractor

    return AlgoKnowledgeExtractor(llm=llm)


# ── start helpers ──────────────────────────────────────────────────────────


async def _log_unbackfilled_hint(admin: CascadeAdmin) -> None:
    """Warn at startup when memory rows still lack a vector.

    The vector column has no scalar index, so this is a full count per
    table; a cheaper "probe first" variant was measured to cost the same on
    a clean store and twice as much on a dirty one.
    """
    total = await admin.count_unbackfilled_rows()
    if total > 0:
        get_logger("everos.cli.server").warning(
            "unbackfilled_memory_rows",
            count=total,
            hint="Run `everos cascade backfill` to include them in "
            "vector/hybrid search (optional).",
        )


def _shielded(
    name: str, stop: Callable[[], Awaitable[None]]
) -> Callable[[], Awaitable[None]]:
    """``stop`` with its failure logged instead of aborting the rest of shutdown."""

    async def _run() -> None:
        try:
            await stop()
        except Exception:
            logger.exception("application_stop_failed", component=name)

    return _run


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUE_VALUES
