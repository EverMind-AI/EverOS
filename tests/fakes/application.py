"""An :class:`~everos.service.Application` assembled from test doubles.

:func:`make_application` gives a route or lifespan test the one object an
entry point holds, with every facade replaced by a recording stub, so the
test drives the HTTP layer without storage, providers or background loops::

    application = make_application()
    app = create_app(application=application)
    ...
    assert application.memory.calls == [...]

``lifecycle`` (shared by every stub) records start / stop calls in order.
"""

from __future__ import annotations

__all__ = [
    "FakeOmeLock",
    "StubAdmin",
    "StubEngine",
    "StubKnowledgeService",
    "StubMemoryService",
    "StubOpsService",
    "make_application",
    "make_backfill_deps",
    "make_knowledge_service",
    "make_memory_service",
    "make_ops",
    "unused_backfill_deps",
]

import datetime as dt
import functools
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

from everos.component.parser import aparse_file
from everos.component.tokenizer import build_tokenizer
from everos.config import OmeSettings, Settings, load_settings
from everos.core.errors import MemoryRootLockedError, QueueSaturatedError
from everos.infra.cascade import CascadeOrchestrator
from everos.infra.cascade.worker import DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS
from everos.infra.ome import OmeLock
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.factories import build_indexes, build_stores
from everos.infra.persistence.index import ProjectionVectorBackfill
from everos.infra.persistence.markdown import (
    MarkdownAgentSkillStore,
    MarkdownKnowledgeStore,
)
from everos.infra.persistence.sqlite import SqliteKnowledgeCatalog
from everos.memory.deps import MemoryDeps
from everos.service import (
    Application,
    BackfillDeps,
    Capabilities,
    FlushResult,
    HealthReport,
    KnowledgeService,
    MemorizeResult,
    MemoryService,
    OpsService,
    Providers,
    Queued,
    QueueHealth,
    TaskItem,
    TaskPage,
)
from everos.service.bootstrap import build_replay_engine

from .deps import bundled_instructions, make_stores, unscripted_llm
from .sources import unused_raw_file_builder
from .storage import make_storage
from .stores import UnusedPort

_VECTOR_TABLES = (
    "episode",
    "atomic_fact",
    "foresight",
    "agent_case",
    "agent_skill",
    "knowledge_topic",
)

_ALL_CAPABILITIES = Capabilities(
    embedding=True,
    rerank=True,
    multimodal_llm=True,
    parser=True,
    index_backend="lancedb",
)


class StubMemoryService:
    """Records add / flush calls; the queue is a per-session counter.

    ``calls`` holds ``(payload, is_final, defer_extraction)``; ``enqueued``
    holds ``(payload, request_id, is_final, defer_extraction)``.
    """

    def __init__(self, lifecycle: list[str] | None = None) -> None:
        self.lifecycle = lifecycle if lifecycle is not None else []
        self.calls: list[tuple[dict[str, Any], bool, bool]] = []
        self.enqueued: list[tuple[dict[str, Any], str, bool, bool]] = []
        self.pending: dict[str, int] = {}
        self.accepting = True
        self.saturated = False
        self.queue_running = True
        self.health_calls = 0
        self.search = AsyncMock()
        self.get = AsyncMock()

    async def start(self) -> None:
        self.lifecycle.append("memory.start")

    async def aclose(self) -> None:
        self.lifecycle.append("memory.aclose")

    async def _memorize(
        self,
        payload: dict[str, Any],
        *,
        is_final: bool = False,
        defer_extraction: bool = False,
    ) -> MemorizeResult:
        self.calls.append((payload, is_final, defer_extraction))
        return MemorizeResult(
            message_count=len(payload["messages"]), status="extracted"
        )

    async def _flush_inline(
        self,
        *,
        session_id: str,
        app_id: str = "default",
        project_id: str = "default",
    ) -> FlushResult:
        payload = {
            "session_id": session_id,
            "app_id": app_id,
            "project_id": project_id,
            "messages": [],
        }
        result = await self._memorize(payload, is_final=True)
        return FlushResult(
            status="extracted" if result.status == "extracted" else "no_extraction"
        )

    @property
    def accepts_async(self) -> bool:
        return self.accepting

    async def has_queued(
        self,
        *,
        session_id: str,
        app_id: str = "default",
        project_id: str = "default",
    ) -> bool:
        return self.pending.get(session_id, 0) > 0

    async def enqueue(
        self,
        payload: dict[str, Any],
        *,
        request_id: str,
        is_final: bool,
        defer_extraction: bool,
    ) -> int:
        if self.saturated:
            raise QueueSaturatedError("memorize queue saturated for session s1")
        self.enqueued.append((payload, request_id, is_final, defer_extraction))
        n = self.pending.get(payload["session_id"], 0) + 1
        self.pending[payload["session_id"]] = n
        return n

    async def add(
        self,
        payload: dict[str, Any],
        *,
        request_id: str,
        async_mode: bool,
        defer_extraction: bool,
    ) -> MemorizeResult | Queued:
        if (async_mode and self.accepting) or await self.has_queued(
            session_id=payload["session_id"]
        ):
            return await self._queue(payload, request_id, False, defer_extraction)
        return await self._memorize(payload, defer_extraction=defer_extraction)

    async def flush(
        self,
        *,
        session_id: str,
        request_id: str,
        app_id: str = "default",
        project_id: str = "default",
    ) -> FlushResult | Queued:
        if await self.has_queued(session_id=session_id):
            payload = {
                "session_id": session_id,
                "app_id": app_id,
                "project_id": project_id,
                "messages": [],
            }
            return await self._queue(payload, request_id, True, False)
        return await self._flush_inline(
            session_id=session_id, app_id=app_id, project_id=project_id
        )

    async def _queue(
        self,
        payload: dict[str, Any],
        request_id: str,
        is_final: bool,
        defer_extraction: bool,
    ) -> Queued:
        pending = await self.enqueue(
            payload,
            request_id=request_id,
            is_final=is_final,
            defer_extraction=defer_extraction,
        )
        return Queued(message_count=len(payload["messages"]), pending=pending)

    async def queue_health(self) -> QueueHealth | None:
        if not self.queue_running:
            return None
        self.health_calls += 1
        return QueueHealth(
            healthy=True,
            reasons=[],
            loop_alive=True,
            accepting=self.accepting,
            pending=sum(self.pending.values()),
            oldest_pending_age_s=1.5 if self.pending else None,
            failed=0,
            loop_restarts=0,
            last_loop_error_at=None,
        )


class StubOpsService:
    """Ops double: records lifecycle; health is the real assembly over its fields.

    ``cascade`` (a stub orchestrator, or ``None`` for not running), ``probed``
    and ``queue_health`` feed :meth:`health` exactly as the bootstrap feeds a
    real :class:`OpsService`.
    """

    def __init__(
        self,
        capabilities: Capabilities = _ALL_CAPABILITIES,
        lifecycle: list[str] | None = None,
        queue_health: Callable[[], Awaitable[QueueHealth | None]] | None = None,
    ) -> None:
        self.capabilities = capabilities
        self.lifecycle = lifecycle if lifecycle is not None else []
        self.cascade: Any = None
        self.probed = False
        self.queue_health = queue_health
        self.ome_trigger = AsyncMock()
        self.cascade_quiesce = AsyncMock(return_value=None)

    async def start(self) -> None:
        self.lifecycle.append("ops.start")

    async def aclose(self) -> None:
        self.lifecycle.append("ops.aclose")

    async def health(self) -> HealthReport:
        ops = make_ops(
            cascade=self.cascade,
            capabilities=self.capabilities,
            probed=self.probed,
            queue_health=self.queue_health,
        )
        return await ops.health()

    async def get_task(self, request_id: str) -> TaskItem | None:
        return None

    async def list_tasks(self, **kwargs: Any) -> TaskPage:
        return TaskPage(
            total=0, page=kwargs["page"], page_size=kwargs["page_size"], items=[]
        )


def StubKnowledgeService() -> Any:  # noqa: N802 - reads as the class it doubles
    """A :class:`KnowledgeService` whose every method is an ``AsyncMock``."""
    return AsyncMock(spec=KnowledgeService)


class StubAdmin:
    """Cascade admin double: records the storage steps of ``Application.start``."""

    backend = "milvus"  # skips the LanceDB-only unbackfilled hint

    def __init__(self, lifecycle: list[str], *, probed: bool = False) -> None:
        self.lifecycle = lifecycle
        self.probed = probed

    async def create_schema(self) -> int:
        self.lifecycle.append("sqlite.create")
        return 0

    async def dispose(self) -> None:
        self.lifecycle.append("sqlite.dispose")

    async def start_index(self) -> None:
        self.lifecycle.append("index.start")

    async def close_index(self) -> None:
        self.lifecycle.append("index.close")

    def embedding_probed(self) -> bool:
        return self.probed


class FakeOmeLock:
    """The memory root's lock, free or held by a pretend server."""

    def __init__(self, *, free: bool = True) -> None:
        self.free = free

    def is_free(self) -> bool:
        return self.free

    @contextmanager
    def hold(self) -> Iterator[None]:
        if not self.free:
            raise MemoryRootLockedError("a pretend server holds the root")
        yield


class StubEngine:
    """OME engine double recording start / stop."""

    def __init__(self, lifecycle: list[str]) -> None:
        self.lifecycle = lifecycle

    async def start(self) -> None:
        self.lifecycle.append("ome.start")

    async def stop(self) -> None:
        self.lifecycle.append("ome.stop")


def make_memory_service(**overrides: Any) -> MemoryService:
    """A real :class:`MemoryService` over in-memory stores and unused ports.

    Keyword arguments replace single constructor arguments (``settings``,
    ``boundary``, ``stores``, ``user_pipeline``, ``bus``, ...); anything not
    replaced fails loudly if the code under test reaches it.
    """
    fields: dict[str, Any] = {
        "settings": Settings(),
        "boundary": UnusedPort("boundary"),
        "multimodal_llm": None,
        "stores": make_stores(),
        "user_pipeline": UnusedPort("user_pipeline"),
        "agent_pipeline": UnusedPort("agent_pipeline"),
        "bus": UnusedPort("bus"),
        "search_manager": UnusedPort("search_manager"),
        "get_manager": UnusedPort("get_manager"),
        "raw_file_builder": unused_raw_file_builder,
    }
    return MemoryService(**{**fields, **overrides})


def make_knowledge_service(
    *,
    root: MemoryRoot | Path | None = None,
    extractor: Any = None,
    embedding: Any = None,
    reranker: Any = None,
    recaller: Any = None,
    search_config: Any = None,
    parse_file: Any = None,
    documents: Any = None,
    topics: Any = None,
    parser_available: bool = False,
    max_upload_bytes: int | None = None,
) -> KnowledgeService:
    """A real :class:`KnowledgeService`; unset providers read as unconfigured.

    An unset ``extractor`` fails the test if anything extracts with it.

    The service runs over the production adapters: a markdown knowledge
    store under ``root`` and a SQLite catalog over ``documents`` / ``topics``
    (the repositories, often mocks). ``root`` defaults to the configured
    memory root (``EVEROS_ROOT``) at call time, so a test that sets the env
    var first gets a service over it.
    """
    if root is None:
        root = MemoryRoot.resolve()
    elif not isinstance(root, MemoryRoot):
        root = MemoryRoot(root)
    return KnowledgeService(
        store=MarkdownKnowledgeStore(root),
        catalog=SqliteKnowledgeCatalog(
            documents or UnusedPort("knowledge documents"),
            topics or UnusedPort("knowledge topics"),
        ),
        extractor=extractor or UnusedPort("knowledge extractor"),
        embedding=embedding,
        reranker=reranker,
        recaller=recaller,
        search_config=search_config or load_settings().knowledge.search,
        parse_file=parse_file or functools.partial(aparse_file, llm=None),
        parser_available=parser_available,
        max_upload_bytes=max_upload_bytes or load_settings().knowledge.max_upload_bytes,
    )


def make_providers(**overrides: Any) -> Providers:
    """Providers for ``build_application`` without touching settings.

    ``llm`` defaults to one that fails the test if called, and ``decider``
    to ``llm`` (as ``[decider]`` falls back to ``[llm]``); every optional
    provider is unconfigured. Keyword arguments replace single providers.
    """
    llm = overrides.pop("llm", None) or unscripted_llm()
    fields: dict[str, Any] = {
        "llm": llm,
        "decider": llm,
        "multimodal_llm": None,
        "embedding": None,
        "rerank": None,
    }
    return Providers(**{**fields, **overrides})


def make_backfill_deps(
    *,
    root: MemoryRoot | Path | None = None,
    embedding: Any = None,
    llm: Any = None,
    **overrides: Any,
) -> BackfillDeps:
    """The production backfill wiring over ``root`` (default: the configured one).

    Real storage, stores, indexes and replay engines, as the bootstrap builds
    them; ``embedding`` defaults to unconfigured and ``llm`` to one that
    fails the test if called. Keyword overrides
    replace single fields (``ome_lock``, ``engine_for``, ``vector_tables``,
    ``cascade``, ...).
    """
    storage = make_storage(root if root is not None else MemoryRoot.resolve())
    deps = MemoryDeps(
        llm=llm or unscripted_llm(),
        embedding=embedding,
        stores=build_stores(storage.root, storage.sqlite),
        indexes=build_indexes(storage.index),
        instructions=bundled_instructions(),
    )
    tokenizer = build_tokenizer()
    vector_tables = storage.index.repositories.with_vectors()
    fields: dict[str, Any] = {
        "embedding": embedding,
        "vectors": ProjectionVectorBackfill(
            vector_tables,
            tokenizer=tokenizer,
            prune_retention=dt.timedelta(
                seconds=DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS
            ),
        ),
        "vector_tables": vector_tables,
        "clusters": deps.stores.cluster,
        "skills": deps.indexes.agent_skill,
        "skill_files": MarkdownAgentSkillStore(storage.root),
        "engine_for": functools.partial(
            build_replay_engine, storage.root, deps, OmeSettings()
        ),
        "ensure_schema": storage.create_schema,
        "cascade": CascadeOrchestrator(
            memory_root=storage.root,
            tokenizer=tokenizer,
            embedding=embedding,
            index=storage.index,
            sqlite=storage.sqlite,
        ),
        "ome_lock": OmeLock(storage.root.ome_db),
    }
    return BackfillDeps(**{**fields, **overrides})


def unused_backfill_deps() -> BackfillDeps:
    """Backfill wiring that fails loudly if reached, over a free root lock."""
    unused = UnusedPort("backfill")
    return BackfillDeps(
        embedding=None,
        vectors=unused,  # type: ignore[arg-type]
        vector_tables=dict.fromkeys(_VECTOR_TABLES, unused),
        clusters=unused,  # type: ignore[arg-type]
        skills=unused,  # type: ignore[arg-type]
        skill_files=unused,  # type: ignore[arg-type]
        engine_for=unused,  # type: ignore[arg-type]
        ensure_schema=unused,  # type: ignore[arg-type]
        cascade=unused,  # type: ignore[arg-type]
        ome_lock=FakeOmeLock(),
    )


def make_ops(
    *,
    cascade: Any = None,
    engine: Any = None,
    inbox: Any = None,
    backfill: BackfillDeps | None = None,
    capabilities: Capabilities = _ALL_CAPABILITIES,
    probed: bool = False,
    queue_health: Callable[[], Awaitable[QueueHealth | None]] | None = None,
    ome_lock: Any = None,
) -> OpsService:
    """A real :class:`OpsService` over only the collaborators a test passes.

    A ``cascade`` given here counts as running, as if the server had started
    it -- the health block and quiesce see it; ``None`` reads as not running.
    ``backfill`` defaults to wiring that fails loudly if reached.
    """
    backfill = backfill or unused_backfill_deps()
    extra: dict[str, Any] = {}
    if queue_health is not None:
        extra["queue_health"] = queue_health
    ops = OpsService(
        settings=load_settings(),
        root=None,  # type: ignore[arg-type] - unused by the server-side paths
        admin=StubAdmin([], probed=probed),  # type: ignore[arg-type]
        cascade=cascade,
        run_cascade=cascade is not None,
        engine=engine,
        ome_lock=ome_lock or backfill.ome_lock,
        inbox=inbox,
        backfill=backfill,
        capabilities=capabilities,
        **extra,
    )
    ops._cascade_attached = cascade is not None
    return ops


def make_application(
    *,
    root: Path | None = None,
    memory: Any = None,
    knowledge: Any = None,
    ops: Any = None,
    capabilities: Capabilities = _ALL_CAPABILITIES,
    lifecycle: list[str] | None = None,
) -> Application:
    """An :class:`Application` over stubs; any facade can be overridden.

    ``root`` defaults to the configured memory root; pass ``tmp_path`` when
    the application will be started (start creates the root's directories).
    """
    events = lifecycle if lifecycle is not None else []
    memory = memory if memory is not None else StubMemoryService(events)
    return Application(
        memory=memory,
        knowledge=knowledge if knowledge is not None else StubKnowledgeService(),
        ops=(
            ops
            if ops is not None
            else StubOpsService(capabilities, events, memory.queue_health)
        ),
        capabilities=capabilities,
        _settings=load_settings(),
        _root=MemoryRoot(root) if root is not None else MemoryRoot.resolve(),
        _admin=StubAdmin(events),  # type: ignore[arg-type] - structural double
        _engine=StubEngine(events),  # type: ignore[arg-type] - structural double
    )
