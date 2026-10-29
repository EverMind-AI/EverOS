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
    "StubAdmin",
    "StubEngine",
    "StubKnowledgeService",
    "StubMemoryService",
    "StubOpsService",
    "make_application",
    "make_backfill_context",
    "make_knowledge_service",
    "make_memory_service",
    "make_ops",
]

import functools
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

from everos.component.parser import aparse_file
from everos.component.tokenizer import build_tokenizer
from everos.config import Settings, load_settings
from everos.core.errors import QueueSaturatedError
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.factories import build_indexes, build_stores
from everos.memory.deps import MemoryDeps
from everos.memory.prompt_slots import PromptLoader
from everos.service import (
    Application,
    Capabilities,
    KnowledgeService,
    MemorizeResult,
    MemoryService,
    OpsService,
    QueueHealth,
    TaskItem,
    TaskPage,
)
from everos.service.backfill import BackfillContext

from .deps import make_stores
from .storage import make_storage
from .stores import UnusedPort

_ALL_CAPABILITIES = Capabilities(
    llm=True,
    decider=True,
    embedding=True,
    rerank=True,
    multimodal_llm=True,
    parser=True,
    index_backend="lancedb",
)


class StubMemoryService:
    """Records memorize / flush calls; the queue is a per-session counter.

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

    async def memorize(
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

    async def flush(
        self,
        *,
        session_id: str,
        app_id: str = "default",
        project_id: str = "default",
    ) -> MemorizeResult:
        payload = {
            "session_id": session_id,
            "app_id": app_id,
            "project_id": project_id,
            "messages": [],
        }
        return await self.memorize(payload, is_final=True)

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
    """Ops double: no cascade, no tasks, no embedding probe; records lifecycle."""

    def __init__(
        self,
        capabilities: Capabilities = _ALL_CAPABILITIES,
        lifecycle: list[str] | None = None,
    ) -> None:
        self.capabilities = capabilities
        self.lifecycle = lifecycle if lifecycle is not None else []
        self.cascade: Any = None
        self.probed = False
        self.ome_trigger = AsyncMock()
        self.cascade_quiesce = AsyncMock(return_value=None)

    async def start(self) -> None:
        self.lifecycle.append("ops.start")

    async def aclose(self) -> None:
        self.lifecycle.append("ops.aclose")

    def embedding_probed(self) -> bool:
        return self.probed

    async def cascade_health(self) -> Any:
        return None if self.cascade is None else await self.cascade.health()

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

    def __init__(self, lifecycle: list[str]) -> None:
        self.lifecycle = lifecycle

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
        return False


class StubEngine:
    """OME engine double recording start / stop."""

    def __init__(self, lifecycle: list[str]) -> None:
        self.lifecycle = lifecycle

    async def start(self) -> None:
        self.lifecycle.append("ome.start")

    async def stop(self) -> None:
        self.lifecycle.append("ome.stop")


_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "src" / "everos" / "config"


def make_memory_service(**overrides: Any) -> MemoryService:
    """A real :class:`MemoryService` over in-memory stores and unused ports.

    Keyword arguments replace single constructor arguments (``settings``,
    ``llm``, ``stores``, ``user_pipeline``, ``bus``, ...); anything not
    replaced fails loudly if the code under test reaches it.
    """
    fields: dict[str, Any] = {
        "settings": Settings(),
        "llm": None,
        "multimodal_llm": None,
        "prompt_loader": PromptLoader(_CONFIG_ROOT),
        "stores": make_stores(),
        "user_pipeline": UnusedPort("user_pipeline"),
        "agent_pipeline": UnusedPort("agent_pipeline"),
        "bus": UnusedPort("bus"),
        "search_manager": UnusedPort("search_manager"),
        "get_manager": UnusedPort("get_manager"),
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
) -> KnowledgeService:
    """A real :class:`KnowledgeService`; unset providers read as unconfigured.

    ``root`` defaults to the configured memory root (``EVEROS_ROOT``) at call
    time, so a test that sets the env var first gets a service over it.
    """
    if root is None:
        root = MemoryRoot.resolve()
    elif not isinstance(root, MemoryRoot):
        root = MemoryRoot(root)
    return KnowledgeService(
        root=root,
        extractor=extractor,
        embedding=embedding,
        reranker=reranker,
        recaller=recaller,
        search_config=search_config or load_settings().knowledge.search,
        parse_file=parse_file or functools.partial(aparse_file, llm=None),
        documents=documents or UnusedPort("knowledge documents"),
        topics=topics or UnusedPort("knowledge topics"),
    )


def make_backfill_context(
    *,
    root: MemoryRoot | Path | None = None,
    embedding: Any = None,
    llm: Any = None,
) -> BackfillContext:
    """The production backfill context over ``root`` (default: the configured one).

    Real storage, stores and indexes, as the CLI builds them; ``embedding`` /
    ``llm`` default to unconfigured.
    """
    storage = make_storage(root if root is not None else MemoryRoot.resolve())
    return BackfillContext(
        root=storage.root,
        embedding=embedding,
        deps=MemoryDeps(
            llm=llm,
            embedding=embedding,
            stores=build_stores(storage.root, storage.sqlite),
            indexes=build_indexes(storage.index),
        ),
        index=storage.index,
        database=storage.database,
        sqlite=storage.sqlite,
        tokenizer=build_tokenizer(),
    )


def make_ops(
    *,
    cascade: Any = None,
    engine: Any = None,
    inbox: Any = None,
    backfill: Any = None,
    capabilities: Capabilities = _ALL_CAPABILITIES,
) -> OpsService:
    """A real :class:`OpsService` over only the collaborators a test passes.

    A ``cascade`` given here counts as running, as if the server had started
    it -- the health block and quiesce see it; ``None`` reads as not running.
    """
    ops = OpsService(
        settings=load_settings(),
        root=None,  # type: ignore[arg-type] - unused by the server-side paths
        admin=StubAdmin([]),  # type: ignore[arg-type] - structural double
        cascade=cascade,
        run_cascade=cascade is not None,
        engine=engine,
        inbox=inbox,
        backfill=backfill,  # type: ignore[arg-type]
        capabilities=capabilities,
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
    return Application(
        memory=memory if memory is not None else StubMemoryService(events),
        knowledge=knowledge if knowledge is not None else StubKnowledgeService(),
        ops=ops if ops is not None else StubOpsService(capabilities, events),
        capabilities=capabilities,
        _settings=load_settings(),
        _root=MemoryRoot(root) if root is not None else MemoryRoot.resolve(),
        _admin=StubAdmin(events),  # type: ignore[arg-type] - structural double
        _engine=StubEngine(events),  # type: ignore[arg-type] - structural double
    )
