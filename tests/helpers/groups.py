"""A real memory stack with group memory seeded in it, for group tests.

Read-path tests seed group records directly, the way the group write path
writes them: through the episode and picture-memory stores with a group
:class:`Scope`, then project the markdown with the cascade handlers into a real
LanceDB index. Write-path tests go through the bootstrap's memory service and
run the group strategies on the events it published (:meth:`GroupWorld.run`),
as the engine would. Everything is real -- markdown, ``system.db``, the index,
the memory service -- and the model and embedder are deterministic doubles.

Usage::

    async with GroupWorld.open(tmp_path) as world:
        await world.roster("trip", [join("u_chen", 0), ...])
        await world.episode("trip", "the plan", start=10, end=20)
        await world.project()
        response = await world.memory.search(SearchRequest(...))
"""

from __future__ import annotations

__all__ = ["GroupWorld", "join", "leave"]

import base64
import contextlib
import io
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from everalgo.testing import FakeLLMClient
from PIL import Image

from everos.component.images import ImageLimits
from everos.component.tokenizer import build_tokenizer
from everos.config import Settings
from everos.infra.cascade.registry import match_kind
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.factories import build_indexes, build_stores
from everos.memory.deps import MemoryDeps, Stores
from everos.memory.models import Episode, Group, ImageMemory, Profile, RosterEvent
from everos.memory.ports import (
    Immediate,
    NoSourceCheck,
    Scope,
    SessionKey,
    StrategySpec,
)
from everos.service import RosterEventInput
from everos.service.bootstrap import _build_memory_service
from tests.fakes import (
    FakeEmbeddingProvider,
    bundled_instructions,
    handler_deps,
    make_providers,
)
from tests.fakes.storage import Storage, make_storage
from tests.fakes.stores import UnusedPort

DIM = 8
APP = "default"
PROJECT = "p"


def join(member: str, at: int, **fields: Any) -> RosterEvent:
    return RosterEvent(op="join", member_id=member, at=at, **fields)


def leave(member: str, at: int) -> RosterEvent:
    return RosterEvent(op="leave", member_id=member, at=at)


def png(shade: int) -> bytes:
    """A small picture whose bytes (and so its hash) depend on ``shade``."""
    image = Image.new("RGB", (16, 16), (shade, 40, 80))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _refuse(*_args: Any, **_kwargs: Any) -> str:
    raise AssertionError("a read path called the model")


class _Bus:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def publish(self, events: Sequence[Any]) -> None:
        self.events.extend(events)


@dataclass
class GroupWorld:
    """The stack and the seeding helpers."""

    root: MemoryRoot
    storage: Storage
    stores: Stores
    memory: Any
    embedding: FakeEmbeddingProvider
    bus: _Bus
    deps: MemoryDeps
    _ran: int = 0

    @classmethod
    @contextlib.asynccontextmanager
    async def open(
        cls,
        tmp_path: Path,
        *,
        llm: Any = None,
        settings: Settings | None = None,
        mode: str = "chat",
    ) -> AsyncIterator[GroupWorld]:
        base = settings or Settings()
        settings = base.model_copy(
            update={
                "embedding": base.embedding.model_copy(update={"dim": DIM}),
                "lancedb": base.lancedb.model_copy(
                    update={"read_consistency_seconds": 0.0}
                ),
                "memorize": base.memorize.model_copy(update={"mode": mode}),
            }
        )
        root = MemoryRoot(tmp_path)
        root.ensure()
        storage = make_storage(root, settings)
        await storage.create_schema()
        embedding = FakeEmbeddingProvider(dim=DIM)
        model = llm or FakeLLMClient(handler=_refuse)
        stores = build_stores(root, storage.sqlite)
        deps = MemoryDeps(
            llm=model,
            embedding=embedding,
            stores=stores,
            indexes=build_indexes(storage.index),
            instructions=bundled_instructions(),
        )
        bus = _Bus()
        memory = _build_memory_service(
            settings,
            make_providers(llm=model, embedding=embedding, rerank=None),
            deps,
            build_tokenizer(),
            bus,
            UnusedPort("index_eraser"),
        )
        try:
            yield cls(
                root=root,
                storage=storage,
                stores=stores,
                memory=memory,
                embedding=embedding,
                bus=bus,
                deps=deps,
            )
        finally:
            await storage.aclose()

    async def run(self, *specs: StrategySpec) -> None:
        """Run ``specs`` on every event published since the last run, in order.

        What the engine does with an ``Immediate`` strategy, without the
        engine: each matching event is handed to the strategy, and the events
        a strategy returns are handled the same way.
        """
        queue = list(self.bus.events[self._ran :])
        self._ran = len(self.bus.events)
        while queue:
            event = queue.pop(0)
            for spec in specs:
                trigger = spec.trigger
                if not isinstance(trigger, Immediate):
                    continue
                if not isinstance(event, tuple(trigger.on)):
                    continue
                if callable(spec.applies_to) and not spec.applies_to(event, self.deps):
                    continue
                queue.extend(await spec.fn(event, self.deps))

    # ── seeding ──────────────────────────────────────────────────────────

    def scope(self, group_id: str) -> Scope:
        return Scope(
            app_id=APP, project_id=PROJECT, owner_id=group_id, owner_type="group"
        )

    def session(self, session_id: str) -> SessionKey:
        return SessionKey(app_id=APP, project_id=PROJECT, session_id=session_id)

    @staticmethod
    def roster_input(**fields: Any) -> RosterEventInput:
        return RosterEventInput(**fields)

    def personal(self, user_id: str) -> Scope:
        return Scope(
            app_id=APP, project_id=PROJECT, owner_id=user_id, owner_type="user"
        )

    async def roster(
        self,
        group_id: str,
        events: Sequence[RosterEvent],
        *,
        visible_from: str = "join",
    ) -> None:
        record = await self.stores.groups.get(self.scope(group_id))
        group = (
            record.group
            if record is not None
            else Group(
                group_id=group_id,
                created_at=0,
                history_visible_from=visible_from,  # type: ignore[arg-type]
            )
        )
        await self.stores.groups.save(self.scope(group_id), group=group, events=events)

    async def episode(
        self,
        group_id: str,
        text: str,
        *,
        start: int,
        end: int,
        senders: Sequence[str] = ("u_chen",),
        session_id: str = "s_group",
        image_ids: Sequence[str] = (),
    ) -> str:
        """A group episode whose segment ran ``[start, end]``; returns its id."""
        ref = await self.stores.episode.append(
            self.scope(group_id),
            Episode(
                owner_id=group_id,
                episode=text,
                summary=text[:40],
                subject=text[:30],
                timestamp=end,
                session_id=session_id,
                sender_ids=list(senders),
                parent_id=f"mc_{group_id}_{start}",
                source_start=start,
                subject_ids=list(senders),
                origin="member",
                image_ids=list(image_ids),
            ),
            only_if=NoSourceCheck(),
        )
        assert ref is not None
        return ref.id

    async def personal_episode(self, user_id: str, text: str, *, at: int) -> str:
        ref = await self.stores.episode.append(
            self.personal(user_id),
            Episode(
                owner_id=user_id,
                episode=text,
                summary=text[:40],
                subject=text[:30],
                timestamp=at,
                session_id=f"s_{user_id}",
                sender_ids=[user_id],
                parent_id=f"mc_{user_id}_{at}",
            ),
            only_if=NoSourceCheck(),
        )
        assert ref is not None
        return ref.id

    async def picture(
        self,
        owner: Scope,
        description: str,
        *,
        at: int,
        shade: int,
        poster: str = "u_chen",
    ) -> str:
        """Store a picture file and one picture memory of it; its content hash."""
        limits = ImageLimits(
            max_bytes=5 * 1024 * 1024,
            max_per_message=4,
            max_request_bytes=10 * 1024 * 1024,
            max_per_request=8,
            formats=("png", "jpeg", "webp", "gif"),
            model_edge=256,
            cache_quality=85,
        )
        stored = await self.stores.images.store(png(shade), limits=limits)
        group = owner.owner_type == "group"
        await self.stores.image_memory.append(
            owner,
            ImageMemory(
                owner_id=owner.owner_id,
                description=description,
                timestamp=at,
                image_id=stored.image_id,
                session_id="s_group" if group else f"s_{owner.owner_id}",
                parent_id=f"mc_pic_{at}",
                sender_ids=[poster] if group else [],
            ),
            only_if=NoSourceCheck(),
        )
        return stored.image_id

    async def profile(self, user_id: str, fact: str) -> None:
        from everos.memory.models import ExplicitInfoItem

        await self.stores.profile.write(
            self.personal(user_id),
            Profile(
                owner_id=user_id,
                summary=fact,
                timestamp=1,
                explicit_info=[ExplicitInfoItem(description=fact, category="misc")],
            ),
        )

    async def project(self) -> None:
        """Project every markdown file into the index, then build FTS."""
        deps = handler_deps(
            memory_root=self.root,
            tokenizer=build_tokenizer(),
            index=self.storage.repos,
            embedding=self.embedding,
            group_roster=self.storage.sqlite.group_roster,
        )
        for path in _markdown(self.root.root):
            rel = path.relative_to(self.root.root).as_posix()
            spec = match_kind(rel)
            if spec is not None:
                await spec.handler_factory(deps).handle_added_or_modified(rel)
        await self.storage.index.ensure_business_indexes()


def _markdown(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*.md")
        if not p.relative_to(root).parts[0].startswith(".")
    )


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")
