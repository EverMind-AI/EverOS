"""Assemble :class:`~everos.memory.deps.MemoryDeps` from in-memory doubles."""

from __future__ import annotations

from datetime import UTC, tzinfo

__all__ = [
    "bundled_instructions",
    "llm_clients",
    "make_deps",
    "make_indexes",
    "make_stores",
    "seed_sources",
    "unscripted_llm",
]

import dataclasses
from pathlib import Path
from typing import Any

from everalgo.agent_memory import CaseJudge
from everalgo.testing import FakeLLMClient

from everos.memory.custom_instructions import CustomInstructions
from everos.memory.deps import Indexes, LLMClients, MemoryDeps, Stores
from everos.memory.models import Episode, MemCell
from everos.memory.ports import Scope

from .embedding import FakeEmbeddingProvider
from .indexes import (
    InMemoryAgentCaseIndex,
    InMemoryAgentSkillIndex,
    InMemoryAtomicFactIndex,
    InMemoryEpisodeIndex,
    InMemoryProfileIndex,
)
from .stores import (
    InMemoryAgentCaseStore,
    InMemoryAgentSkillStore,
    InMemoryAtomicFactStore,
    InMemoryClusterStore,
    InMemoryEpisodeStore,
    InMemoryForesightStore,
    InMemoryMemcellStore,
    InMemoryProfileStore,
    InMemoryReflectionReportStore,
    UnusedPort,
)

_DEFAULT: Any = object()
_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "src" / "everos" / "config"


def bundled_instructions() -> CustomInstructions:
    """The custom instructions the package ships (every operator disabled)."""
    return CustomInstructions(_CONFIG_ROOT)


def unscripted_llm() -> FakeLLMClient:
    """An LLM that fails the test if anything calls it."""

    def _refuse(*_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("unexpected LLM call")

    return FakeLLMClient(handler=_refuse)


def llm_clients(llm: Any = None, **per_task: Any) -> LLMClients:
    """``llm`` for every task, except the tasks named in ``per_task``.

    ``llm_clients(fake)`` is the single-model deployment and
    ``llm_clients(fake, profile=other)`` points one task elsewhere. Without
    ``llm`` every task gets a refusing client of its own, so a test can tell
    which task's client a consumer was handed.
    """
    tasks = {
        field.name: llm if llm is not None else unscripted_llm()
        for field in dataclasses.fields(LLMClients)
    }
    return LLMClients(**{**tasks, **per_task})


def make_stores(**overrides: Any) -> Stores:
    """Fresh in-memory stores; keyword arguments replace single stores."""
    fields: dict[str, Any] = {
        "episode": InMemoryEpisodeStore(),
        "atomic_fact": InMemoryAtomicFactStore(),
        "foresight": InMemoryForesightStore(),
        "profile": InMemoryProfileStore(),
        "agent_case": InMemoryAgentCaseStore(),
        "agent_skill": InMemoryAgentSkillStore(),
        "session_buffer": UnusedPort("session_buffer"),
        "memcell": InMemoryMemcellStore(),
        "cluster": InMemoryClusterStore(),
        "memorize_inbox": UnusedPort("memorize_inbox"),
        "reflection_report": InMemoryReflectionReportStore(),
        "owner_dir": UnusedPort("owner_dir"),
        "session_purge": UnusedPort("session_purge"),
    }
    return Stores(**{**fields, **overrides})


def make_indexes(**overrides: Any) -> Indexes:
    """Fresh in-memory indexes; keyword arguments replace single indexes."""
    fields: dict[str, Any] = {
        "episode": InMemoryEpisodeIndex(),
        "atomic_fact": InMemoryAtomicFactIndex(),
        "agent_case": InMemoryAgentCaseIndex(),
        "agent_skill": InMemoryAgentSkillIndex(),
        "profile": InMemoryProfileIndex(),
        "knowledge_topic": UnusedPort("knowledge_topic"),
    }
    return Indexes(**{**fields, **overrides})


def make_deps(
    *,
    llm: Any = _DEFAULT,
    llms: LLMClients | None = None,
    embedding: Any = _DEFAULT,
    stores: Stores | None = None,
    indexes: Indexes | None = None,
    instructions: CustomInstructions | None = None,
    case_judge: CaseJudge | None = None,
    tz: tzinfo = UTC,
) -> MemoryDeps:
    """Dependencies backed by in-memory doubles.

    ``llm`` serves every task; without it each task gets its own LLM that
    refuses every call. ``llms`` (see :func:`llm_clients`) gives tasks clients
    of their own and wins over ``llm``. ``embedding`` defaults to a
    :class:`FakeEmbeddingProvider`; pass ``None`` to model an unconfigured
    provider. ``case_judge`` stays ``None`` (no agent-case gate) unless given.
    """
    if llms is None:
        llms = llm_clients(None if llm is _DEFAULT else llm)
    return MemoryDeps(
        llms=llms,
        embedding=FakeEmbeddingProvider() if embedding is _DEFAULT else embedding,
        stores=stores if stores is not None else make_stores(),
        indexes=indexes if indexes is not None else make_indexes(),
        instructions=instructions or bundled_instructions(),
        case_judge=case_judge,
        tz=tz,
    )


def seed_sources(
    deps: MemoryDeps,
    *,
    memcell_id: str,
    episode_entry_id: str | None = None,
    owner_id: str = "u_alice",
    app_id: str = "default",
    project_id: str = "default",
) -> MemoryDeps:
    """Seed what a derived write checks for: the memcell and, if named, the
    episode in md (the fake episode store). Returns ``deps`` for chaining.

    In production every derived write has these sources; a fixture that
    leaves them out describes a deleted conversation, whose writes are
    skipped.
    """
    deps.stores.memcell.seed(memcell_id, MemCell(items=[], timestamp=0))  # type: ignore[attr-defined]
    if episode_entry_id is not None:
        scope = Scope(
            app_id=app_id, project_id=project_id, owner_id=owner_id, owner_type="user"
        )
        deps.stores.episode.seed(  # type: ignore[attr-defined]
            scope,
            episode_entry_id,
            Episode(
                owner_id=owner_id,
                episode="seeded",
                timestamp=0,
                subject="seeded",
                summary="seeded",
                parent_id=memcell_id,
            ),
        )
    return deps
