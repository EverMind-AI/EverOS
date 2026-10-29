"""Assemble :class:`~everos.memory.deps.MemoryDeps` from in-memory doubles."""

from __future__ import annotations

__all__ = ["make_deps", "make_indexes", "make_stores", "unscripted_llm"]

from typing import Any

from everalgo.testing import FakeLLMClient

from everos.memory.deps import Indexes, MemoryDeps, Stores

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


def unscripted_llm() -> FakeLLMClient:
    """An LLM that fails the test if anything calls it."""

    def _refuse(*_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("unexpected LLM call")

    return FakeLLMClient(handler=_refuse)


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
    embedding: Any = _DEFAULT,
    stores: Stores | None = None,
    indexes: Indexes | None = None,
) -> MemoryDeps:
    """Dependencies backed by in-memory doubles.

    ``llm`` defaults to an LLM that refuses every call and ``embedding`` to a
    :class:`FakeEmbeddingProvider`; pass ``None`` for either to model an
    unconfigured provider.
    """
    return MemoryDeps(
        llm=unscripted_llm() if llm is _DEFAULT else llm,
        embedding=FakeEmbeddingProvider() if embedding is _DEFAULT else embedding,
        stores=stores if stores is not None else make_stores(),
        indexes=indexes if indexes is not None else make_indexes(),
    )
