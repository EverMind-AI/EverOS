"""The dependencies a strategy works through, as one injectable value.

Strategies are functions of ``(event, deps)``; :class:`MemoryDeps` is the
``deps``. It holds ports only — the domain declares the storage protocols,
the LLM contract comes from everalgo and the embedding contract from
``component`` — so a strategy can be exercised against in-memory doubles
and the service layer decides which adapters stand behind it.

``llm`` / ``embedding`` are ``None`` when the provider is not configured.
Embedding is optional by design (Tier 1 runs without it, and the strategies
that need it return no events); the LLM is not, so :meth:`MemoryDeps.require_llm`
raises the same configuration error a missing provider raises elsewhere.
Neither changes while the process runs: a tier change needs a restart.
"""

from __future__ import annotations

__all__ = ["Indexes", "MemoryDeps", "Stores"]

from dataclasses import dataclass

from everalgo.llm import LLMClient

from everos.component.embedding import EmbeddingProvider
from everos.core.errors import ProviderNotConfiguredError
from everos.memory.ports import (
    AgentCaseIndex,
    AgentCaseStore,
    AgentSkillIndex,
    AgentSkillStore,
    AtomicFactIndex,
    AtomicFactStore,
    ClusterStore,
    EpisodeIndex,
    EpisodeStore,
    ForesightStore,
    KnowledgeTopicIndex,
    MemcellStore,
    MemorizeInboxStore,
    ProfileIndex,
    ProfileStore,
    ReflectionReportStore,
    SessionBufferStore,
)


@dataclass(frozen=True)
class Stores:
    """Every write-side store, typed by its port."""

    episode: EpisodeStore
    atomic_fact: AtomicFactStore
    foresight: ForesightStore
    profile: ProfileStore
    agent_case: AgentCaseStore
    agent_skill: AgentSkillStore
    session_buffer: SessionBufferStore
    memcell: MemcellStore
    cluster: ClusterStore
    memorize_inbox: MemorizeInboxStore
    reflection_report: ReflectionReportStore


@dataclass(frozen=True)
class Indexes:
    """Every read-side index of the configured engine, typed by its port."""

    episode: EpisodeIndex
    atomic_fact: AtomicFactIndex
    agent_case: AgentCaseIndex
    agent_skill: AgentSkillIndex
    profile: ProfileIndex
    knowledge_topic: KnowledgeTopicIndex


@dataclass(frozen=True)
class MemoryDeps:
    """What a strategy may use.

    Args:
        llm: Extraction LLM; ``None`` when not configured.
        embedding: Embedding provider; ``None`` when not configured.
        stores: Write-side stores.
        indexes: Read-side indexes.
    """

    llm: LLMClient | None
    embedding: EmbeddingProvider | None
    stores: Stores
    indexes: Indexes

    def require_llm(self) -> LLMClient:
        """Return the LLM, or raise when none is configured.

        Raises:
            ProviderNotConfiguredError: No LLM is configured.
        """
        if self.llm is None:
            raise ProviderNotConfiguredError(provider="llm")
        return self.llm
