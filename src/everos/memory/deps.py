"""The dependencies a strategy works through, as one injectable value.

Strategies are functions of ``(event, deps)``; :class:`MemoryDeps` is the
``deps``. It holds ports only — the domain declares the storage protocols,
the LLM contract comes from everalgo and the embedding contract from
``component`` — so a strategy can be exercised against in-memory doubles
and the service layer decides which adapters stand behind it.

``llms`` holds one LLM client per task and is always complete: the
bootstrap refuses to build without an LLM. Each strategy calls its own task's
client, so an operator can point one task at another model in configuration
without the domain knowing. ``embedding`` is ``None`` when the provider is not
configured -- optional by design (Tier 1 runs without it, and the strategies
that need it return no events) -- and ``case_judge`` is ``None`` unless the
agent-case gate is switched on. Neither changes while the process runs: a tier
change needs a restart.
"""

from __future__ import annotations

__all__ = ["Indexes", "LLMClients", "MemoryDeps", "Stores"]

from dataclasses import dataclass
from datetime import UTC, tzinfo
from typing import TYPE_CHECKING

from everalgo.agent_memory import CaseJudge
from everalgo.llm import LLMClient

from everos.component.embedding import EmbeddingProvider
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
    OwnerDirStore,
    ProfileIndex,
    ProfileStore,
    ReflectionReportStore,
    SenderNameStore,
    SessionBufferStore,
    SessionPurgeStore,
)
from everos.memory.profile_batcher import ProfileBatcher

if TYPE_CHECKING:
    from everos.memory.custom_instructions import CustomInstructions


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
    owner_dir: OwnerDirStore
    session_purge: SessionPurgeStore
    sender_name: SenderNameStore


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
class LLMClients:
    """One LLM client per task the application calls a model for.

    The attribute names are the configuration's task names (``[llm.<task>]``);
    tasks configured alike may share one client instance.

    Args:
        boundary: Cuts a session's messages into memcells.
        episode: Extracts each memcell's episode.
        summary: Writes the episode's display summary.
        atomic_fact: Extracts atomic facts.
        foresight: Extracts foresight.
        profile: Extracts the user profile.
        reflection: Reflects over episodes.
        agent_case: Extracts agent cases and answers the case gate.
        agent_skill: Extracts agent skills and clusters them.
        knowledge: Extracts knowledge-document topics.
        search: Agentic retrieval and the LLM rerank lane.
        decider: The multi-round retrieval decider.
    """

    boundary: LLMClient
    episode: LLMClient
    summary: LLMClient
    atomic_fact: LLMClient
    foresight: LLMClient
    profile: LLMClient
    reflection: LLMClient
    agent_case: LLMClient
    agent_skill: LLMClient
    knowledge: LLMClient
    search: LLMClient
    decider: LLMClient


@dataclass(frozen=True)
class MemoryDeps:
    """What a strategy may use.

    Args:
        llms: One LLM client per task.
        profile_batcher: The per-call episode cap of user-profile extraction
            and its marks of episodes folded ahead; one for the process.
        embedding: Embedding provider; ``None`` when not configured.
        stores: Write-side stores.
        indexes: Read-side indexes.
        instructions: Custom instructions appended to the operators' rules.
        case_judge: Answers the agent-case gate question; ``None`` runs
            agent-case extraction without the gate.
        tz: The zone message times are shown to the model in (``[memory]
            timezone``).
    """

    llms: LLMClients
    profile_batcher: ProfileBatcher
    embedding: EmbeddingProvider | None
    stores: Stores
    indexes: Indexes
    instructions: CustomInstructions
    case_judge: CaseJudge | None = None
    tz: tzinfo = UTC
