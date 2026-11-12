"""Assemble :class:`~everos.memory.deps.MemoryDeps` from in-memory doubles."""

from __future__ import annotations

__all__ = [
    "bundled_instructions",
    "make_deps",
    "make_indexes",
    "make_stores",
    "seed_sources",
    "unscripted_llm",
]

from pathlib import Path
from typing import Any

from everalgo.agent_memory import CaseJudge
from everalgo.testing import FakeLLMClient

from everos.component.images import ImageLimits
from everos.memory.custom_instructions import CustomInstructions
from everos.memory.deps import Indexes, MemoryDeps, Stores
from everos.memory.models import Episode, MemCell
from everos.memory.ports import Scope

from .embedding import FakeEmbeddingProvider
from .indexes import (
    InMemoryAgentCaseIndex,
    InMemoryAgentSkillIndex,
    InMemoryEpisodeIndex,
    InMemoryGroupProfileIndex,
    InMemoryImageMemoryIndex,
    InMemoryProfileIndex,
)
from .stores import (
    InMemoryAgentCaseStore,
    InMemoryAgentSkillStore,
    InMemoryClusterStore,
    InMemoryEpisodeStore,
    InMemoryForgetLogStore,
    InMemoryGroupProfileStore,
    InMemoryGroupStore,
    InMemoryImageMemoryStore,
    InMemoryImageStore,
    InMemoryMemcellStore,
    InMemoryMessageForgetStore,
    InMemoryProfileStore,
    InMemoryReflectionReportStore,
    InMemorySceneStore,
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


def make_stores(**overrides: Any) -> Stores:
    """Fresh in-memory stores; keyword arguments replace single stores."""
    fields: dict[str, Any] = {
        "episode": InMemoryEpisodeStore(),
        "image_memory": InMemoryImageMemoryStore(),
        "images": InMemoryImageStore(),
        "forget_log": InMemoryForgetLogStore(),
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
        "groups": InMemoryGroupStore(),
        "scene": InMemorySceneStore(),
        "group_profile": InMemoryGroupProfileStore(),
        "message_forget": InMemoryMessageForgetStore(),
    }
    return Stores(**{**fields, **overrides})


def make_indexes(**overrides: Any) -> Indexes:
    """Fresh in-memory indexes; keyword arguments replace single indexes."""
    fields: dict[str, Any] = {
        "episode": InMemoryEpisodeIndex(),
        "agent_case": InMemoryAgentCaseIndex(),
        "agent_skill": InMemoryAgentSkillIndex(),
        "image_memory": InMemoryImageMemoryIndex(),
        "profile": InMemoryProfileIndex(),
        "knowledge_topic": UnusedPort("knowledge_topic"),
        "group_profile": InMemoryGroupProfileIndex(),
    }
    return Indexes(**{**fields, **overrides})


def make_deps(
    *,
    llm: Any = _DEFAULT,
    embedding: Any = _DEFAULT,
    stores: Stores | None = None,
    indexes: Indexes | None = None,
    instructions: CustomInstructions | None = None,
    case_judge: CaseJudge | None = None,
    picture_limits: ImageLimits | None = None,
) -> MemoryDeps:
    """Dependencies backed by in-memory doubles.

    ``llm`` defaults to an LLM that refuses every call and ``embedding`` to a
    :class:`FakeEmbeddingProvider`; pass ``None`` for either to model an
    unconfigured provider. ``case_judge`` stays ``None`` (no agent-case
    gate) unless given; ``picture_limits`` stays ``None`` (a profile update
    reads no pictures) unless given.
    """
    return MemoryDeps(
        llm=unscripted_llm() if llm is _DEFAULT else llm,
        embedding=FakeEmbeddingProvider() if embedding is _DEFAULT else embedding,
        stores=stores if stores is not None else make_stores(),
        indexes=indexes if indexes is not None else make_indexes(),
        instructions=instructions or bundled_instructions(),
        case_judge=case_judge,
        picture_limits=picture_limits,
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
