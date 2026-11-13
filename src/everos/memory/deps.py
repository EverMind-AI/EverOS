"""The dependencies a strategy works through, as one injectable value.

Strategies are functions of ``(event, deps)``; :class:`MemoryDeps` is the
``deps``. It holds ports only — the domain declares the storage protocols,
the LLM contract comes from everalgo and the embedding contract from
``component`` — so a strategy can be exercised against in-memory doubles
and the service layer decides which adapters stand behind it.

``llm`` is always present: the bootstrap refuses to build without one.
``embedding`` is ``None`` when the provider is not configured -- optional by
design (Tier 1 runs without it, and the strategies that need it return no
events). It does not change while the process runs: a tier change needs a
restart.
"""

from __future__ import annotations

__all__ = ["Indexes", "MemoryDeps", "Stores"]

from dataclasses import dataclass
from typing import TYPE_CHECKING

from everalgo.llm import LLMClient

from everos.component.embedding import EmbeddingProvider
from everos.component.images import ImageLimits
from everos.memory.ports import (
    ClusterStore,
    EpisodeIndex,
    EpisodeStore,
    ForgetLogStore,
    GroupProfileIndex,
    GroupProfileStore,
    GroupStore,
    ImageMemoryIndex,
    ImageMemoryStore,
    ImageStore,
    MemcellStore,
    MemorizeInboxStore,
    MessageForgetStore,
    OwnerDirStore,
    ProfileIndex,
    ProfileStore,
    ProjectLinkStore,
    SceneStore,
    SessionBufferStore,
    SessionPurgeStore,
)

if TYPE_CHECKING:
    from everos.memory.custom_instructions import CustomInstructions


@dataclass(frozen=True)
class Stores:
    """Every write-side store, typed by its port."""

    episode: EpisodeStore
    image_memory: ImageMemoryStore
    images: ImageStore
    forget_log: ForgetLogStore
    profile: ProfileStore
    session_buffer: SessionBufferStore
    memcell: MemcellStore
    cluster: ClusterStore
    memorize_inbox: MemorizeInboxStore
    owner_dir: OwnerDirStore
    session_purge: SessionPurgeStore
    groups: GroupStore
    project_links: ProjectLinkStore
    """Which groups belong to a project (docs/group-memory.md, section 4.4)."""
    scene: SceneStore
    group_profile: GroupProfileStore
    message_forget: MessageForgetStore


@dataclass(frozen=True)
class Indexes:
    """Every read-side index of the configured engine, typed by its port."""

    episode: EpisodeIndex
    image_memory: ImageMemoryIndex
    profile: ProfileIndex
    group_profile: GroupProfileIndex


@dataclass(frozen=True)
class MemoryDeps:
    """What a strategy may use.

    Args:
        llm: Extraction LLM.
        embedding: Embedding provider; ``None`` when not configured.
        stores: Write-side stores.
        indexes: Read-side indexes.
        instructions: Custom instructions appended to the operators' rules.
        picture_limits: ``[images]``, the scale a profile update reads a
            segment's pictures at; ``None`` when ``llm`` reads no images
            (``[extraction] vision`` off), and the update then reads text only.
    """

    llm: LLMClient
    embedding: EmbeddingProvider | None
    stores: Stores
    indexes: Indexes
    instructions: CustomInstructions
    picture_limits: ImageLimits | None = None
