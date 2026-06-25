"""Domain events emitted by memory pipelines, consumed by OME strategies."""

from __future__ import annotations

from everalgo.types import MemCell

from everos.infra.ome.events import BaseEvent


class UserPipelineStarted(BaseEvent):
    """Fired at the start of :class:`UserMemoryPipeline.run`, once per cell.

    Hot-path emit, so atomic_fact / foresight / clustering strategies can
    start in parallel with the in-pipeline Episode extraction. Carries the
    algo-side ``MemCell`` so crash recovery has the full payload (OME
    serialises events to JSON via Pydantic v2 nested-model handling).
    """

    memcell_id: str
    session_id: str
    app_id: str = "default"
    project_id: str = "default"
    memcell: MemCell


class AgentPipelineStarted(BaseEvent):
    """Fired at the start of :class:`AgentMemoryPipeline.run`, once per cell.

    Only emitted in ``mode="agent"`` (the agent pipeline does not run in
    chat mode). Subscribers handle the agent-side processing chain
    (case extraction, agent-skill clustering) in parallel with the user
    chain. Payload mirrors :class:`UserPipelineStarted`.
    """

    memcell_id: str
    session_id: str
    app_id: str = "default"
    project_id: str = "default"
    memcell: MemCell


class EpisodeExtracted(BaseEvent):
    """Fired once per Episode after :class:`UserMemoryPipeline` writes its md.

    Carries ``episode_text`` so downstream clustering can embed it without
    racing the cascade (cascade also embeds, but at the LanceDB layer —
    keeping a copy on the event is cheaper than polling LanceDB until the
    row appears). ``episode_timestamp_ms`` rides along so the cluster
    strategy can stamp the algo-side ``Cluster.last_ts`` without a second
    md read. One memcell can produce multiple episodes (one per user
    sender), so this event fires per-episode, not per-memcell.
    """

    memcell_id: str
    episode_entry_id: str
    episode_text: str
    episode_timestamp_ms: int
    owner_id: str
    session_id: str | None = None
    app_id: str = "default"
    project_id: str = "default"
    source: str = "pipeline"


class AgentCaseExtracted(BaseEvent):
    """Fired by ``extract_agent_case`` after the AgentCase md is written.

    Carries ``task_intent`` so the skill-clustering strategy can embed it
    directly, and ``quality_score`` so the strategy can short-circuit
    before any embedding work when the case is below algo's quality floor
    (``AgentCaseExtractor`` also short-circuits internally; this is the
    upstream gate that saves an LLM call too). ``case_timestamp_ms``
    drives the algo-side ``Cluster.last_ts`` for the time-window filter
    in :func:`everalgo.clustering.cluster_by_geometry`.
    """

    memcell_id: str
    case_entry_id: str
    task_intent: str
    quality_score: float
    case_timestamp_ms: int
    agent_id: str
    app_id: str = "default"
    project_id: str = "default"


class ProfileClusterUpdated(BaseEvent):
    """Fired after the user-memory cluster strategy has merged a new
    memcell into a cluster.

    Drives the profile-extraction strategy; ``cluster_id`` is the new
    or merged cluster the source memcell now belongs to.
    """

    memcell_id: str
    cluster_id: str
    owner_id: str
    app_id: str = "default"
    project_id: str = "default"


class SkillClusterUpdated(BaseEvent):
    """Fired after the agent-case cluster strategy has merged a new
    case into a cluster.

    Drives the agent-skill extraction strategy; ``cluster_id`` is the
    new or merged cluster the source case now belongs to.
    """

    case_entry_id: str
    cluster_id: str
    agent_id: str
    app_id: str = "default"
    project_id: str = "default"
