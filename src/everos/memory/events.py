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

    Carries the full case body (``task_intent`` / ``approach`` / ``key_insight``)
    so downstream strategies do not need to read LanceDB — they receive strong-
    consistency data on the event bus, avoiding the cascade lag that made
    ``extract_agent_skill`` retry-then-dead-letter on every run.

    ``quality_score`` lets ``trigger_skill_clustering`` short-circuit before any
    embedding work when the case is below algo's quality floor.
    ``case_timestamp_ms`` drives the algo-side ``Cluster.last_ts`` for the
    time-window filter in :func:`everalgo.clustering.cluster_by_geometry`.
    """

    memcell_id: str
    case_entry_id: str
    task_intent: str
    approach: str = ""
    """Case's Approach section verbatim. Defaults empty for back-compat with
    pending 1.2.2 events in the OME run_record queue."""
    key_insight: str | None = None
    """Case's optional KeyInsight section. Defaults None for the same
    back-compat reason as ``approach``."""
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

    Drives the agent-skill extraction strategy. Carries a snapshot of the
    triggering case body (``task_intent`` / ``approach`` / ``key_insight`` /
    ``quality_score`` / ``case_timestamp_ms``) plus ``case_vector`` (already
    embedded by ``trigger_skill_clustering``) so ``extract_agent_skill`` can
    build its algo input without a LanceDB probe that races cascade.
    """

    case_entry_id: str
    cluster_id: str
    agent_id: str
    app_id: str = "default"
    project_id: str = "default"
    task_intent: str = ""
    """Case task_intent for algo-side rendering. Default empty for back-compat
    with 1.2.2 payloads in the OME run_record queue; 1.2.3+ emitters populate it."""
    approach: str = ""
    key_insight: str | None = None
    quality_score: float = 0.0
    case_timestamp_ms: int = 0
    case_vector: list[float] | None = None
    """Case task_intent embedding, produced by trigger_skill_clustering when it
    embeds for cluster matching. Passed through so extract_agent_skill does not
    need a second embedding call for the > MAX_SKILLS_IN_PROMPT top-k branch."""
