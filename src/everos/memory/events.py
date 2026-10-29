"""Domain events produced by the memory pipelines and strategies.

Every event is an :class:`~everos.memory.ports.Event`: frozen, and
round-trippable through JSON, because the scheduler persists the triggering
event of each run and replays it after a crash. Fields added later carry
defaults so payloads persisted by an older release still validate.
"""

from __future__ import annotations

from typing import Any, Literal

from everalgo.types import MemCell
from pydantic import BaseModel, ConfigDict, Field

from everos.memory.ports import Event


class UserPipelineStarted(Event):
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


class AgentPipelineStarted(Event):
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


class ReflectedEpisode(BaseModel):
    """One cluster member a Reflection merge replaces, located in markdown."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entry_id: str
    md_path: str
    parent_id: str


class ReflectionContext(BaseModel):
    """What finishing one Reflection merge needs, captured when it was merged.

    The merge step writes the merged episode and publishes it for fact
    extraction; the finishing step runs only after the facts exist, so the
    snapshot it acts on travels with the events instead of living in a
    waiting coroutine.

    Attributes:
        cluster_id: The cluster that was merged.
        mode: ``"init"`` for a first merge, ``"update"`` when the cluster
            already held a merged episode.
        source_members: ``(member_id, member_type)`` pairs at merge time.
        episodes: The member episodes as they were read for the merge.
        merged_episode: Text of the merged episode (the new centroid source).
        last_ts: Newest member timestamp, epoch milliseconds.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cluster_id: str
    mode: Literal["init", "update"]
    source_members: list[tuple[str, str]]
    episodes: list[ReflectedEpisode]
    merged_episode: str
    last_ts: int


class EpisodeExtracted(Event):
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
    reflection: ReflectionContext | None = None
    """Set on Reflection's merged episode (``source="reflection"``); carried
    through to :class:`AtomicFactsExtracted` so the step that finishes the
    merge knows what to deprecate."""
    language: str | None = None
    """The episode's detected ``OutputLanguage`` value; the extractions this
    event drives write in it. ``None`` lets each extractor infer its own."""


class AtomicFactsExtracted(Event):
    """Fired by ``extract_atomic_facts`` once an episode's facts are written.

    Fired even when the episode yielded no facts: Reflection finishes a
    merge on this event, and a merged episode without facts must still
    retire the members it replaced.
    """

    episode_entry_id: str
    owner_id: str
    app_id: str = "default"
    project_id: str = "default"
    source: str = "pipeline"
    fact_entry_ids: list[str] = Field(default_factory=list)
    reflection: ReflectionContext | None = None


class AgentCaseExtracted(Event):
    """Fired by ``extract_agent_case`` after the AgentCase md is written.

    Carries the full case body (``task_intent`` / ``approach`` / ``key_insight``)
    so downstream strategies do not need to read LanceDB — they receive strong-
    consistency data on the event bus, avoiding the cascade lag that made
    ``extract_agent_skill`` retry-then-dead-letter on every run.

    ``quality_score`` and ``case_gate`` let ``trigger_skill_clustering``
    short-circuit before any embedding work when algo would not distil the
    case into a skill.
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
    case_gate: dict[str, Any] | None = None
    """The case's admission decision when a correctness gate admitted it;
    ``None`` when no gate ran (see :attr:`AgentCase.case_gate`)."""


class ProfileClusterUpdated(Event):
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
    language: str | None = None
    """Carried over from the triggering :class:`EpisodeExtracted`."""


class SkillClusterUpdated(Event):
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
    case_gate: dict[str, Any] | None = None
    """Passed through from :attr:`AgentCaseExtracted.case_gate`."""
