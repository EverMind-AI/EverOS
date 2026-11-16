"""Domain events produced by the memory pipelines and strategies.

Every event is an :class:`~everos.memory.ports.Event`: frozen, and
round-trippable through JSON, because the scheduler persists the triggering
event of each run and replays it after a crash. Fields added later carry
defaults, and a reshaped event maps its old payload onto the new shape, so
payloads persisted by an older release still validate.
"""

from __future__ import annotations

from typing import Any, Literal

from everalgo.types import MemCell
from pydantic import BaseModel, ConfigDict, Field, model_validator

from everos.memory.ports import Event, Scope


class UserPipelineStarted(Event):
    """Fired at the start of :class:`UserMemoryPipeline.run`, once per cell.

    Hot-path emit, so the strategies that read the memcell itself
    (foresight) can start in parallel with the in-pipeline Episode
    extraction. Carries the algo-side ``MemCell`` so crash recovery has the
    full payload (OME serialises events to JSON via Pydantic v2 nested-model
    handling).
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


class EpisodeCopy(BaseModel):
    """One owner's copy of an extracted episode, as written to their files.

    Entry ids are sequences within each owner's daily file, so a copy's id
    cannot be derived from the memcell and travels on the event.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    owner_id: str
    entry_id: str
    owner_name: str | None = None
    """How the owner's chat lines were labelled in the source memcell -- their
    ``sender_name``, else ``owner_id`` -- so the narrative names the owner by
    it. ``None`` where no pipeline run labelled them (Reflection's merged
    episodes, backfill's synthesized events, a payload from before the field)."""


class EpisodeExtracted(Event):
    """Fired once per memcell after its episode is written to markdown.

    The pipeline extracts one episode per memcell and writes a copy of it
    into the files of every user sender; ``copies`` lists the copies actually
    written (a write skipped because the memcell was deleted is not listed,
    and a cell with no written copy emits no event). The copies share the
    text, timestamp and language, so work done once per episode (atomic
    facts, the clustering embedding, the summary) consumes this event and
    fans out over the copies itself, while work on one owner's state (the
    profile) consumes :class:`EpisodeExtractedForOwner`, which
    ``split_episode_by_owner`` derives from it. Reflection's merged episode
    is an event with one copy (``source="reflection"``).

    Carries ``episode_text`` so the strategies need no markdown read and do
    not race the cascade, and ``episode_timestamp_ms`` so clustering can
    stamp ``Cluster.last_ts``.

    A payload persisted by a release that emitted one event per copy
    (``owner_id`` / ``episode_entry_id`` / ``owner_name`` at the top level)
    validates as a one-copy event.
    """

    memcell_id: str
    episode_text: str
    episode_timestamp_ms: int
    copies: list[EpisodeCopy] = Field(min_length=1)
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

    def owner_scope(self, copy: EpisodeCopy) -> Scope:
        """The user scope ``copy`` was written under."""
        return Scope(
            app_id=self.app_id,
            project_id=self.project_id,
            owner_id=copy.owner_id,
            owner_type="user",
        )

    @model_validator(mode="before")
    @classmethod
    def _one_copy_from_per_copy_payload(cls, data: Any) -> Any:
        """Map a per-copy payload of an older release onto ``copies``."""
        if not isinstance(data, dict) or "episode_entry_id" not in data:
            return data
        fields = dict(data)
        copy = {
            "owner_id": fields.pop("owner_id", None),
            "entry_id": fields.pop("episode_entry_id"),
            "owner_name": fields.pop("owner_name", None),
        }
        return {**fields, "copies": [copy]}


class EpisodeExtractedForOwner(Event):
    """One owner's copy of a pipeline :class:`EpisodeExtracted`.

    Emitted by ``split_episode_by_owner`` for work that reads and rewrites one
    owner's state (the profile), so each owner is a run of its own, retried
    on its own.
    """

    memcell_id: str
    episode_entry_id: str
    owner_id: str
    owner_name: str | None = None
    """As :attr:`EpisodeCopy.owner_name`."""
    app_id: str = "default"
    project_id: str = "default"
    language: str | None = None
    """As :attr:`EpisodeExtracted.language`."""


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
