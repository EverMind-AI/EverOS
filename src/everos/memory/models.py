"""Domain models shared across the memory layer.

These models live in the domain layer so service / pipeline / ingest can
all consume the same canonical shapes.

Algorithm-side models (``MemCell``, ``Episode``, ``ChatMessage``) are
owned by ``everalgo.types`` and are not re-defined here. Re-export here for
the ``from everos.memory import MemCell, Episode`` convenience.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from collections.abc import Sequence
from typing import Any, Literal

from everalgo.types import AgentCase as AlgoAgentCase
from everalgo.types import AtomicFact as AlgoAtomicFact
from everalgo.types import ChatMessage as AlgoMessage
from everalgo.types import Episode as AlgoEpisode
from everalgo.types import Foresight as AlgoForesight
from everalgo.types import MemCell as MemCell
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from everos.core.observability.logging import get_logger

logger = get_logger(__name__)

_Role = Literal["user", "assistant", "tool"]
EpisodeParentType = Literal["memcell", "cluster"]


class ToolCall(BaseModel):
    """OpenAI Chat Completions tool_call shape (kept verbatim)."""

    id: str
    type: str = "function"
    function: dict[str, str]  # {"name": ..., "arguments": json_str}


class CanonicalMessage(BaseModel):
    """Canonical internal message after ingest normalisation.

    Carries enough metadata to be persisted into ``unprocessed_buffer``,
    adapted into ``everalgo.types.Message`` for the algo layer, and
    reconstructed back when ``adetect`` returns a ``tail``.

    Field split (mirrors src_old ``RawMessage.content_items``):

    - ``content_items`` holds the raw ``ContentItem`` array verbatim
      (text / image / audio / doc / pdf / html / email). Currently only
      ``type="text"`` is parsed downstream; the field still keeps the
      original structure so a future multimodal parser can reach back
      without losing data.
    - ``text`` is the derived concatenation of ``content_items[*].text``
      for entries with ``type="text"`` — what the LLM-facing extractors
      and md writer consume.
    """

    message_id: str
    session_id: str
    sender_id: str
    sender_name: str | None = None
    role: _Role
    timestamp: dt.datetime
    content_items: list[dict[str, Any]] = Field(default_factory=list)
    text: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class IngestResult(BaseModel):
    """Output of ``ingest.process()`` — handed to pipelines.

    ``unparsed_non_text_count`` reports the number of non-text
    ``ContentItem`` entries kept on the records (a parser hook will
    consume them once it lands; today only ``type="text"`` is parsed).
    """

    session_id: str
    app_id: str = "default"
    project_id: str = "default"
    """App / project scope for this add cycle (request-level; default
    ``"default"``). Threaded to the boundary ledger + writers so memcells,
    buffer rows, and md paths all land in the right space."""
    messages: list[CanonicalMessage]
    unparsed_non_text_count: int = 0


class Episode(BaseModel):
    """Domain Episode — algo-emitted business fields + everos context.

    Composed (not inherited) from :class:`everalgo.types.Episode`. everos
    keeps the *semantic* fields algo emits (``owner_id``, narrative
    ``episode`` text, ``subject``, ``timestamp``) and adds engineering
    context (``session_id``, ``sender_ids``, ``parent_id``). The global
    episode id is derived later by cascade from
    ``<scope_id>_<entry_id_in_md>`` — algo no longer mints an id of its
    own.

    ``summary`` is a short display preview of ``episode`` and is required,
    as everalgo makes it; an episode stored before it existed reads back
    with ``""``. ``subject`` is the episode's title.

    ``parent_id`` is the source memcell id. The new everalgo types no
    longer carry ``parent_id`` on Episode / Foresight / AtomicFact, so
    everos fills it from the memcell currently being processed (the
    pipeline knows the id — it created the memcell). ``parent_type`` is
    ``"cluster"`` for an episode Reflection merged out of a cluster, in
    which case ``parent_id`` is that cluster's id.
    """

    owner_id: str
    episode: str
    summary: str
    timestamp: int
    subject: str = ""

    # everos engineering metadata.
    session_id: str | None = None
    sender_ids: list[str] = Field(default_factory=list)
    parent_id: str
    parent_type: EpisodeParentType = "memcell"

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_algo(
        cls,
        algo_episode: AlgoEpisode,
        *,
        owner_id: str,
        session_id: str | None,
        sender_ids: list[str],
        parent_id: str,
    ) -> Episode:
        """Build a domain Episode from an algo Episode plus engineering context.

        ``owner_id`` is caller-supplied so the same generic algo Episode
        (produced with ``sender_id=None`` to save an LLM round-trip per
        sender) can fan out to one md per user sender. The pipeline runs
        the algo once per cell and then loops the senders here, each
        getting an ``Episode`` rooted at its own ``owner_id``. Any
        ``owner_id`` algo's model might carry is dropped — the algo's
        value is ``None`` in the generic path; even when it isn't, the
        caller's context is authoritative.

        ``parent_id`` is required for the same reason: the caller always
        knows the source memcell id. Anything algo's model carries via
        ``extra='allow'`` is dropped in favour of the caller-supplied value.
        """
        data = algo_episode.model_dump(exclude={"parent_id", "owner_id"})
        data["owner_id"] = owner_id
        data["session_id"] = session_id
        data["sender_ids"] = list(sender_ids)
        data["parent_id"] = parent_id
        return cls.model_validate(data)


class AtomicFact(BaseModel):
    """Domain AtomicFact — algo-emitted business fields + everos context.

    Composed (not inherited) from :class:`everalgo.types.AtomicFact`. Mirrors
    :class:`Episode`: everos keeps the *semantic* fields algo emits
    (``owner_id`` / ``content`` / ``timestamp``) and adds engineering context
    (``session_id`` / ``parent_id``) so md writer + cascade can audit-link
    back to the source episode.

    No ``sender_ids``: an atomic fact is a statement about its ``owner_id``;
    the surrounding participants are not part of the fact itself. (Episode
    keeps ``sender_ids`` because the narrative is *about* the conversation
    as a whole.)

    ``parent_id`` is the source episode entry_id, supplied by the caller
    because the new everalgo types no longer carry it on AtomicFact.
    """

    owner_id: str
    content: str
    timestamp: int

    # everos engineering metadata.
    session_id: str | None = None
    parent_id: str

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_algo(
        cls,
        algo_fact: AlgoAtomicFact,
        *,
        owner_id: str,
        session_id: str | None,
        parent_id: str,
    ) -> AtomicFact:
        """Build a domain AtomicFact from an algo AtomicFact plus context.

        ``owner_id`` is supplied by the caller (not read from ``algo_fact``)
        because atomic_fact extraction uses a subject-agnostic prompt — one
        LLM call produces facts per owner. The algo-side ``owner_id`` is
        therefore a placeholder; the caller knows the real one. Same
        rationale for ``parent_id``: algo no longer carries the source
        episode entry_id; caller injects the authoritative value (any
        ``extra='allow'`` smuggled values are dropped).
        """
        data = algo_fact.model_dump(exclude={"parent_id", "owner_id"})
        data["owner_id"] = owner_id
        data["session_id"] = session_id
        data["parent_id"] = parent_id
        return cls.model_validate(data)


class Foresight(BaseModel):
    """Domain Foresight — algo-emitted business fields + everos context.

    Composed (not inherited) from :class:`everalgo.types.Foresight`. Mirrors
    :class:`Episode`: everos keeps the semantic fields algo emits
    (``owner_id`` / ``foresight`` / ``evidence`` / ``timestamp`` plus the
    optional time-window trio) and adds engineering context
    (``session_id`` / ``parent_id``).

    Extraction is per-sender (like Episode, unlike AtomicFact's
    subject-agnostic fan-out): a foresight is a forward-looking statement
    *about* a specific user, so the algo is invoked once per user sender
    and the emitted ``owner_id`` already matches — no fan-out override
    needed.

    No ``sender_ids``: a foresight is a prediction about its ``owner_id``;
    other participants in the source conversation are not part of the
    foresight itself.
    """

    owner_id: str
    foresight: str
    evidence: str
    timestamp: int
    start_time: str | None = None
    end_time: str | None = None
    duration_days: int | None = None

    # everos engineering metadata.
    session_id: str | None = None
    parent_id: str

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_algo(
        cls,
        algo_foresight: AlgoForesight,
        *,
        session_id: str | None,
        parent_id: str,
    ) -> Foresight:
        """Build a domain Foresight from an algo Foresight plus context.

        Per-sender extraction: the algo's ``owner_id`` is authoritative
        (it was invoked with the target sender). Only the engineering
        metadata is injected here. Any algo-side ``parent_id`` smuggled
        through ``extra='allow'`` is dropped in favour of the caller's.
        """
        data = algo_foresight.model_dump(exclude={"parent_id"})
        data["session_id"] = session_id
        data["parent_id"] = parent_id
        return cls.model_validate(data)


class AgentCase(BaseModel):
    """Domain AgentCase — algo-emitted business fields + everos context.

    Composed (not inherited) from :class:`everalgo.types.AgentCase`; the two
    convert through :mod:`everos.memory.mappers.agent_case`. everos keeps the
    semantic fields algo emits (``task_intent`` / ``approach`` /
    ``quality_score`` / ``key_insight`` / ``timestamp``) and adds engineering
    context (``owner_id`` = agent_id, ``session_id`` / ``parent_id``).

    ``case_gate`` is the admission decision when the case passed everalgo's
    correctness gate (``None`` when no gate ran). It travels with the case to
    skill extraction, which treats an admitted case as worth a skill whatever
    its quality score; the case markdown does not record it.
    """

    owner_id: str
    task_intent: str
    approach: str
    quality_score: float
    key_insight: str | None = None
    timestamp: int
    case_gate: dict[str, Any] | None = None

    # everos engineering metadata.
    session_id: str
    parent_id: str

    model_config = ConfigDict(extra="allow")


class ExplicitInfoItem(BaseModel):
    """One fact a source states about the profile's owner.

    Same shape as everalgo's ``ExplicitInfoItem``; extra keys a host attaches
    ride along.

    Attributes:
        description: The fact as one short sentence.
        category: The dimension the fact is filed under.
        evidence: The source excerpt that grounds it.
        observed_at: Epoch ms of the source that last established it.
    """

    description: str
    category: str | None = None
    evidence: str | None = None
    observed_at: int | None = None

    model_config = ConfigDict(extra="allow")


class ImplicitTraitItem(BaseModel):
    """One stable pattern inferred about the profile's owner.

    Same shape as everalgo's ``ImplicitTraitItem``; extra keys ride along.

    Attributes:
        description: The pattern as one short sentence.
        trait: The trait label it is filed under.
        basis: The signals the inference rests on.
        observed_at: Epoch ms of the source that last established it.
    """

    description: str
    trait: str | None = None
    basis: str | None = None
    observed_at: int | None = None

    model_config = ConfigDict(extra="allow")


class Profile(BaseModel):
    """Domain user profile — one per owner, rewritten on every synthesis.

    An item given as a bare sentence (how early profiles were stored) is the
    item with only its ``description``. Constructing a profile is strict;
    reading one back from storage goes through :meth:`from_stored`.
    """

    owner_id: str
    summary: str
    timestamp: int
    explicit_info: list[ExplicitInfoItem] = Field(default_factory=list)
    implicit_traits: list[ImplicitTraitItem] = Field(default_factory=list)

    @field_validator("explicit_info", "implicit_traits", mode="before")
    @classmethod
    def _sentences_are_items(cls, items: object) -> object:
        if not isinstance(items, list):
            return items
        return [_item_data(i) for i in items]

    @classmethod
    def from_stored(
        cls,
        *,
        owner_id: str,
        summary: str,
        timestamp: int,
        explicit_info: Sequence[object],
        implicit_traits: Sequence[object],
    ) -> Profile:
        """A stored profile, without the items that are no longer items.

        Early profiles could hold an item with no ``description``; one bad
        item must not fail every read of its profile, nor reach everalgo on
        the next update. Each such item is dropped with a
        ``profile_entry_dropped`` warning.
        """
        return cls(
            owner_id=owner_id,
            summary=summary,
            timestamp=timestamp,
            explicit_info=_readable_items(ExplicitInfoItem, explicit_info, owner_id),
            implicit_traits=_readable_items(
                ImplicitTraitItem, implicit_traits, owner_id
            ),
        )


_DROPPED_PREVIEW_CHARS = 80


def _item_data(item: object) -> object:
    """A bare sentence is the item with only its description."""
    return {"description": item} if isinstance(item, str) else item


def _readable_items[ItemT: BaseModel](
    model: type[ItemT], items: Sequence[object], owner_id: str
) -> list[ItemT]:
    kept: list[ItemT] = []
    for item in items:
        try:
            kept.append(model.model_validate(_item_data(item)))
        except ValidationError:
            logger.warning(
                "profile_entry_dropped",
                owner_id=owner_id,
                bucket=model.__name__,
                entry=_preview(item),
            )
    return kept


def _preview(item: object) -> str:
    text = (
        item
        if isinstance(item, str)
        else json.dumps(item, ensure_ascii=False, default=str)
    )
    return text[:_DROPPED_PREVIEW_CHARS]


class AgentSkill(BaseModel):
    """Domain agent skill — a reusable procedure aggregated from cases.

    Mirrors the fields of everalgo-core 0.6 ``AgentSkill``; stage 3/4
    converges the two. ``id`` survives a rename, which is how a renamed
    skill is told apart from a new one.
    """

    id: str
    cluster_id: str = ""
    name: str = ""
    description: str = ""
    content: str = ""
    confidence: float = 0.0
    maturity_score: float = 0.6
    source_case_ids: list[str] = Field(default_factory=list)


class Cluster(BaseModel):
    """Domain cluster snapshot — a group of related memories of one owner.

    Mirrors the fields of everalgo-core 0.6 ``clustering.Cluster`` with the
    centroid as a plain float tuple, so the domain carries no numpy; stage
    3/4 converges the two.
    """

    model_config = ConfigDict(frozen=True)

    id: str | None = None
    centroid: tuple[float, ...]
    count: int = 1
    last_ts: int
    preview: list[str] = Field(default_factory=list)
    members: list[str] = Field(default_factory=list)


class KnowledgeMemory(BaseModel):
    """Domain knowledge memory — one node of an extracted topic tree.

    Mirrors everalgo-knowledge ``KnowledgeMemory`` (flattened DFS form; the
    node at ``topic_index=0`` is the document root and carries the title as
    ``topic`` and the document summary). :mod:`everos.memory.mappers.knowledge`
    converts between the two.
    """

    doc_id: str
    topic_index: int
    topic: str
    summary: str
    content: str = ""
    block_refs: list[int] = Field(default_factory=list)
    depth: int
    parent_index: int | None = None
    children_index: list[int] = Field(default_factory=list)
    topic_path: str
    content_labels: list[str] = Field(default_factory=list)
    category_id: str = ""


class KnowledgeCategory(BaseModel):
    """One entry of a project's knowledge taxonomy."""

    id: str
    description: str = ""


class KnowledgeDocument(BaseModel):
    """Domain knowledge document — the metadata of one uploaded document.

    ``md_path`` is the document's ``index.md`` relative to the memory root,
    as recorded by whoever read it; the knowledge store uses it to find the
    document's directory. This is a deliberate exception to the rule that
    domain models carry no storage path: a knowledge document is an entity
    rooted at its directory, so its location is part of its identity --
    repairing a directory that escaped ``knowledge/``, the replace backup and
    the uploaded original are all found through it. Owner-scoped memories
    keep identity apart, in ``MemoryRef``.

    ``created_at`` / ``updated_at`` are set by the catalog that records the
    document; a document read straight from its markdown has neither.
    """

    doc_id: str
    app_id: str
    project_id: str
    category_id: str
    title: str
    summary: str = ""
    source_name: str | None = None
    source_type: str | None = None
    md_path: str | None = None
    created_at: dt.datetime | None = None
    updated_at: dt.datetime | None = None


class KnowledgeTopic(BaseModel):
    """Domain knowledge topic — one node of a document's topic tree.

    Knowledge is project-level, not owned by a user or agent, so the topic
    carries its own ``app_id`` / ``project_id`` instead of an owner
    ``Scope``. ``content``, ``content_labels`` and ``children_topic_ids`` are
    only held by the catalog; a search hit leaves them empty.
    """

    topic_id: str
    doc_id: str
    app_id: str
    project_id: str
    category_id: str
    topic_name: str
    topic_path: str
    depth: int
    summary: str
    parent_topic_id: str | None = None
    content: str = ""
    content_labels: list[str] = Field(default_factory=list)
    children_topic_ids: list[str] = Field(default_factory=list)
    created_at: dt.datetime
    updated_at: dt.datetime


ClusterKind = Literal["user_memory", "agent_case"]
ClusterMemberType = Literal["episode", "case"]


def mint_cluster_id() -> str:
    """Mint a fresh cluster id (``cl_<12hex>``)."""
    return f"cl_{uuid.uuid4().hex[:12]}"


class MemcellRecord(BaseModel):
    """One boundary-detected slice of a conversation, ready to persist.

    Attributes:
        memcell_id: everos-minted identifier of the slice.
        cell: The algorithm-side cell (items + timestamp).
        raw_type: Source shape of the slice (conversation / agent trace).
        message_ids: Ids of the buffered messages the slice consumed.
        sender_ids: Distinct senders in the slice, first-seen order.
    """

    memcell_id: str
    cell: MemCell
    raw_type: str
    message_ids: list[str]
    sender_ids: list[str]


class MemorizeTask(BaseModel):
    """One accepted asynchronous memorize request in the durable queue.

    ``id`` orders a session's requests; ``status`` is one of ``queued`` /
    ``processing`` / ``success`` / ``failed``. ``payload_json`` is emptied
    once the request is delivered.
    """

    id: int
    request_id: str
    app_id: str
    project_id: str
    session_id: str
    is_final: bool
    defer_extraction: bool
    payload_json: str
    message_count: int
    status: str
    attempt: int
    last_error: str | None
    created_at: dt.datetime
    updated_at: dt.datetime


class ReflectionReport(BaseModel):
    """Audit record of one completed Reflection merge.

    Attributes:
        id: Report identifier.
        cluster_id: Cluster that was reflected.
        owner_id: Owner of the cluster.
        app_id: Application scope.
        project_id: Project scope.
        mode: ``"init"`` or ``"update"``.
        source_members: ``(member_id, member_type)`` pairs consumed.
        merged_entry_id: Entry id of the merged episode.
        deprecated_fact_count: Atomic facts marked deprecated.
    """

    id: str
    cluster_id: str
    owner_id: str
    app_id: str
    project_id: str
    mode: Literal["init", "update"]
    source_members: list[tuple[str, str]]
    merged_entry_id: str
    deprecated_fact_count: int

    @property
    def source_count(self) -> int:
        return len(self.source_members)


__all__ = [
    "AgentCase",
    "AgentSkill",
    "AlgoAgentCase",
    "AlgoAtomicFact",
    "AlgoEpisode",
    "AlgoForesight",
    "AlgoMessage",
    "AtomicFact",
    "CanonicalMessage",
    "Cluster",
    "ClusterKind",
    "ClusterMemberType",
    "Episode",
    "EpisodeParentType",
    "ExplicitInfoItem",
    "Foresight",
    "ImplicitTraitItem",
    "IngestResult",
    "MemCell",
    "MemcellRecord",
    "MemorizeTask",
    "Profile",
    "ReflectionReport",
    "ToolCall",
    "mint_cluster_id",
]
