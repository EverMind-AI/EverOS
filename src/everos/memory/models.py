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

from everos.core.errors import RosterConflictError
from everos.core.observability.logging import get_logger

logger = get_logger(__name__)

_Role = Literal["user", "assistant", "tool"]
EpisodeParentType = Literal["memcell", "cluster"]
Origin = Literal["member", "assistant", "mixed"]
"""Whose messages a group record rests on (docs/group-memory.md, section 4.5)."""


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
      (text / image / audio / doc / pdf / html / email). A picture item here
      carries the client's ``name`` and, once ingest has stored it, its
      ``image_id`` / dimensions — which is where an image memory's provenance
      comes from, since the rendered ``text`` keeps only the hash.
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
    client_message_id: str | None = None
    """The caller's own ``message_id`` (unique within the session), kept for a
    group batch: message-level forget and source quotes address messages by it
    (docs/group-memory.md, section 5.2). ``None`` on one-to-one batches, which
    keep today's generated ids only."""


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
    group_id: str | None = None
    """The group the session is bound to (D18); ``None`` on the one-to-one path.
    A pipeline reads it to know whose memory a cell is."""


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

    ``image_ids`` are the pictures this episode's body refers to, and it is
    what makes that reference resolvable: the body names a picture by id and
    never restates its description (the picture has its own memory), so a
    reader — the answering model above all — needs the episode to say which
    pictures it means. Recorded at write time from the cell's content items,
    never parsed back out of the text: a body is model-written prose and an id
    in it is a reference, not a field.
    """

    owner_id: str
    episode: str
    summary: str
    timestamp: int
    subject: str = ""

    # everos engineering metadata.
    session_id: str | None = None
    sender_ids: list[str] = Field(default_factory=list)
    image_ids: list[str] = Field(default_factory=list)
    """Content hashes of the pictures the body refers to, in the order they
    arrived. Empty for an episode that carried none, and for every episode
    written before pictures existed."""
    parent_id: str
    parent_type: EpisodeParentType = "memcell"
    language: str | None = None
    """The ``OutputLanguage`` value the episode is written in, detected from
    its text; downstream extractions are pinned to it. ``None`` when unknown
    (an episode stored before it existed, or text with no letters)."""

    # Group attribution (docs/group-memory.md, section 4.5). One-to-one
    # episodes keep the defaults, and markdown writes none of the three then.
    subject_ids: list[str] = Field(default_factory=list)
    """Members the episode is about. Empty on a one-to-one episode."""
    origin: Origin | None = None
    """``member`` / ``assistant`` / ``mixed``: whose messages the segment holds.
    ``None`` on a one-to-one episode."""
    source_start: int | None = None
    """Epoch ms of the segment's first message. ``None`` reads as ``timestamp``
    (the segment's last message): a membership window must contain the whole
    segment for a member to see it."""

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
        language: str | None = None,
        image_ids: Sequence[str] = (),
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
        data["language"] = language
        data["image_ids"] = list(image_ids)
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


class ImageMemory(BaseModel):
    """Domain image memory — what one picture showed, as one memory.

    One record per picture *occurrence*, not per file: the file is
    content-addressed and stored once, while its time and provenance belong to
    the message that carried it. The same picture in two sessions is one file
    and two of these — which is why ``image_id`` (the file) and ``id`` (this
    record) are separate identities.

    The record is a peer of :class:`Episode`: it competes for retrieval on its
    own, can be selected as core memory on its own, and can be forgotten on its
    own. An episode body references it by ``image_id`` rather than copying its
    description, so replacing a picture's description never rewrites the
    episode.

    ``status`` mirrors the other memory kinds rather than inventing a lifecycle:
    ``current`` is the live record, ``superseded`` names the entry that replaced
    it, and ``forgotten`` is a record whose picture was forgotten. Only
    ``current`` records compete for retrieval by default.
    """

    owner_id: str
    description: str
    """What the picture shows, written by the extraction model while it looked
    at the picture. This is the text the picture is retrieved by when no
    vector is available, so it names what distinguishes this picture."""

    timestamp: int
    ocr: str = ""
    """Text visible in the picture. Empty for a photograph of a scene; the
    reason a screenshot of a sign is searchable by the words on it."""

    image_id: str
    """SHA-256 of the picture file. Stable across occurrences and the handle
    ``forget`` takes."""

    name: str = ""
    """What the client called the picture (``IMG_2231.jpg``), as sent.

    Provenance of the *occurrence*, not of the file: the same bytes may arrive
    as ``IMG_2231.jpg`` in one conversation and ``board.png`` in another, and
    the two arriving names are different facts about different messages. Empty
    when the client sent none — an inline part usually carries no filename.
    Kept because a caller mapping a memory back to its source (an evaluation
    harness matching a dataset's picture ids) has only this to go on.
    """

    original_relpath: str = ""
    """Memory-root-relative path of the original, ``figures/<aa>/<hash>.<ext>``.
    Recorded rather than derived so a result can hand the caller a path even
    when the picture file itself is gone."""

    media_type: str = "jpeg"
    width: int = 0
    height: int = 0

    # everos engineering metadata.
    session_id: str | None = None
    parent_id: str = ""
    """Source memcell id — the segment the picture hung in."""
    parent_type: EpisodeParentType = "memcell"
    entry_id: str | None = None
    """``im_<YYYYMMDD>_<NNNN>`` once written; ``None`` before the store
    allocates it."""
    md_path: str | None = None
    status: Literal["current", "superseded", "forgotten"] = "current"
    superseded_by: str | None = None
    """Entry id of the record that replaced this one."""

    # Group attribution (docs/group-memory.md, section 4.5); one-to-one keeps
    # the defaults and markdown writes none of them.
    sender_ids: list[str] = Field(default_factory=list)
    """Who posted the picture (one id). Empty on a one-to-one picture memory."""
    subject_ids: list[str] = Field(default_factory=list)
    origin: Origin | None = None

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
    primary_language: str | None = None
    """The ``OutputLanguage`` value the profile is written in, frozen by the
    first extraction that knew it and never changed automatically after."""

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
        primary_language: str | None = None,
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
            primary_language=primary_language,
        )


MEMBER_ITEM_KEYS: tuple[str, ...] = (
    "item_id",
    "source_memcell_ids",
    "source_start",
    "source_end",
)
"""The provenance keys every item of a :class:`MemberProfile` carries.

``item_id`` stays while the item's description does; ``source_memcell_ids``
are the group segments it was read from and ``source_start`` /
``source_end`` (epoch ms) their span, which a read compares with the asker's
membership windows (docs/group-memory.md, section 9.1)."""


class MemberProfile(BaseModel):
    """One member's profile inside one group (docs/group-memory.md, 7.4, D6).

    ``profile`` is the profile operator's document about ``member_id``, read
    only from this group's segments in which the member spoke; it never holds
    the member's personal profile (``users/<m>/user.md``) or another group's.
    Its items carry :data:`MEMBER_ITEM_KEYS` as extra keys.

    Attributes:
        group_id: The group.
        member_id: Whom the profile is about.
        profile: The document; ``profile.owner_id`` is ``member_id``.
        pending_memcell_ids: Segments the member spoke in that no update has
            read yet (the ``[groups] member_profile_every`` throttle).
        history: Items a later update replaced or dropped, each with the
            ``replaced_at`` epoch ms of that update.
    """

    group_id: str
    member_id: str
    profile: Profile
    pending_memcell_ids: list[str] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)


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

    Identified by ``doc_id`` alone: where its markdown lives is the knowledge
    store's business (``WrittenDocument`` reports it to the catalog).

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


# ── Groups (docs/group-memory.md, sections 4.2-4.3) ─────────────────────

MemberKind = Literal["human", "assistant"]
AccessRole = Literal["owner", "admin", "member", "assistant"]
"""The role the caller declares; it decides who manages the roster, never
who reads more (section 9.1)."""
RosterOp = Literal["join", "leave", "update"]
RosterSource = Literal["api", "inline", "implicit"]
HistoryVisibleFrom = Literal["join", "creation"]
"""``join`` (default): a member sees what happened inside their membership
windows. ``creation``: a current member sees the group's history from its start
(section 15, open questions)."""

MANAGER_ROLES: frozenset[str] = frozenset({"owner", "admin"})
"""Access roles that may manage a group's roster and meta."""


class Group(BaseModel):
    """A group's meta: what ``group.md``'s frontmatter holds besides the log.

    Attributes:
        group_id: Path-safe id, unique within one ``(app_id, project_id)``.
        name: Shown to the extraction model in the roster block.
        description: Likewise.
        created_at: Epoch ms; the first message's time for an implicit group.
        created_by: Member id of the creator; ``None`` for an implicit group.
        extra: Stored, never interpreted.
        history_visible_from: See :data:`HistoryVisibleFrom`.
    """

    group_id: str
    name: str = ""
    description: str = ""
    created_at: int
    created_by: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    history_visible_from: HistoryVisibleFrom = "join"


class RosterEvent(BaseModel):
    """One change to a group's roster, appended in order and never rewritten.

    ``join`` opens an interval at ``at``; ``leave`` closes the open one;
    ``update`` changes ``display_name`` / ``aliases`` / ``role`` / ``title``
    from ``at`` on. Fields an event leaves ``None`` are not changed (a join
    falls back to the member's previous interval, then to the defaults).

    ``event_id`` is the entry id in ``group.md`` (``gr_<YYYYMMDD>_<seq>``),
    set by the store when the event is written.
    """

    op: RosterOp
    member_id: str
    at: int
    source: RosterSource = "api"
    kind: MemberKind | None = None
    display_name: str | None = None
    aliases: list[str] | None = None
    role: AccessRole | None = None
    title: str | None = None
    event_id: str | None = None


class MembershipInterval(BaseModel):
    """One member's half-open stay in a group: ``[joined_at, left_at)``.

    ``left_at`` is ``None`` while the member is in the group. A member who left
    and rejoined has two intervals; ``seq`` counts them from 1. The descriptive
    fields carry the values in force at the end of the interval.
    """

    member_id: str
    seq: int
    kind: MemberKind
    display_name: str
    aliases: list[str] = Field(default_factory=list)
    role: AccessRole
    title: str = ""
    joined_at: int
    left_at: int | None = None
    source: RosterSource = "api"

    def contains(self, at: int) -> bool:
        """Whether ``at`` (epoch ms) lies inside this interval."""
        return self.joined_at <= at and (self.left_at is None or at < self.left_at)


def fold_roster(events: Sequence[RosterEvent]) -> list[MembershipInterval]:
    """The membership intervals ``events`` describe, validated.

    Events apply in ``at`` order; events with the same ``at`` keep their log
    order, so an importer's backdated join lands where it belongs while two
    events of one request apply as sent. The result is sorted by
    ``(member_id, seq)``.

    Raises:
        RosterConflictError: A ``join`` for a member with an open interval, or
            a ``leave`` / ``update`` for one without (409, section 4.3).
    """
    ordered = sorted(enumerate(events), key=lambda pair: (pair[1].at, pair[0]))
    open_by_member: dict[str, MembershipInterval] = {}
    closed: list[MembershipInterval] = []
    seqs: dict[str, int] = {}
    last: dict[str, MembershipInterval] = {}
    for _, event in ordered:
        current = open_by_member.get(event.member_id)
        if event.op == "join":
            if current is not None:
                raise RosterConflictError(
                    f"{event.member_id!r} is already a member at {event.at}"
                )
            previous = last.get(event.member_id)
            kind = event.kind or (previous.kind if previous else None)
            if kind is None:
                kind = "assistant" if event.role == "assistant" else "human"
            role = event.role or (previous.role if previous else None)
            if role is None:
                role = "assistant" if kind == "assistant" else "member"
            seqs[event.member_id] = seqs.get(event.member_id, 0) + 1
            open_by_member[event.member_id] = MembershipInterval(
                member_id=event.member_id,
                seq=seqs[event.member_id],
                kind=kind,
                display_name=event.display_name
                or (previous.display_name if previous else event.member_id),
                aliases=list(
                    event.aliases
                    if event.aliases is not None
                    else (previous.aliases if previous else [])
                ),
                role=role,
                title=event.title
                if event.title is not None
                else (previous.title if previous else ""),
                joined_at=event.at,
                source=event.source,
            )
            continue
        if current is None:
            raise RosterConflictError(
                f"{event.member_id!r} is not a member at {event.at}; cannot {event.op}"
            )
        if event.op == "leave":
            ended = current.model_copy(update={"left_at": event.at})
            closed.append(ended)
            last[event.member_id] = ended
            del open_by_member[event.member_id]
            continue
        changes = {
            name: value
            for name, value in (
                ("display_name", event.display_name),
                ("aliases", event.aliases),
                ("role", event.role),
                ("title", event.title),
            )
            if value is not None
        }
        open_by_member[event.member_id] = current.model_copy(update=changes)
    for interval in open_by_member.values():
        last[interval.member_id] = interval
    return sorted(
        [*closed, *open_by_member.values()], key=lambda i: (i.member_id, i.seq)
    )


class Roster(BaseModel):
    """A group's membership intervals, with the questions every read asks.

    Built from :func:`fold_roster`; holds no events, so it cannot answer "the
    roster at time t" for the descriptive fields -- only who was a member when.
    """

    intervals: list[MembershipInterval] = Field(default_factory=list)

    def current(self, member_id: str) -> MembershipInterval | None:
        """The member's open interval, or ``None`` when they are not in the group."""
        for interval in self.intervals:
            if interval.member_id == member_id and interval.left_at is None:
                return interval
        return None

    def is_current_member(self, member_id: str) -> bool:
        return self.current(member_id) is not None

    def current_members(self) -> list[MembershipInterval]:
        return [i for i in self.intervals if i.left_at is None]

    def intervals_of(self, member_id: str) -> list[MembershipInterval]:
        return [i for i in self.intervals if i.member_id == member_id]

    def interval_at(self, member_id: str, at: int) -> MembershipInterval | None:
        """The member's interval that contains ``at``, if any."""
        for interval in self.intervals_of(member_id):
            if interval.contains(at):
                return interval
        return None

    def windows_of(self, member_id: str) -> list[tuple[int, int | None]]:
        """The member's windows ``[from, to)``, adjacent intervals merged."""
        windows: list[tuple[int, int | None]] = []
        for interval in sorted(self.intervals_of(member_id), key=lambda i: i.joined_at):
            if windows and windows[-1][1] == interval.joined_at:
                windows[-1] = (windows[-1][0], interval.left_at)
            else:
                windows.append((interval.joined_at, interval.left_at))
        return windows

    def has_managers(self) -> bool:
        """Whether any current member holds an owner or admin role."""
        return any(i.role in MANAGER_ROLES for i in self.current_members())


__all__ = [
    "MANAGER_ROLES",
    "MEMBER_ITEM_KEYS",
    "AccessRole",
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
    "Group",
    "HistoryVisibleFrom",
    "ImageMemory",
    "ImplicitTraitItem",
    "IngestResult",
    "MemCell",
    "MemberKind",
    "MemberProfile",
    "MembershipInterval",
    "MemcellRecord",
    "MemorizeTask",
    "Origin",
    "Profile",
    "ReflectionReport",
    "Roster",
    "RosterEvent",
    "RosterOp",
    "RosterSource",
    "ToolCall",
    "fold_roster",
    "mint_cluster_id",
]
