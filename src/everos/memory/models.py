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
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from everalgo.types import ChatMessage as AlgoMessage
from everalgo.types import Episode as AlgoEpisode
from everalgo.types import MemCell as MemCell
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from everos.core.errors import RosterConflictError
from everos.core.observability.logging import get_logger

logger = get_logger(__name__)

_Role = Literal["user", "assistant", "tool"]
EpisodeParentType = Literal["memcell", "cluster", "scene"]
"""What an episode row rests on: a memcell (a segment), or a scene -- a scene
version, which the episode table holds next to the owner's episodes
(docs/scenes.md). ``cluster`` is read only: an episode the Reflection merge of
an earlier 2.0 build wrote over a cluster; nothing writes one now."""
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

    ``rejected_count`` counts the pictures ingest refused (over a limit, not a
    picture, unreadable), ``failed_count`` the non-text items the parser could
    not turn into text; ``unparsed_non_text_count`` is their sum. Each such
    item is counted once, stays on its message's content items, and adds
    nothing to its ``text``. ``skipped_count`` counts the batch's messages
    ingest was told to leave out: the ones the group already held.
    """

    session_id: str
    app_id: str = "default"
    project_id: str = "default"
    """App / project scope for this add cycle (request-level; default
    ``"default"``). Threaded to the boundary ledger + writers so memcells,
    buffer rows, and md paths all land in the right space."""
    messages: list[CanonicalMessage]
    unparsed_non_text_count: int = 0
    rejected_count: int = 0
    failed_count: int = 0
    group_id: str | None = None
    """The group the session is bound to (D18); ``None`` on the one-to-one path.
    A pipeline reads it to know whose memory a cell is."""
    skipped_count: int = 0


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
    longer carry ``parent_id`` on Episode, so everos fills it from the
    memcell currently being processed (the pipeline knows the id — it
    created the memcell). ``parent_type`` is ``"scene"`` for a scene
    version, whose ``parent_id`` is the scene id.

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
    """The ``OutputLanguage`` value the episode is written in: its
    conversation's, resolved from the human messages and named to the model
    (read off the episode's own text only for a conversation without a
    letter); downstream extractions are pinned to it. ``None`` when unknown
    (an episode stored before it existed, or no text anywhere)."""

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

    # A version a person wrote through ``POST /memory/update`` (EDIT_FIELDS).
    supersedes: str | None = None
    edited_by: str | None = None
    edited_at: int | None = None
    held: bool = False

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

    # A version a person wrote through ``POST /memory/update`` (EDIT_FIELDS).
    supersedes: str | None = None
    edited_by: str | None = None
    edited_at: int | None = None
    held: bool = False

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
    """One person's profile in one space, rewritten on every update.

    The space is a user's own one-to-one memory (``users/<u>/user.md``, the
    person is the user) or a group (``groups/<g>/members/<m>.md``, the person
    is a member, read only from that group's segments -- docs/group-memory.md,
    7.4, D6): one document either way, written by the one profile operator.
    Its items carry :data:`MEMBER_ITEM_KEYS` as extra keys, and
    :data:`MEMBER_VERSION_KEYS` when they replaced an item.

    An item given as a bare sentence (how early profiles were stored) is the
    item with only its ``description``. Constructing a profile is strict;
    reading one back from storage goes through :meth:`from_stored`.
    """

    owner_id: str
    """The person the profile is about: the user, or the member in a group."""
    summary: str
    timestamp: int
    explicit_info: list[ExplicitInfoItem] = Field(default_factory=list)
    implicit_traits: list[ImplicitTraitItem] = Field(default_factory=list)
    primary_language: str | None = None
    """The ``OutputLanguage`` value the profile is written in, frozen by the
    first extraction that knew it and never changed automatically after."""
    history: list[dict[str, Any]] = Field(default_factory=list)
    """Items an update replaced or retired, newest first, each with
    ``replaced_at`` and what replaced it (``superseded_by``, ``deprecated_by``
    and ``reason``, ``merged_into``, or ``reason``)."""
    pending_memcell_ids: list[str] = Field(default_factory=list)
    """Segments the person spoke in that no update has read yet: a group's
    profile is updated every ``[groups] member_profile_every`` of them. Always
    empty in a one-to-one space, which reads every segment at once."""

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
        history: Sequence[object] = (),
        pending_memcell_ids: Sequence[str] = (),
        md_path: str | None = None,
    ) -> Profile:
        """A stored profile, without the items that are no longer items.

        Early profiles could hold an item with no ``description``, and a hand
        edit can break one; one bad item must not fail every read of its
        profile, nor reach everalgo on the next update. Each such item is
        dropped with a ``profile_entry_dropped`` warning naming the file
        (``md_path``, when the caller read one), the bucket, the item's
        position in it and why it was dropped.
        """
        return cls(
            owner_id=owner_id,
            summary=summary,
            timestamp=timestamp,
            explicit_info=_readable_items(
                ExplicitInfoItem, explicit_info, owner_id, md_path
            ),
            implicit_traits=_readable_items(
                ImplicitTraitItem, implicit_traits, owner_id, md_path
            ),
            primary_language=primary_language,
            history=[dict(entry) for entry in history if isinstance(entry, dict)],
            pending_memcell_ids=list(pending_memcell_ids),
        )


class Scene(BaseModel):
    """One version of a scene: a user's (``users/<u>/scenes/``) or a group's.

    A scene is an owner's episodes about one thing with a narrative written
    over them (docs/scenes.md; a group's, docs/group-memory.md, section 7.7).
    It links to its episodes (``member_ids``, their entry ids) and never
    changes them. A change is a new version that supersedes the previous one;
    the old version stays readable in markdown.

    Its span runs from its first member's first message (``source_start``)
    to its newest member's last (``timestamp``): a group read shows the scene
    only to an asker whose window contains all of it.
    """

    owner_id: str
    """The user or the group."""
    scene_id: str
    version: int
    member_ids: list[str]
    subject: str
    summary: str
    narrative: str
    timestamp: int
    source_start: int
    sender_ids: list[str] = Field(default_factory=list)
    subject_ids: list[str] = Field(default_factory=list)
    origin: Origin | None = None
    language: str | None = None
    # A version a person wrote through ``POST /memory/update`` (EDIT_FIELDS).
    supersedes: str | None = None
    edited_by: str | None = None
    edited_at: int | None = None
    held: bool = False


EDIT_FIELDS: tuple[str, ...] = ("supersedes", "edited_by", "edited_at", "held")
"""What a version a person wrote through ``POST /memory/update`` carries.

On an episode, a picture memory or a scene version they are fields; on a
profile item (``user.md``, a member's profile) the last three are extra keys
next to the item's own ``supersedes`` / ``since``. ``supersedes`` is the id of
the version the edit replaced, ``edited_by`` who wrote it, ``edited_at`` when
(epoch ms). ``held`` is the plan's "a person's edit is not changed by the model
again": a profile update keeps a held item as it is, and a scene update leaves
a held scene version alone, until a caller releases it (an update with
``hold = false``). A model-written version carries none of them."""

EDIT_KEYS: tuple[str, ...] = ("edited_by", "edited_at", "held")
"""The :data:`EDIT_FIELDS` a profile item carries as extra keys; its
``supersedes`` is the profile's own version link (:data:`MEMBER_VERSION_KEYS`)."""


def is_held(item: object) -> bool:
    """Whether a person's version holds ``item`` against the model's updates.

    ``item`` is a daily-log record (an attribute), a profile item (an extra
    key) or an item's dict.
    """
    if isinstance(item, Mapping):
        return item.get("held") is True
    if isinstance(item, BaseModel) and "held" not in type(item).model_fields:
        return (item.model_extra or {}).get("held") is True
    return getattr(item, "held", False) is True


MEMBER_ITEM_KEYS: tuple[str, ...] = (
    "item_id",
    "source_memcell_ids",
    "source_start",
    "source_end",
)
"""The provenance keys every profile item carries, in either space.

``item_id`` stays while the item's description does; ``source_memcell_ids``
are the segments it was read from and ``source_start`` / ``source_end``
(epoch ms) their span, which a group read compares with the asker's
membership windows (docs/group-memory.md, section 9.1)."""

MEMBER_VERSION_KEYS: tuple[str, ...] = ("supersedes", "since")
"""The keys a profile fact carries when it is a new value of another.

``supersedes`` is the ``item_id`` of the ``explicit_info`` item an update
rewrote into this one and ``since`` the epoch ms of that update; the replaced
item's ``history`` entry names this one in ``superseded_by``
(docs/group-memory.md, 7.4). The personal profile (``user.md``) links its
facts the same way."""

MEMBER_PREVIOUS_KEY = "previous"
"""The key under which a read of a profile lists the values an item
replaced, newest first: ``item_id``, ``description``, ``until`` (epoch ms)
and the old item's ``source_start`` / ``source_end`` each."""

MEMBER_PREVIOUS_LIMIT = 2
"""Replaced values a read shows per profile item."""


def replaced_chains(
    items: Sequence[Mapping[str, Any]],
    history: Sequence[object],
    *,
    limit: int = MEMBER_PREVIOUS_LIMIT,
) -> list[list[Mapping[str, Any]]]:
    """For each profile item, the history entries of the values it replaced.

    Newest first, at most ``limit``: the entry its ``supersedes`` names, then
    the one that entry names, and so on. A link is followed only to an entry
    that names the newer item back (``superseded_by``) and records when it was
    replaced; the chain stops at the first link that does not hold, so an
    entry a forget removed ends it.
    """
    replaced: dict[str, Mapping[str, Any]] = {}
    for entry in history:  # newest first: the first entry of an id is its last
        if isinstance(entry, Mapping) and isinstance(entry.get("item_id"), str):
            replaced.setdefault(entry["item_id"], entry)
    chains: list[list[Mapping[str, Any]]] = []
    for item in items:
        chain: list[Mapping[str, Any]] = []
        newer, link = item.get("item_id"), item.get("supersedes")
        while len(chain) < limit and isinstance(link, str) and link in replaced:
            entry = replaced[link]
            if entry.get("superseded_by") != newer or not isinstance(
                entry.get("replaced_at"), int
            ):
                break
            if any(e is entry for e in chain):
                break
            chain.append(entry)
            newer, link = link, entry.get("supersedes")
        chains.append(chain)
    return chains


def with_previous_values(
    items: Sequence[object],
    history: Sequence[object],
    *,
    limit: int = MEMBER_PREVIOUS_LIMIT,
) -> list[object]:
    """Profile items as a read shows them, each with the values it replaced.

    An item that replaced others gains :data:`MEMBER_PREVIOUS_KEY`, read from
    ``history`` through :func:`replaced_chains`; every other item, and
    anything that is not an item, is returned as it is. What a forget took out
    of ``history`` is no longer there to show.
    """
    mappings = [item if isinstance(item, Mapping) else {} for item in items]
    shown: list[object] = []
    for item, chain in zip(
        items, replaced_chains(mappings, history, limit=limit), strict=True
    ):
        if not chain or not isinstance(item, Mapping):
            shown.append(item)
            continue
        previous = [
            {
                "item_id": entry["item_id"],
                "description": entry.get("description"),
                "until": entry["replaced_at"],
                "source_start": entry.get("source_start"),
                "source_end": entry.get("source_end"),
            }
            for entry in chain
        ]
        shown.append({**item, MEMBER_PREVIOUS_KEY: previous})
    return shown


FORGOTTEN_MARK = "forgotten"
"""The ``deprecated_by`` value a forget writes, as opposed to the entry id of a
successor: a superseding item and a forget are different states, and a reader
must be able to tell "replaced by this entry" from "forgotten"."""

GroupProfileKind = Literal["topic", "decision", "dissent", "role"]
"""The four kinds of group-profile item (docs/group-memory.md, section 7.5)."""
TopicStatus = Literal["exploring", "disagreement", "consensus", "implemented"]
ObservedRole = Literal[
    "decision_maker",
    "opinion_leader",
    "topic_initiator",
    "execution_promoter",
    "core_contributor",
    "coordinator",
    "info_summarizer",
]
"""Roles a member is observed to play in the group's conversations -- never the
access role the caller declares (section 4.3)."""
GroupProfileStatus = Literal[
    "exploring",
    "disagreement",
    "consensus",
    "implemented",
    "current",
    "open",
    "resolved",
    "active",
    "retired",
    "superseded",
    "forgotten",
]
"""What a read says about an item: a topic's own status, ``current`` for a
decision, ``open`` / ``resolved`` for dissent, ``active`` / ``retired`` for a
role -- or ``superseded`` once a later item names it, ``forgotten`` once a
forget marked it."""


class ProfileSource(BaseModel):
    """One verified piece of evidence behind a group-profile item.

    Every field is taken from the segment, never from the model: the model
    names a segment, a message and a quote; code checks the quote is a
    verbatim part of that message and copies the speaker and the time from it.

    Attributes:
        memcell_id: The segment.
        message_id: The message's id inside the segment (the client's id when
            it sent one).
        speaker_id: The message's sender.
        at: The message's time, epoch ms.
        quote: The verbatim part of the message, whitespace normalised.
    """

    memcell_id: str
    message_id: str
    speaker_id: str
    at: int
    quote: str


class GroupProfileItem(BaseModel):
    """One append-only item of a group's profile (docs/group-memory.md, 7.5, D7).

    A change never edits an item: it appends a new one whose ``supersedes``
    names the entry it replaces. A topic keeps its ``topic_id`` across
    versions; a dissent is resolved and a role retired by a new version with
    ``resolved`` / ``retired`` set. Which fields a kind uses:

    * ``topic``: ``topic_id``, ``name``, ``summary``, ``topic_status``;
    * ``decision``: ``statement``, ``topic_id`` (optional), ``decided_by``,
      ``agreed_by``, ``assignees``, ``applies_to`` (empty for the whole
      group);
    * ``dissent``: ``statement``, ``held_by``, ``about`` (a decision's entry
      id or a topic id), ``resolved``, ``resolved_by``;
    * ``role``: ``member_id``, ``role``, ``confidence``, ``retired``.

    Every item carries its ``sources`` (verified evidence), the attribution
    fields of section 4.5 (``sender_ids`` are the evidence's speakers,
    ``origin`` their roster kind) and the span of its sources, which a read
    compares with the asker's windows like any group record's.

    ``item_id`` (the entry id, ``gp_<YYYYMMDD>_<seq>``), ``status`` and
    ``superseded_by`` are read-side: set when an item is read back, never
    written.
    """

    owner_id: str
    """The group."""
    kind: GroupProfileKind
    statement: str = ""
    topic_id: str | None = None
    name: str = ""
    summary: str = ""
    topic_status: TopicStatus | None = None
    decided_by: list[str] = Field(default_factory=list)
    agreed_by: list[str] = Field(default_factory=list)
    assignees: list[str] = Field(default_factory=list)
    applies_to: list[str] = Field(default_factory=list)
    """Members a decision applies to; empty means the whole group."""
    held_by: list[str] = Field(default_factory=list)
    about: str | None = None
    resolved: bool = False
    resolved_by: list[str] = Field(default_factory=list)
    member_id: str | None = None
    role: ObservedRole | None = None
    confidence: Literal["strong", "weak"] | None = None
    retired: bool = False
    supersedes: str | None = None
    """Entry id of the item this one replaces."""
    sources: list[ProfileSource] = Field(default_factory=list)
    subject_ids: list[str] = Field(default_factory=list)
    sender_ids: list[str] = Field(default_factory=list)
    origin: Origin | None = None
    source_start: int
    source_end: int
    observed_at: int
    """When the update that wrote the item ran: its batch's last segment."""

    # Read side.
    item_id: str | None = None
    status: GroupProfileStatus | None = None
    superseded_by: str | None = None

    @property
    def source_memcell_ids(self) -> list[str]:
        """The segments the item rests on, in evidence order, each once."""
        return list(dict.fromkeys(source.memcell_id for source in self.sources))

    def read_back(self, *, item_id: str, mark: str | None) -> GroupProfileItem:
        """The item as a read returns it: its entry id and the status its mark gives.

        ``mark`` is the entry's ``deprecated_by`` (in markdown, its line in
        ``deprecated_entries``): :data:`FORGOTTEN_MARK` reads ``forgotten``,
        another value ``superseded`` by the entry it names, none the item's
        own status. The markdown store and the index read it the same way.
        """
        if mark == FORGOTTEN_MARK:
            status, superseded_by = "forgotten", None
        elif mark:
            status, superseded_by = "superseded", mark
        else:
            status, superseded_by = self.own_status(), None
        return self.model_copy(
            update={
                "item_id": item_id,
                "status": status,
                "superseded_by": superseded_by,
            }
        )

    def own_status(self) -> GroupProfileStatus:
        """The status the item states itself, before any later item replaced it."""
        if self.kind == "topic":
            return self.topic_status or "exploring"
        if self.kind == "decision":
            return "current"
        if self.kind == "dissent":
            return "resolved" if self.resolved else "open"
        return "retired" if self.retired else "active"


_DROPPED_PREVIEW_CHARS = 80


def _item_data(item: object) -> object:
    """A bare sentence is the item with only its description."""
    return {"description": item} if isinstance(item, str) else item


def _readable_items[ItemT: BaseModel](
    model: type[ItemT], items: Sequence[object], owner_id: str, md_path: str | None
) -> list[ItemT]:
    kept: list[ItemT] = []
    for index, item in enumerate(items):
        try:
            kept.append(model.model_validate(_item_data(item)))
        except ValidationError as exc:
            logger.warning(
                "profile_entry_dropped",
                owner_id=owner_id,
                md_path=md_path,
                bucket=model.__name__,
                index=index,
                reason=_validation_reason(exc),
                entry=_preview(item),
            )
    return kept


def _validation_reason(exc: ValidationError) -> str:
    """What a dropped item failed, one ``field: message`` per error."""
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'item'}: {error['msg']}"
        for error in exc.errors()
    )[:200]


def _preview(item: object) -> str:
    text = (
        item
        if isinstance(item, str)
        else json.dumps(item, ensure_ascii=False, default=str)
    )
    return text[:_DROPPED_PREVIEW_CHARS]


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


ClusterKind = Literal["group_scene", "user_scene"]
ClusterMemberType = Literal["episode"]


def mint_cluster_id() -> str:
    """Mint a fresh cluster id (``cl_<12hex>``)."""
    return f"cl_{uuid.uuid4().hex[:12]}"


class MemcellRecord(BaseModel):
    """One boundary-detected slice of a conversation, ready to persist.

    Attributes:
        memcell_id: everos-minted identifier of the slice.
        cell: The algorithm-side cell (items + timestamp).
        raw_type: Source shape of the slice (``Conversation``).
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
        deliverer_id: The identity the gateway asserts for the group's key:
            the one program that forwards the group's chat (section 5.1).
            While it is set, only it adds to the group's sessions or
            flushes them, and it holds every right of an owner, reading the
            whole history (section 9.1). It is no member: it has no
            interval, is never listed and never reaches a prompt. ``None``
            (the default, and always for an implicit group): anyone may
            deliver, as before.
    """

    group_id: str
    name: str = ""
    description: str = ""
    created_at: int
    created_by: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    history_visible_from: HistoryVisibleFrom = "join"
    deliverer_id: str | None = None


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

    ``deliverer_id`` is the group's deliverer (:attr:`Group.deliverer_id`),
    which every :class:`~everos.memory.ports.GroupRecord` copies here so that
    the access checks, which all read the roster, see it. It is no member:
    no interval, never in :meth:`current_members`.
    """

    intervals: list[MembershipInterval] = Field(default_factory=list)
    deliverer_id: str | None = None

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
    "EDIT_FIELDS",
    "EDIT_KEYS",
    "FORGOTTEN_MARK",
    "MANAGER_ROLES",
    "MEMBER_ITEM_KEYS",
    "MEMBER_PREVIOUS_KEY",
    "MEMBER_PREVIOUS_LIMIT",
    "MEMBER_VERSION_KEYS",
    "AccessRole",
    "AlgoEpisode",
    "AlgoMessage",
    "CanonicalMessage",
    "Cluster",
    "ClusterKind",
    "ClusterMemberType",
    "Episode",
    "EpisodeParentType",
    "ExplicitInfoItem",
    "Group",
    "GroupProfileItem",
    "GroupProfileKind",
    "GroupProfileStatus",
    "HistoryVisibleFrom",
    "ImageMemory",
    "ImplicitTraitItem",
    "IngestResult",
    "MemCell",
    "MemberKind",
    "MembershipInterval",
    "MemcellRecord",
    "MemorizeTask",
    "ObservedRole",
    "Origin",
    "Profile",
    "ProfileSource",
    "Roster",
    "RosterEvent",
    "RosterOp",
    "RosterSource",
    "Scene",
    "ToolCall",
    "TopicStatus",
    "fold_roster",
    "is_held",
    "mint_cluster_id",
    "replaced_chains",
    "with_previous_values",
]
