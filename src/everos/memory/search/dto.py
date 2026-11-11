"""Public DTOs for ``POST /api/v2/memory/search``.

Contract per the final design:

* The only search method is ``llm_multiround`` (multi-round retrieval):
  a request naming any other method is refused with a message that says
  so, and so is a request for an agent's memory, which multi-round
  retrieval does not cover.
* ``owner_type`` is a hard partition. ``user`` returns ``episodes`` and
  ``images`` (and optionally ``profiles``); a group scope (one group, a
  pool of groups, or a project's groups) returns the group's episodes and
  pictures (and the member-in-group ``profiles``), with its group profile,
  constraint block and ``coverage``. Every ``data.*`` array always exists;
  routes not applicable to the current scope stay ``[]``.
* Item-side ``owner_type`` / ``type`` fields are intentionally narrowed
  to the currently-emitted Literal so callers get a tight schema.

The :class:`FilterNode` model is intentionally permissive
(``extra="allow"``) because the DSL has an open key shape; the
allow-list / safety validation runs in :mod:`everos.memory.search.filters`
at compile time, not via Pydantic.
"""

from __future__ import annotations

import datetime as _dt
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SearchMethod(StrEnum):
    """The search method. Multi-round retrieval is the only one.

    The keyword, vector, hybrid and agentic methods were removed with the
    atomic facts the vector and hybrid routes ranked: a request naming one is
    refused (see :data:`MULTIROUND_ONLY`).
    """

    LLM_MULTIROUND = "llm_multiround"
    """LLM-guided iterative multi-round retrieval (per-sub-query RRF blocks).

    Each round fuses every current sub-query's BM25+vector recall with RRF
    INDEPENDENTLY (round 0 = one original-question block), shows an injected
    decider the labelled blocks plus the core accumulated so far, and lets it
    pin core and issue gap-covering sub-queries. Episodes, scenes and picture
    memories compete in one candidate pool. The final injection is
    core-first + each sub-query's top-1 guarantee + max-RRF-score fill; there
    is no cross-encoder anywhere. The decider is pluggable -- a prompted LLM
    by default, a trained policy in Phase 2, which makes this loop the RL
    environment. Needs an LLM; without an embedding provider every recall is
    BM25 only (keyword recall). User and group memory.
    """


MULTIROUND_ONLY = (
    "EverOS 2.0 searches memory with multi-round retrieval only: "
    "method must be 'llm_multiround'"
)
"""What a request naming a removed method is told."""

NO_AGENT_SEARCH = (
    "agent memory is not searchable: multi-round retrieval, the only search "
    "method, covers user and group memory; list an agent's cases and skills "
    "with POST /api/v2/memory/get"
)
"""What a request for an agent's memory is told."""


class FilterNode(BaseModel):
    """One Filters DSL node.

    Recursive ``AND`` / ``OR`` arrays mix with arbitrary scalar fields at
    the same level. Pydantic only checks the combinators; field-level
    safety is enforced when compiling the node to a LanceDB ``where``
    string in :mod:`everos.memory.search.filters`.
    """

    model_config = ConfigDict(extra="allow")

    AND: list[FilterNode] | None = None
    OR: list[FilterNode] | None = None


# ── Request ──────────────────────────────────────────────────────────────


class QuestionPicture(BaseModel):
    """A picture attached to the question itself, not to the memory.

    Same content-item shape as ``/memory/add`` (``uri`` or ``base64``, plus
    ``ext`` and an optional ``name``), because a caller that can send a picture
    to one endpoint should not have to learn a second shape for the other.

    A question picture is **content-addressed in ``figures/`` and establishes
    nothing**: no picture memory, no reference count, no profile item. It exists
    for one search — the decider looks at it every round — and "what I asked
    with" is not a fact about the user. Storing it by hash is what lets the
    decider read it from the same place it reads a candidate, and lets the same
    question asked twice pay for the file once.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["image"] = "image"
    """Carried so a caller can reuse an ``/memory/add`` content item unchanged.

    Redundant here — an item under ``images`` is a picture by definition — but
    making a caller strip one field from a payload it already built is how
    "the same shape" stops being true.
    """

    base64: str | None = None
    uri: str | None = None
    ext: str | None = None
    name: str | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> QuestionPicture:
        if bool(self.base64) == bool(self.uri):
            raise ValueError("a question picture needs exactly one of base64 / uri")
        return self


class SearchRequest(BaseModel):
    """Request body for ``POST /api/v2/memory/search``.

    Callers name the memory owner with ``user_id`` or a group scope
    (``group_id`` / ``group_ids`` / ``project_scope``) -- exactly one.
    Internally the manager + compile_filters keep using ``owner_id`` /
    ``owner_type`` (the storage tables' columns); those are exposed as
    derived properties so the rename only affects the wire contract, not the
    internal recall plumbing.

    An ``agent_id`` is refused with :data:`NO_AGENT_SEARCH`: agent cases and
    skills are extracted and listed by ``/memory/get``, but multi-round
    retrieval does not rank them.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str | None = Field(default=None, min_length=1)
    """Memory owner for user memory (episodes, pictures, profiles); exactly one
    owner scope must be set (``user_id``, ``group_id``, ``group_ids`` or
    ``project_scope``)."""
    group_id: str | None = Field(default=None, min_length=1)
    group_ids: list[str] | None = Field(default=None, min_length=1, max_length=16)
    """A group scope (docs/group-memory.md, section 5.3): one group, or several
    searched as one ranked pool. Needs an asker who is a current member of
    each; the asker sees only what lies inside their membership windows."""
    asker_id: str | None = Field(default=None, min_length=1)
    """Who asks. Required for a group scope unless the deployment's identity
    header carries it. With ``user_id`` it is optional and must equal it."""
    audience: Literal["asker", "group"] = "asker"
    """``asker`` (default): the asker's own view. ``group``: only what every
    current member can see -- for a reply posted to the whole group."""
    app_id: str = "default"
    project_id: str = "default"
    """App / project scope (default ``"default"``). Pinned into the LanceDB
    ``where`` so a search never crosses into another space's rows."""
    query: str = Field(min_length=1)
    method: SearchMethod = SearchMethod.LLM_MULTIROUND
    """Kept so a caller that names the method keeps working; any other value
    than ``llm_multiround`` is refused with :data:`MULTIROUND_ONLY`."""
    top_k: int = -1
    include_profile: bool = False
    include_group_profile: bool | None = None
    """Attach the visible current group-profile items (section 5.3). Unset:
    on for a group scope; a user scope has none to attach."""
    constraints: bool | None = None
    """Return the constraint block (section 8.4). Unset: on for a group scope;
    a user scope has none."""
    project_scope: bool = False
    """Search every group of ``project_id`` the asker currently belongs to, as
    one pool (section 8.7); needs ``asker_id`` and no other owner field."""
    images: list[QuestionPicture] = Field(
        default_factory=list,
        max_length=4,
        description=(
            "Pictures that are part of the question. They are shown to a "
            "multi-round decider on every round; they create no memory and no "
            "reference."
        ),
    )
    filters: FilterNode | None = None

    @model_validator(mode="before")
    @classmethod
    def _refuse_agent_scope(cls, data: Any) -> Any:
        if isinstance(data, dict) and (
            data.get("agent_id") is not None or data.get("context_group_id") is not None
        ):
            raise ValueError(NO_AGENT_SEARCH)
        return data

    @field_validator("method", mode="before")
    @classmethod
    def _multiround_only(cls, value: Any) -> Any:
        if value != SearchMethod.LLM_MULTIROUND.value:
            raise ValueError(f"{MULTIROUND_ONLY} (got {str(value)!r})")
        return value

    @model_validator(mode="after")
    def _validate_one_scope(self) -> SearchRequest:
        scopes = [
            self.user_id is not None,
            self.group_id is not None,
            self.group_ids is not None,
            self.project_scope,
        ]
        if sum(scopes) != 1:
            raise ValueError(
                "exactly one of user_id / group_id / group_ids / project_scope "
                "must be provided"
            )
        return self

    @model_validator(mode="after")
    def _validate_top_k(self) -> SearchRequest:
        if self.top_k == 0 or self.top_k < -1 or self.top_k > 100:
            raise ValueError("top_k must be -1 or in 1..100")
        return self

    @property
    def scope_group_ids(self) -> tuple[str, ...]:
        """The groups of a group scope, in request order; ``()`` otherwise."""
        if self.group_id is not None:
            return (self.group_id,)
        return tuple(dict.fromkeys(self.group_ids or ()))

    @property
    def owner_id(self) -> str:
        """Derived from whichever owner scope is set (the first group of a pool).

        The one-scope validator guarantees exactly one is set, so the ``or``
        falls through to a real string (never the ``""`` default).
        """
        groups = self.scope_group_ids
        return self.user_id or (groups[0] if groups else "")

    @property
    def owner_type(self) -> Literal["user", "group"]:
        """``"user"`` or ``"group"``, by the owner scope set.

        A project scope is a pool of groups: ``group``.
        """
        if self.user_id is not None:
            return "user"
        return "group"


# ── Item DTOs ────────────────────────────────────────────────────────────


class AudienceItem(BaseModel):
    """Who knew a group record (docs/group-memory.md, section 8.6).

    Computed at read time from the roster and the record's sources; shown only
    on records the asker can see.
    """

    model_config = ConfigDict(extra="forbid")

    present: list[str] = Field(default_factory=list)
    """Members whose windows cover all of the record's sources."""
    joined_after: list[str] = Field(default_factory=list)
    """Current members who joined after its last source."""


class SearchSource(BaseModel):
    """Where a group record comes from: its segment and the span it covers.

    ``from`` and ``to`` are epoch ms of the segment's first and last message
    (a scene's: of its first member's first and its newest member's last);
    ``memcell_id`` is the segment, ``None`` for a scene, which has several.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    memcell_id: str | None = None
    from_: int = Field(alias="from")
    to: int


class SearchEpisodeItem(BaseModel):
    """Episode hit — always user-scoped in the current emission contract.

    ``type`` is narrowed to ``"Conversation"`` because the only emitted
    episode shape today is conversation-derived; widen when other
    sources ship. Item kind is encoded by class name (no ``owner_type``
    field on the wire), so episode results never carry ambiguity.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: str | None
    """Owning user; ``None`` for a group episode (and on malformed rows)."""
    owner_type: Literal["group"] | None = None
    """``group`` on a group episode; unset on one-to-one memory."""
    group_id: str | None = None
    """Owning group, for a group episode; ``None`` on one-to-one memory."""
    source: SearchSource | None = None
    """The group episode's segment and span (docs/group-memory.md, 5.3); unset
    on one-to-one memory."""
    app_id: str = "default"
    project_id: str = "default"
    session_id: str | None = None
    """``None`` for merged episodes (Reflection aggregation products)."""
    timestamp: _dt.datetime
    sender_ids: list[str] = Field(default_factory=list)
    subject_ids: list[str] = Field(default_factory=list)
    """Members a group episode is about; empty on one-to-one memory."""
    origin: Literal["member", "assistant", "mixed"] | None = None
    """Whose messages a group episode rests on; unset on one-to-one memory."""
    image_ids: list[str] = Field(default_factory=list)
    """Pictures the body refers to, by content hash, in arrival order.

    The body names them (``(img-1: <hash>)``) and never restates what they
    show, so this is the key to their content. A referenced picture that is
    also selected as core memory appears at the top level, not here; one that
    was forgotten resolves to nothing, which is the correct answer for a
    forget.
    """
    images: list[SearchImageItem] = Field(default_factory=list)
    """The referenced pictures' contents, for the pictures not already selected.

    Attached so a reader of the episode — the answering model above all — gets
    what the body points at without a second lookup. A forgotten picture is
    omitted rather than revived: the reference stays, its content does not.
    """
    summary: str
    subject: str
    episode: str
    type: Literal["Conversation", "Scene"]
    """``Scene`` for a group scene version: a narrative over several of the
    group's episodes, competing with them in the same ranking."""
    score: float
    audience: AudienceItem | None = None
    """Who heard it, on a group episode (section 8.6); unset otherwise."""


class SearchImageItem(BaseModel):
    """Image-memory hit — one appearance of one picture.

    ``id`` and ``image_id`` are different identities on purpose: ``id`` names
    this *record* (a time, a provenance, a description), while ``image_id``
    names the *file*, which is stored once however many records refer to it and
    is what ``forget`` takes.

    ``original_path`` and ``thumbnail_path`` are memory-root-relative. The
    caller resolves them against its own root to hand the answering model the
    original — never the thumbnail, which exists so the decider can look at
    many pictures cheaply.

    ``name`` is the file name the client sent with this occurrence, when it
    sent one. It is not an identity (the same picture may arrive under two
    names) but it is the only handle a caller has for saying which picture of
    its own input this memory is about.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    entry_id: str
    image_id: str
    user_id: str | None
    group_id: str | None = None
    """Owning group, for a group picture memory."""
    app_id: str = "default"
    project_id: str = "default"
    session_id: str | None = None
    timestamp: _dt.datetime
    sender_ids: list[str] = Field(default_factory=list)
    """Who posted the picture, on a group picture memory."""
    subject_ids: list[str] = Field(default_factory=list)
    origin: Literal["member", "assistant", "mixed"] | None = None
    source: SearchSource | None = None
    """A group picture memory's segment, and the time of the message that
    carried the picture (``from`` = ``to``); unset on one-to-one memory."""
    audience: AudienceItem | None = None
    """Who was there when it was posted (group scopes, section 8.6)."""
    description: str
    """What the picture shows — the half of its memory a reader gets as text."""

    ocr: str = ""
    """Text visible in the picture. Always present as a field, empty when the
    picture carries none, so a caller renders ``description + ocr`` without
    branching on whether the model happened to find text."""

    name: str = ""
    original_path: str
    thumbnail_path: str
    media_type: str = ""
    width: int = 0
    height: int = 0
    status: Literal["current", "superseded", "forgotten"] = "current"
    label: str = ""
    """``img-N`` the referring episode's body uses, when this item is attached
    to an episode rather than ranked. Empty for a ranked result: a picture
    selected on its own merit has no label in any body."""
    score: float


class SearchProfileItem(BaseModel):
    """A profile attached with ``include_profile``.

    On a user scope, the owner's profile (at most one). On a group scope, the
    member-in-group profiles of the asker and of the members the query names
    (``owner_type = group``, ``group_id``, ``member_id``), holding only the
    items the asker's windows allow -- never a personal profile.

    ``score`` is ``None`` for direct fetches (``include_profile=true``
    on its own does no ranking); a future query-aware lookup may fill
    it in.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: str | None
    owner_type: Literal["group"] | None = None
    group_id: str | None = None
    member_id: str | None = None
    """The member a group profile is about."""
    app_id: str = "default"
    project_id: str = "default"
    profile_data: dict[str, object]
    score: float | None = None


class UnprocessedMessageDTO(BaseModel):
    """A raw message still in the boundary-detection buffer.

    No extracted memcell yet, no owner inference yet (attribution
    happens at boundary detection). Returned by ``/search`` **only when**
    ``filters.session_id`` is present as a top-level eq predicate —
    unprocessed messages have no ``user_id`` / ``agent_id`` to filter
    on, so session is the only meaningful query dimension.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    """Original ``message_id`` from ``/add``."""
    app_id: str = "default"
    project_id: str = "default"
    session_id: str
    sender_id: str
    sender_name: str | None = None
    role: Literal["user", "assistant", "tool"]
    content: str | list[dict[str, object]]
    """``str`` for the single-text-item shorthand; ``list`` of opaque
    objects for the original multi-modal payload (mirrors
    ``MessageItem.content`` from the /add side)."""
    timestamp: _dt.datetime
    tool_calls: list[dict[str, object]] | None = None
    tool_call_id: str | None = None


class ProfileSourceItem(BaseModel):
    """One verified piece of evidence: a message of a segment, quoted verbatim."""

    model_config = ConfigDict(extra="forbid")

    memcell_id: str
    message_id: str
    speaker_id: str
    at: int
    """The message's time, epoch ms."""
    quote: str


class GroupProfileEntry(BaseModel):
    """One group-profile item (docs/group-memory.md, section 7.5).

    ``kind`` decides which fields are filled: a topic its ``topic_id`` /
    ``name`` / ``summary`` / ``topic_status``; a decision its ``statement``
    and the people fields; a dissent ``statement`` / ``held_by`` / ``about``;
    a role ``member_id`` / ``role`` / ``confidence``. ``status`` is the
    item's own (``current``, ``open``, ``active``, a topic's status ...) or
    ``superseded`` once a later item replaced it.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    """Index row id, ``<group_id>_<item_id>``."""
    item_id: str
    group_id: str
    kind: Literal["topic", "decision", "dissent", "role"]
    status: str
    statement: str = ""
    topic_id: str | None = None
    name: str = ""
    summary: str = ""
    topic_status: str | None = None
    decided_by: list[str] = Field(default_factory=list)
    agreed_by: list[str] = Field(default_factory=list)
    assignees: list[str] = Field(default_factory=list)
    applies_to: list[str] = Field(default_factory=list)
    held_by: list[str] = Field(default_factory=list)
    about: str | None = None
    resolved_by: list[str] = Field(default_factory=list)
    member_id: str | None = None
    role: str | None = None
    confidence: str | None = None
    supersedes: str | None = None
    superseded_by: str | None = None
    subject_ids: list[str] = Field(default_factory=list)
    sender_ids: list[str] = Field(default_factory=list)
    origin: Literal["member", "assistant", "mixed"] | None = None
    sources: list[ProfileSourceItem] = Field(default_factory=list)
    source: SearchSource
    """The span of the item's sources (``memcell_id`` is ``None``: an item may
    rest on several segments)."""
    observed_at: int
    audience: AudienceItem | None = None


class ConstraintSourceItem(BaseModel):
    """The message a constraint quotes: verbatim, from its segment."""

    model_config = ConfigDict(extra="forbid")

    memcell_id: str
    message_id: str
    speaker_id: str
    at: int
    quote: str


class ConstraintScopeItem(BaseModel):
    """Where a constraint applies: the group, or some of its members."""

    model_config = ConfigDict(extra="forbid")

    group_id: str
    applies_to: Literal["group"] | list[str] = "group"


class SupersededStatement(BaseModel):
    """An earlier version a current constraint replaced."""

    model_config = ConfigDict(extra="forbid")

    statement: str
    at: int
    """When the earlier version's last source was said, epoch ms."""


class ConstraintItem(BaseModel):
    """One thing an answer must respect (docs/group-memory.md, section 8.4).

    Built after the search loop from the core: a group-profile item linked to
    it (a decision, a dissent, a topic's status) whose quote and speakers were
    verified when it was written, or -- for a core record no item covers -- a
    metadata constraint quoting the record's best-matching source message.
    Every quote is a verbatim part of the message it cites.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    """``c1``, ``c2`` ... in the block's order."""
    statement: str
    kind: Literal["decision", "dissent", "topic", "episode", "picture"]
    record_id: str
    """The item id (``gp_...``) or the episode / picture memory id it rests on."""
    subject_ids: list[str] = Field(default_factory=list)
    scope: ConstraintScopeItem
    status: str
    """``current`` / ``disputed`` / ``superseded`` for a decision, ``open`` /
    ``resolved`` for a dissent, a topic's own status, ``current`` for a
    record."""
    supersedes: list[SupersededStatement] = Field(default_factory=list)
    decided_by: list[str] = Field(default_factory=list)
    agreed_by: list[str] = Field(default_factory=list)
    held_by: list[str] = Field(default_factory=list)
    origin: Literal["member", "assistant", "mixed"] | None = None
    source: ConstraintSourceItem
    applies: bool | None = None
    """Set by the model layer (``[search] constraints = "model"``): whether the
    constraint bears on the question. ``None`` under the rules layer."""
    audience: AudienceItem | None = None


class SearchGroupProfile(BaseModel):
    """The visible current group-profile items of a group scope, by kind."""

    model_config = ConfigDict(extra="forbid")

    topics: list[GroupProfileEntry] = Field(default_factory=list)
    decisions: list[GroupProfileEntry] = Field(default_factory=list)
    dissent: list[GroupProfileEntry] = Field(default_factory=list)
    roles: list[GroupProfileEntry] = Field(default_factory=list)


# ── Response envelope ────────────────────────────────────────────────────


class SearchWindow(BaseModel):
    """One of the asker's membership windows in one group, ``[from, to)`` ms."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    group_id: str
    from_: int | None = Field(default=None, alias="from")
    to: int | None = None


class SearchScope(BaseModel):
    """The scope a group search was answered in (docs/group-memory.md, 5.3)."""

    model_config = ConfigDict(extra="forbid")

    owner_type: Literal["group"] = "group"
    group_ids: list[str]
    asker_id: str
    audience: Literal["asker", "group"] = "asker"
    windows: list[SearchWindow] = Field(default_factory=list)


class SearchDegradation(BaseModel):
    """A part of this search that did not run as configured.

    Today one kind: a multi-round round the decider did not decide. Its
    candidates were still ranked, but which of them are core was not the
    decider's choice, and the loop stopped there.
    """

    model_config = ConfigDict(extra="forbid")

    component: Literal["decider"] = "decider"
    round: int
    """The multi-round round, 0-based."""
    outcome: Literal["fallback", "salvaged"]
    """``fallback``: every attempt failed; the round's core is the top
    ``[decider] fallback_core`` candidates by fused score. ``salvaged``: every
    attempt failed to parse; the core was read from a reply that was cut
    short (``[decider] salvage_truncated_core``)."""
    reason: str
    """The last attempt's error, e.g. ``NotFoundError: Error code: 404 ...``;
    anything shaped like a credential is masked."""
    attempts: int
    """Requests sent to the decider for the round."""


class SearchData(BaseModel):
    """Body of ``response.data``.

    All arrays are always present so client code can iterate without
    branching on ``owner_type``. Routes not applicable to the request's
    owner type stay as ``[]``. ``unprocessed_messages`` is filled only
    when ``filters.session_id`` is present as a top-level eq scalar —
    in-flight buffer rows are scope-tagged but unattributed (no
    ``user_id``), so session is the only meaningful query dimension.
    """

    model_config = ConfigDict(extra="forbid")

    episodes: list[SearchEpisodeItem] = Field(default_factory=list)
    images: list[SearchImageItem] = Field(default_factory=list)
    """Picture memories that survived the same ranking as the episodes: the
    two kinds compete in one candidate pool."""
    profiles: list[SearchProfileItem] = Field(default_factory=list)
    unprocessed_messages: list[UnprocessedMessageDTO] = Field(default_factory=list)
    """In-flight messages still in the boundary-detection buffer for
    the ``filters.session_id`` (if supplied as a top-level eq scalar);
    otherwise stays empty."""
    scope: SearchScope | None = None
    """The group scope and the asker's windows; ``None`` for a user search,
    whose scope is the owner the request named."""
    group_profile: SearchGroupProfile | None = None
    """The visible current group-profile items of a group scope
    (``include_group_profile``); ``None`` otherwise."""
    constraints: list[ConstraintItem] = Field(default_factory=list)
    """What an answer must respect (``constraints``, group scopes)."""
    coverage: Literal["complete", "partial", "none"] | None = None
    """How far a group scope's multi-round search got: ``complete`` (the
    decider stopped with a core), ``partial`` (the round budget ran out, or
    the loop ended on a round the decider did not decide), ``none`` (no
    core). ``None`` for a user scope, whose response stays as it was."""
    degraded: list[SearchDegradation] = Field(default_factory=list)
    """What did not run as configured -- a decider round that failed and
    fell back, with the reason. Empty when the search ran as configured."""


class SearchResponse(BaseModel):
    """Top-level response envelope."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    data: SearchData
