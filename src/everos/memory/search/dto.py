"""Public DTOs for ``POST /api/v2/memory/search``.

Contract per the final design:

* ``owner_type`` is a hard partition. ``user`` returns ``episodes``
  (and optionally ``profiles``); ``agent`` returns ``agent_cases`` +
  ``agent_skills``. The four ``data.*`` arrays always exist; routes not
  applicable to the current ``owner_type`` stay as ``[]``.
* ``atomic_facts`` are **nested** inside :class:`SearchEpisodeItem`,
  never returned as a top-level array.
* Item-side ``owner_type`` / ``type`` fields are intentionally narrowed
  to the currently-emitted Literal so callers get a tight schema. Loosen
  them only when a new emission path (agent episodes, agent profiles)
  ships.

The :class:`FilterNode` model is intentionally permissive
(``extra="allow"``) because the DSL has an open key shape; the
allow-list / safety validation runs in :mod:`everos.memory.search.filters`
at compile time, not via Pydantic.
"""

from __future__ import annotations

import datetime as _dt
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SearchMethod(StrEnum):
    """Public method enum. RRF / LR / vector_anchored are hidden under HYBRID."""

    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"
    AGENTIC = "agentic"
    LLM_MULTIROUND = "llm_multiround"
    """LLM-guided iterative multi-round episode retrieval (per-sub-query RRF blocks).

    Each round fuses every current sub-query's BM25+vector recall with RRF
    INDEPENDENTLY (round 0 = one original-question block), shows an injected
    decider the labelled blocks plus the core accumulated so far, and lets it
    pin core and issue gap-covering sub-queries. The final injection is
    core-first + each sub-query's top-1 guarantee + max-RRF-score fill; there is
    no cross-encoder anywhere. Unlike AGENTIC's fixed round1+round2 this is
    genuinely iterative, and the decider is pluggable -- a prompted LLM by
    default, a trained policy in Phase 2, which makes this loop the RL
    environment. Needs an LLM + embedding provider. User memory only.
    """


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

    Callers identify the memory owner via ``user_id`` XOR ``agent_id`` —
    exactly one must be set. Internally the manager + compile_filters keep
    using ``owner_id`` / ``owner_type`` (the storage tables' columns);
    those are exposed as derived properties so the rename only affects
    the wire contract, not the internal recall plumbing.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str | None = Field(default=None, min_length=1)
    agent_id: str | None = Field(default=None, min_length=1)
    """Memory owner — provide ``user_id`` for user-memory (episodes /
    profiles) or ``agent_id`` for agent-memory (cases / skills); exactly
    one owner scope must be set (``user_id``, ``agent_id``, ``group_id`` or
    ``group_ids``)."""
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
    context_group_id: str | None = Field(default=None, min_length=1)
    """With ``agent_id`` only: read the agent's memory in this group's
    context. Cases the agent learned in a group are returned only in that
    group's context (docs/group-memory.md, section 9.1), to a current member
    of it -- the agent itself unless ``asker_id`` names another."""
    app_id: str = "default"
    project_id: str = "default"
    """App / project scope (default ``"default"``). Pinned into the LanceDB
    ``where`` so a search never crosses into another space's rows."""
    query: str = Field(min_length=1)
    method: SearchMethod = SearchMethod.HYBRID
    top_k: int = -1
    radius: float | None = Field(default=None, ge=0.0, le=1.0)
    """Cosine floor on dense recall, applied before any fusion or rerank.

    Every method's vector route honours it (VECTOR, HYBRID, AGENTIC);
    KEYWORD has no cosine and ignores it. ``None`` with ``top_k=-1`` falls
    back to the server default, see ``_effective_radius``.
    """
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    """Post-fusion relevance floor for the episode HYBRID path.

    Applied after heap-expand against the LR-calibrated score in ``[0, 1]``:
    items scoring below this value are dropped. Independent of ``radius``
    (which gates raw cosine at recall time); ``None`` disables the floor.
    Only the episode hybrid path consumes it — other methods ignore it.
    """
    include_profile: bool = False
    enable_llm_rerank: bool = Field(
        default=False,
        description=(
            "Opt-in LLM rerank pass for HYBRID. Applies to agent_case "
            "and agent_skill fusion only; the episode hybrid path "
            "has built-in fact eviction and ignores this flag. "
            "Ignored by keyword / vector / agentic."
        ),
    )
    images: list[QuestionPicture] = Field(
        default_factory=list,
        max_length=4,
        description=(
            "Pictures that are part of the question. They are shown to a "
            "multi-round decider on every round; they create no memory and no "
            "reference. Routes that cannot use them ignore them."
        ),
    )
    filters: FilterNode | None = None

    @model_validator(mode="after")
    def _validate_user_xor_agent(self) -> SearchRequest:
        scopes = [
            self.user_id is not None,
            self.agent_id is not None,
            self.group_id is not None,
            self.group_ids is not None,
        ]
        if sum(scopes) != 1:
            raise ValueError(
                "exactly one of user_id / agent_id / group_id / group_ids "
                "must be provided"
            )
        return self

    @model_validator(mode="after")
    def _validate_context_group(self) -> SearchRequest:
        if self.context_group_id is not None and self.agent_id is None:
            raise ValueError("context_group_id is valid with agent_id only")
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

        The xor validator guarantees exactly one is set, so the ``or`` falls
        through to a real string (never the ``""`` default).
        """
        groups = self.scope_group_ids
        return self.user_id or self.agent_id or (groups[0] if groups else "")

    @property
    def owner_type(self) -> Literal["user", "agent", "group"]:
        """``"user"``, ``"agent"`` or ``"group"``, by the owner scope set."""
        if self.user_id is not None:
            return "user"
        if self.agent_id is not None:
            return "agent"
        return "group"


# ── Item DTOs ────────────────────────────────────────────────────────────


class SearchAtomicFactItem(BaseModel):
    """A single atomic fact nested inside its parent episode."""

    model_config = ConfigDict(extra="forbid")

    id: str
    content: str
    score: float


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
    group_id: str | None = None
    """Owning group, for a group episode; ``None`` on one-to-one memory."""
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
    atomic_facts: list[SearchAtomicFactItem] = Field(default_factory=list)


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
    """Owner profile — at most one per response, only for user owners.

    ``score`` is ``None`` for direct fetches (``include_profile=true``
    on its own does no ranking); a future query-aware lookup may fill
    it in.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: str | None
    app_id: str = "default"
    project_id: str = "default"
    profile_data: dict[str, object]
    score: float | None = None


class SearchAgentCaseItem(BaseModel):
    """Agent case hit — always agent-scoped."""

    model_config = ConfigDict(extra="forbid")

    id: str
    entry_id: str
    """The case's id inside its owner's markdown (``ac_<YYYYMMDD>_<NNNN>``);
    what a skill's ``source_case_ids`` refers to. ``id`` is the index row key,
    ``<agent_id>_<entry_id>``."""
    agent_id: str
    app_id: str = "default"
    project_id: str = "default"
    session_id: str
    task_intent: str
    approach: str
    quality_score: float
    key_insight: str | None = None
    timestamp: _dt.datetime
    score: float


class SearchAgentSkillItem(BaseModel):
    """Agent skill hit — always agent-scoped."""

    model_config = ConfigDict(extra="forbid")

    id: str
    agent_id: str
    app_id: str = "default"
    project_id: str = "default"
    name: str
    description: str
    content: str
    confidence: float
    maturity_score: float
    source_case_ids: list[str] = Field(default_factory=list)
    score: float


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
    """Picture memories that survived the same ranking as the episodes.

    Populated for both owner types — a picture can arrive in a chat or in an
    agent's trace — because the picture is evidence whichever track it came
    from."""
    profiles: list[SearchProfileItem] = Field(default_factory=list)
    agent_cases: list[SearchAgentCaseItem] = Field(default_factory=list)
    agent_skills: list[SearchAgentSkillItem] = Field(default_factory=list)
    unprocessed_messages: list[UnprocessedMessageDTO] = Field(default_factory=list)
    """In-flight messages still in the boundary-detection buffer for
    the ``filters.session_id`` (if supplied as a top-level eq scalar);
    otherwise stays empty."""
    scope: SearchScope | None = None
    """The group scope and the asker's windows; ``None`` for a user or agent
    search, whose scope is the owner the request named."""


class SearchResponse(BaseModel):
    """Top-level response envelope."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    data: SearchData
