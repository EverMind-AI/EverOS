"""Public DTOs for ``POST /api/v1/memory/search``.

Contract per the final design:

* ``owner_type`` is a hard partition. ``user`` returns ``episodes``
  (and optionally ``profiles``); ``agent`` returns ``agent_cases`` +
  ``agent_skills``. The five ``data.*`` arrays always exist; routes not
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


class SearchRequest(BaseModel):
    """Request body for ``POST /api/v1/memory/search``.

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
    one must be set."""
    app_id: str = "default"
    project_id: str = "default"
    """App / project scope (default ``"default"``). Pinned into the LanceDB
    ``where`` so a search never crosses into another space's rows."""
    query: str = Field(min_length=1)
    method: SearchMethod = SearchMethod.HYBRID
    top_k: int = -1
    radius: float | None = Field(default=None, ge=0.0, le=1.0)
    include_profile: bool = False
    enable_llm_rerank: bool = Field(
        default=False,
        description=(
            "Opt-in LLM rerank pass for HYBRID. Applies to agent_case "
            "and agent_skill fusion only; the episode hierarchy path "
            "has built-in fact eviction and ignores this flag. "
            "Ignored by keyword / vector / agentic."
        ),
    )
    filters: FilterNode | None = None

    @model_validator(mode="after")
    def _validate_user_xor_agent(self) -> SearchRequest:
        if (self.user_id is None) == (self.agent_id is None):
            raise ValueError("exactly one of user_id / agent_id must be provided")
        return self

    @model_validator(mode="after")
    def _validate_top_k(self) -> SearchRequest:
        if self.top_k == 0 or self.top_k < -1 or self.top_k > 100:
            raise ValueError("top_k must be -1 or in 1..100")
        return self

    @property
    def owner_id(self) -> str:
        """Derived from whichever of ``user_id`` / ``agent_id`` is set.

        The xor validator guarantees exactly one is non-None, so the
        ``or`` falls through to a real string (never the ``""`` default).
        """
        return self.user_id or self.agent_id or ""

    @property
    def owner_type(self) -> Literal["user", "agent"]:
        """``"user"`` if ``user_id`` is set, else ``"agent"``."""
        return "user" if self.user_id is not None else "agent"


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
    """Owning user (``None`` only on malformed cascade rows)."""
    app_id: str = "default"
    project_id: str = "default"
    session_id: str
    timestamp: _dt.datetime
    sender_ids: list[str] = Field(default_factory=list)
    summary: str
    subject: str
    episode: str
    type: Literal["Conversation"]
    score: float
    atomic_facts: list[SearchAtomicFactItem] = Field(default_factory=list)


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


class SearchData(BaseModel):
    """Body of ``response.data``.

    All five arrays are always present so client code can iterate without
    branching on ``owner_type``. Routes not applicable to the request's
    owner type stay as ``[]``. ``unprocessed_messages`` is filled only
    when ``filters.session_id`` is present as a top-level eq scalar —
    in-flight buffer rows are scope-tagged but unattributed (no
    ``user_id``), so session is the only meaningful query dimension.
    """

    model_config = ConfigDict(extra="forbid")

    episodes: list[SearchEpisodeItem] = Field(default_factory=list)
    profiles: list[SearchProfileItem] = Field(default_factory=list)
    agent_cases: list[SearchAgentCaseItem] = Field(default_factory=list)
    agent_skills: list[SearchAgentSkillItem] = Field(default_factory=list)
    unprocessed_messages: list[UnprocessedMessageDTO] = Field(default_factory=list)
    """In-flight messages still in the boundary-detection buffer for
    the ``filters.session_id`` (if supplied as a top-level eq scalar);
    otherwise stays empty."""


class SearchResponse(BaseModel):
    """Top-level response envelope."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    data: SearchData
