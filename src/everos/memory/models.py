"""Domain models shared across the memory layer.

These models live in the domain layer so service / pipeline / ingest can
all consume the same canonical shapes.

Algorithm-side models (``MemCell``, ``Episode``, ``ChatMessage``) are
owned by ``everalgo.types`` and are not re-defined here. Re-export here for
the ``from everos.memory import MemCell, Episode`` convenience.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from everalgo.types import AgentCase as AlgoAgentCase
from everalgo.types import AtomicFact as AlgoAtomicFact
from everalgo.types import ChatMessage as AlgoMessage
from everalgo.types import Episode as AlgoEpisode
from everalgo.types import Foresight as AlgoForesight
from everalgo.types import MemCell as MemCell
from pydantic import BaseModel, ConfigDict, Field

_Role = Literal["user", "assistant", "tool"]
_Track = Literal["user_memory", "agent_memory"]


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


class PipelineOutcome(BaseModel):
    """Return type of every pipeline's ``run()``."""

    track: _Track
    status: Literal["accumulated", "extracted", "skipped"]
    message_count: int
    extracted_md_paths: list[str] = Field(default_factory=list)


class Episode(BaseModel):
    """Domain Episode — algo-emitted business fields + everos context.

    Composed (not inherited) from :class:`everalgo.types.Episode`. everos
    keeps the *semantic* fields algo emits (``owner_id``, narrative
    ``episode`` text, ``subject``, ``timestamp``) and adds engineering
    context (``session_id``, ``sender_ids``, ``parent_id``). The global
    episode id is derived later by cascade from
    ``<scope_id>_<entry_id_in_md>`` — algo no longer mints an id of its
    own.

    ``parent_id`` is the source memcell id. The new everalgo types no
    longer carry ``parent_id`` on Episode / Foresight / AtomicFact, so
    everos fills it from the memcell currently being processed (the
    pipeline knows the id — it created the memcell).
    """

    owner_id: str
    episode: str
    timestamp: int

    # everos engineering metadata.
    session_id: str | None = None
    sender_ids: list[str] = Field(default_factory=list)
    parent_id: str

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
    (``owner_id`` / ``fact`` / ``timestamp``) and adds engineering context
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
    fact: str
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

        The algo type exposes the fact sentence as ``content``; everos's
        domain field is ``fact``. This boundary is where that rename is
        bridged (mirrors how :meth:`Episode.from_algo` adapts algo fields
        into everos's vocabulary).
        """
        data = algo_fact.model_dump(exclude={"parent_id", "owner_id"})
        data["fact"] = data.pop("content")
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

    Composed (not inherited) from :class:`everalgo.types.AgentCase`. Mirrors
    :class:`Episode` / :class:`AtomicFact` / :class:`Foresight`: everos
    keeps the semantic fields algo emits (``task_intent`` / ``approach`` /
    ``quality_score`` / ``key_insight`` / ``timestamp``) and adds
    engineering context (``owner_id`` = agent_id, ``session_id`` /
    ``parent_id``).

    ``owner_id`` is supplied by the caller because algo's AgentCase has no
    ``owner_id`` field — the strategy infers the agent identity from the
    source MemCell (assistant's ``sender_id``).

    Single output per memcell (algo returns ``[]`` or ``[case]``); no
    fan-out semantics.
    """

    owner_id: str
    task_intent: str
    approach: str
    quality_score: float
    key_insight: str | None = None
    timestamp: int

    # everos engineering metadata.
    session_id: str
    parent_id: str

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_algo(
        cls,
        algo_case: AlgoAgentCase,
        *,
        owner_id: str,
        session_id: str,
        parent_id: str,
    ) -> AgentCase:
        """Build a domain AgentCase from an algo AgentCase plus context.

        ``owner_id`` is caller-supplied (agent_id derived from the source
        memcell's assistant sender). Algo's ``id`` (a uuid4 hex) is
        dropped — md writer mints the authoritative entry_id. ``key_insight``
        normalises algo's ``""`` to ``None`` so the optional KeyInsight
        section is omitted in md when there's nothing to record.
        """
        data = algo_case.model_dump(exclude={"id", "parent_id", "owner_id"})
        data["owner_id"] = owner_id
        data["session_id"] = session_id
        data["parent_id"] = parent_id
        if not data.get("key_insight"):
            data["key_insight"] = None
        return cls.model_validate(data)


__all__ = [
    "AgentCase",
    "AlgoAgentCase",
    "AlgoAtomicFact",
    "AlgoEpisode",
    "AlgoForesight",
    "AlgoMessage",
    "AtomicFact",
    "CanonicalMessage",
    "Episode",
    "Foresight",
    "IngestResult",
    "MemCell",
    "PipelineOutcome",
    "ToolCall",
]
