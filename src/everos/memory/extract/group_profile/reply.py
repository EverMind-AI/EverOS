"""The reply the group-profile update call returns: a list of operations.

docs/group-memory.md, section 7.5. Every operation is flat -- one shape with
every field, empty when the operation does not use it -- so one strict JSON
schema covers all eight. The model names segments, messages and quotes; it
never names a speaker or a time, which code takes from the message
(:mod:`.verify`).

Parsing is lenient about form, never about meaning (section 14, Step 2
notes): models in JSON mode leave out the keys of fields an operation does
not use, write ``null`` for them, or write one id where a list is expected.
So every field but ``op`` reads as empty when it is missing or ``null``, and
a list field given one string reads as a list of that string (``""`` as no
one). What the values mean is :mod:`.verify`'s to judge: an operation whose
evidence is missing is dropped as ``no_evidence``, never repaired. The JSON
schema the call sends is unchanged -- every field is still required and
carries no default -- so a provider that enforces it sees the same request.
"""

from __future__ import annotations

__all__ = ["OPERATIONS", "Evidence", "GroupProfileReply", "Operation"]

from typing import Literal

from everalgo.llm import StrictOutputModel
from pydantic import Field, field_validator

from everos.memory.prompts import LENIENT_REPLY

OperationName = Literal[
    "topic.new",
    "topic.update",
    "decision.new",
    "decision.supersede",
    "dissent.open",
    "dissent.resolve",
    "role.add",
    "role.retire",
]
OPERATIONS: tuple[str, ...] = (
    "topic.new",
    "topic.update",
    "decision.new",
    "decision.supersede",
    "dissent.open",
    "dissent.resolve",
    "role.add",
    "role.retire",
)

_TEXTS = (
    "target",
    "topic_id",
    "name",
    "summary",
    "status",
    "statement",
    "about",
    "member_id",
    "role",
    "confidence",
)
_PEOPLE = ("decided_by", "agreed_by", "assignees", "applies_to", "held_by")


def _text(value: object) -> object:
    return "" if value is None else value


def _people(value: object) -> object:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return value


class Evidence(StrictOutputModel):
    """One cited message: the segment, the message id, words copied from it."""

    model_config = LENIENT_REPLY

    segment: str = ""
    message: str = ""
    quote: str = ""

    @field_validator("segment", "message", "quote", mode="before")
    @classmethod
    def _null_is_empty(cls, value: object) -> object:
        return _text(value)


class Operation(StrictOutputModel):
    """One change to the profile; fields an operation does not use are empty."""

    model_config = LENIENT_REPLY

    op: OperationName
    target: str = ""
    """``topic.update``: the topic id. ``decision.supersede`` /
    ``dissent.resolve`` / ``role.retire``: the id of the item replaced."""
    topic_id: str = ""
    name: str = ""
    summary: str = ""
    status: str = ""
    statement: str = ""
    decided_by: list[str] = Field(default_factory=list)
    agreed_by: list[str] = Field(default_factory=list)
    assignees: list[str] = Field(default_factory=list)
    applies_to: list[str] = Field(default_factory=list)
    held_by: list[str] = Field(default_factory=list)
    about: str = ""
    member_id: str = ""
    role: str = ""
    confidence: str = ""
    evidence: list[Evidence] = Field(default_factory=list)

    @field_validator(*_TEXTS, mode="before")
    @classmethod
    def _null_is_empty(cls, value: object) -> object:
        return _text(value)

    @field_validator(*_PEOPLE, mode="before")
    @classmethod
    def _one_id_is_a_list(cls, value: object) -> object:
        return _people(value)

    @field_validator("evidence", mode="before")
    @classmethod
    def _no_evidence_is_empty(cls, value: object) -> object:
        return [] if value is None else value


class GroupProfileReply(StrictOutputModel):
    """The whole reply: the operations, possibly none."""

    operations: list[Operation]
