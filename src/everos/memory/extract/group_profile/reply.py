"""The reply the group-profile update call returns: a list of operations.

docs/group-memory.md, section 7.5. Every operation is flat -- each field is
present, empty when the operation does not use it -- so one strict JSON schema
covers all eight. The model names segments, messages and quotes; it never
names a speaker or a time, which code takes from the message
(:mod:`.verify`).
"""

from __future__ import annotations

__all__ = ["OPERATIONS", "Evidence", "GroupProfileReply", "Operation"]

from typing import Literal

from everalgo.llm import StrictOutputModel

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


class Evidence(StrictOutputModel):
    """One cited message: the segment, the message id, words copied from it."""

    segment: str
    message: str
    quote: str


class Operation(StrictOutputModel):
    """One change to the profile; fields an operation does not use are empty."""

    op: OperationName
    target: str
    """``topic.update``: the topic id. ``decision.supersede`` /
    ``dissent.resolve`` / ``role.retire``: the id of the item replaced."""
    topic_id: str
    name: str
    summary: str
    status: str
    statement: str
    decided_by: list[str]
    agreed_by: list[str]
    assignees: list[str]
    applies_to: list[str]
    held_by: list[str]
    about: str
    member_id: str
    role: str
    confidence: str
    evidence: list[Evidence]


class GroupProfileReply(StrictOutputModel):
    """The whole reply: the operations, possibly none."""

    operations: list[Operation]
