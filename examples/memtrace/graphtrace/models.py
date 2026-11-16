"""Typed case, exploration state, and attribution contracts for the example."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from everos.core.utils.datetime import get_utc_now


def _timestamp() -> str:
    return get_utc_now().isoformat()


class FailedQueryCase(BaseModel):
    """One failed QA chain, with decoded user-facing text."""

    query_full_node_id: str
    query: str
    golden_answers: list[str]
    prediction: str
    judge_response: str
    judge_op_id: str


class GraphTraceNodeRecord(BaseModel):
    """A versioned variable and the provenance of its exploration."""

    full_node_id: str
    created_at: str
    category: str = "variable"
    comment: str | None = None
    state: Literal["to_explore", "exploring", "explored"] = "to_explore"
    added_at: str = Field(default_factory=_timestamp)
    discovered_from: str | None = None
    discovered_by_op_id: str | None = None
    explored_at: str | None = None
    outcome: str | None = None

    @field_validator("full_node_id")
    @classmethod
    def validate_version(cls, value: str) -> str:
        name, separator, version = value.rpartition("@")
        if not name or not separator or not version.isdigit():
            raise ValueError("A full node id must have the form name@integer.")
        return value

    def priority(self) -> tuple[str, str]:
        return self.created_at, self.full_node_id


class GraphTraceState(BaseModel):
    """Keep the active node separate so adding older nodes cannot displace it."""

    to_explore_nodes: dict[str, GraphTraceNodeRecord] = Field(default_factory=dict)
    exploring_node: GraphTraceNodeRecord | None = None
    exploration_history: dict[str, GraphTraceNodeRecord] = Field(default_factory=dict)
    finished_at: str | None = None
    outcome: str | None = None

    def add(self, record: GraphTraceNodeRecord, capacity: int) -> None:
        node_id = record.full_node_id
        if (
            node_id in self.to_explore_nodes
            or node_id in self.exploration_history
            or (self.exploring_node and self.exploring_node.full_node_id == node_id)
        ):
            raise ValueError(
                f"Variable {node_id} is already queued, active, or explored."
            )
        if len(self.to_explore_nodes) >= capacity:
            raise ValueError("The frontier is full; delete a queued node first.")
        self.to_explore_nodes[node_id] = record

    def delete(self, node_id: str) -> None:
        if node_id not in self.to_explore_nodes:
            raise ValueError("Only a queued node can be deleted.")
        del self.to_explore_nodes[node_id]

    def start_next(self) -> GraphTraceNodeRecord:
        if self.exploring_node is None:
            if not self.to_explore_nodes:
                raise ValueError(
                    "The frontier is empty. Finish or restart exploration."
                )
            record = min(
                self.to_explore_nodes.values(), key=lambda item: item.priority()
            )
            del self.to_explore_nodes[record.full_node_id]
            record.state = "exploring"
            self.exploring_node = record
        return self.exploring_node

    def finish_node(self, outcome: str) -> None:
        if self.exploring_node is None:
            raise ValueError("There is no active node.")
        record = self.exploring_node
        record.state = "explored"
        record.outcome = outcome
        record.explored_at = _timestamp()
        self.exploration_history[record.full_node_id] = record
        self.exploring_node = None

    def finish(self, outcome: str) -> None:
        if self.exploring_node is not None:
            self.finish_node(outcome)
        self.outcome = outcome
        self.finished_at = _timestamp()


class AttributionResult(BaseModel):
    """The same single-operation output schema used by MemTrace."""

    model_config = ConfigDict(extra="forbid")

    error_type: Literal[
        "ExtractionError",
        "UpdateError",
        "DeletionError",
        "RetrievalError",
        "ResponseError",
    ]
    op_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("The reason must not be blank.")
        return value


class AttributionConfig(BaseModel):
    """Bounded example runtime; the character budget is not a tokenizer."""

    max_iters: int = Field(default=200, ge=1)
    max_trace_nodes: int = Field(default=16, ge=1)
    num_starting_points: int = Field(default=8, ge=1)
    context_chars: int = Field(default=100_000, ge=1)
