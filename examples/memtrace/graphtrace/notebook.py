"""Seven graph tools and state hints, independent of an agent framework."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import TYPE_CHECKING, Any, Literal

from .cases import decode_snapshot
from .models import GraphTraceNodeRecord, GraphTraceState
from .scope import same_tenant

if TYPE_CHECKING:
    from smartcomment.runtime import ExecNetwork
    from smartcomment.runtime.operation import RuntimeOp

DUMMY_ROOT = "__memtrace_root__@0"
PAGE_LIMIT = 128_000
DEFAULT_PAGE = 16_000


def paginate(text: str, offset: int = 1, limit: int = DEFAULT_PAGE) -> str:
    """Return one-based character pagination and the next readable offset."""
    if offset < 1 or limit < 1:
        raise ValueError("offset and limit must be positive integers.")
    limit = min(limit, PAGE_LIMIT)
    end = min(offset - 1 + limit, len(text))
    content = text[offset - 1 : end]
    status = f"Characters {offset}-{end} of {len(text)}."
    if offset > len(text):
        status = f"No content at offset {offset}; total characters: {len(text)}."
    elif end < len(text):
        status += f" Truncated; next offset={end + 1}."
    else:
        status += " End of content."
    return content + "\n\n" + status


class GraphTraceNotebook:
    """Explore an existing graph; tools never mutate or re-export that graph."""

    def __init__(
        self,
        graph: ExecNetwork,
        max_trace_nodes: int = 16,
        query_full_node_id: str | None = None,
    ) -> None:
        if max_trace_nodes < 1:
            raise ValueError("max_trace_nodes must be positive.")
        self.graph = graph
        self.max_trace_nodes = max_trace_nodes
        self.state: GraphTraceState | None = None
        self.final_state: GraphTraceState | None = None
        self.insufficient_evidence: str | None = None
        self.inspected_operations: set[str] = set()
        self.read_nodes: set[str] = set()
        self._page_coverage: dict[tuple[str, str, bool], list[tuple[int, int]]] = {}
        self.page_limit = DEFAULT_PAGE
        self.query_node = (
            graph.get_variable(query_full_node_id) if query_full_node_id else None
        )
        query_value = decode_snapshot(self.query_node.value) if self.query_node else {}
        self.request_id = (
            query_value.get("request_id") if isinstance(query_value, dict) else None
        )
        self.query_identity = (
            query_value.get("id") if isinstance(query_value, dict) else None
        )
        sequences = Counter(
            op.metadata["event_seq"]
            for op in graph.search_operations()
            if isinstance(op.metadata.get("event_seq"), int)
        )
        self.unique_event_sequences = all(count == 1 for count in sequences.values())

    def _state(self) -> GraphTraceState:
        if self.state is None:
            raise ValueError("Initialize the execution graph before using this tool.")
        return self.state

    def _record(self, node_id: str, op_id: str | None = None) -> GraphTraceNodeRecord:
        if node_id == DUMMY_ROOT:
            return GraphTraceNodeRecord(
                full_node_id=DUMMY_ROOT, created_at="", category="synthetic_root"
            )
        try:
            node = self.graph.get_variable(node_id)
        except KeyError as exc:
            raise ValueError(f"Unknown variable {node_id}.") from exc
        if self.query_node is not None and not same_tenant(node, self.query_node):
            raise ValueError("The variable belongs to another tenant.")
        active = self.state.exploring_node if self.state else None
        return GraphTraceNodeRecord(
            full_node_id=node.full_node_id,
            created_at=node.created_at,
            category=node.category,
            comment=node.comment,
            discovered_from=active.full_node_id if active else None,
            discovered_by_op_id=op_id,
        )

    def initialize_execution_graph(
        self, initial_full_node_ids: list[str] | None = None
    ) -> str:
        """Initialize with selected variables, or a synthetic root if omitted."""
        if self.state is not None:
            raise ValueError("The execution graph is already initialized.")
        ids = list(dict.fromkeys(initial_full_node_ids or [DUMMY_ROOT]))
        records = [self._record(node_id) for node_id in ids]
        state = GraphTraceState()
        for record in records:
            state.add(record, self.max_trace_nodes)
        self.state = state
        self.insufficient_evidence = None
        return f"Initialized graph with {len(records)} frontier nodes."

    def _operation_key(self, op: RuntimeOp) -> tuple[Any, ...]:
        seq = op.metadata.get("event_seq")
        if self.unique_event_sequences:
            # Offline QA operations lack event_seq and were appended after recording.
            return (
                not isinstance(seq, int),
                seq if isinstance(seq, int) else 0,
                op.created_at,
                op.op_id,
            )
        return (op.created_at, seq if isinstance(seq, int) else 0, op.op_id)

    def _render_operation(self, op: RuntimeOp, node_id: str, values: bool) -> str:
        subgraph = self.graph.filter_by_operation(op.op_id)
        sources = {edge.source_full_node_id for edge in subgraph.edges}
        targets = {edge.target_full_node_id for edge in subgraph.edges}
        inputs = [
            node.full_node_id
            for node in subgraph.nodes
            if node.full_node_id not in targets
        ]
        outputs = [
            node.full_node_id for node in subgraph.nodes if node.full_node_id in targets
        ]
        successors = sorted(
            {
                edge.target_full_node_id
                for edge in subgraph.edges
                if edge.source_full_node_id == node_id
            }
        )
        role = (
            "intermediate"
            if node_id in sources and node_id in targets
            else "input"
            if node_id in sources
            else "output"
            if node_id in targets
            else "isolated"
        )
        ordering = {
            key: op.metadata[key]
            for key in ("event_seq", "observed_at", "source", "trigger")
            if key in op.metadata
        }
        return "\n".join(
            [
                f"# Operation `{op.op_id}`",
                f"Name: {op.op_name}; Category: {op.category}; "
                f"Created: {op.created_at}",
                f"Comment: {op.comment or ''}",
                f"Observation ordering: {json.dumps(ordering)}",
                f"Focus variable: {node_id}; Role: {role}",
                f"Inputs: {sorted(inputs)}",
                f"Produced variables: {sorted(outputs)}",
                f"Direct successors: {successors}",
                "Final output variables: "
                + str(sorted(node.full_node_id for node in subgraph.get_leaf_nodes())),
                subgraph.to_xml(include_metadata=True, include_variable_value=values),
            ]
        )

    def explore_graph_trace_node(
        self,
        offset: int = 1,
        limit: int = DEFAULT_PAGE,
        include_variable_value: bool = True,
    ) -> str:
        """Inspect the active/earliest node's producing and consuming operations."""
        if offset < 1 or limit < 1:
            raise ValueError("offset and limit must be positive.")
        record = self._state().start_next()
        if record.full_node_id == DUMMY_ROOT:
            roots = sorted(
                (
                    node
                    for node in self.graph.get_root_nodes()
                    if self.query_node is None
                    or (
                        same_tenant(node, self.query_node)
                        and node.created_at <= self.query_node.created_at
                    )
                ),
                key=lambda node: (node.created_at, node.full_node_id),
            )
            text = "# Synthetic root: choose roots using update_to_explore_nodes\n"
            text += "\n".join(
                node.to_xml(
                    include_metadata=True, include_variable_value=include_variable_value
                )
                for node in roots
            )
        else:
            operations = sorted(
                self.graph.get_operations_by_variable(record.full_node_id),
                key=self._operation_key,
            )
            text = f"# Graph trace node {record.full_node_id}\n\n"
            spans = []
            for op in operations:
                if not self.operation_allowed(op):
                    continue
                start = len(text)
                text += self._render_operation(
                    op, record.full_node_id, include_variable_value
                )
                spans.append((op, start, len(text)))
                text += "\n\n---\n\n"
            end = min(offset - 1 + min(limit, self.page_limit), len(text))
            for op, start, stop in spans:
                left, right = max(start, offset - 1), min(stop, end)
                if left >= right:
                    continue
                key = (op.op_id, record.full_node_id, include_variable_value)
                coverage = self._page_coverage.setdefault(key, [])
                coverage.append((left - start, right - start))
                covered = 0
                for left, right in sorted(coverage):
                    if left > covered:
                        break
                    covered = max(covered, right)
                self._page_coverage[key] = [(0, covered)] + [
                    (left, right) for left, right in coverage if left > covered
                ]
                if covered >= stop - start:
                    self.inspected_operations.add(op.op_id)
                    if include_variable_value:
                        self.read_nodes.update(
                            n.full_node_id
                            for n in self.graph.filter_by_operation(op.op_id).nodes
                        )
        return paginate(text, offset, min(limit, self.page_limit))

    def operation_allowed(self, op: RuntimeOp) -> bool:
        """Keep the current request plus earlier construction history in scope."""
        if self.query_node is None:
            return True
        scope = self.query_node.metadata.get("tenant_scope")
        if scope is not None and op.metadata.get("tenant_scope") != scope:
            return False
        if op.category in ("retrieval", "response", "evaluation"):
            if op.metadata.get("query_id"):
                return op.metadata["query_id"] == self.query_identity
            if self.request_id and op.metadata.get("request_id"):
                return op.metadata["request_id"] == self.request_id
            queries = [
                node
                for node in self.graph.filter_by_operation(op.op_id).nodes
                if node.class_name == "query"
            ]
            if queries:
                return all(
                    node.full_node_id == self.query_node.full_node_id
                    for node in queries
                )
            if (
                op.created_at > self.query_node.created_at
                and op.category == "retrieval"
            ):
                return False
        elif op.created_at > self.query_node.created_at:
            return False
        return True

    def update_to_explore_nodes(
        self,
        action: Literal["add", "delete"],
        full_node_id: str,
        discovered_by_op_id: str | None = None,
    ) -> str:
        """Maintain the frontier and record which inspected operation exposed a node."""
        state = self._state()
        if action == "delete":
            state.delete(full_node_id)
        elif action == "add":
            if discovered_by_op_id is not None:
                try:
                    subgraph = self.graph.filter_by_operation(discovered_by_op_id)
                except KeyError as exc:
                    raise ValueError("Unknown discovery operation.") from exc
                if full_node_id not in {node.full_node_id for node in subgraph.nodes}:
                    raise ValueError(
                        "The discovery operation does not contain this node."
                    )
            state.add(
                self._record(full_node_id, discovered_by_op_id), self.max_trace_nodes
            )
        else:
            raise ValueError("action must be add or delete.")
        return f"Frontier action {action} succeeded for {full_node_id}."

    def view_node_value(
        self,
        full_node_id: str,
        pattern: str | None = None,
        max_surrounding_chars: int = 120,
        offset: int = 1,
        limit: int = DEFAULT_PAGE,
    ) -> str:
        """Read stored text, or paginate regex matches with source character ranges."""
        self._state()
        if offset < 1 or limit < 1 or max_surrounding_chars < 0:
            raise ValueError("Invalid pagination or surrounding-character count.")
        try:
            node = self.graph.get_variable(full_node_id)
        except KeyError as exc:
            raise ValueError(f"Unknown variable {full_node_id}.") from exc
        if self.query_node is not None and not same_tenant(node, self.query_node):
            raise ValueError("The variable belongs to another tenant.")
        value = node.value
        if pattern is None:
            if offset <= len(value):
                self.read_nodes.add(full_node_id)
            return paginate(value, offset, min(limit, self.page_limit))
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Invalid regex: {exc}") from exc
        limit = min(limit, PAGE_LIMIT, self.page_limit)
        parts = []
        size = 0
        next_offset = None
        for match in regex.finditer(value, offset - 1):
            start = max(0, match.start() - max_surrounding_chars)
            end = min(len(value), match.end() + max_surrounding_chars)
            snippet = f"Match at {match.start() + 1}-{match.end()}: {value[start:end]}"
            if parts and size + len(snippet) + 1 > limit:
                next_offset = match.start() + 1
                break
            if len(snippet) > limit:
                # Never silently discard a large match; direct reading stays available.
                snippet = snippet[:limit] + " [snippet truncated; read value directly]"
            parts.append(snippet)
            size += len(snippet) + 1
        if parts:
            self.read_nodes.add(full_node_id)
        status = (
            f"More matches; next offset={next_offset}."
            if next_offset
            else "End of matches."
        )
        return "\n".join(parts or ["No matches."]) + "\n" + status

    def finish_node_exploration(self, outcome: str) -> str:
        """Move the active node into immutable exploration history."""
        self._state().finish_node(outcome)
        return "Node exploration finished."

    def finish_graph_exploration(
        self, outcome: str, insufficient_evidence: bool = False
    ) -> str:
        """Archive final state; allow a later explicit initialization to restart."""
        state = self._state()
        state.finish(outcome)
        self.final_state = state.model_copy(deep=True)
        self.state = None
        if insufficient_evidence:
            self.insufficient_evidence = outcome
            return (
                "Exploration stopped: insufficient evidence; no attribution submitted."
            )
        return "Graph exploration finished. Submit attribution or explicitly restart."

    def view_exploration_history(
        self, offset: int = 1, limit: int = DEFAULT_PAGE
    ) -> str:
        """Return completed node records without returning full node values."""
        return paginate(
            json.dumps(
                {
                    key: node.model_dump()
                    for key, node in self._state().exploration_history.items()
                },
                indent=2,
            ),
            offset,
            min(limit, self.page_limit),
        )

    def hint(self) -> str:
        """Generate a fresh state hint rather than accumulating stale hints."""
        if self.state is None:
            return (
                "Graph exploration finished; call submit_attribution, or initialize "
                "again if more evidence is needed."
                if self.final_state
                else "Call initialize_execution_graph before exploring."
            )
        state = self.state
        active = state.exploring_node
        queued = sorted(
            state.to_explore_nodes.values(), key=lambda node: node.priority()
        )

        def brief(node: GraphTraceNodeRecord) -> dict[str, Any]:
            return {
                "full_node_id": node.full_node_id,
                "category": node.category,
                "comment": (node.comment or "")[:240],
                "discovered_by_op_id": node.discovered_by_op_id,
            }

        summary = {
            "active": brief(active) if active else None,
            "frontier_count": len(queued),
            "frontier": [brief(node) for node in queued[:16]],
            "completed_count": len(state.exploration_history),
            "completed": [
                {
                    "full_node_id": node.full_node_id,
                    "outcome": (node.outcome or "")[:400],
                }
                for node in list(state.exploration_history.values())[-4:]
            ],
        }
        actions = (
            "Inspect more pages or selected values, add downstream nodes, then "
            "finish_node_exploration."
            if active
            else "Explore the earliest node or finish_graph_exploration if ready."
        )
        return (
            "<system-hint>\n" + json.dumps(summary) + "\n" + actions + "</system-hint>"
        )

    def snapshot(self) -> dict[str, Any] | None:
        state = self.state or self.final_state
        return state.model_dump() if state is not None else None


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


def graph_tools() -> list[dict[str, Any]]:
    """Expose the notebook tools through OpenAI function calling."""
    pagination = {
        "offset": {"type": "integer", "minimum": 1, "default": 1},
        "limit": {"type": "integer", "minimum": 1, "default": DEFAULT_PAGE},
    }
    node_id = {"type": "string", "description": "Exact versioned full_node_id."}
    outcome = {"outcome": {"type": "string"}}
    return [
        _tool(
            "initialize_execution_graph",
            "Initialize or restart graph exploration.",
            {"initial_full_node_ids": {"type": "array", "items": node_id}},
        ),
        _tool(
            "explore_graph_trace_node",
            "Inspect operations involving the active node.",
            {
                **pagination,
                "include_variable_value": {"type": "boolean", "default": True},
            },
        ),
        _tool(
            "update_to_explore_nodes",
            "Add/delete one future exploration node.",
            {
                "action": {"type": "string", "enum": ["add", "delete"]},
                "full_node_id": node_id,
                "discovered_by_op_id": {"type": "string"},
            },
            ["action", "full_node_id"],
        ),
        _tool(
            "view_node_value",
            "Read a value or search it with a regex.",
            {
                **pagination,
                "full_node_id": node_id,
                "pattern": {"type": "string"},
                "max_surrounding_chars": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 120,
                },
            },
            ["full_node_id"],
        ),
        _tool(
            "finish_node_exploration",
            "Finish the active node with findings.",
            outcome,
            ["outcome"],
        ),
        _tool(
            "finish_graph_exploration",
            "Finish graph exploration before submitting.",
            {
                **outcome,
                "insufficient_evidence": {
                    "type": "boolean",
                    "default": False,
                    "description": "Stop if critical evidence is absent.",
                },
            },
            ["outcome"],
        ),
        _tool(
            "view_exploration_history",
            "Read completed node exploration records.",
            pagination,
        ),
    ]
