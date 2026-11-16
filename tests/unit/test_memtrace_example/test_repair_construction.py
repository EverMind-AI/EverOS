"""Offline repairs remove false boundary duplicates without rewriting history."""

from __future__ import annotations

import copy
import json

from smartcomment import (
    comment_graph,
    comment_link,
    comment_op_scope,
    comment_session,
    comment_variable,
)
from smartcomment.runtime import ExecNetwork

from examples.memtrace.repair_construction import repair_payload


def _legacy_graph() -> tuple[dict, dict]:
    """Model an episode and two foresights repeatedly observing one real boundary."""
    members = [f"m_session_{index}" for index in range(4)]
    boundary = {
        "app_id": "default",
        "project_id": "default",
        "session_id": "session",
        "message_ids": members,
        "sender_ids": ["alice", "assistant"],
    }
    with comment_graph(user_id="alice") as graph, comment_session(session_name="test"):

        def variable(value: dict, category: str):
            return comment_variable(
                value,
                to_runtime=True,
                id_strategy=lambda value: value["id"],
                encoding_fn=lambda value: json.dumps(value, indent=2, sort_keys=True),
                category=category,
            )

        messages = [variable({"id": name}, "message") for name in members]
        buffer = {"id": "buffer-session-alice", **boundary, "owner_id": "alice"}
        buffer["message_ids"] = members[:2]
        first = variable(buffer, "message_buffer")
        with comment_op_scope(
            op_name="memory_add_session",
            category="initialization",
            metadata={"event_seq": 1},
        ):
            for message in messages[:2]:
                comment_link(source=message, target=first)
        buffer["message_ids"] = members
        second = variable(buffer, "message_buffer")
        with comment_op_scope(
            op_name="memory_add_session",
            category="initialization",
            metadata={"event_seq": 2},
        ):
            comment_link(source=first, target=second)
            for message in messages[2:]:
                comment_link(source=message, target=second)
        for index in range(3):
            cell = {
                "id": "mc_1",
                **boundary,
                "sender_ids": boundary["sender_ids"] if index == 0 else [],
            }
            with comment_op_scope(
                op_name="boundary_detection_mc_1",
                category="boundary_detection",
                metadata={"event_seq": index + 3},
            ):
                parent = comment_variable(
                    cell,
                    to_runtime=True,
                    id_strategy=lambda value: value["id"],
                    encoding_fn=lambda value: json.dumps(
                        value, indent=2, sort_keys=True
                    ),
                    category="memory_entry",
                    metadata={"membership": "exact"},
                )
                for message in messages:
                    comment_link(source=message, target=parent)
            child = variable({"id": f"alice_entry_{index}"}, "memory_entry")
            with comment_op_scope(op_name=f"entry_{index}", category="extraction"):
                comment_link(source=parent, target=child)
        # A later add must not replace the earlier consumed buffer in a repair.
        buffer["message_ids"] = [*members, "later_message"]
        third = variable(buffer, "message_buffer")
        with comment_op_scope(
            op_name="memory_add_session",
            category="initialization",
            metadata={"event_seq": 10},
        ):
            comment_link(source=second, target=third)
    return graph.export_graph(), {"mc_1": boundary}


def test_repair_merges_only_repeated_exact_boundaries() -> None:
    """One edge per member, one real buffer input, and every child survive repair."""
    original, boundaries = _legacy_graph()
    before = copy.deepcopy(original)
    repaired, audit = repair_payload(original, boundaries)
    assert original == before
    graph = ExecNetwork.import_graph(repaired)
    assert len(graph.get_versions("mc_1")) == 1
    assert json.loads(graph.get_latest_variable("mc_1").raw_value)["sender_ids"] == [
        "alice",
        "assistant",
    ]
    inputs = [
        edge for edge in graph.get_all_edges() if edge.target_full_node_id == "mc_1@1"
    ]
    assert len(inputs) == 5
    assert {edge.source_full_node_id for edge in inputs} == {
        *(f"m_session_{index}@1" for index in range(4)),
        "buffer-session-alice@2",
    }
    assert len(graph.search_operations(category="boundary_detection")) == 1
    assert (
        len(
            [
                edge
                for edge in graph.get_all_edges()
                if edge.source_full_node_id == "mc_1@1"
            ]
        )
        == 3
    )
    assert audit["removed_nodes"] == 1 and audit["added_buffer_edges"] == 1
    again, repeated = repair_payload(repaired, boundaries)
    assert again == repaired
    assert not repeated["cells"]


def test_repair_preserves_actual_membership_changes() -> None:
    """A different message boundary is genuine history and cannot be collapsed."""
    original, boundaries = _legacy_graph()
    changed = next(
        node for node in original["data"]["nodes"] if node["full_node_id"] == "mc_1@2"
    )
    value = json.loads(changed["value"])
    value["message_ids"] = value["message_ids"][:2]
    changed["value"] = json.dumps(value, indent=2, sort_keys=True)
    repaired, audit = repair_payload(original, boundaries)
    assert repaired == original
    assert not audit["cells"]
