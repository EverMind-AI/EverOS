"""Display truncation preserves returned evidence and owner-prefixed members."""

from __future__ import annotations

import json

import pytest
from smartcomment import (
    comment_graph,
    comment_link,
    comment_op_scope,
    comment_session,
    comment_variable,
)
from smartcomment.runtime import ExecNetwork

from examples.memtrace.export_png import _display_payload


@pytest.mark.parametrize("selected_query", [None, "query-alice-first"])
@pytest.mark.parametrize("request_local", [False, True])
def test_display_preserves_all_returns_and_correct_candidate_prefix(
    selected_query: str | None,
    request_local: bool,
) -> None:
    """Top-one display retains a listed non-hit and every returned fact/episode."""
    with comment_graph() as graph, comment_session(session_name="display"):

        def variable(name: str, value: dict, category: str, cls: str | None = None):
            return comment_variable(
                {"id": name, **value},
                to_runtime=True,
                id_strategy=lambda snapshot: snapshot["id"],
                encoding_fn=json.dumps,
                category=category,
                class_name=cls,
            )

        query = variable(
            "query-alice-first", {"user_id": "alice"}, "message & query", "query"
        )
        pool = variable(
            "candidates_episode_hierarchy-query-alice-first",
            {
                "ids": ["af_shown", "af_returned", "af_hidden"],
                "scores": [0.9, 0.8, 0.7],
                "sources": ["", "", ""],
                "total_count": 3,
                "count": 3,
            },
            "memory_context",
            "reranked_results",
        )
        recall = variable(
            "candidates_episode_sparse-query-alice-first",
            {"ids": ["ep_other_query"], "scores": [0.5], "sources": ["keyword"]},
            "memory_context",
            "sparse_retrieval_results",
        )
        # An earlier snapshot mentions evidence returned by another query;
        # it must not appear as an isolated node in this query's diagram.
        variable("alice_ep_other_query", {}, "memory_entry")
        entries = {
            eid: variable(
                f"hit-query-alice-first-{eid}" if request_local else f"alice_{eid}",
                {"memory_identity": f"alice_{eid}", "query_id": "query-alice-first"}
                if request_local
                else {},
                "retrieval_result" if request_local else "memory_entry",
                "retrieval_hit" if request_local else None,
            )
            for eid in ("af_shown", "af_returned", "af_hidden", "ep_wrapper")
        }
        context = variable("context-query-alice-first", {}, "memory_context", "context")
        with comment_op_scope(op_name="retrieve", category="retrieval"):
            comment_link(source=query, target=recall)
            comment_link(source=recall, target=pool)
            for entry in entries.values():
                comment_link(
                    source=pool, target=entry, edge_metadata={"membership": True}
                )
            for eid in ("af_returned", "ep_wrapper"):
                comment_link(source=entries[eid], target=context)
        variable("query-alice-other", {"user_id": "alice"}, "message & query", "query")

    payload = graph.export_graph()
    before = json.dumps(payload, sort_keys=True)
    display = _display_payload(payload, 1, selected_query)
    restored = ExecNetwork.import_graph(display)
    names = {node.name for node in restored.get_all_nodes()}
    assert {
        entries[eid].name for eid in ("af_shown", "af_returned", "ep_wrapper")
    } <= names
    pairs = {
        (edge["source_full_node_id"], edge["target_full_node_id"])
        for edge in display["data"]["edges"]
    }
    for eid in ("af_shown", "af_returned", "ep_wrapper"):
        assert (pool.full_node_id, entries[eid].full_node_id) in pairs
    assert (pool.full_node_id, entries["af_hidden"].full_node_id) not in pairs
    if selected_query:
        assert "query-alice-other" not in names
        assert "alice_ep_other_query" not in names
    pool_node = next(n for n in display["data"]["nodes"] if n["name"] == pool.name)
    assert json.loads(pool_node["value"])["ids"] == ["af_shown"]
    assert json.dumps(payload, sort_keys=True) == before


@pytest.mark.parametrize("candidate_count", [0, 3])
def test_display_keeps_auxiliary_branch_to_context(candidate_count: int) -> None:
    """Pool truncation must keep context assembly, including an empty read."""
    with comment_graph() as graph, comment_session(session_name="branches"):
        query = comment_variable(
            {"id": "query-alice-one", "user_id": "alice"},
            to_runtime=True,
            id_strategy=lambda value: value["id"],
            encoding_fn=json.dumps,
            category="message & query",
            class_name="query",
        )
        pool = comment_variable(
            {
                "id": "candidates_unprocessed__load_unprocessed-query-alice-one",
                "ids": [f"m_{i}" for i in range(candidate_count)],
                "total_count": candidate_count,
            },
            to_runtime=True,
            id_strategy=lambda value: value["id"],
            encoding_fn=json.dumps,
            category="memory_context",
        )
        context = comment_variable(
            {"id": "context-query-alice-one"},
            to_runtime=True,
            id_strategy=lambda value: value["id"],
            encoding_fn=json.dumps,
            category="memory_context",
            class_name="context",
        )
        with comment_op_scope(op_name="_load_unprocessed", category="retrieval"):
            comment_link(source=query, target=pool)
        with comment_op_scope(op_name="context_assembly", category="retrieval"):
            comment_link(source=pool, target=context)
    display = _display_payload(graph.export_graph(), 1, query.name)
    ExecNetwork.import_graph(display)
    assert (pool.full_node_id, context.full_node_id) in {
        (edge["source_full_node_id"], edge["target_full_node_id"])
        for edge in display["data"]["edges"]
    }
