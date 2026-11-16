"""Pin buffer drains, historical boundary inputs and independent search branches."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from smartcomment.runtime import ExecNetwork

from everos.infra.memtrace.tracker import (
    MemTraceTracker,
    _boundary_result_identity,
)
from everos.memory.search.dto import SearchProfileItem, UnprocessedMessageDTO
from everos.memory.search.trace import collect_search_trace, record_branch

from .test_memtrace_tracker import (
    _TS,
    _TS_MS,
    _load_export,
    _message,
    _payload,
    _req,
    _resp,
)


@pytest.fixture
def tracker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> MemTraceTracker:
    monkeypatch.setenv("EVEROS_MEMTRACE_DIR", str(tmp_path / "memtrace"))
    return MemTraceTracker()


@pytest.mark.parametrize("restart", [False, True])
async def test_two_flushes_drain_separate_buffer_versions(
    tracker: MemTraceTracker,
    restart: bool,
) -> None:
    ids = [f"m_sess_a_{_TS_MS + i}_000" for i in range(3)]
    common = {
        "app_id": "default",
        "project_id": "default",
        "session_id": "sess_a",
        "observed_at": _TS.isoformat(),
    }
    for i in range(2):
        await tracker.on_memorize(
            _payload(_message(str(i), _TS_MS + i)),
            is_final=False,
            defer_extraction=True,
            result=None,
        )
    for i in range(2):
        consumed = ids[:2] if i == 0 else ids[2:]
        await tracker.on_memorize(
            _payload(),
            is_final=True,
            defer_extraction=False,
            result=SimpleNamespace(status="extracted", message_count=0),
            request_id=f"flush-{i}",
            construction_events=[
                {
                    **common,
                    "phase": "buffer_loaded",
                    "before_message_ids": consumed,
                    "message_ids": consumed,
                    "sender_ids": ["alice"],
                },
                {**common, "phase": "buffer_persisted", "message_ids": []},
                {
                    **common,
                    "phase": "boundary",
                    "cells": [
                        {
                            "id": f"mc_{i}",
                            "message_ids": consumed,
                            "sender_ids": ["alice"],
                            "content": "real cell",
                        },
                    ],
                },
            ],
        )
        if i == 0:
            if restart:
                tracker = MemTraceTracker()
            await tracker.on_memorize(
                _payload(_message("new", _TS_MS + 2)),
                is_final=False,
                defer_extraction=True,
                result=None,
            )
    payload = _load_export(tracker, ("default", "default", "alice", "sess_a"))
    graph = ExecNetwork.import_graph(payload)
    buffers = graph.get_versions("buffer-sess_a-alice")
    assert all(
        n.comment == "The session messages held in the buffer at this point."
        for n in buffers
    )
    assert [json.loads(n.raw_value)["message_ids"] for n in buffers] == [
        [],
        ids[:1],
        ids[:2],
        [],
        ids[2:],
        [],
    ]
    for i, version in enumerate((3, 5)):
        consumed = ids[:2] if i == 0 else ids[2:]
        edges = [
            e for e in graph.get_all_edges() if e.target_full_node_id == f"mc_{i}@1"
        ]
        result_id = (
            "boundary_detection_result:"
            f"{_boundary_result_identity('sess_a', consumed)}@1"
        )
        assert [e.source_full_node_id for e in edges] == [result_id]
        assert [
            e.source_full_node_id
            for e in graph.get_all_edges()
            if e.target_full_node_id == result_id
        ] == [f"buffer-sess_a-alice@{version}"]
        op = next(
            o
            for o in graph.get_all_operations()
            if o.metadata.get("request_id") == f"flush-{i}"
            and o.op_name == "memory_flush_sess_a"
        )
        transitions = [e for e in graph.get_all_edges() if e.op_id == op.op_id]
        assert [
            (e.source_full_node_id, e.target_full_node_id) for e in transitions
        ] == [
            (f"buffer-sess_a-alice@{version}", f"buffer-sess_a-alice@{version + 1}"),
        ]
    assert not graph.get_versions("COMMENT:NONE")


async def test_deferred_outcome_and_flush_decision_share_one_node(
    tracker: MemTraceTracker,
) -> None:
    """A flush over the deferred messages versions the same outcome node."""
    ids = [f"m_sess_a_{_TS_MS}_000", f"m_sess_a_{_TS_MS + 1}_001"]
    common = {
        "app_id": "default",
        "project_id": "default",
        "session_id": "sess_a",
        "observed_at": _TS.isoformat(),
    }
    await tracker.on_memorize(
        _payload(_message("one", _TS_MS), _message("two", _TS_MS + 1)),
        is_final=False,
        defer_extraction=True,
        result=None,
        request_id="add-0",
        construction_events=[
            {
                **common,
                "phase": "boundary",
                "reason": "deferred",
                "detection_performed": False,
                "cells": [],
            }
        ],
    )
    await tracker.on_memorize(
        _payload(),
        is_final=True,
        defer_extraction=False,
        result=SimpleNamespace(status="extracted", message_count=0),
        request_id="flush-0",
        construction_events=[
            {
                **common,
                "phase": "buffer_loaded",
                "before_message_ids": ids,
                "message_ids": ids,
                "sender_ids": ["alice"],
            },
            {**common, "phase": "buffer_persisted", "message_ids": []},
            {
                **common,
                "phase": "boundary",
                "detection_performed": True,
                "detection": {
                    "message_ids": ids,
                    "is_final": True,
                    "mode": "chat",
                    "model_result": {
                        "reasoning": "The topic closes.",
                        "boundaries": [2],
                        "should_wait": False,
                    },
                },
                "cells": [
                    {
                        "id": "mc_0",
                        "message_ids": ids,
                        "sender_ids": ["alice"],
                        "content": "real cell",
                    }
                ],
            },
        ],
    )
    payload = _load_export(tracker, ("default", "default", "alice", "sess_a"))
    graph = ExecNetwork.import_graph(payload)
    outcomes = graph.search_variables(class_name="boundary_detection_result")
    identity = _boundary_result_identity("sess_a", ids)
    assert {node.name for node in outcomes} == {identity}
    deferred, split = sorted(outcomes, key=lambda node: node.version)
    assert (deferred.version, split.version) == (1, 2)
    deferred_value = json.loads(deferred.raw_value)
    assert deferred_value["decision"] == "deferred"
    assert deferred_value["detection_performed"] is False
    assert deferred_value["request_id"] == "add-0"
    assert deferred.category == "boundary_outcome"
    split_value = json.loads(split.raw_value)
    assert split_value["decision"] == "split"
    assert split_value["request_id"] == "flush-0"
    assert split_value["cell_ids"] == ["mc_0"]
    assert split.category == "llm_response"
    edges = graph.get_all_edges()
    buffer_id = next(
        e.source_full_node_id
        for e in edges
        if e.target_full_node_id == deferred.full_node_id
    )
    assert buffer_id.startswith("buffer-sess_a-alice@")
    assert any(
        e.source_full_node_id == buffer_id
        and e.target_full_node_id == split.full_node_id
        for e in edges
    )
    assert any(
        e.source_full_node_id == split.full_node_id
        and e.target_full_node_id == "mc_0@1"
        for e in edges
    )


async def test_nonfinal_boundary_keeps_real_tail_and_empty_flush_is_versioned(
    tracker: MemTraceTracker,
) -> None:
    await tracker.on_memorize(
        _payload(_message("one", _TS_MS), _message("tail", _TS_MS + 1)),
        is_final=False,
        defer_extraction=False,
        result=None,
        construction_events=[
            {
                "phase": "buffer_persisted",
                "app_id": "default",
                "project_id": "default",
                "session_id": "sess_a",
                "observed_at": _TS.isoformat(),
                "message_ids": [f"m_sess_a_{_TS_MS + 1}_001"],
            }
        ],
    )
    assert tracker._buffers[("default", "default", "alice", "sess_a")][
        "message_ids"
    ] == [f"m_sess_a_{_TS_MS + 1}_001"]
    empty = {
        "phase": "buffer_persisted",
        "app_id": "default",
        "project_id": "default",
        "session_id": "sess_a",
        "observed_at": _TS.isoformat(),
        "message_ids": [],
    }
    for rid in ("drain", "empty-flush"):
        await tracker.on_memorize(
            _payload(),
            is_final=True,
            defer_extraction=False,
            result=None,
            request_id=rid,
            construction_events=[empty],
        )
    graph = tracker._graphs[("default", "default", "alice", "sess_a")]
    versions = graph.get_versions("buffer-sess_a-alice")
    assert [len(json.loads(n.raw_value)["message_ids"]) for n in versions] == [
        0,
        2,
        1,
        0,
        0,
    ]


async def test_profile_is_untraced_and_pending_messages_feed_context(
    tracker: MemTraceTracker,
) -> None:
    profile = SearchProfileItem(
        id="profile_alice",
        user_id="alice",
        profile_data={"summary": "camping"},
    )
    message = UnprocessedMessageDTO(
        id="m_sess_a_pending_000",
        session_id="sess_a",
        sender_id="alice",
        role="user",
        content="new trip",
        timestamp=_TS,
    )
    with collect_search_trace() as stages:
        record_branch("_fetch_profile", "profile", [profile])
        record_branch(
            "_load_unprocessed", "unprocessed", [message], session_id="sess_a"
        )
    response = _resp(unprocessed=[message])
    response.data.profiles = [profile]
    await tracker.on_search(
        _req(include_profile=True), response, stages=stages, request_id="branches"
    )
    payload = _load_export(tracker)
    graph = ExecNetwork.import_graph(payload)
    query = graph.search_variables(class_name="query")[0]
    context = graph.search_variables(class_name="context")[0]
    edge_pairs = {
        (e.source_full_node_id, e.target_full_node_id) for e in graph.get_all_edges()
    }
    assert not graph.search_operations(name_pattern="^_fetch_profile_")
    assert not graph.search_variables(class_name="retrieval_hit")
    assert not graph.search_variables(category="message")
    op = graph.search_operations(name_pattern="^_load_unprocessed_")[0]
    assert op.metadata["status"] == "success"
    edges = graph.filter_by_operation(op.op_id).edges
    assert len(edges) == 1 and edges[0].source_full_node_id == query.full_node_id
    result = graph.get_variable(edges[0].target_full_node_id)
    assert json.loads(result.raw_value)["items"] == [message.model_dump(mode="json")]
    assert result.category == "retrieval_result"
    assert (result.full_node_id, context.full_node_id) in edge_pairs
    passages = json.loads(context.raw_value)["passages"]
    assert json.loads(passages[0]["text"]) == profile.profile_data


@pytest.mark.parametrize("include_profile", [False, True])
async def test_empty_profile_branch_does_not_feed_context(
    tracker: MemTraceTracker,
    include_profile: bool,
) -> None:
    status = "success" if include_profile else "skipped"
    with collect_search_trace() as stages:
        record_branch("_fetch_profile", "profile", [], status=status)
    await tracker.on_search(
        _req(include_profile=include_profile),
        _resp(),
        stages=stages,
        request_id="empty-profile",
    )
    graph = ExecNetwork.import_graph(_load_export(tracker))
    query = graph.search_variables(class_name="query")[0]
    context = graph.search_variables(class_name="context")[0]
    assert not graph.search_operations(name_pattern="^_fetch_profile_")
    assert not graph.search_variables(class_name="retrieval_results")
    assert not graph.search_variables(class_name="retrieval_hit")
    edge_pairs = {
        (e.source_full_node_id, e.target_full_node_id) for e in graph.get_all_edges()
    }
    assert (query.full_node_id, context.full_node_id) not in edge_pairs
