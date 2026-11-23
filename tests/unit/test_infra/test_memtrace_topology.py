"""Pin shared extraction inputs, compact search branches and readable exports."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from smartcomment.runtime import ExecNetwork

from everos.infra.memtrace.qa import inject_qa_dir
from everos.infra.memtrace.tracker import MemTraceTracker
from everos.memory.search.dto import SearchProfileItem, UnprocessedMessageDTO
from everos.memory.search.trace import collect_search_trace, record_branch

from .test_memtrace_tracker import (
    _EPISODE_MD,
    _TS,
    _load_export,
    _req,
    _resp,
)


@pytest.fixture(autouse=True)
def _memtrace_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tracing is opt-in; these tests exercise the enabled tracker."""
    monkeypatch.setenv("EVEROS_MEMTRACE_ENABLED", "1")


_INPUT = {"items": [{"role": "user", "content": "Camping on Friday."}]}
_KEY = ("default", "default", "alice", "sess_a")


async def _empty_memcells(ids: list[str]) -> list[SimpleNamespace]:
    return []


memcell_repo = SimpleNamespace(find_by_ids=_empty_memcells)


@pytest.fixture(autouse=True)
def _inject_memcell_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    original_init = MemTraceTracker.__init__

    def init(tracker: MemTraceTracker) -> None:
        original_init(tracker)
        tracker.set_memcell_repo(memcell_repo)

    monkeypatch.setattr(MemTraceTracker, "__init__", init)


@pytest.fixture
def tracker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> MemTraceTracker:
    monkeypatch.setenv("EVEROS_MEMTRACE_DIR", str(tmp_path / "memtrace"))
    return MemTraceTracker()


def _write_foresights(root: Path) -> None:
    path = root / "users/alice/.foresights/2026-01-01.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\nuser_id: alice\n---\n"
        + "\n".join(
            f"<!-- entry:fs_{i} -->\n**owner_id**: alice\n"
            "**session_id**: sess_a\n**parent_type**: memcell\n"
            "**parent_id**: mc_1\n### Foresight\n"
            f"Prepare for trip {i}.\n<!-- /entry:fs_{i} -->"
            for i in range(2)
        )
    )


def _archive(**overrides: object) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            "app_id": "default",
            "project_id": "default",
            "session_id": "sess_a",
            "message_ids_json": "[]",
            "sender_ids_json": '["alice"]',
            "payload_json": json.dumps(_INPUT),
            **overrides,
        }
    )


def _assert_descriptions(payload: dict) -> None:
    for collection in ("nodes", "operations", "edges"):
        for record in payload["data"][collection]:
            assert record.get("comment"), (collection, record)
            assert "_load_unprocessed" not in record["comment"]
            assert "_fetch_profile" not in record["comment"]


@pytest.mark.parametrize("restart", [False, True])
async def test_episode_and_late_foresights_share_the_original_input(
    tracker: MemTraceTracker,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restart: bool,
) -> None:
    async def archive(ids: list[str]) -> list[SimpleNamespace]:
        return [_archive()]

    monkeypatch.setattr(memcell_repo, "find_by_ids", archive)
    common = {
        "app_id": "default",
        "project_id": "default",
        "session_id": "sess_a",
        "observed_at": _TS.isoformat(),
    }
    events = [
        {
            **common,
            "phase": "boundary",
            "cells": [
                {
                    "id": "mc_1",
                    "message_ids": [],
                    "sender_ids": ["alice"],
                    "content": _INPUT,
                }
            ],
        },
        {
            **common,
            "phase": "extraction_attempt",
            "memcell_id": "mc_1",
            "owners": ["alice"],
        },
        {
            **common,
            "phase": "extraction_result",
            "memcell_id": "mc_1",
            "owners": ["alice"],
            "content": "Alice went camping in July.",
        },
        {
            **common,
            "phase": "persisted_episode",
            "memcell_id": "mc_1",
            "owners": ["alice"],
            "entry_id": "ep_1",
        },
    ]
    path = tmp_path / "users/alice/episodes/2026-01-01.md"
    path.parent.mkdir(parents=True)
    path.write_text(_EPISODE_MD)
    await tracker.on_memorize(
        {"session_id": "sess_a", "messages": []},
        is_final=True,
        defer_extraction=False,
        result=None,
        request_id="build",
        construction_events=events,
    )
    if restart:
        tracker = MemTraceTracker()
    _write_foresights(tmp_path)
    await tracker.sync_now()
    payload = _load_export(tracker, _KEY)
    graph = ExecNetwork.import_graph(payload)
    cell = graph.get_versions("cell-input-mc_1")
    assert len(cell) == 1
    assert json.loads(cell[0].raw_value)["content"] == _INPUT
    assert not graph.search_variables(category="extraction_attempt")
    assert not graph.search_operations(name_pattern="^extraction_attempt_")
    result = graph.search_variables(category="extraction_result")[0]
    pairs = [
        (e.source_full_node_id, e.target_full_node_id) for e in graph.get_all_edges()
    ]
    assert pairs.count(("mc_1@1", cell[0].full_node_id)) == 1
    assert (cell[0].full_node_id, result.full_node_id) in pairs
    assert (result.full_node_id, "alice_ep_1@1") in pairs
    for i in range(2):
        assert pairs.count((cell[0].full_node_id, f"alice_fs_{i}@1")) == 1
        assert ("mc_1@1", f"alice_fs_{i}@1") not in pairs
    _assert_descriptions(payload)


@pytest.mark.parametrize(
    "source",
    [
        "valid",
        "missing",
        "invalid",
        "wrong_app",
        "wrong_project",
        "wrong_session",
        "error",
    ],
)
async def test_foresight_input_recovery_is_exact_or_explicitly_unknown(
    tracker: MemTraceTracker,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    async def archive(ids: list[str]) -> list[SimpleNamespace]:
        if source == "missing":
            return []
        if source == "error":
            raise RuntimeError("database unavailable")
        overrides = {}
        if source.startswith("wrong_"):
            overrides[f"{source.removeprefix('wrong_')}_id"] = "other"
        if source == "invalid":
            overrides["payload_json"] = "invalid JSON"
        return [_archive(**overrides)]

    monkeypatch.setattr(memcell_repo, "find_by_ids", archive)
    _write_foresights(tmp_path)
    await tracker.sync_now()
    payload = _load_export(tracker, _KEY)
    graph = ExecNetwork.import_graph(payload)
    cell = graph.get_versions("cell-input-mc_1")
    assert len(cell) == 1
    assert json.loads(cell[0].raw_value)["content"] == (
        _INPUT if source == "valid" else None
    )
    assert cell[0].metadata["content_observed"] is (source == "valid")
    observe = graph.search_operations(name_pattern="^observe_cell_")[0]
    assert observe.metadata["observation_only"] is True
    _assert_descriptions(payload)


@pytest.mark.parametrize("scenario", ["success", "empty", "skipped", "error"])
async def test_search_branches_have_one_described_result_without_profile_nodes(
    tracker: MemTraceTracker, scenario: str
) -> None:
    status = "success" if scenario == "empty" else scenario
    messages = (
        [
            UnprocessedMessageDTO(
                id=f"pending-{i}",
                session_id="sess_a",
                sender_id="alice",
                role="user",
                content=f"Trip detail {i}",
                timestamp=_TS,
            )
            for i in range(2)
        ]
        if scenario == "success"
        else []
    )
    profile = SearchProfileItem(
        id="profile_alice", user_id="alice", profile_data={"summary": "camping"}
    )
    with collect_search_trace() as stages:
        record_branch("_fetch_profile", "profile", [profile])
        record_branch("_load_unprocessed", "unprocessed", messages, status=status)
    if status == "error":
        await tracker.on_search_error(
            _req(), stages=stages, error=RuntimeError("read failed"), request_id=status
        )
    else:
        response = _resp(unprocessed=messages)
        response.data.profiles = [profile]
        await tracker.on_search(_req(), response, stages=stages, request_id=status)
    payload = _load_export(tracker)
    graph = ExecNetwork.import_graph(payload)
    assert not graph.search_operations(name_pattern="^_fetch_profile_")
    assert not graph.search_variables(class_name="retrieval_hit")
    assert not graph.search_variables(category="message")
    result = graph.search_variables(class_name="retrieval_results")
    assert len(result) == 1 and result[0].category == "retrieval_result"
    assert result[0].metadata["status"] == status
    assert json.loads(result[0].raw_value)["items"] == [
        m.model_dump(mode="json") for m in messages
    ]
    if status == "error":
        assert not graph.search_variables(class_name="context")
    else:
        context = graph.search_variables(class_name="context")[0]
        pairs = {
            (e.source_full_node_id, e.target_full_node_id)
            for e in graph.get_all_edges()
        }
        assert (result[0].full_node_id, context.full_node_id) in pairs
        pending_edge = next(
            e
            for e in graph.get_all_edges()
            if e.source_full_node_id == result[0].full_node_id
            and e.target_full_node_id == context.full_node_id
        )
        assert (
            pending_edge.comment
            == {
                "success": "Include the pending session messages in the context.",
                "empty": "Add no pending messages because the session buffer is empty.",
                "skipped": "Add no pending messages because no session was specified.",
            }[scenario]
        )
        assert json.loads(json.loads(context.raw_value)["passages"][0]["text"]) == (
            profile.profile_data
        )
    _assert_descriptions(payload)


async def test_live_and_offline_qa_have_descriptions(tracker: MemTraceTracker) -> None:
    for request_id in ("live", "offline"):
        await tracker.on_search(_req(), _resp(), request_id=request_id)
    await tracker.on_answer(
        app_id="default",
        project_id="default",
        owner="alice",
        request_id="live",
        answer="Friday",
        answer_context="Camping on Friday.",
    )
    await tracker.on_evaluation(
        app_id="default",
        project_id="default",
        owner="alice",
        request_id="live",
        golden_answers=["Friday"],
        judge_score=1.0,
    )
    inject_qa_dir(
        tracker.out_dir,
        [
            {
                "user_id": "alice",
                "request_id": "offline",
                "answer": "Friday",
                "golden_answers": ["Friday"],
                "judge_score": 1.0,
            }
        ],
    )
    _assert_descriptions(_load_export(tracker))
