"""Unit tests for the MemTrace tracker.

Pins the MemTraceBench-compatible export produced by
:meth:`MemTraceTracker.on_search` / :meth:`MemTraceTracker.on_memorize`:

* without stage records the legacy ``query -> hits -> context`` chain is
  kept and no ``candidates_*`` pool nodes appear;
* with stage records each stage becomes a ``candidates_{kind}_{phase}``
  pool node, edges chain ``query -> recall pools -> fusion pools -> last
  pool -> hits`` with only terminal pools expanding candidates and no
  duplicate hit edges. Request-local hits reference canonical write-path
  ``{owner}_{entry_id}`` revisions when the response content matches;
* ``unprocessed_messages`` share one retrieval result linked into the context;
* graph files are named with collision-proof ``{safe}-{digest}``
  fragments, one file per ``(app, project, owner, bucket)`` key, and
  ``graph_id`` carries the same digest fragments;
* a diff round exports each dirtied graph exactly once, entry moves across
  files produce no versions, downtime deletions are reconciled on the
  first post-restart diff, and ``/add`` messages carry the ingest
  ``m_<session>_<ts_ms>_<idx>`` identity (an unparseable timestamp falls
  back to ``m_<session>_invalid_<idx>`` flagged ``timestamp_invalid``);
* flush requests record a request-level ``memory_flush`` op (including
  empty flushes, in the ``unknown``-owner graph) and failed requests
  record ``status: "error"`` ops (``on_memorize_error`` /
  ``on_search_error``, the latter keeping the partial stage pools);
* operation metadata carries ``event_seq`` / ``observed_at`` (and
  ``source`` / ``trigger`` on file-diff ops), stage pools carry
  ``total_count`` / ``truncated``, query identities derive from the
  request id with a sequence suffix on collision, memcell membership is
  tri-state (``exact`` / ``approximate`` / ``lookup_failed``), and
  ``sync_now`` flushes a final shutdown diff.

The tracker resolves its root / export dir from ``EVEROS_ROOT`` /
``EVEROS_MEMTRACE_DIR`` at construction; the root conftest pins
``EVEROS_ROOT`` to ``tmp_path`` and the fixture below pins the export dir,
so every test exports into an isolated directory.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import importlib
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from everos.infra.memtrace.tracker import GraphKey, MemTraceTracker
from everos.memory.search.dto import (
    SearchAtomicFactItem,
    SearchData,
    SearchEpisodeItem,
    SearchMethod,
    SearchRequest,
    SearchResponse,
    UnprocessedMessageDTO,
)
from everos.memory.search.trace import (
    CandidateBrief,
    SearchStageRecord,
)


@pytest.fixture(autouse=True)
def _memtrace_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tracing is opt-in; these tests exercise the enabled tracker."""
    monkeypatch.setenv("EVEROS_MEMTRACE_ENABLED", "1")


async def _empty_memcells(ids: list[str]) -> list[Any]:
    return []


memcell_repo = SimpleNamespace(find_by_ids=_empty_memcells)


@pytest.fixture(autouse=True)
def _inject_memcell_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    original_init = MemTraceTracker.__init__

    def init(tracker: MemTraceTracker) -> None:
        original_init(tracker)
        tracker.set_memcell_repo(memcell_repo)

    monkeypatch.setattr(MemTraceTracker, "__init__", init)


_TS = _dt.datetime(2026, 1, 1, tzinfo=_dt.UTC)
_TS_MS = 1767225600000  # 2026-01-01T00:00:00Z in unix milliseconds.

# Minimal episode markdown with one entry block; picked up by the
# tracker's file diff as the write-path record of ``alice_ep_1``.
_EPISODE_MD = """---
user_id: alice
---
<!-- entry:ep_1 -->
**owner_id**: alice
**session_id**: sess_a
**timestamp**: 2026-01-01T00:00:00+00:00
**parent_type**: memcell
**parent_id**: mc_1
**sender_ids**: [alice]
### Subject
Camping
### Summary
Camping trip
### Content
Alice went camping in July.
<!-- /entry:ep_1 -->
"""

# Two entries carved out of two different memcells of the same session.
_TWO_MEMCELL_MD = """---
user_id: alice
---
<!-- entry:ep_1 -->
**owner_id**: alice
**session_id**: sess_a
**timestamp**: 2026-01-01T00:00:00+00:00
**parent_type**: memcell
**parent_id**: mc_1
**sender_ids**: [alice]
### Content
First cell.
<!-- /entry:ep_1 -->
<!-- entry:ep_2 -->
**owner_id**: alice
**session_id**: sess_a
**timestamp**: 2026-01-01T00:05:00+00:00
**parent_type**: memcell
**parent_id**: mc_2
**sender_ids**: [alice]
### Content
Second cell.
<!-- /entry:ep_2 -->
"""

# Three parentless entries of one owner/session: same graph, one diff round.
_THREE_ENTRY_MD = """---
user_id: alice
---
<!-- entry:ep_1 -->
**owner_id**: alice
**session_id**: sess_a
**timestamp**: 2026-01-01T00:00:00+00:00
**sender_ids**: [alice]
### Content
One.
<!-- /entry:ep_1 -->
<!-- entry:ep_2 -->
**owner_id**: alice
**session_id**: sess_a
**timestamp**: 2026-01-01T00:01:00+00:00
**sender_ids**: [alice]
### Content
Two.
<!-- /entry:ep_2 -->
<!-- entry:ep_3 -->
**owner_id**: alice
**session_id**: sess_a
**timestamp**: 2026-01-01T00:02:00+00:00
**sender_ids**: [alice]
### Content
Three.
<!-- /entry:ep_3 -->
"""


@pytest.fixture
def tracker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> MemTraceTracker:
    """A tracker exporting into an isolated per-test directory."""
    monkeypatch.setenv("EVEROS_MEMTRACE_DIR", str(tmp_path / "memtrace"))
    return MemTraceTracker()


def _req(**kwargs) -> SearchRequest:
    options: dict[str, Any] = {
        "user_id": "alice",
        "query": "camping",
        "method": SearchMethod.HYBRID,
        "top_k": 5,
    }
    options.update(kwargs)
    return SearchRequest(**options)


def _episode_item(eid: str, *, score: float = 0.9) -> SearchEpisodeItem:
    return SearchEpisodeItem(
        id=eid,
        user_id="alice",
        session_id="sess_a",
        timestamp=_TS,
        sender_ids=["alice"],
        subject="s",
        summary="s",
        episode=f"body {eid}",
        type="Conversation",
        score=score,
        atomic_facts=[
            SearchAtomicFactItem(id=f"{eid}_f1", content=f"fact {eid}", score=score)
        ],
    )


def _resp(
    *episode_ids: str, unprocessed: list[UnprocessedMessageDTO] | None = None
) -> SearchResponse:
    return SearchResponse(
        request_id="r" * 32,
        data=SearchData(
            episodes=[_episode_item(eid) for eid in episode_ids],
            unprocessed_messages=unprocessed or [],
        ),
    )


def _stages() -> list[SearchStageRecord]:
    return [
        SearchStageRecord(
            phase="sparse",
            kind="episode",
            candidates=(
                CandidateBrief(id="ep_1", score=0.8, source="keyword"),
                CandidateBrief(id="ep_2", score=0.7, source="keyword"),
            ),
        ),
        SearchStageRecord(
            phase="dense",
            kind="episode",
            candidates=(CandidateBrief(id="ep_1", score=0.9, source="vector"),),
        ),
        SearchStageRecord(
            phase="fusion",
            kind="episode",
            candidates=(CandidateBrief(id="ep_1", score=0.95, source=""),),
            metadata={"fusion_mode": "rrf", "enable_rerank": False},
        ),
    ]


def _payload(*messages: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": "sess_a",
        "app_id": "default",
        "project_id": "default",
        "messages": list(messages),
    }


def _message(content: str, timestamp: Any, *, sender: str = "alice") -> dict[str, Any]:
    return {
        "sender_id": sender,
        "role": "user",
        "content": content,
        "timestamp": timestamp,
    }


def _load_export(
    tracker: MemTraceTracker,
    key: GraphKey = ("default", "default", "alice", "global"),
) -> dict:
    return json.loads(tracker._graph_path(key).read_text())


def _nodes_by_name(data: dict) -> dict[str, list[dict]]:
    by_name: dict[str, list[dict]] = {}
    for node in data["data"]["nodes"]:
        by_name.setdefault(node["name"], []).append(node)
    return by_name


def _fnid(node: dict) -> str:
    class_name = node.get("class_name")
    full_name = f"{class_name}:{node['name']}" if class_name else node["name"]
    return f"{full_name}@{node['version']}"


def _edge_pairs(data: dict) -> set[tuple[str, str]]:
    return {
        (e["source_full_node_id"], e["target_full_node_id"])
        for e in data["data"]["edges"]
    }


def _hit_node(data: dict, identity: str, query_id: str | None = None) -> dict:
    hits = [
        node
        for node in data["data"]["nodes"]
        if node.get("class_name") == "retrieval_hit"
        and json.loads(node["value"]).get("memory_identity") == identity
        and (query_id is None or json.loads(node["value"]).get("query_id") == query_id)
    ]
    assert len(hits) == 1
    return hits[0]


async def test_on_search_without_stages_keeps_legacy_chain(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    await tracker.on_search(_req(), _resp("ep_1"))
    data = _load_export(tracker)
    nodes = _nodes_by_name(data)
    assert not any(name.startswith("candidates_") for name in nodes)
    query_ids = [name for name in nodes if name.startswith("query-alice-")]
    assert len(query_ids) == 1
    query_fnid = _fnid(nodes[query_ids[0]][0])
    edges = _edge_pairs(data)
    # Legacy query -> hit edges (episode + its nested atomic fact).
    assert (query_fnid, _fnid(_hit_node(data, "alice_ep_1"))) in edges
    assert (query_fnid, _fnid(_hit_node(data, "alice_ep_1_f1"))) in edges
    # Hits feed the assembled context.
    context_fnid = _fnid(nodes[f"context-{query_ids[0]}"][0])
    assert (_fnid(_hit_node(data, "alice_ep_1")), context_fnid) in edges
    assert (_fnid(_hit_node(data, "alice_ep_1_f1")), context_fnid) in edges


async def test_on_search_with_stages_records_pools_and_chain(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    await tracker.on_search(_req(), _resp("ep_1"), stages=_stages())
    data = _load_export(tracker)
    nodes = _nodes_by_name(data)
    query_id = next(name for name in nodes if name.startswith("query-alice-"))
    query_fnid = _fnid(nodes[query_id][0])

    # One pool node per recorded stage, carrying the capped triple lists.
    for phase, count in (("sparse", 2), ("dense", 1), ("fusion", 1)):
        pool = nodes[f"candidates_episode_{phase}-{query_id}"][0]
        assert pool["category"] == "memory_context"
        value = json.loads(pool["value"])
        assert value["phase"] == phase and value["kind"] == "episode"
        assert value["count"] == count
        assert len(value["ids"]) == len(value["scores"]) == len(value["sources"])
    fusion_pool = nodes[f"candidates_episode_fusion-{query_id}"][0]
    assert fusion_pool["metadata"]["fusion_mode"] == "rrf"
    assert fusion_pool["metadata"]["enable_rerank"] is False

    edges = _edge_pairs(data)
    sparse_fnid = _fnid(nodes[f"candidates_episode_sparse-{query_id}"][0])
    dense_fnid = _fnid(nodes[f"candidates_episode_dense-{query_id}"][0])
    fusion_fnid = _fnid(fusion_pool)
    # query -> recall pools; recall pools -> fusion pool.
    assert (query_fnid, sparse_fnid) in edges
    assert (query_fnid, dense_fnid) in edges
    assert (sparse_fnid, fusion_fnid) in edges
    assert (dense_fnid, fusion_fnid) in edges
    # Intermediate pools retain their snapshots without expanding entries.
    assert (sparse_fnid, _fnid(_hit_node(data, "alice_ep_1"))) not in edges
    assert (sparse_fnid, "alice_ep_2@1") not in edges
    assert (dense_fnid, _fnid(_hit_node(data, "alice_ep_1"))) not in edges
    assert (fusion_fnid, _fnid(_hit_node(data, "alice_ep_1"))) in edges
    # Last pool -> every hit; no direct query -> hit edge anymore.
    hit_fnid = _fnid(_hit_node(data, "alice_ep_1"))
    fact_fnid = _fnid(_hit_node(data, "alice_ep_1_f1"))
    assert (fusion_fnid, hit_fnid) in edges
    assert (fusion_fnid, fact_fnid) in edges
    assert (query_fnid, hit_fnid) not in edges

    assert "alice_ep_2" not in nodes
    terminal_edges = [
        edge
        for edge in data["data"]["edges"]
        if edge["source_full_node_id"] == fusion_fnid
    ]
    assert Counter(e["target_full_node_id"] for e in terminal_edges) == {
        hit_fnid: 1,
        fact_fnid: 1,
    }
    listed = next(e for e in terminal_edges if e["target_full_node_id"] == hit_fnid)
    projected = next(e for e in terminal_edges if e["target_full_node_id"] == fact_fnid)
    assert listed["metadata"]["membership"] is True
    assert listed["metadata"]["score"] == 0.95
    assert "response_only" not in listed["metadata"]
    assert projected["metadata"]["response_only"] is True


@pytest.mark.parametrize("failed", [False, True])
async def test_only_terminal_pool_per_kind_expands_candidates(
    tracker: MemTraceTracker, failed: bool
) -> None:
    """Interleaved routes, including failures, expand only their last pools."""
    stages = [
        SearchStageRecord(
            phase=phase,
            kind=kind,
            candidates=(CandidateBrief(id=eid, score=0.8, source=""),),
        )
        for phase, kind, eid in (
            ("sparse", "episode", "ep_dropped"),
            ("dense", "agent_case", "case_dropped"),
            ("fusion", "episode", "ep_kept"),
            ("fusion", "agent_case", "case_kept"),
        )
    ]
    if failed:
        await tracker.on_search_error(
            _req(), error=RuntimeError("failed"), stages=stages
        )
    else:
        await tracker.on_search(_req(), _resp(), stages=stages)
    data = _load_export(tracker)
    nodes = _nodes_by_name(data)
    assert "alice_ep_dropped" not in nodes
    assert "alice_case_dropped" not in nodes
    for kind, eid in (("episode", "ep_kept"), ("agent_case", "case_kept")):
        pool = next(
            versions[0]
            for name, versions in nodes.items()
            if name.startswith(f"candidates_{kind}_fusion-")
        )
        entry = _hit_node(data, f"alice_{eid}")
        assert json.loads(entry["value"])["source"] == "candidate_pool"
        assert (_fnid(pool), _fnid(entry)) in _edge_pairs(data)


async def test_hierarchy_response_episode_stays_connected_without_hit_edge(
    tracker: MemTraceTracker,
) -> None:
    """Fact-only hierarchy snapshots still connect their response episode."""
    stage = SearchStageRecord(
        phase="hierarchy",
        kind="episode",
        candidates=(CandidateBrief(id="ep_1_f1", score=0.9, source=""),),
        total_count=1,
    )
    await tracker.on_search(_req(), _resp("ep_1"), stages=[stage])
    data = _load_export(tracker)
    nodes = _nodes_by_name(data)
    pool = next(v[0] for n, v in nodes.items() if n.startswith("candidates_"))
    assert json.loads(pool["value"])["ids"] == ["ep_1_f1"]
    outgoing = {
        edge["target_full_node_id"]: edge["metadata"]
        for edge in data["data"]["edges"]
        if edge["source_full_node_id"] == _fnid(pool)
    }
    assert len(outgoing) == 2
    assert outgoing[_fnid(_hit_node(data, "alice_ep_1"))]["response_only"] is True
    assert "response_only" not in outgoing[_fnid(_hit_node(data, "alice_ep_1_f1"))]
    for name in ("alice_ep_1", "alice_ep_1_f1"):
        hit = _hit_node(data, name)
        assert hit["version"] == 1
        assert any(
            edge["source_full_node_id"] == _fnid(hit)
            and edge["target_full_node_id"].startswith("context:")
            for edge in data["data"]["edges"]
        )


async def test_capped_terminal_pool_keeps_unlisted_returned_entries(
    tracker: MemTraceTracker,
) -> None:
    """Truncating the stage snapshot cannot disconnect an actual response."""
    stage = SearchStageRecord(
        phase="fusion",
        kind="episode",
        candidates=(CandidateBrief(id="ep_1", score=0.8, source=""),),
        total_count=120,
    )
    await tracker.on_search(_req(), _resp("ep_2"), stages=[stage])
    data = _load_export(tracker)
    nodes = _nodes_by_name(data)
    pool = next(v[0] for n, v in nodes.items() if n.startswith("candidates_"))
    assert pool["metadata"]["truncated"] is True
    assert json.loads(pool["value"])["ids"] == ["ep_1"]
    assert (_fnid(pool), _fnid(_hit_node(data, "alice_ep_1"))) in _edge_pairs(data)
    assert (_fnid(pool), _fnid(_hit_node(data, "alice_ep_2"))) in _edge_pairs(data)


async def test_duplicate_candidates_and_hits_have_one_membership_edge(
    tracker: MemTraceTracker,
) -> None:
    """Repeated identities cannot reintroduce duplicate terminal edges."""
    candidate = CandidateBrief(id="ep_1", score=0.8, source="")
    stage = SearchStageRecord(
        phase="fusion", kind="episode", candidates=(candidate, candidate)
    )
    await tracker.on_search(_req(), _resp("ep_1", "ep_1"), stages=[stage])
    data = _load_export(tracker)
    counts = Counter(
        edge["target_full_node_id"]
        for edge in data["data"]["edges"]
        if edge["source_full_node_id"].startswith("hybrid_retrieval_results:")
    )
    assert counts == {
        _fnid(_hit_node(data, "alice_ep_1")): 1,
        _fnid(_hit_node(data, "alice_ep_1_f1")): 1,
    }


async def test_candidate_stitches_onto_tracked_entry(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    """A candidate known to the file index keeps its full write-path snapshot.

    The entry node shares the ``{owner}_{entry_id}`` identity across the
    extraction graph (``alice`` / ``sess_a``) and the search graph
    (``alice`` / ``global``).
    """
    md_path = tmp_path / "users" / "alice" / "episodes" / "2026-01-01.md"
    md_path.parent.mkdir(parents=True)
    md_path.write_text(_EPISODE_MD, encoding="utf-8")

    response = _resp("ep_1")
    response.data.episodes[0].episode = "Alice went camping in July."
    await tracker.on_search(_req(), response, stages=_stages())

    # The write-path diff recorded the extraction into the session graph.
    extraction_nodes = _nodes_by_name(
        _load_export(tracker, ("default", "default", "alice", "sess_a"))
    )
    assert "alice_ep_1" in extraction_nodes

    search_nodes = _nodes_by_name(_load_export(tracker))
    entry = search_nodes["alice_ep_1"][0]
    value = json.loads(entry["value"])
    # Full tracked snapshot, not the lightweight candidate stub.
    assert value["content"] == "Alice went camping in July."
    assert value["kind"] == "episode"


async def test_atomic_fact_entry_carries_fact_text(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    """A kind-specific body section (``### Fact``) still reaches the graph.

    The AtomicFact writer emits ``### Fact`` instead of ``### Content``; the
    entry snapshot must fold it into ``content`` so the exported graph keeps
    the fact text for downstream attribution.
    """
    md_path = tmp_path / "users" / "alice" / ".atomic_facts" / "af-2026-01-01.md"
    md_path.parent.mkdir(parents=True)
    md_path.write_text(_ATOMIC_FACT_MD, encoding="utf-8")

    await tracker.on_search(_req(), _resp("ep_1"))

    extraction_nodes = _nodes_by_name(
        _load_export(tracker, ("default", "default", "alice", "sess_a"))
    )
    assert "alice_af_1" in extraction_nodes
    value = json.loads(extraction_nodes["alice_af_1"][0]["value"])
    assert value["content"] == "Alice went camping in July."
    assert value["kind"] == "atomic_fact"


# Atomic-fact daily log with the kind-specific ``### Fact`` body section
# (the AtomicFact writer emits no ``### Content``).
_ATOMIC_FACT_MD = """---
user_id: alice
---
<!-- entry:af_1 -->
**owner_id**: alice
**session_id**: sess_a
**timestamp**: 2026-01-01T00:00:00+00:00
**parent_type**: memcell
**parent_id**: mc_1
**sender_ids**: [alice]
### Fact
Alice went camping in July.
<!-- /entry:af_1 -->
"""


async def test_unprocessed_messages_link_into_context(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    unprocessed = [
        UnprocessedMessageDTO(
            id="m-1",
            session_id="sess_a",
            sender_id="alice",
            sender_name="Alice",
            role="user",
            content="still buffering",
            timestamp=_TS,
        )
    ]
    await tracker.on_search(_req(), _resp("ep_1", unprocessed=unprocessed))
    data = _load_export(tracker)
    nodes = _nodes_by_name(data)
    result = next(
        versions[0]
        for name, versions in nodes.items()
        if name.startswith("candidates_unprocessed_")
    )
    assert result["category"] == "retrieval_result"
    assert result["class_name"] == "retrieval_results"
    assert json.loads(result["value"])["items"][0]["content"] == "still buffering"
    assert "m-1" not in nodes
    query_id = next(name for name in nodes if name.startswith("query-alice-"))
    context_fnid = _fnid(nodes[f"context-{query_id}"][0])
    assert (_fnid(result), context_fnid) in _edge_pairs(data)


async def test_disabled_tracker_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EVEROS_MEMTRACE_ENABLED", "0")
    monkeypatch.setenv("EVEROS_MEMTRACE_DIR", str(tmp_path / "memtrace"))
    tracker = MemTraceTracker()
    await tracker.on_search(_req(), _resp("ep_1"), stages=_stages())
    assert not (tmp_path / "memtrace").exists()


async def test_tracing_is_off_unless_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("EVEROS_MEMTRACE_ENABLED")
    monkeypatch.setenv("EVEROS_MEMTRACE_DIR", str(tmp_path / "memtrace"))
    tracker = MemTraceTracker()
    assert not tracker.enabled
    await tracker.on_search(_req(), _resp("ep_1"), stages=_stages())
    assert not (tmp_path / "memtrace").exists()


async def test_empty_daily_log_records_removed_entries_as_deletions(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    md_path = tmp_path / "users" / "alice" / "episodes" / "2026-01-01.md"
    md_path.parent.mkdir(parents=True)
    md_path.write_text(_EPISODE_MD, encoding="utf-8")
    await tracker.on_search(_req(), _resp())

    md_path.write_text("---\nuser_id: alice\n---\n", encoding="utf-8")
    await tracker.on_search(_req(), _resp())

    nodes = _nodes_by_name(
        _load_export(tracker, ("default", "default", "alice", "sess_a"))
    )
    versions = sorted(nodes["alice_ep_1"], key=lambda node: node["version"])
    assert [node["version"] for node in versions] == [1, 2]
    assert json.loads(versions[-1]["value"])["deleted"] is True


async def test_deleted_whole_file_artifact_is_recorded(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    profile_path = tmp_path / "users" / "alice" / "user.md"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(
        "---\nuser_id: alice\n---\n# User profile\n", encoding="utf-8"
    )
    await tracker.on_search(_req(), _resp())

    profile_path.unlink()
    await tracker.on_search(_req(), _resp())

    nodes = _nodes_by_name(_load_export(tracker))
    versions = sorted(nodes["profile-alice"], key=lambda node: node["version"])
    assert [node["version"] for node in versions] == [1, 2]
    assert json.loads(versions[-1]["value"])["deleted"] is True


def test_parse_failure_is_retried_without_a_file_change(
    tracker: MemTraceTracker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    md_path = tmp_path / "users" / "alice" / "episodes" / "2026-01-01.md"
    md_path.parent.mkdir(parents=True)
    md_path.write_text(_EPISODE_MD, encoding="utf-8")
    tracker_mod = importlib.import_module("everos.infra.memtrace.tracker")
    real_parse = tracker_mod.parse_memory_file
    calls = 0

    def _flaky_parse(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("transient partial write")
        return real_parse(*args, **kwargs)

    monkeypatch.setattr(tracker_mod, "parse_memory_file", _flaky_parse)

    first_changed, _ = tracker._scan()
    second_changed, _ = tracker._scan()

    assert first_changed == []
    assert len(second_changed) == 1
    assert calls == 2


async def test_app_project_scopes_export_to_distinct_graphs(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    for app_id, content in (("app_a", "A camping trip."), ("app_b", "B trip.")):
        md_path = (
            tmp_path
            / app_id
            / "proj"
            / "users"
            / "alice"
            / "episodes"
            / "2026-01-01.md"
        )
        md_path.parent.mkdir(parents=True)
        md_path.write_text(
            _EPISODE_MD.replace("Alice went camping in July.", content),
            encoding="utf-8",
        )

    for app, text in (("app_a", "A camping trip."), ("app_b", "B trip.")):
        response = _resp("ep_1")
        response.data.episodes[0].episode = text
        await tracker.on_search(_req(app_id=app, project_id="proj"), response)

    graph_a = _nodes_by_name(
        _load_export(tracker, ("app_a", "proj", "alice", "global"))
    )
    graph_b = _nodes_by_name(
        _load_export(tracker, ("app_b", "proj", "alice", "global"))
    )
    assert json.loads(graph_a["alice_ep_1"][0]["value"])["content"] == (
        "A camping trip."
    )
    assert json.loads(graph_b["alice_ep_1"][0]["value"])["content"] == "B trip."


async def test_skill_main_file_uses_search_result_identity(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    skill_path = (
        tmp_path
        / "default_app"
        / "default_project"
        / "agents"
        / "bot"
        / "skills"
        / "plan"
        / "SKILL.md"
    )
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(
        "---\nagent_id: bot\nname: plan\n---\n# Plan\n", encoding="utf-8"
    )

    await tracker._sync_files()

    nodes = _nodes_by_name(
        _load_export(tracker, ("default", "default", "bot", "global"))
    )
    assert "bot_plan" in nodes
    assert (
        "artifact-bot-default_app/default_project/agents/bot/skills/plan/SKILL.md"
        not in nodes
    )


async def test_skill_reference_is_not_recorded_as_an_independent_skill(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    ref_path = (
        tmp_path
        / "default_app/default_project/agents/bot/skills/skill_plan"
        / "references/notes.md"
    )
    ref_path.parent.mkdir(parents=True)
    ref_path.write_text("Notes for the plan skill.\n", encoding="utf-8")

    await tracker._sync_files()

    nodes = _nodes_by_name(
        _load_export(tracker, ("default", "default", "bot", "global"))
    )
    artifact = next(iter(nodes.values()))[0]
    assert json.loads(artifact["value"])["kind"] == "skill_reference"


async def test_restart_reuses_unchanged_entry_and_records_later_update(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    md_path = tmp_path / "users" / "alice" / "episodes" / "2026-01-01.md"
    md_path.parent.mkdir(parents=True)
    md_path.write_text(_EPISODE_MD, encoding="utf-8")
    await tracker._sync_files()

    restarted = MemTraceTracker()
    await restarted._sync_files()
    session_key = ("default", "default", "alice", "sess_a")
    initial = _nodes_by_name(_load_export(restarted, session_key))
    assert len(initial["alice_ep_1"]) == 1

    md_path.write_text(
        _EPISODE_MD.replace("Alice went camping in July.", "Alice stayed home."),
        encoding="utf-8",
    )
    await restarted._sync_files()
    updated = _nodes_by_name(_load_export(restarted, session_key))
    assert len(updated["alice_ep_1"]) == 2
    assert "Alice stayed home." in updated["alice_ep_1"][-1]["value"]


async def test_restart_continues_the_existing_message_buffer(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    await tracker._record_messages(
        "default", "default", "sess_a", [_message("first", 1)]
    )
    await tracker._flush_dirty_graphs()
    restarted = MemTraceTracker()
    await restarted._record_messages(
        "default", "default", "sess_a", [_message("second", 2)]
    )
    await restarted._flush_dirty_graphs()

    nodes = _nodes_by_name(
        _load_export(restarted, ("default", "default", "alice", "sess_a"))
    )
    versions = sorted(nodes["buffer-sess_a-alice"], key=lambda n: n["version"])
    assert [node["version"] for node in versions] == [1, 2, 3]
    assert json.loads(versions[0]["value"])["message_ids"] == []
    assert len(json.loads(versions[-1]["value"])["message_ids"]) == 2


@pytest.mark.parametrize("restart", [False, True])
async def test_both_chat_roles_feed_each_buffer_version(
    tracker: MemTraceTracker, restart: bool
) -> None:
    """Both user and assistant inputs reach initialization and mutation."""
    expected_ids = []
    for batch in range(2):
        if batch and restart:
            tracker = MemTraceTracker()
        messages = [
            _message(f"user {batch}", _TS_MS + batch * 10),
            {
                **_message(
                    f"assistant {batch}", _TS_MS + batch * 10 + 1, sender="assistant"
                ),
                "role": "assistant",
            },
        ]
        await tracker.on_memorize(
            _payload(*messages),
            is_final=False,
            defer_extraction=True,
            result=None,
            request_id=f"add-{batch}",
        )
        batch_ids = [
            f"m_sess_a_{_TS_MS + batch * 10 + idx}_{idx:03d}" for idx in range(2)
        ]
        expected_ids.extend(batch_ids)
        for owner in ("alice", "assistant"):
            data = _load_export(tracker, ("default", "default", owner, "sess_a"))
            nodes = _nodes_by_name(data)
            buffer = max(nodes[f"buffer-sess_a-{owner}"], key=lambda n: n["version"])
            assert buffer["version"] == batch + 2
            assert json.loads(buffer["value"])["message_ids"] == expected_ids
            for message_id in batch_ids:
                assert (f"{message_id}@1", _fnid(buffer)) in _edge_pairs(data)
            ops = _ops_by_name(data)["memory_add_sess_a"]
            assert all(op["metadata"]["message_count"] == 2 for op in ops)


async def test_complete_message_batches_remain_scope_isolated(
    tracker: MemTraceTracker,
) -> None:
    """Full conversation buffers remain partitioned by app, project, session."""
    for idx, (app, project, session) in enumerate(
        (("a", "p", "s"), ("b", "p", "s"), ("a", "q", "s"), ("a", "p", "t"))
    ):
        messages = [
            _message("user", _TS_MS + idx * 10),
            {
                **_message("reply", _TS_MS + idx * 10 + 1, sender="assistant"),
                "role": "assistant",
            },
        ]
        await tracker.on_memorize(
            {
                "app_id": app,
                "project_id": project,
                "session_id": session,
                "messages": messages,
            },
            is_final=False,
            defer_extraction=True,
            result=None,
        )
        data = _load_export(tracker, (app, project, "alice", session))
        nodes = _nodes_by_name(data)
        buffer = json.loads(nodes[f"buffer-{session}-alice"][-1]["value"])
        assert buffer["message_ids"] == [
            f"m_{session}_{_TS_MS + idx * 10 + offset}_{offset:03d}"
            for offset in range(2)
        ]


async def test_shutdown_records_foresight_body_evidence_and_memcell(
    tracker: MemTraceTracker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late foresight write keeps its source chain and later file versions."""
    messages = [
        _message("I will go camping.", _TS_MS),
        {
            **_message("Pack warm clothes.", _TS_MS + 1, sender="assistant"),
            "role": "assistant",
        },
    ]
    await tracker.on_memorize(
        _payload(*messages), is_final=False, defer_extraction=True, result=None
    )
    message_ids = [f"m_sess_a_{_TS_MS + idx}_{idx:03d}" for idx in range(2)]

    async def find_boundary(ids: list[str]) -> list[SimpleNamespace]:
        assert ids == ["mc_1"]
        return [
            SimpleNamespace(
                app_id="default",
                project_id="default",
                message_ids_json=json.dumps(message_ids),
            )
        ]

    monkeypatch.setattr(memcell_repo, "find_by_ids", find_boundary)
    markdown = """---
user_id: alice
---
<!-- entry:fs_1 -->
**owner_id**: alice
**session_id**: sess_a
**timestamp**: 2026-01-01T00:00:00+00:00
**parent_type**: memcell
**parent_id**: mc_1
### Foresight
Alice may need warm clothes for camping.
### Evidence
Alice plans to camp in January.
<!-- /entry:fs_1 -->
"""
    path = tmp_path / "users" / "alice" / ".foresights" / "foresight-2026-01-01.md"
    path.parent.mkdir(parents=True)
    path.write_text(markdown, encoding="utf-8")
    await tracker.sync_now()
    key = ("default", "default", "alice", "sess_a")
    data = _load_export(tracker, key)
    nodes = _nodes_by_name(data)
    value = json.loads(nodes["alice_fs_1"][0]["value"])
    assert value["kind"] == "foresight"
    assert value["owner_id"] == "alice" and value["session_id"] == "sess_a"
    assert value["content"] == (
        "Alice may need warm clothes for camping.\n\nAlice plans to camp in January."
    )
    assert ("mc_1@1", "alice_fs_1@1") not in _edge_pairs(data)
    assert ("mc_1@1", "cell-input-mc_1@1") in _edge_pairs(data)
    assert ("cell-input-mc_1@1", "alice_fs_1@1") in _edge_pairs(data)
    assert nodes["mc_1"][0]["metadata"]["membership"] == "exact"
    for message_id in message_ids:
        assert (f"{message_id}@1", "mc_1@1") not in _edge_pairs(data)
    _assert_recovered_boundary(data, "buffer-sess_a-alice@2", "mc_1")
    op = _ops_by_name(data)["foresight_extraction_fs_1"][0]
    assert op["metadata"]["trigger"] == "shutdown"
    path.write_text(markdown.replace("warm clothes", "a tent"), encoding="utf-8")
    await tracker.sync_now()
    path.unlink()
    await tracker.sync_now()
    versions = _nodes_by_name(_load_export(tracker, key))["alice_fs_1"]
    assert [n["version"] for n in versions] == [1, 2, 3]
    assert json.loads(versions[-1]["value"])["deleted"] is True


async def test_graph_file_names_are_collision_proof(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    """Ids that ``_safe_name`` folds together still get distinct graph files."""
    owners = ("张三", "李四", "a@b", "a_b")
    for owner in owners:
        await tracker.on_search(_req(user_id=owner), _resp("ep_1"))

    expected = {
        tracker._graph_path(("default", "default", owner, "global")) for owner in owners
    }
    assert len(expected) == 4
    assert set((tmp_path / "memtrace").glob("*.json")) == expected
    # Each graph holds exactly its own owner's query.
    for owner in owners:
        path = tracker._graph_path(("default", "default", owner, "global"))
        nodes = _nodes_by_name(json.loads(path.read_text()))
        queries = [name for name in nodes if name.startswith("query-")]
        assert len(queries) == 1
        assert queries[0].startswith(f"query-{owner}-")


async def test_sync_files_exports_each_graph_once(
    tracker: MemTraceTracker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A diff round dirtying one graph for several entries exports it once."""
    exports: list[GraphKey] = []
    real_export = MemTraceTracker._export

    def counting_export(self: MemTraceTracker, key: GraphKey, graph: object) -> None:
        exports.append(key)
        real_export(self, key, graph)

    monkeypatch.setattr(MemTraceTracker, "_export", counting_export)
    md_path = tmp_path / "users" / "alice" / "episodes" / "2026-01-01.md"
    md_path.parent.mkdir(parents=True)
    md_path.write_text(_THREE_ENTRY_MD, encoding="utf-8")

    await tracker._sync_files()

    assert exports == [("default", "default", "alice", "sess_a")]


def _assert_recovered_boundary(data: dict, buffer_id: str, cell_id: str) -> None:
    """A recovered cell also has a result between its buffer and membership."""
    result_id = f"boundary_detection_result:boundary-observed-{cell_id}-v1@1"
    edges = _edge_pairs(data)
    assert (buffer_id, result_id) in edges
    assert (result_id, cell_id + "@1") in edges
    assert (buffer_id, cell_id + "@1") not in edges
    assert {source for source, target in edges if target == cell_id + "@1"} == {
        result_id
    }
    result = next(n for n in data["data"]["nodes"] if n["full_node_id"] == result_id)
    assert result["metadata"]["observation_only"] is True
    assert json.loads(result["value"])["detection_performed"] is None


async def test_memcell_membership_uses_recorded_boundary_data(
    tracker: MemTraceTracker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two memcells extracted in one round each link only their own messages."""
    message_ids = [f"m_sess_a_{_TS_MS + idx}_{idx:03d}" for idx in range(4)]
    boundary = {"mc_1": message_ids[:2], "mc_2": message_ids[2:]}

    async def fake_find_by_ids(ids: list[str]) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                app_id="default",
                project_id="default",
                message_ids_json=json.dumps(boundary[memcell_id]),
            )
            for memcell_id in ids
            if memcell_id in boundary
        ]

    monkeypatch.setattr(memcell_repo, "find_by_ids", fake_find_by_ids)
    md_path = tmp_path / "users" / "alice" / "episodes" / "2026-01-01.md"
    md_path.parent.mkdir(parents=True)
    md_path.write_text(_TWO_MEMCELL_MD, encoding="utf-8")

    payload = _payload(*[_message(f"msg {idx}", _TS_MS + idx) for idx in range(4)])
    await tracker.on_memorize(
        payload, is_final=False, defer_extraction=False, result=None
    )

    data = _load_export(tracker, ("default", "default", "alice", "sess_a"))
    nodes = _nodes_by_name(data)
    assert json.loads(nodes["mc_1"][0]["value"])["message_ids"] == message_ids[:2]
    assert json.loads(nodes["mc_2"][0]["value"])["message_ids"] == message_ids[2:]
    assert nodes["mc_1"][0]["metadata"]["membership"] == "exact"
    edges = _edge_pairs(data)
    assert (f"{message_ids[0]}@1", "mc_1@1") not in edges
    assert (f"{message_ids[1]}@1", "mc_1@1") not in edges
    assert (f"{message_ids[2]}@1", "mc_2@1") not in edges
    assert (f"{message_ids[3]}@1", "mc_2@1") not in edges
    # No cross-claiming: neither memcell links the other cell's messages.
    assert (f"{message_ids[2]}@1", "mc_1@1") not in edges
    assert (f"{message_ids[0]}@1", "mc_2@1") not in edges
    _assert_recovered_boundary(data, "buffer-sess_a-alice@2", "mc_1")
    _assert_recovered_boundary(data, "buffer-sess_a-alice@2", "mc_2")


@pytest.mark.parametrize("restart", [False, True])
async def test_foresights_reuse_one_memcell_boundary(
    tracker: MemTraceTracker,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restart: bool,
) -> None:
    """Derived entries cannot duplicate membership or invent sender-only versions."""
    messages = [
        _message("user", _TS_MS),
        {**_message("reply", _TS_MS + 1, sender="assistant"), "role": "assistant"},
    ]
    message_ids = [f"m_sess_a_{_TS_MS}_000", f"m_sess_a_{_TS_MS + 1}_001"]

    async def boundary(ids: list[str]) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                app_id="default",
                project_id="default",
                message_ids_json=json.dumps(message_ids),
                sender_ids_json=json.dumps(["alice", "assistant"]),
            )
        ]

    monkeypatch.setattr(memcell_repo, "find_by_ids", boundary)
    await tracker.on_memorize(
        _payload(*messages), is_final=False, defer_extraction=True, result=None
    )
    episode = tmp_path / "users/alice/episodes/2026-01-01.md"
    episode.parent.mkdir(parents=True)
    episode.write_text(_EPISODE_MD)
    await tracker.sync_now()
    if restart:
        tracker = MemTraceTracker()
    await tracker.on_memorize(
        _payload(_message("later pending message", _TS_MS + 10)),
        is_final=False,
        defer_extraction=True,
        result=None,
    )
    foresights = tmp_path / "users/alice/.foresights/2026-01-01.md"
    foresights.parent.mkdir(parents=True)
    foresights.write_text(
        "---\nuser_id: alice\n---\n"
        + "\n".join(
            f"<!-- entry:fs_{index} -->\n"
            "**owner_id**: alice\n**session_id**: sess_a\n"
            "**parent_type**: memcell\n**parent_id**: mc_1\n"
            f"### Foresight\nPrediction {index}.\n<!-- /entry:fs_{index} -->"
            for index in range(5)
        )
    )
    await tracker.sync_now()
    data = _load_export(tracker, ("default", "default", "alice", "sess_a"))
    nodes = _nodes_by_name(data)
    assert len(nodes["mc_1"]) == 1
    assert json.loads(nodes["mc_1"][0]["value"])["sender_ids"] == ["alice", "assistant"]
    assert len(_ops_by_name(data)["boundary_detection_mc_1"]) == 1
    counts = Counter(
        (edge["source_full_node_id"], edge["target_full_node_id"])
        for edge in data["data"]["edges"]
    )
    for message_id in message_ids:
        assert counts[(message_id + "@1", "mc_1@1")] == 0
    _assert_recovered_boundary(data, "buffer-sess_a-alice@2", "mc_1")
    assert counts[("buffer-sess_a-alice@3", "mc_1@1")] == 0
    assert counts[("buffer-sess_a-alice@2", "buffer-sess_a-alice@3")] == 1
    assert counts[("mc_1@1", "cell-input-mc_1@1")] == 1
    assert all(
        counts[("cell-input-mc_1@1", f"alice_fs_{index}@1")] == 1
        and counts[("mc_1@1", f"alice_fs_{index}@1")] == 0
        for index in range(5)
    )


async def test_later_boundary_uses_the_buffer_after_the_next_add(
    tracker: MemTraceTracker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two flush rounds attach different cells to the buffer each round consumed."""
    boundary = {
        "mc_1": [f"m_sess_a_{_TS_MS}_000"],
        "mc_2": [f"m_sess_a_{_TS_MS + 1}_000"],
    }

    async def lookup(ids: list[str]) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                app_id="default",
                project_id="default",
                message_ids_json=json.dumps(boundary[ids[0]]),
                sender_ids_json='["alice"]',
            )
        ]

    monkeypatch.setattr(memcell_repo, "find_by_ids", lookup)
    episode = tmp_path / "users/alice/episodes/2026-01-01.md"
    episode.parent.mkdir(parents=True)
    await tracker.on_memorize(
        _payload(_message("first", _TS_MS)),
        is_final=False,
        defer_extraction=True,
        result=None,
    )
    episode.write_text(_EPISODE_MD)
    await tracker.sync_now()
    await tracker.on_memorize(
        _payload(_message("second", _TS_MS + 1)),
        is_final=False,
        defer_extraction=True,
        result=None,
    )
    episode.write_text(_TWO_MEMCELL_MD)
    await tracker.sync_now()
    data = _load_export(tracker, ("default", "default", "alice", "sess_a"))
    edges = _edge_pairs(data)
    _assert_recovered_boundary(data, "buffer-sess_a-alice@2", "mc_1")
    _assert_recovered_boundary(data, "buffer-sess_a-alice@3", "mc_2")
    assert ("buffer-sess_a-alice@3", "mc_1@1") not in edges
    assert len(_ops_by_name(data)["boundary_detection_mc_1"]) == 1


async def test_exact_memcell_boundary_is_not_downgraded_by_later_lookup_failure(
    tracker: MemTraceTracker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exact boundary upgrades the approximation once and survives read errors."""
    state = "missing"

    async def lookup(ids: list[str]) -> list[SimpleNamespace]:
        if state == "missing":
            return []
        if state == "failed":
            raise RuntimeError("temporary read error")
        return [
            SimpleNamespace(
                app_id="default",
                project_id="default",
                message_ids_json=json.dumps([f"m_sess_a_{_TS_MS}_000"]),
                sender_ids_json='["alice"]',
            )
        ]

    monkeypatch.setattr(memcell_repo, "find_by_ids", lookup)
    await tracker.on_memorize(
        _payload(_message("first", _TS_MS)),
        is_final=False,
        defer_extraction=True,
        result=None,
    )
    episode = tmp_path / "users/alice/episodes/2026-01-01.md"
    episode.parent.mkdir(parents=True)
    episode.write_text(_EPISODE_MD)
    await tracker.sync_now()
    for index, next_state in enumerate(("exact", "failed"), 2):
        state = next_state
        episode.write_text(
            _EPISODE_MD
            + _EPISODE_MD.split("---\n", 2)[-1].replace("ep_1", f"ep_{index}")
        )
        await tracker.sync_now()
    data = _load_export(tracker, ("default", "default", "alice", "sess_a"))
    nodes = _nodes_by_name(data)
    assert len(nodes["mc_1"]) == 2
    assert nodes["mc_1"][-1]["metadata"]["membership"] == "exact"
    assert len(_ops_by_name(data)["boundary_detection_mc_1"]) == 2


async def test_memcell_membership_falls_back_with_approximate_flag(
    tracker: MemTraceTracker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a memcell row the session heuristic links unbound messages."""

    async def no_rows(ids: list[str]) -> list[SimpleNamespace]:
        return []

    monkeypatch.setattr(memcell_repo, "find_by_ids", no_rows)
    md_path = tmp_path / "users" / "alice" / "episodes" / "2026-01-01.md"
    md_path.parent.mkdir(parents=True)
    md_path.write_text(_EPISODE_MD, encoding="utf-8")

    payload = _payload(_message("first", _TS_MS), _message("second", _TS_MS + 1))
    await tracker.on_memorize(
        payload, is_final=False, defer_extraction=False, result=None
    )

    nodes = _nodes_by_name(
        _load_export(tracker, ("default", "default", "alice", "sess_a"))
    )
    memcell = json.loads(nodes["mc_1"][0]["value"])
    assert memcell["message_ids"] == [
        f"m_sess_a_{_TS_MS}_000",
        f"m_sess_a_{_TS_MS + 1}_001",
    ]
    assert nodes["mc_1"][0]["metadata"]["membership"] == "approximate"


async def test_restart_reconciles_downtime_deletion(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    """A file deleted while the process was down gets a deletion version."""
    md_path = tmp_path / "users" / "alice" / "episodes" / "2026-01-01.md"
    md_path.parent.mkdir(parents=True)
    md_path.write_text(_EPISODE_MD, encoding="utf-8")
    await tracker._sync_files()

    md_path.unlink()  # Deleted during "downtime".

    restarted = MemTraceTracker()
    await restarted._sync_files()

    nodes = _nodes_by_name(
        _load_export(restarted, ("default", "default", "alice", "sess_a"))
    )
    versions = sorted(nodes["alice_ep_1"], key=lambda node: node["version"])
    assert [node["version"] for node in versions] == [1, 2]
    assert json.loads(versions[-1]["value"])["deleted"] is True


async def test_concurrent_search_requests_share_one_graph(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    """Concurrent get-or-create for a fresh key must not lose updates."""
    await asyncio.gather(
        tracker.on_search(_req(), _resp("ep_1"), request_id="r" * 31 + "1"),
        tracker.on_search(_req(), _resp("ep_2"), request_id="r" * 31 + "2"),
    )

    nodes = _nodes_by_name(_load_export(tracker))
    query_ids = [name for name in nodes if name.startswith("query-alice-")]
    assert len(query_ids) == 2
    data = _load_export(tracker)
    assert _hit_node(data, "alice_ep_1")
    assert _hit_node(data, "alice_ep_2")


async def test_entry_move_between_files_records_no_update(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    """A pure cross-file move keeps the node version and updates the index."""
    first = tmp_path / "users" / "alice" / "episodes" / "2026-01-01.md"
    second = tmp_path / "users" / "alice" / "episodes" / "2026-01-02.md"
    first.parent.mkdir(parents=True)
    first.write_text(_EPISODE_MD, encoding="utf-8")
    await tracker._sync_files()

    first.unlink()
    second.write_text(_EPISODE_MD, encoding="utf-8")
    await tracker._sync_files()

    nodes = _nodes_by_name(
        _load_export(tracker, ("default", "default", "alice", "sess_a"))
    )
    assert [node["version"] for node in nodes["alice_ep_1"]] == [1]
    scoped = ("default", "default", "alice_ep_1")
    assert tracker._entry_file[scoped] == str(second.relative_to(tmp_path))


async def test_later_hit_records_content_after_candidate_stub(
    tracker: MemTraceTracker,
) -> None:
    stages = [
        SearchStageRecord(
            "sparse", "episode", (CandidateBrief("ep_9", 0.8, "keyword"),)
        )
    ]
    await tracker.on_search(_req(), _resp(), stages=stages, request_id="candidate")
    await tracker.on_search(_req(), _resp("ep_9"), request_id="hit")
    data = _load_export(tracker)
    stub = _hit_node(data, "alice_ep_9", "query-alice-candidate")
    hit = _hit_node(data, "alice_ep_9", "query-alice-hit")
    assert json.loads(stub["value"])["source"] == "candidate_pool"
    assert json.loads(hit["value"])["content"] == "body ep_9"
    assert _fnid(stub) != _fnid(hit)


async def test_memorize_message_identity_matches_ingest_format(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    """``/add`` messages carry the ingest ``m_<session>_<ts_ms>_<idx>`` id.

    Seconds / ISO-8601 inputs normalise onto the same millisecond identity;
    a message without a usable timestamp gets a stable fallback identity
    flagged ``timestamp_invalid`` instead of being dropped.
    """
    payload = _payload(
        _message("ms", _TS_MS),
        _message("seconds", _TS_MS // 1000),
        _message("iso", "2026-01-01T00:00:00+00:00"),
        {"sender_id": "alice", "role": "user", "content": "kept"},
    )
    await tracker.on_memorize(
        payload, is_final=False, defer_extraction=True, result=None
    )

    nodes = _nodes_by_name(
        _load_export(tracker, ("default", "default", "alice", "sess_a"))
    )
    base = f"m_sess_a_{_TS_MS}"
    assert f"{base}_000" in nodes
    assert f"{base}_001" in nodes
    assert f"{base}_002" in nodes
    fallback = json.loads(nodes["m_sess_a_invalid_003"][0]["value"])
    assert fallback["timestamp_invalid"] is True
    assert fallback["raw_timestamp"] is None
    assert fallback["content"] == "kept"
    buffer = json.loads(nodes["buffer-sess_a-alice"][-1]["value"])
    assert buffer["message_ids"] == [
        f"{base}_000",
        f"{base}_001",
        f"{base}_002",
        "m_sess_a_invalid_003",
    ]
    # Timestamps render as ISO strings that round-trip to the same instant.
    value = json.loads(nodes[f"{base}_000"][0]["value"])
    moment = _dt.datetime.fromisoformat(value["timestamp"])
    assert int(moment.timestamp() * 1000) == _TS_MS


def _ops_by_name(data: dict) -> dict[str, list[dict]]:
    by_name: dict[str, list[dict]] = {}
    for op in data["data"]["operations"]:
        by_name.setdefault(op["op_name"], []).append(op)
    return by_name


async def test_flush_records_request_level_op(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    """A flush gets its own ``memory_flush`` op wired to the session buffer."""
    payload = _payload(_message("first", _TS_MS))
    await tracker.on_memorize(
        payload, is_final=False, defer_extraction=True, result=None
    )
    result = SimpleNamespace(status="extracted", message_count=1)
    await tracker.on_memorize(
        _payload(),
        is_final=True,
        defer_extraction=False,
        result=result,
        request_id="req-flush",
    )

    data = _load_export(tracker, ("default", "default", "alice", "sess_a"))
    flush_ops = _ops_by_name(data)["memory_flush_sess_a"]
    assert len(flush_ops) == 1
    flush_op = flush_ops[0]
    assert flush_op["category"] == "extraction"
    metadata = flush_op["metadata"]
    assert metadata["is_final"] is True
    assert metadata["status"] == "extracted"
    assert metadata["message_count"] == 1
    assert metadata["request_id"] == "req-flush"
    assert metadata["event_seq"] >= 1
    assert "observed_at" in metadata
    # A unique receipt marks completion when the real buffer state is unavailable.
    edges = _edge_pairs(data)
    assert any(source.startswith("buffer-sess_a-alice@") for source, _ in edges)


async def test_empty_flush_is_recorded_in_unknown_graph(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    """A flush with no known buffer still leaves an input-less flush op."""
    result = SimpleNamespace(status="skipped", message_count=0)
    await tracker.on_memorize(
        _payload(), is_final=True, defer_extraction=False, result=result
    )

    data = _load_export(tracker, ("default", "default", "unknown", "sess_a"))
    flush_op = _ops_by_name(data)["memory_flush_sess_a"][0]
    assert flush_op["metadata"]["status"] == "skipped"
    assert flush_op["metadata"]["message_count"] == 0


async def test_on_memorize_error_records_error_op(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    """A failed add leaves a ``status: error`` request-level op."""
    await tracker.on_memorize_error(
        _payload(_message("x", _TS_MS)),
        is_final=False,
        defer_extraction=False,
        error=ValueError("pipeline blew up"),
        request_id="req-err",
    )

    data = _load_export(tracker, ("default", "default", "alice", "sess_a"))
    error_op = _ops_by_name(data)["memory_add_sess_a"][0]
    assert error_op["category"] == "initialization"
    metadata = error_op["metadata"]
    assert metadata["status"] == "error"
    assert metadata["error_type"] == "ValueError"
    assert metadata["error_message"] == "pipeline blew up"
    assert metadata["request_id"] == "req-err"
    assert metadata["is_final"] is False


async def test_on_memorize_error_flush_records_flush_op(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    """A failed flush records a ``memory_flush`` error op, not an add op."""
    await tracker.on_memorize_error(
        _payload(),
        is_final=True,
        defer_extraction=False,
        error=RuntimeError("flush failed"),
    )

    data = _load_export(tracker, ("default", "default", "unknown", "sess_a"))
    ops = _ops_by_name(data)
    assert "memory_add_sess_a" not in ops
    error_op = ops["memory_flush_sess_a"][0]
    assert error_op["category"] == "extraction"
    assert error_op["metadata"]["status"] == "error"
    assert error_op["metadata"]["error_type"] == "RuntimeError"


async def test_on_search_error_records_query_pools_and_error_op(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    """A failed search keeps the query node and the partial stage pools."""
    await tracker.on_search_error(
        _req(),
        stages=_stages(),
        error=RuntimeError("search blew up"),
        request_id="req-search-err",
    )

    data = _load_export(tracker)
    nodes = _nodes_by_name(data)
    query_id = "query-alice-req-search-err"
    query = json.loads(nodes[query_id][0]["value"])
    assert query["request_id"] == "req-search-err"
    assert nodes[query_id][0]["metadata"]["request_id"] == "req-search-err"
    # The pools collected before the failure are recorded and chained.
    assert f"candidates_episode_sparse-{query_id}" in nodes
    query_fnid = _fnid(nodes[query_id][0])
    sparse_fnid = _fnid(nodes[f"candidates_episode_sparse-{query_id}"][0])
    assert (query_fnid, sparse_fnid) in _edge_pairs(data)

    error_op = _ops_by_name(data)["memory_search_0001"][0]
    assert error_op["category"] == "retrieval"
    metadata = error_op["metadata"]
    assert metadata["status"] == "error"
    assert metadata["error_type"] == "RuntimeError"
    assert metadata["error_message"] == "search blew up"
    assert metadata["request_id"] == "req-search-err"


async def test_stage_pool_records_truncation_metadata(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    """A capped pool carries its pre-truncation size and the truncated flag."""
    stages = [
        SearchStageRecord(
            phase="sparse",
            kind="episode",
            candidates=(CandidateBrief(id="ep_1", score=0.8, source="keyword"),),
            total_count=120,
        )
    ]
    await tracker.on_search(_req(), _resp(), stages=stages, request_id="req-trunc")

    nodes = _nodes_by_name(_load_export(tracker))
    pool = nodes["candidates_episode_sparse-query-alice-req-trunc"][0]
    assert pool["metadata"]["total_count"] == 120
    assert pool["metadata"]["truncated"] is True
    value = json.loads(pool["value"])
    assert value["total_count"] == 120
    assert value["count"] == 1


async def test_query_identity_uses_request_id_with_collision_suffix(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    """Same request id twice: the second query gets a sequence suffix."""
    await tracker.on_search(_req(), _resp("ep_1"), request_id="req-1")
    await tracker.on_search(_req(), _resp("ep_1"), request_id="req-1")

    nodes = _nodes_by_name(_load_export(tracker))
    assert "query-alice-req-1" in nodes
    assert "query-alice-req-1-0002" in nodes


async def test_memcell_membership_marks_lookup_failed(
    tracker: MemTraceTracker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing SQLite read is marked ``lookup_failed``, not ``approximate``."""

    async def raising_find_by_ids(ids: list[str]) -> list[SimpleNamespace]:
        raise RuntimeError("db down")

    monkeypatch.setattr(memcell_repo, "find_by_ids", raising_find_by_ids)
    md_path = tmp_path / "users" / "alice" / "episodes" / "2026-01-01.md"
    md_path.parent.mkdir(parents=True)
    md_path.write_text(_EPISODE_MD, encoding="utf-8")

    payload = _payload(_message("first", _TS_MS))
    await tracker.on_memorize(
        payload, is_final=False, defer_extraction=False, result=None
    )

    nodes = _nodes_by_name(
        _load_export(tracker, ("default", "default", "alice", "sess_a"))
    )
    metadata = nodes["mc_1"][0]["metadata"]
    assert metadata["membership"] == "lookup_failed"
    assert metadata["membership_detail"] == "lookup_failed"
    # The heuristic still linked the session's unbound messages.
    memcell = json.loads(nodes["mc_1"][0]["value"])
    assert memcell["message_ids"] == [f"m_sess_a_{_TS_MS}_000"]


async def test_memcell_membership_marks_invalid_payload(
    tracker: MemTraceTracker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unparsable memcell row falls back with ``invalid_payload`` detail."""

    async def fake_find_by_ids(ids: list[str]) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                app_id="default", project_id="default", message_ids_json="not-json"
            )
        ]

    monkeypatch.setattr(memcell_repo, "find_by_ids", fake_find_by_ids)
    md_path = tmp_path / "users" / "alice" / "episodes" / "2026-01-01.md"
    md_path.parent.mkdir(parents=True)
    md_path.write_text(_EPISODE_MD, encoding="utf-8")

    payload = _payload(_message("first", _TS_MS))
    await tracker.on_memorize(
        payload, is_final=False, defer_extraction=False, result=None
    )

    nodes = _nodes_by_name(
        _load_export(tracker, ("default", "default", "alice", "sess_a"))
    )
    metadata = nodes["mc_1"][0]["metadata"]
    assert metadata["membership"] == "approximate"
    assert metadata["membership_detail"] == "invalid_payload"


async def test_graph_export_carries_digest_graph_id(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    """``graph_id`` carries the same ``{safe}-{digest}`` fragments as the file."""
    await tracker.on_search(_req(user_id="a@b"), _resp("ep_1"))
    data = _load_export(tracker, ("default", "default", "a@b", "global"))
    # Every one of the four key-component fragments ends in an 8-char digest.
    assert re.search(r"^everos-filetrace-(?:.+-[0-9a-f]{8}){4}$", data["graph_id"])
    assert "-7505d64a-" in data["graph_id"]  # sha1("default")[:8]


async def test_sync_now_flushes_dirty_graphs(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    """Shutdown sync diffs the tree and exports with ``trigger: shutdown``."""
    md_path = tmp_path / "users" / "alice" / "episodes" / "2026-01-01.md"
    md_path.parent.mkdir(parents=True)
    md_path.write_text(_EPISODE_MD, encoding="utf-8")

    await tracker.sync_now()

    data = _load_export(tracker, ("default", "default", "alice", "sess_a"))
    assert "alice_ep_1" in _nodes_by_name(data)
    extraction_op = _ops_by_name(data)["episode_extraction_ep_1"][0]
    metadata = extraction_op["metadata"]
    assert metadata["source"] == "file_diff"
    assert metadata["trigger"] == "shutdown"
    assert metadata["event_seq"] >= 1
    assert "observed_at" in metadata


async def test_stage_pools_are_explicit_retrieval_ops(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    """Every stage pool is produced by its own retrieval op (native style)."""
    await tracker.on_search(
        _req(), _resp("ep_1"), stages=_stages(), request_id="req-ops"
    )
    data = _load_export(tracker)
    nodes = _nodes_by_name(data)
    query_id = next(name for name in nodes if name.startswith("query-alice-"))

    ops = {op["op_name"]: op for op in data["data"]["operations"]}
    edges_by_op: dict[str, set[tuple[str, str]]] = {}
    for edge in data["data"]["edges"]:
        edges_by_op.setdefault(edge["op_id"], set()).add(
            (edge["source_full_node_id"], edge["target_full_node_id"])
        )

    query_fnid = _fnid(nodes[query_id][0])
    sparse_fnid = _fnid(nodes[f"candidates_episode_sparse-{query_id}"][0])
    dense_fnid = _fnid(nodes[f"candidates_episode_dense-{query_id}"][0])
    fusion_fnid = _fnid(nodes[f"candidates_episode_fusion-{query_id}"][0])
    expected = {
        "episode_sparse_0001": (
            {(query_fnid, sparse_fnid)},
            {"phase": "sparse", "kind": "episode"},
        ),
        "episode_dense_0001": (
            {(query_fnid, dense_fnid)},
            {"phase": "dense", "kind": "episode"},
        ),
        "episode_fusion_0001": (
            {(sparse_fnid, fusion_fnid), (dense_fnid, fusion_fnid)},
            {"phase": "fusion", "kind": "episode"},
        ),
    }
    for op_name, (edge_pairs, metadata) in expected.items():
        op = ops[op_name]
        assert op["category"] == "retrieval"
        assert edges_by_op[op["op_id"]] == edge_pairs
        for key, value in metadata.items():
            assert op["metadata"][key] == value
        assert op["metadata"]["event_seq"] >= 1
        assert op["metadata"]["request_id"] == "req-ops"


async def test_stage_pool_class_names_follow_native_vocabulary(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    """Pool nodes keep ``memory_context`` but carry a native-style class."""
    await tracker.on_search(_req(), _resp("ep_1"), stages=_stages())
    nodes = _nodes_by_name(_load_export(tracker))
    query_id = next(name for name in nodes if name.startswith("query-alice-"))
    expected = {
        "sparse": "sparse_retrieval_results",
        "dense": "dense_retrieval_results",
        "fusion": "hybrid_retrieval_results",
    }
    for phase, class_name in expected.items():
        pool = nodes[f"candidates_episode_{phase}-{query_id}"][0]
        assert pool["category"] == "memory_context"
        assert pool["class_name"] == class_name


async def test_on_answer_and_evaluation_record_qa_chain(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    """The QA hooks append the native prediction / evaluation chain."""
    await tracker.on_search(
        _req(), _resp("ep_1"), stages=_stages(), request_id="req-qa"
    )
    await tracker.on_answer(
        app_id="default",
        project_id="default",
        owner="alice",
        request_id="req-qa",
        answer="She went camping in July.",
        prompt="answer prompt text",
    )
    await tracker.on_evaluation(
        app_id="default",
        project_id="default",
        owner="alice",
        request_id="req-qa",
        golden_answers=["July"],
        judge_response="VERDICT: 1 correct",
        judge_score=1.0,
    )

    data = _load_export(tracker)
    nodes = _nodes_by_name(data)
    query_id = "query-alice-req-qa"
    prompt_node = nodes[f"prompt-{query_id}"][0]
    assert prompt_node["category"] == "prompt"
    assert json.loads(prompt_node["value"])["prompt"] == "answer prompt text"
    prediction = nodes[f"prediction-{query_id}"][0]
    assert prediction["category"] == "llm_response"
    assert prediction["class_name"] == "prediction"
    assert json.loads(prediction["value"])["answer"] == "She went camping in July."
    golden = nodes[f"golden-{query_id}"][0]
    assert golden["category"] == "golden_answers"
    assert golden["class_name"] == "golden_answers"
    assert json.loads(golden["value"])["answers"] == ["July"]
    judge = nodes[f"judge-{query_id}"][0]
    assert judge["class_name"] == "judge_response"
    assert json.loads(judge["value"])["score"] == 1.0

    edges = _edge_pairs(data)
    query_fnid = _fnid(nodes[query_id][0])
    context_fnid = _fnid(nodes[f"context-{query_id}"][0])
    prompt_fnid = _fnid(prompt_node)
    prediction_fnid = _fnid(prediction)
    golden_fnid = _fnid(golden)
    judge_fnid = _fnid(judge)
    assert (query_fnid, prediction_fnid) in edges
    answer_context_fnid = _fnid(nodes[f"answer-context-{query_id}"][0])
    assert (context_fnid, answer_context_fnid) in edges
    assert (answer_context_fnid, prediction_fnid) in edges
    assert (context_fnid, prediction_fnid) not in edges
    assert (prompt_fnid, prediction_fnid) in edges
    assert (query_fnid, golden_fnid) in edges
    assert (prediction_fnid, judge_fnid) in edges
    assert (golden_fnid, judge_fnid) in edges

    ops = {op["op_name"]: op for op in data["data"]["operations"]}
    assert ops[f"answer_generation_{query_id}"]["category"] == "response"
    assert ops[f"inject_golden_answers_{query_id}"]["category"] == "evaluation"
    assert ops[f"judge_answer_{query_id}"]["category"] == "evaluation"


async def test_qa_hooks_fail_safe_on_unknown_request(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    """A QA record for an unrecorded search is a logged no-op, not an error."""
    await tracker.on_answer(
        app_id="default",
        project_id="default",
        owner="alice",
        request_id="req-missing",
        answer="x",
    )
    await tracker.on_evaluation(
        app_id="default",
        project_id="default",
        owner="alice",
        request_id="req-missing",
        golden_answers=["y"],
    )
    assert not tracker._graph_path(("default", "default", "alice", "global")).exists()


async def test_qa_hooks_resolve_collision_suffixed_query(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    """A repeated request id resolves to the base query node deterministically."""
    await tracker.on_search(_req(), _resp("ep_1"), request_id="req-dup")
    await tracker.on_search(_req(), _resp("ep_1"), request_id="req-dup")
    await tracker.on_answer(
        app_id="default",
        project_id="default",
        owner="alice",
        request_id="req-dup",
        answer="a",
    )
    nodes = _nodes_by_name(_load_export(tracker))
    assert "query-alice-req-dup" in nodes
    assert "query-alice-req-dup-0002" in nodes
    assert "prediction-query-alice-req-dup" not in nodes
    await tracker.on_answer(
        app_id="default",
        project_id="default",
        owner="alice",
        request_id="req-dup",
        answer="a",
        query_full_node_id="query:query-alice-req-dup-0002@1",
    )
    nodes = _nodes_by_name(_load_export(tracker))
    assert "prediction-query-alice-req-dup-0002" in nodes
