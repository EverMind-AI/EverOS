"""Unit tests for ``SearchManager`` with in-memory scripted indexes.

These tests exercise the orchestration without touching LanceDB: every index
is a scripted stand-in and the decider is a scripted model. Multi-round
retrieval is the only route; the manager's job is to:

* resolve the read scope and pin it into every recall's ``where``,
* run the multi-round loop through its own decider client,
* honour ``top_k`` (``-1`` caps at 100),
* report the top score without a hit verdict (the multi-round score is not
  calibrated),
* attach the pictures a selected episode refers to without padding them into
  the ranking.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

import pytest
from everalgo.testing import FakeLLMClient

from everos.component.tokenizer import Tokenizer
from everos.memory.access import AccessPolicy
from everos.memory.models import CanonicalMessage, ImageMemory
from everos.memory.ports import MemoryRef, Scope, SearchHit, SessionKey, Stored
from everos.memory.search.dto import SearchRequest
from everos.memory.search.manager import SearchManager
from everos.memory.search.recall import EpisodeRecaller
from everos.memory.search.recall.base import RecallerDeps
from tests.fakes import InMemoryGroupStore, static_trace_path, static_tuning
from tests.fakes._filter import matches

from ._records import ScriptedIndex, hit, stored_episode

# ── Stubs ───────────────────────────────────────────────────────────────


def _keyed(stored: Any, record_id: str) -> Any:
    """Give a stored record the global id ``record_id``."""
    return dataclasses.replace(
        stored, ref=dataclasses.replace(stored.ref, id=record_id)
    )


def _episode_row(eid: str, score: float = 0.8) -> SearchHit[Any]:
    """An episode match whose global id and entry id are both ``eid``."""
    return hit(_keyed(stored_episode(eid, parent_id=f"mc_{eid}"), eid), score)


def _tokenizer() -> Tokenizer:
    tok = MagicMock(spec=Tokenizer)
    tok.tokenize.return_value = ["hi"]
    return tok


class _StubProfileRecaller:
    async def fetch(self, owner_id: str, **_: Any) -> list[Any]:
        return []


class _EmptySessionBuffer:
    """No session has buffered messages."""

    async def list(self, session: SessionKey) -> list[CanonicalMessage]:
        return []


class _StubEmbedding:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    async def embed(self, text: str) -> list[float]:
        return [0.1] * self.dim

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1] * self.dim for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return await self.embed(text)


class _Decider:
    """Keeps the first ``keep`` candidates and stops; records every prompt."""

    def __init__(self, keep: int = 10) -> None:
        self.keep = keep
        self.prompts: list[str] = []

    def __call__(self, messages: Any, **_kwargs: Any) -> str:
        self.prompts.extend(str(getattr(m, "content", "")) for m in messages)
        return json.dumps({"core": list(range(self.keep)), "next_queries": []})


def _never_called(messages: Any, **_kwargs: Any) -> str:
    raise AssertionError("a user search never calls the [llm] client")


# ── Fixtures ────────────────────────────────────────────────────────────


def _build_manager(
    *,
    episodes: list[SearchHit[Any]] | None = None,
    index: ScriptedIndex | None = None,
    embedding: Any = None,
    decider: _Decider | None = None,
    image_recaller: Any = None,
    image_store: Any = None,
) -> SearchManager:
    """A manager over the real episode recaller reading a scripted index.

    ``episodes`` answer both the BM25 and the vector recall. The embedding
    defaults to a stub (multi-round needs one); pass ``embedding=False`` for
    none.
    """
    rows = episodes or []
    return SearchManager(
        access=AccessPolicy(InMemoryGroupStore()),
        episode_recaller=EpisodeRecaller(
            RecallerDeps(tokenizer=_tokenizer()),
            index or ScriptedIndex(sparse=rows, dense=rows, found=rows),  # type: ignore[arg-type]
        ),
        profile_recaller=_StubProfileRecaller(),  # type: ignore[arg-type]
        session_buffer=_EmptySessionBuffer(),  # type: ignore[arg-type]
        embedding=None if embedding is False else (embedding or _StubEmbedding()),
        # Only a group search's constraint binding calls the [llm] client.
        llm_client=FakeLLMClient(handler=_never_called),
        decider_client=FakeLLMClient(handler=decider or _Decider()),
        tuning_source=static_tuning(),
        trace_path_source=static_trace_path(),
        image_recaller=image_recaller,
        image_store=image_store,
    )


def _user_req(**kwargs: Any) -> SearchRequest:
    return SearchRequest(user_id="alice", query="hi", **kwargs)


# ── Multi-round route ───────────────────────────────────────────────────


async def test_user_search_returns_the_deciders_episodes() -> None:
    decider = _Decider()
    mgr = _build_manager(episodes=[_episode_row("ep_1")], decider=decider)
    resp = await mgr.search(_user_req())
    assert len(resp.request_id) == 32 and all(
        c in "0123456789abcdef" for c in resp.request_id
    )
    assert [e.id for e in resp.data.episodes] == ["ep_1"]
    assert resp.data.episodes[0].user_id == "alice"
    assert resp.data.episodes[0].type == "Conversation"
    assert resp.data.profiles == []
    # The decider ran on the question and saw the recalled episode.
    assert decider.prompts and "hi" in decider.prompts[0]
    assert "body ep_1" in decider.prompts[0] or "ep_1" in decider.prompts[0]


async def test_search_uses_propagated_request_id_when_bound() -> None:
    """When a request id is bound upstream (middleware), ``search`` reuses it
    instead of minting a fresh one, so the response id matches the trace."""
    from everos.core.context import reset_request_id, set_request_id

    mgr = _build_manager(episodes=[_episode_row("ep_1")])
    token = set_request_id("deadbeef" * 4)
    try:
        resp = await mgr.search(_user_req())
        assert resp.request_id == "deadbeef" * 4
    finally:
        reset_request_id(token)


async def test_no_results() -> None:
    resp = await _build_manager().search(_user_req())
    assert resp.data.episodes == [] and resp.data.images == []


async def test_every_recall_is_pinned_to_the_owner() -> None:
    """The read scope's predicate is in every ``where`` the loop sends."""
    from everos.infra.persistence.lancedb import render_predicate

    index = ScriptedIndex(sparse=[_episode_row("ep_1")], dense=[_episode_row("ep_1")])
    await _build_manager(index=index).search(_user_req())
    searches = [call for call in index.calls if call[0] in ("sparse", "dense")]
    assert {call[0] for call in searches} == {"sparse", "dense"}
    for call in searches:
        rendered = render_predicate(call[2])
        assert "owner_id = 'alice'" in rendered
        assert "owner_type = 'user'" in rendered


async def test_search_without_embedding_recalls_by_keyword_only() -> None:
    """A deployment without an embedding model searches by BM25 alone.

    It used to refuse every search (422 ``PROVIDER_NOT_CONFIGURED``). The loop,
    the decider and the ranking are the same; only the vector leg is absent.
    """
    index = ScriptedIndex(
        sparse=[_episode_row("ep_1")], dense=[_episode_row("ep_dense_only")]
    )
    decider = _Decider()
    mgr = _build_manager(index=index, embedding=False, decider=decider)

    resp = await mgr.search(_user_req())

    assert [e.id for e in resp.data.episodes] == ["ep_1"]
    assert resp.data.degraded == []
    assert {call[0] for call in index.calls} == {"sparse"}
    assert decider.prompts, "the decider still decides"


async def test_embed_query_rejects_a_width_that_disagrees_with_dim() -> None:
    """A provider returning a width other than its declared ``dim`` must fail
    at the embed step, not inside LanceDB.

    Sending a mismatched query vector into the engine surfaces as an opaque
    ``ValueError: Invalid input, No vector column found to match…`` only after
    the query has been set up — 13-14s per request in a soak run, as an
    unhandled 500 with a ~6k-line traceback. Here it is microseconds and a
    named error. ``ConfigurationError`` (server-side) rather than
    ``InvalidInputError``: callers only ever send query *text*, so the width is
    entirely our provider's doing.
    """
    from everos.core.errors import ConfigurationError

    class _LyingEmbedding(_StubEmbedding):
        async def embed(self, text: str) -> list[float]:
            return [0.0] * (self.dim // 2)  # declares dim, returns half of it

    mgr = _build_manager(embedding=_LyingEmbedding(dim=8))

    with pytest.raises(ConfigurationError) as excinfo:
        await mgr._embed_query("anything")

    msg = str(excinfo.value)
    assert "4-dimension" in msg, msg
    assert "dim=8" in msg, msg


async def test_a_mismatched_width_fails_the_search_by_name() -> None:
    """The width check guards the multi-round recall, not only the helper."""
    from everos.core.errors import ConfigurationError

    class _LyingEmbedding(_StubEmbedding):
        async def embed(self, text: str) -> list[float]:
            return [0.0] * (self.dim // 2)

    mgr = _build_manager(
        episodes=[_episode_row("ep_1")], embedding=_LyingEmbedding(dim=8)
    )
    with pytest.raises(ConfigurationError):
        await mgr.search(_user_req())


async def test_top_k_truncates_results() -> None:
    rows = [_episode_row(f"ep_{i}", score=0.9 - i / 100) for i in range(5)]
    resp = await _build_manager(episodes=rows).search(_user_req(top_k=2))
    assert len(resp.data.episodes) == 2


async def test_top_k_minus_one_caps_at_100() -> None:
    assert SearchManager._top_k(-1) == 100
    assert SearchManager._top_k(7) == 7


async def test_recall_scores_carry_no_hit_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The multi-round score is the max RRF score across sub-queries, not a
    calibrated relevance: the top score is reported, a hit verdict never."""
    import everos.memory.search.manager as mgr_mod

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(mgr_mod, "current_trace_ids", lambda: ("t" * 32, "s" * 16))
    monkeypatch.setattr(mgr_mod, "emit_recall_scores", lambda **kw: calls.append(kw))

    resp = await _build_manager(episodes=[_episode_row("ep_1")]).search(_user_req())
    assert len(calls) == 1
    assert calls[0]["method"] == "llm_multiround"
    assert calls[0]["hit"] is None
    assert calls[0]["top_score"] == pytest.approx(resp.data.episodes[0].score)


async def test_search_emits_memory_search_span() -> None:
    """search() opens an everos.memory.search retriever span carrying the
    langfuse.* attribute contract (observation type / user id / metadata)."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from everos.config.settings import ObservabilitySettings
    from everos.core.observability.tracing import (
        force_flush,
        init_tracing,
        shutdown_tracing,
    )

    exporter = InMemorySpanExporter()
    init_tracing(
        ObservabilitySettings(enabled=True, endpoint="http://collector.invalid"),
        span_processor=SimpleSpanProcessor(exporter),
    )
    try:
        mgr = _build_manager(episodes=[_episode_row("ep_1")])
        await mgr.search(_user_req())
        force_flush()
        spans = {s.name: s for s in exporter.get_finished_spans()}
        assert "everos.memory.search" in spans
        attrs = spans["everos.memory.search"].attributes
        assert attrs["langfuse.observation.type"] == "retriever"
        assert attrs["langfuse.user.id"] == "alice"
        assert attrs["langfuse.trace.metadata.owner_type"] == "user"
        assert list(attrs["langfuse.trace.tags"]) == ["everos", "memory"]
    finally:
        shutdown_tracing()


@pytest.fixture
def _search_spans():  # type: ignore[no-untyped-def]
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from everos.config.settings import ObservabilitySettings
    from everos.core.observability.tracing import init_tracing, shutdown_tracing

    exporter = InMemorySpanExporter()
    shutdown_tracing()
    init_tracing(
        ObservabilitySettings(enabled=True, endpoint="http://collector.invalid"),
        span_processor=SimpleSpanProcessor(exporter),
    )
    yield exporter
    shutdown_tracing()


def _span_index(exporter: Any) -> dict[str, Any]:
    from everos.core.observability.tracing import force_flush

    force_flush()
    return {s.name: s for s in exporter.get_finished_spans()}


async def test_search_sets_top_score_without_hit(_search_spans: Any) -> None:
    mgr = _build_manager(episodes=[_episode_row("ep_1")])
    resp = await mgr.search(_user_req())
    attrs = _span_index(_search_spans)["everos.memory.search"].attributes
    assert attrs["everos.search.top_score"] == pytest.approx(
        resp.data.episodes[0].score
    )
    assert "everos.search.hit" not in attrs


async def test_search_top_score_zero_without_results(_search_spans: Any) -> None:
    await _build_manager().search(_user_req())
    attrs = _span_index(_search_spans)["everos.memory.search"].attributes
    assert attrs["everos.search.top_score"] == pytest.approx(0.0)
    assert "everos.search.hit" not in attrs


async def test_search_enqueues_recall_scores(
    _search_spans: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When tracing is active, search hands the top score to the score sink
    with the retriever span's trace_id (032x) + observation_id (016x)."""
    import everos.memory.search.manager as mgr_mod

    captured: dict[str, Any] = {}

    def fake_emit(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(mgr_mod, "emit_recall_scores", fake_emit)
    await _build_manager(episodes=[_episode_row("ep_1")]).search(_user_req())

    assert captured["hit"] is None
    assert captured["method"] == "llm_multiround"
    assert len(captured["trace_id"]) == 32
    assert len(captured["observation_id"]) == 16


async def test_search_captures_query_when_content_on(_search_spans: Any) -> None:
    from everos.core.observability.tracing import set_capture_content

    set_capture_content(True)
    try:
        await _build_manager(episodes=[_episode_row("ep_1")]).search(_user_req())
    finally:
        set_capture_content(False)

    attrs = _span_index(_search_spans)["everos.memory.search"].attributes
    assert json.loads(attrs["langfuse.observation.input"])["query"] == "hi"


async def test_search_captures_returned_hits_when_content_on(
    _search_spans: Any,
) -> None:
    """capture_content on → the search span records the returned episode ids,
    not just the query."""
    from everos.core.observability.tracing import set_capture_content

    set_capture_content(True)
    try:
        await _build_manager(episodes=[_episode_row("ep_1")]).search(_user_req())
    finally:
        set_capture_content(False)

    attrs = _span_index(_search_spans)["everos.memory.search"].attributes
    assert json.loads(attrs["langfuse.observation.output"]) == {"episodes": ["ep_1"]}


async def test_search_omits_query_when_content_off(_search_spans: Any) -> None:
    await _build_manager(episodes=[_episode_row("ep_1")]).search(_user_req())
    attrs = _span_index(_search_spans)["everos.memory.search"].attributes
    assert "langfuse.observation.input" not in attrs


class _PictureIndex:
    """The picture index as far as resolution needs it: current rows by file."""

    def __init__(self, *rows: Stored[ImageMemory]) -> None:
        self.rows = list(rows)

    async def sparse_search(self, terms, where, *, limit: int):
        return []

    async def dense_search(self, vector, where, *, limit: int):
        return []

    async def find(self, where, *, limit: int | None = None):
        return [
            SearchHit(stored=row, score=None)
            for row in self.rows
            if matches(where, _picture_columns(row))
        ][:limit]


def _picture_columns(row: Stored[ImageMemory]) -> dict[str, Any]:
    """The index columns a read scope's predicate filters a picture on."""
    return {
        "owner_id": row.scope.owner_id,
        "owner_type": row.scope.owner_type,
        "app_id": row.scope.app_id,
        "project_id": row.scope.project_id,
        "image_id": row.item.image_id,
        "deprecated_by": row.item.superseded_by,
    }


async def test_a_referenced_picture_is_attached_and_not_padded_into_the_ranking() -> (
    None
):
    """The ranked list is what the ranking chose; a reference is content, not a hit.

    An episode kept without its pictures must still reach the reader with their
    description and in-image text, and those pictures must not appear in
    ``data.images`` as if a ranker had selected them.
    """
    from everos.memory.search.recall import ImageMemoryRecaller

    image_id = "a" * 64
    stored = Stored(
        ref=MemoryRef(id="alice_im_1", entry_id="im_1", md_path="images.md"),
        scope=Scope(
            app_id="default", project_id="default", owner_id="alice", owner_type="user"
        ),
        item=ImageMemory(
            owner_id="alice",
            description="a sign reading EST. 2019",
            timestamp=1_747_216_800_000,
            ocr="EST. 2019",
            image_id=image_id,
            original_relpath=f"figures/aa/{image_id}.png",
            parent_id="mc_1",
            entry_id="im_1",
        ),
    )
    episode = _episode_row("ep_1")
    episode.stored.item.image_ids = [image_id]
    manager = _build_manager(
        episodes=[episode],
        image_recaller=ImageMemoryRecaller(
            RecallerDeps(tokenizer=_tokenizer()),  # type: ignore[arg-type]
            _PictureIndex(stored),  # type: ignore[arg-type]
        ),
    )

    response = await manager.search(_user_req())

    assert response.data.images == []
    attached = response.data.episodes[0].images
    assert [item.image_id for item in attached] == [image_id]
    assert attached[0].ocr == "EST. 2019"
