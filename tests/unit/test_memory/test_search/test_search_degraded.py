"""A search says when the decider did not decide a round, and why.

Before, a round whose every decider attempt failed fell back to the top
``[decider] fallback_core`` fused candidates and the search answered 200 like
any other: the reason was in the server log only. Twelve sweep arms once
reported 87-93% while every decider call 404'd. The response now carries a
``degraded`` entry per undecided round, with the last error as the reason;
a decided search carries none.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from everalgo.llm import LLMMessage
from everalgo.testing import FakeLLMClient

from everos.component.llm.openai_provider import RawReply
from everos.component.tokenizer import Tokenizer
from everos.config.settings import DeciderSettings
from everos.memory.access import AccessPolicy
from everos.memory.models import CanonicalMessage
from everos.memory.ports import SearchHit, SessionKey, all_of, eq
from everos.memory.search.dto import SearchDegradation, SearchRequest
from everos.memory.search.llm_multiround import (
    DeciderFailure,
    RoundDecision,
    _failure_of,
    run_llm_multiround,
)
from everos.memory.search.manager import SearchManager
from everos.memory.search.recall import EpisodeRecaller
from everos.memory.search.recall.base import RecallerDeps
from tests.fakes import InMemoryGroupStore, static_trace_path, static_tuning

from ._records import ScriptedIndex, hit, stored_episode

_FAST = DeciderSettings(retries=1, retry_backoff_seconds=0.0)
"""Two attempts per round and no backoff sleep."""


def _row(eid: str, score: float = 0.8) -> SearchHit[Any]:
    stored = stored_episode(eid, parent_id=f"mc_{eid}")
    keyed = dataclasses.replace(stored, ref=dataclasses.replace(stored.ref, id=eid))
    return hit(keyed, score)


def _tokenizer() -> Tokenizer:
    tok = MagicMock(spec=Tokenizer)
    tok.tokenize.return_value = ["hi"]
    return tok


class _NoProfiles:
    async def fetch(self, owner_id: str, **_: Any) -> list[Any]:
        return []


class _NoBuffer:
    async def list(self, session: SessionKey) -> list[CanonicalMessage]:
        return []


class _Embedding:
    dim = 4

    async def embed_query(self, text: str) -> list[float]:
        return [0.1] * self.dim


class _RawEndpoint:
    """A decider endpoint reached through ``achat_raw``; answers ``replies`` in turn."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls = 0

    async def achat_raw(self, messages: list[LLMMessage], **_: Any) -> RawReply:
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return RawReply(text=reply, finish_reason="length", usage=None)

    async def achat(self, messages: list[LLMMessage], **_: Any) -> str:
        raise AssertionError("the decider uses achat_raw")


def _manager(decider_client: Any, rows: list[SearchHit[Any]]) -> SearchManager:
    return SearchManager(
        access=AccessPolicy(InMemoryGroupStore()),
        episode_recaller=EpisodeRecaller(
            RecallerDeps(tokenizer=_tokenizer()),
            ScriptedIndex(sparse=rows, dense=rows, found=rows),  # type: ignore[arg-type]
        ),
        profile_recaller=_NoProfiles(),  # type: ignore[arg-type]
        session_buffer=_NoBuffer(),  # type: ignore[arg-type]
        embedding=_Embedding(),  # type: ignore[arg-type]
        llm_client=FakeLLMClient(handler=lambda *a, **k: ""),
        decider_client=decider_client,
        tuning_source=static_tuning(_FAST),
        trace_path_source=static_trace_path(),
    )


def _refuse(message: str) -> Any:
    def handler(*_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError(message)

    return handler


async def test_a_failed_decider_is_reported_with_its_reason() -> None:
    rows = [_row("ep_1"), _row("ep_2", 0.5)]
    client = FakeLLMClient(
        handler=_refuse("NotFoundError: Error code: 404 - model 'nope' does not exist")
    )
    resp = await _manager(client, rows).search(
        SearchRequest(user_id="alice", query="hi")
    )

    assert [e.id for e in resp.data.episodes] == ["ep_1", "ep_2"]  # still ranked
    assert resp.data.degraded == [
        SearchDegradation(
            component="decider",
            round=0,
            outcome="fallback",
            reason="RuntimeError: NotFoundError: Error code: 404 - model 'nope' "
            "does not exist",
            attempts=2,
        )
    ]
    assert client.call_count == 2


async def test_a_decided_search_reports_nothing() -> None:
    client = FakeLLMClient(
        handler=lambda *a, **k: json.dumps({"core": [0], "next_queries": []})
    )
    resp = await _manager(client, [_row("ep_1")]).search(
        SearchRequest(user_id="alice", query="hi")
    )
    assert [e.id for e in resp.data.episodes] == ["ep_1"]
    assert resp.data.degraded == []
    assert json.loads(resp.model_dump_json())["data"]["degraded"] == []


async def test_a_salvaged_round_is_reported_as_salvaged() -> None:
    endpoint = _RawEndpoint(['{"core": [1, 0],'])
    resp = await _manager(endpoint, [_row("ep_1"), _row("ep_2", 0.5)]).search(
        SearchRequest(user_id="alice", query="hi")
    )
    (entry,) = resp.data.degraded
    assert (entry.round, entry.outcome, entry.attempts) == (0, "salvaged", 2)
    assert "no JSON object" in entry.reason


def test_a_credential_in_the_error_is_masked() -> None:
    decision = RoundDecision(
        stop=True,
        queries=[],
        core=[0],
        failed=True,
        error="AuthenticationError: Incorrect API key provided: sk-proj-AbC123xyz789",
        attempts=4,
    )
    failure = _failure_of(2, decision)
    assert failure == DeciderFailure(
        round_idx=2,
        outcome="fallback",
        reason="AuthenticationError: Incorrect API key provided: ***",
        attempts=4,
    )
    assert "AbC123" not in failure.reason


def test_a_decided_round_is_not_a_failure() -> None:
    assert _failure_of(0, RoundDecision(stop=True, queries=[], core=[0])) is None


async def test_a_loop_ended_by_a_fallback_is_not_complete() -> None:
    """A group search's ``coverage`` said ``complete`` for a fallback core."""
    rows = [_row("ep_1")]
    index = ScriptedIndex(sparse=rows, dense=rows, found=rows)
    result = await run_llm_multiround(
        "hi",
        owner_id="alice",
        where=all_of(eq("owner_id", "alice")),
        episode_recaller=EpisodeRecaller(
            RecallerDeps(tokenizer=_tokenizer()),
            index,  # type: ignore[arg-type]
        ),
        embed_query_fn=_Embedding().embed_query,
        llm=FakeLLMClient(handler=_refuse("APITimeoutError: Request timed out.")),
        top_k=10,
        tuning_source=static_tuning(_FAST),
        trace_path_source=static_trace_path(),
    )
    assert result.core == ["ep_1"]
    assert result.coverage == "partial"
    assert result.rounds == 1
    assert [f.outcome for f in result.failures] == ["fallback"]


@pytest.mark.parametrize("failed_round", [False, True])
async def test_the_trace_records_the_reason(tmp_path: Any, failed_round: bool) -> None:
    rows = [_row("ep_1")]
    reply = json.dumps({"core": [0], "next_queries": []})
    handler = _refuse("boom") if failed_round else (lambda *a, **k: reply)
    trace = tmp_path / "trace.jsonl"
    await run_llm_multiround(
        "hi",
        owner_id="alice",
        where=all_of(eq("owner_id", "alice")),
        episode_recaller=EpisodeRecaller(
            RecallerDeps(tokenizer=_tokenizer()),
            ScriptedIndex(sparse=rows, dense=rows, found=rows),  # type: ignore[arg-type]
        ),
        embed_query_fn=_Embedding().embed_query,
        llm=FakeLLMClient(handler=handler),
        top_k=10,
        tuning_source=static_tuning(_FAST),
        trace_path_source=static_trace_path(str(trace)),
    )
    first = json.loads(trace.read_text().splitlines()[0])
    if failed_round:
        assert first["decider_failed"] is True
        assert first["decider_error"] == "RuntimeError: boom"
    else:
        assert "decider_error" not in first and "decider_failed" not in first
