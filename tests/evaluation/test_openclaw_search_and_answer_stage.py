"""
Task 5: search() / answer() on the adapter, plus question_id passthrough
and latency/context metadata capture in search_stage / answer_stage.
"""
import logging

import pytest

from evaluation.src.adapters.openclaw_adapter import OpenClawAdapter
from evaluation.src.core.data_models import (
    Conversation,
    QAPair,
    SearchResult,
)
from evaluation.src.core.stages.answer_stage import (
    build_context,
    estimate_tokens,
    run_answer_stage,
)
from evaluation.src.core.stages.search_stage import run_search_stage


logger = logging.getLogger("test")


# ------------------------------------------------------------ build_context
def test_build_context_prefers_formatted_context():
    sr = SearchResult(
        query="q",
        conversation_id="c0",
        results=[{"content": "fallback", "score": 1.0, "metadata": {}}],
        retrieval_metadata={"formatted_context": "native context", "top_k": 1},
    )
    assert build_context(sr) == "native context"


def test_estimate_tokens_is_positive_for_nonempty_text():
    assert estimate_tokens("hello world") > 0
    assert estimate_tokens("") == 0


# --------------------------------------------------- search_stage passthrough
@pytest.mark.asyncio
async def test_search_stage_passes_question_id_to_adapter(monkeypatch):
    received = []

    class FakeAdapter:
        config: dict = {"search": {"num_workers": 1}}
        num_workers = 1

        async def search(self, query, conv_id, index, **kwargs):
            received.append({"query": query, "conv_id": conv_id, **kwargs})
            return SearchResult(
                query=query,
                conversation_id=conv_id,
                results=[],
                retrieval_metadata={"formatted_context": "ctx"},
            )

    adapter = FakeAdapter()
    qas = [
        QAPair(
            question_id="q1",
            question="what?",
            answer="",
            metadata={"conversation_id": "c0"},
        )
    ]
    convs = [Conversation(conversation_id="c0", messages=[], metadata={})]

    results = await run_search_stage(adapter, qas, {}, convs, None, logger)
    assert len(results) == 1
    assert received[0]["question_id"] == "q1"
    assert received[0]["conv_id"] == "c0"


# --------------------------------------------- answer_stage metadata capture
@pytest.mark.asyncio
async def test_answer_stage_captures_latency_and_context_metadata():
    answer_calls = []

    class FakeAdapter:
        config: dict = {}

        async def answer(self, query, context, **kwargs):
            answer_calls.append({"query": query, "context": context, **kwargs})
            return "hello"

    adapter = FakeAdapter()
    qas = [
        QAPair(
            question_id="q1",
            question="what?",
            answer="gold",
            metadata={"conversation_id": "c0"},
        )
    ]
    search_results = [
        SearchResult(
            query="what?",
            conversation_id="c0",
            results=[],
            retrieval_metadata={
                "formatted_context": "some context",
                "retrieval_latency_ms": 12.5,
                "retrieval_route": "search_then_get",
                "backend_mode": "hybrid",
            },
        )
    ]

    results = await run_answer_stage(adapter, qas, search_results, None, logger)
    assert len(results) == 1
    meta = results[0].metadata
    assert meta["answer_latency_ms"] is not None
    assert meta["answer_latency_ms"] >= 0
    assert meta["final_context_chars"] == len("some context")
    assert meta["final_context_tokens"] > 0
    assert meta["retrieval_latency_ms"] == 12.5
    assert meta["retrieval_route"] == "search_then_get"
    assert meta["backend_mode"] == "hybrid"
    # adapter.answer must have received question_id
    assert answer_calls[0]["question_id"] == "q1"


# ------------------------------------------- adapter.search via stub bridge
@pytest.mark.asyncio
async def test_openclaw_search_uses_bridge_and_attaches_source_sessions(
    tmp_path, monkeypatch
):
    adapter = OpenClawAdapter(
        {
            "adapter": "openclaw",
            "dataset_name": "locomo",
            "llm": {"provider": "openai", "model": "gpt-4o-mini"},
            "openclaw": {
                "retrieval_route": "search_only",
                "backend_mode": "fts_only",
            },
            "search": {"top_k": 5, "max_inflight_queries_per_conversation": 1},
        },
        output_dir=tmp_path,
    )

    async def fake_arun_bridge(bridge_script, payload, timeout=600.0):
        assert payload["command"] == "search"
        return {
            "ok": True,
            "command": "search",
            "hits": [
                {
                    "score": 0.9,
                    "snippet": "Caroline moved from Chicago.",
                    "artifact_locator": {
                        "kind": "memory_file_range",
                        "path_rel": "native_store/memory/2023-06-09.md",
                        "line_start": 10,
                        "line_end": 12,
                    },
                    "metadata": {"source_sessions": ["S3"]},
                }
            ],
        }

    monkeypatch.setattr(
        "evaluation.src.adapters.openclaw_adapter.arun_bridge",
        fake_arun_bridge,
    )

    index = {
        "type": "openclaw_sandboxes",
        "run_id": "run-x",
        "root_dir": str(tmp_path),
        "conversations": {
            "c0": {
                "conversation_id": "c0",
                "workspace_dir": str(tmp_path / "c0"),
                "resolved_config_path": str(tmp_path / "c0" / "openclaw.resolved.json"),
                "native_store_dir": str(tmp_path / "c0" / "native_store"),
                "backend_mode": "fts_only",
                "retrieval_route": "search_only",
            }
        },
    }

    result = await adapter.search(
        "where did Caroline move from?", "c0", index, question_id="q1"
    )
    assert result.conversation_id == "c0"
    assert len(result.results) == 1
    assert result.results[0]["metadata"]["source_sessions"] == ["S3"]
    assert result.retrieval_metadata["retrieval_route"] == "search_only"
    assert result.retrieval_metadata["backend_mode"] == "fts_only"
    assert result.retrieval_metadata["retrieval_latency_ms"] >= 0
    assert result.retrieval_metadata.get("formatted_context")


# ------------------------------------------------ adapter.answer mocks LLM
@pytest.mark.asyncio
async def test_openclaw_answer_uses_shared_prompt_and_llm(tmp_path, monkeypatch):
    adapter = OpenClawAdapter(
        {
            "adapter": "openclaw",
            "llm": {"provider": "openai", "model": "gpt-4o-mini"},
            "openclaw": {},
        },
        output_dir=tmp_path,
    )

    captured = {}

    async def fake_generate(prompt):
        captured["prompt"] = prompt
        return "mocked answer"

    monkeypatch.setattr(adapter, "_generate_answer", fake_generate)

    answer = await adapter.answer(
        query="what is X?", context="X is Y.", question_id="q1"
    )
    assert answer == "mocked answer"
    assert "X is Y." in captured["prompt"]
    assert "what is X?" in captured["prompt"]
