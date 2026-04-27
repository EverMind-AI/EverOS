# -*- coding: utf-8 -*-
"""
Real LLM integration tests for TokenUsageCollector.

These tests make actual LLM/embedding API calls — they require:
- LLM backend configured (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL in .env)
- Embedding backend configured (deepinfra fallback)

Run with:
    PYTHONPATH=src uv run python -m pytest tests/test_token_usage_collector_real_llm.py -v -s

Use -s to see the token usage output in real time.
Skip with: pytest -m "not integration"
"""

import os
import sys
import pytest

# Load .env before imports
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from core.component.token_usage_collector import TokenUsageCollector
from core.component.llm.llm_adapter.message import ChatMessage, MessageRole
from core.component.llm.llm_adapter.completion import ChatCompletionRequest


class SpyCollector(TokenUsageCollector):
    """Records all add() calls for assertion."""

    def __init__(self):
        self.calls = []

    def add(
        self, model, input_tokens, output_tokens, *, call_type="llm", request_id=None
    ):
        self.calls.append(
            {
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "call_type": call_type,
            }
        )
        print(
            f"  [SpyCollector] add: model={model}, type={call_type}, in={input_tokens}, out={output_tokens}"
        )

    def get_totals(self):
        return {}

    def reset(self):
        self.calls.clear()


# ============================================================
# Skip conditions
# ============================================================

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")

skip_no_llm = pytest.mark.skipif(
    not LLM_API_KEY or not LLM_BASE_URL,
    reason="LLM_API_KEY and LLM_BASE_URL not configured in .env",
)


# ============================================================
# OpenAI Adapter (real call via OpenRouter)
# ============================================================


@skip_no_llm
class TestOpenAIAdapterReal:
    """Real OpenAI-compatible adapter tests (uses OpenRouter/configured LLM)."""

    def _make_adapter(self):
        from core.component.llm.llm_adapter.openai_adapter import OpenAIAdapter

        return OpenAIAdapter(
            {"api_key": LLM_API_KEY, "base_url": LLM_BASE_URL, "timeout": 60}
        )

    @pytest.mark.asyncio
    async def test_non_stream_real(self):
        """Real non-streaming call: verify token counts > 0."""
        from unittest.mock import patch

        adapter = self._make_adapter()
        spy = SpyCollector()

        with patch(
            "core.component.llm.llm_adapter.openai_adapter.get_bean_by_type",
            return_value=spy,
        ):
            request = ChatCompletionRequest(
                messages=[
                    ChatMessage(role=MessageRole.USER, content="Say hello in one word.")
                ],
                model=LLM_MODEL,
                stream=False,
                max_tokens=10,
            )
            response = await adapter.chat_completion(request)

        print(f"  Response: {response.choices[0]['message']['content']}")
        print(f"  Usage: {response.usage}")

        assert len(spy.calls) == 1, f"Expected 1 call, got {spy.calls}"
        call = spy.calls[0]
        assert (
            call["input_tokens"] > 0
        ), f"Expected input_tokens > 0, got {call['input_tokens']}"
        assert (
            call["output_tokens"] > 0
        ), f"Expected output_tokens > 0, got {call['output_tokens']}"
        assert call["call_type"] == "llm"
        print(f"  PASS: input={call['input_tokens']}, output={call['output_tokens']}")

    @pytest.mark.asyncio
    async def test_stream_real(self):
        """Real streaming call: verify token counts > 0 from final chunk."""
        from unittest.mock import patch

        adapter = self._make_adapter()
        spy = SpyCollector()

        with patch(
            "core.component.llm.llm_adapter.openai_adapter.get_bean_by_type",
            return_value=spy,
        ):
            request = ChatCompletionRequest(
                messages=[
                    ChatMessage(role=MessageRole.USER, content="Say hello in one word.")
                ],
                model=LLM_MODEL,
                stream=True,
                max_tokens=10,
            )
            gen = await adapter.chat_completion(request)
            chunks = [c async for c in gen]

        print(f"  Streamed content: {''.join(chunks)}")

        assert len(spy.calls) == 1, f"Expected 1 call, got {spy.calls}"
        call = spy.calls[0]
        assert (
            call["input_tokens"] > 0
        ), f"Expected input_tokens > 0, got {call['input_tokens']}"
        assert (
            call["output_tokens"] > 0
        ), f"Expected output_tokens > 0, got {call['output_tokens']}"
        print(f"  PASS: input={call['input_tokens']}, output={call['output_tokens']}")


# ============================================================
# Anthropic Adapter (real call via Anthropic API)
# ============================================================

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# The adapter posts to {base_url}/v1/messages, so base_url should NOT end with /v1
_raw_anthropic_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
ANTHROPIC_BASE_URL = _raw_anthropic_url.rstrip("/").removesuffix("/v1")

skip_no_anthropic = pytest.mark.skipif(
    not ANTHROPIC_API_KEY, reason="ANTHROPIC_API_KEY not configured in .env"
)


@skip_no_anthropic
class TestAnthropicAdapterReal:
    """Real Anthropic adapter tests."""

    def _make_adapter(self):
        from core.component.llm.llm_adapter.anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(
            {
                "api_key": ANTHROPIC_API_KEY,
                "base_url": ANTHROPIC_BASE_URL,
                "timeout": 60,
            }
        )

    @pytest.mark.asyncio
    async def test_non_stream_real(self):
        """Real non-streaming Anthropic call: verify token counts > 0."""
        from unittest.mock import patch

        adapter = self._make_adapter()
        spy = SpyCollector()

        with patch(
            "core.component.llm.llm_adapter.anthropic_adapter.get_bean_by_type",
            return_value=spy,
        ):
            request = ChatCompletionRequest(
                messages=[
                    ChatMessage(role=MessageRole.USER, content="Say hello in one word.")
                ],
                model="claude-haiku-4-5-20251001",
                stream=False,
                max_tokens=10,
            )
            response = await adapter.chat_completion(request)

        print(f"  Response: {response.choices[0]['message']['content']}")
        print(f"  Usage: {response.usage}")

        assert len(spy.calls) == 1, f"Expected 1 call, got {spy.calls}"
        call = spy.calls[0]
        assert (
            call["input_tokens"] > 0
        ), f"Expected input_tokens > 0, got {call['input_tokens']}"
        assert (
            call["output_tokens"] > 0
        ), f"Expected output_tokens > 0, got {call['output_tokens']}"
        assert call["call_type"] == "llm"
        print(f"  PASS: input={call['input_tokens']}, output={call['output_tokens']}")

    @pytest.mark.asyncio
    async def test_stream_real(self):
        """Real streaming Anthropic call: verify usage from SSE events.

        Note: Some proxies don't send the `message_delta` event with output_tokens.
        The real Anthropic API does. We assert input_tokens > 0 (from message_start)
        and output_tokens >= 0 (may be 0 if proxy omits message_delta).
        """
        from unittest.mock import patch

        adapter = self._make_adapter()
        spy = SpyCollector()

        with patch(
            "core.component.llm.llm_adapter.anthropic_adapter.get_bean_by_type",
            return_value=spy,
        ):
            request = ChatCompletionRequest(
                messages=[
                    ChatMessage(role=MessageRole.USER, content="Say hello in one word.")
                ],
                model="claude-haiku-4-5-20251001",
                stream=True,
                max_tokens=10,
            )
            gen = await adapter.chat_completion(request)
            chunks = [c async for c in gen]

        print(f"  Streamed content: {''.join(chunks)}")

        assert len(spy.calls) == 1, f"Expected 1 call, got {spy.calls}"
        call = spy.calls[0]
        assert (
            call["input_tokens"] > 0
        ), f"Expected input_tokens > 0, got {call['input_tokens']}"
        assert (
            call["output_tokens"] > 0
        ), f"Expected output_tokens > 0, got {call['output_tokens']}"
        print(f"  PASS: input={call['input_tokens']}, output={call['output_tokens']}")


# ============================================================
# Gemini Adapter (real call via Google AI)
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

skip_no_gemini = pytest.mark.skipif(
    not GEMINI_API_KEY, reason="GEMINI_API_KEY not configured in .env"
)


@skip_no_gemini
class TestGeminiAdapterReal:
    """Real Gemini adapter tests."""

    def _make_adapter(self):
        from core.component.llm.llm_adapter.gemini_adapter import GeminiAdapter

        return GeminiAdapter(
            {"api_key": GEMINI_API_KEY, "default_model": "gemini-2.5-flash"}
        )

    @pytest.mark.asyncio
    async def test_non_stream_real(self):
        """Real non-streaming Gemini call: verify token counts > 0."""
        from unittest.mock import patch

        adapter = self._make_adapter()
        spy = SpyCollector()

        with patch(
            "core.component.llm.llm_adapter.gemini_adapter.get_bean_by_type",
            return_value=spy,
        ):
            request = ChatCompletionRequest(
                messages=[
                    ChatMessage(role=MessageRole.USER, content="Say hello in one word.")
                ],
                model="gemini-2.5-flash",
                stream=False,
                max_tokens=100,  # Gemini needs room for thinking tokens
            )
            response = await adapter.chat_completion(request)

        print(f"  Response: {response.choices[0]['message']['content']}")
        print(f"  Usage: {response.usage}")

        assert len(spy.calls) == 1, f"Expected 1 call, got {spy.calls}"
        call = spy.calls[0]
        assert (
            call["input_tokens"] > 0
        ), f"Expected input_tokens > 0, got {call['input_tokens']}"
        assert (
            call["output_tokens"] > 0
        ), f"Expected output_tokens > 0, got {call['output_tokens']}"
        assert call["call_type"] == "llm"
        print(f"  PASS: input={call['input_tokens']}, output={call['output_tokens']}")

    @pytest.mark.asyncio
    async def test_stream_real(self):
        """Real streaming Gemini call: verify token counts > 0 from last chunk."""
        from unittest.mock import patch

        adapter = self._make_adapter()
        spy = SpyCollector()

        with patch(
            "core.component.llm.llm_adapter.gemini_adapter.get_bean_by_type",
            return_value=spy,
        ):
            request = ChatCompletionRequest(
                messages=[
                    ChatMessage(role=MessageRole.USER, content="Say hello in one word.")
                ],
                model="gemini-2.5-flash",
                stream=True,
                max_tokens=100,  # Gemini needs room for thinking tokens
            )
            gen = await adapter.chat_completion(request)
            chunks = [c async for c in gen]

        print(f"  Streamed content: {''.join(chunks)}")

        assert len(spy.calls) == 1, f"Expected 1 call, got {spy.calls}"
        call = spy.calls[0]
        assert (
            call["input_tokens"] > 0
        ), f"Expected input_tokens > 0, got {call['input_tokens']}"
        assert (
            call["output_tokens"] > 0
        ), f"Expected output_tokens > 0, got {call['output_tokens']}"
        print(f"  PASS: input={call['input_tokens']}, output={call['output_tokens']}")


# ============================================================
# Embedding (real call via DeepInfra)
# ============================================================

DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY", "")

skip_no_embedding = pytest.mark.skipif(
    not DEEPINFRA_API_KEY, reason="DEEPINFRA_API_KEY not configured in .env"
)


@skip_no_embedding
class TestEmbeddingReal:
    """Real embedding test (uses DeepInfra)."""

    @pytest.mark.asyncio
    async def test_embedding_real(self):
        """Real embedding call: verify collector.add(call_type='embedding') is called."""
        from unittest.mock import patch
        from agentic_layer.vectorize_deepinfra import (
            DeepInfraVectorizeService,
            DeepInfraVectorizeConfig,
        )

        config = DeepInfraVectorizeConfig(api_key=DEEPINFRA_API_KEY)
        service = DeepInfraVectorizeService(config=config)
        spy = SpyCollector()

        with patch("agentic_layer.vectorize_base.get_bean_by_type", return_value=spy):
            embedding = await service.get_embedding("hello world")

        print(f"  Embedding shape: {embedding.shape}")

        assert len(spy.calls) == 1, f"Expected 1 call, got {spy.calls}"
        call = spy.calls[0]
        assert call["call_type"] == "embedding"
        assert call["input_tokens"] >= 0  # some providers return 0
        print(f"  PASS: type=embedding, input_tokens={call['input_tokens']}")


# ============================================================
# Legacy OpenAI Provider (real call)
# ============================================================


@skip_no_llm
class TestLegacyProviderReal:
    """Real legacy OpenAIProvider test."""

    @pytest.mark.asyncio
    async def test_legacy_provider_real(self):
        """Real legacy provider call: verify collector.add() called with real tokens."""
        from unittest.mock import patch
        from memory_layer.llm.openai_provider import OpenAIProvider

        provider = OpenAIProvider(
            base_url=LLM_BASE_URL, api_key=LLM_API_KEY, model=LLM_MODEL
        )
        spy = SpyCollector()

        with patch(
            "memory_layer.llm.openai_provider.get_bean_by_type", return_value=spy
        ):
            result = await provider.generate("Say hello in one word.")

        print(f"  Response: {result}")

        assert len(spy.calls) == 1, f"Expected 1 call, got {spy.calls}"
        call = spy.calls[0]
        assert call["input_tokens"] > 0
        assert call["output_tokens"] > 0
        assert call["model"] == LLM_MODEL
        print(
            f"  PASS: model={call['model']}, input={call['input_tokens']}, output={call['output_tokens']}"
        )
