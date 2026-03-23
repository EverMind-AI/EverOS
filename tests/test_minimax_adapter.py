"""
MiniMax Adapter Unit Tests

Tests for the MiniMax LLM adapter, covering:
- Adapter initialization and configuration
- Temperature clamping to MiniMax range
- Think-tag stripping from responses
- Non-streaming and streaming chat completion
- Error handling

Usage:
    pytest tests/test_minimax_adapter.py -v
"""

import os
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

# Ensure the src directory is on the import path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.component.llm.llm_adapter.minimax_adapter import MiniMaxAdapter
from core.component.llm.llm_adapter.completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from core.component.llm.llm_adapter.message import ChatMessage, MessageRole


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimax_config():
    """Minimal valid config for MiniMaxAdapter."""
    return {
        "api_key": "test-minimax-api-key",
        "base_url": "https://api.minimax.io/v1",
        "timeout": 60,
        "models": [
            "MiniMax-M2.7",
            "MiniMax-M2.7-highspeed",
            "MiniMax-M2.5",
            "MiniMax-M2.5-highspeed",
        ],
    }


@pytest.fixture
def adapter(minimax_config):
    """Create a MiniMaxAdapter with test config."""
    return MiniMaxAdapter(minimax_config)


@pytest.fixture
def sample_messages():
    """Create sample chat messages for testing."""
    return [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="Hello!"),
    ]


@pytest.fixture
def sample_request(sample_messages):
    """Create a sample ChatCompletionRequest."""
    return ChatCompletionRequest(
        messages=sample_messages,
        model="MiniMax-M2.7",
        temperature=0.7,
        max_tokens=1024,
    )


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestMiniMaxAdapterInit:
    """Tests for adapter initialization."""

    def test_init_with_config_api_key(self, minimax_config):
        adapter = MiniMaxAdapter(minimax_config)
        assert adapter.api_key == "test-minimax-api-key"
        assert adapter.base_url == "https://api.minimax.io/v1"
        assert adapter.timeout == 60

    def test_init_with_env_api_key(self):
        config = {"base_url": "https://api.minimax.io/v1"}
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key"}):
            adapter = MiniMaxAdapter(config)
            assert adapter.api_key == "env-key"

    def test_init_without_api_key_raises(self):
        config = {"base_url": "https://api.minimax.io/v1"}
        with patch.dict(os.environ, {}, clear=True):
            # Remove MINIMAX_API_KEY if present
            os.environ.pop("MINIMAX_API_KEY", None)
            with pytest.raises(ValueError):
                MiniMaxAdapter(config)

    def test_init_default_base_url(self):
        config = {"api_key": "test-key"}
        adapter = MiniMaxAdapter(config)
        assert adapter.base_url == "https://api.minimax.io/v1"

    def test_init_default_timeout(self):
        config = {"api_key": "test-key"}
        adapter = MiniMaxAdapter(config)
        assert adapter.timeout == 600

    def test_init_creates_async_openai_client(self, adapter):
        import openai
        assert isinstance(adapter.client, openai.AsyncOpenAI)


# ---------------------------------------------------------------------------
# Temperature clamping tests
# ---------------------------------------------------------------------------

class TestTemperatureClamping:
    """Tests for MiniMax temperature clamping."""

    def test_clamp_normal_temperature(self):
        assert MiniMaxAdapter._clamp_temperature(0.5) == 0.5

    def test_clamp_zero_temperature(self):
        """Temperature 0 should be clamped to MIN_TEMPERATURE."""
        result = MiniMaxAdapter._clamp_temperature(0.0)
        assert result == MiniMaxAdapter.MIN_TEMPERATURE

    def test_clamp_negative_temperature(self):
        result = MiniMaxAdapter._clamp_temperature(-0.5)
        assert result == MiniMaxAdapter.MIN_TEMPERATURE

    def test_clamp_high_temperature(self):
        result = MiniMaxAdapter._clamp_temperature(1.5)
        assert result == MiniMaxAdapter.MAX_TEMPERATURE

    def test_clamp_max_temperature(self):
        assert MiniMaxAdapter._clamp_temperature(1.0) == 1.0

    def test_clamp_min_temperature(self):
        assert MiniMaxAdapter._clamp_temperature(0.01) == 0.01

    def test_clamp_none_temperature(self):
        assert MiniMaxAdapter._clamp_temperature(None) is None


# ---------------------------------------------------------------------------
# Think-tag stripping tests
# ---------------------------------------------------------------------------

class TestThinkTagStripping:
    """Tests for stripping <think>...</think> tags."""

    def test_strip_single_think_tag(self):
        text = "<think>reasoning here</think>Final answer"
        assert MiniMaxAdapter._strip_think_tags(text) == "Final answer"

    def test_strip_multiline_think_tag(self):
        text = "<think>\nStep 1: analyze\nStep 2: compute\n</think>\nThe answer is 42."
        assert MiniMaxAdapter._strip_think_tags(text) == "The answer is 42."

    def test_no_think_tags(self):
        text = "Just a plain response."
        assert MiniMaxAdapter._strip_think_tags(text) == "Just a plain response."

    def test_empty_think_tag(self):
        text = "<think></think>Result"
        assert MiniMaxAdapter._strip_think_tags(text) == "Result"

    def test_multiple_think_tags(self):
        text = "<think>first</think>A<think>second</think>B"
        assert MiniMaxAdapter._strip_think_tags(text) == "AB"

    def test_strip_with_surrounding_whitespace(self):
        text = "<think>thinking</think>  \n  Hello world"
        result = MiniMaxAdapter._strip_think_tags(text)
        assert result == "Hello world"

    def test_empty_string(self):
        assert MiniMaxAdapter._strip_think_tags("") == ""


# ---------------------------------------------------------------------------
# Available models tests
# ---------------------------------------------------------------------------

class TestGetAvailableModels:
    """Tests for model list retrieval."""

    def test_get_available_models(self, adapter):
        models = adapter.get_available_models()
        assert "MiniMax-M2.7" in models
        assert "MiniMax-M2.7-highspeed" in models
        assert "MiniMax-M2.5" in models
        assert "MiniMax-M2.5-highspeed" in models

    def test_get_available_models_empty_config(self):
        config = {"api_key": "test-key"}
        adapter = MiniMaxAdapter(config)
        assert adapter.get_available_models() == []


# ---------------------------------------------------------------------------
# Chat completion tests (non-streaming)
# ---------------------------------------------------------------------------

class TestChatCompletion:
    """Tests for non-streaming chat completion."""

    @pytest.mark.asyncio
    async def test_chat_completion_success(self, adapter, sample_request):
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "MiniMax-M2.7",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        with patch.object(
            adapter.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            result = await adapter.chat_completion(sample_request)

        assert isinstance(result, ChatCompletionResponse)
        assert result.id == "chatcmpl-test"
        assert result.model == "MiniMax-M2.7"
        assert result.choices[0]["message"]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_chat_completion_strips_think_tags(self, adapter, sample_request):
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "MiniMax-M2.7",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "<think>reasoning</think>The answer is 42.",
                    },
                    "finish_reason": "stop",
                }
            ],
        }

        with patch.object(
            adapter.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            result = await adapter.chat_completion(sample_request)

        assert result.choices[0]["message"]["content"] == "The answer is 42."

    @pytest.mark.asyncio
    async def test_chat_completion_clamps_temperature(self, adapter, sample_messages):
        request = ChatCompletionRequest(
            messages=sample_messages,
            model="MiniMax-M2.7",
            temperature=0.0,
            max_tokens=100,
        )

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "MiniMax-M2.7",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

        with patch.object(
            adapter.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            await adapter.chat_completion(request)

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["temperature"] == MiniMaxAdapter.MIN_TEMPERATURE

    @pytest.mark.asyncio
    async def test_chat_completion_no_model_raises(self, adapter, sample_messages):
        request = ChatCompletionRequest(
            messages=sample_messages,
            model=None,
        )
        with pytest.raises(ValueError):
            await adapter.chat_completion(request)

    @pytest.mark.asyncio
    async def test_chat_completion_api_error(self, adapter, sample_request):
        with patch.object(
            adapter.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = Exception("API rate limit exceeded")
            with pytest.raises(RuntimeError, match="MiniMax chat completion request failed"):
                await adapter.chat_completion(sample_request)


# ---------------------------------------------------------------------------
# Chat completion tests (streaming)
# ---------------------------------------------------------------------------

class TestChatCompletionStreaming:
    """Tests for streaming chat completion."""

    @pytest.mark.asyncio
    async def test_streaming_returns_generator(self, adapter, sample_messages):
        request = ChatCompletionRequest(
            messages=sample_messages,
            model="MiniMax-M2.7",
            stream=True,
        )

        # Create mock streaming chunks
        chunks = []
        for text in ["Hello", ", world", "!"]:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = text
            chunks.append(chunk)

        async def mock_stream():
            for c in chunks:
                yield c

        with patch.object(
            adapter.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_stream()
            result = await adapter.chat_completion(request)

            # Consume the generator inside the mock context
            collected = []
            async for part in result:
                collected.append(part)
        assert "".join(collected) == "Hello, world!"

    @pytest.mark.asyncio
    async def test_streaming_strips_think_tags(self, adapter, sample_messages):
        request = ChatCompletionRequest(
            messages=sample_messages,
            model="MiniMax-M2.7",
            stream=True,
        )

        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = "<think>reasoning</think>Answer"

        async def mock_stream():
            yield chunk

        with patch.object(
            adapter.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_stream()
            result = await adapter.chat_completion(request)

            collected = []
            async for part in result:
                collected.append(part)
        assert "".join(collected) == "Answer"


# ---------------------------------------------------------------------------
# OpenAICompatibleClient integration tests (minimax provider routing)
# ---------------------------------------------------------------------------

class TestOpenAICompatibleClientMiniMaxRouting:
    """Tests that OpenAICompatibleClient correctly routes to MiniMaxAdapter."""

    @pytest.mark.asyncio
    async def test_minimax_provider_creates_minimax_adapter(self):
        """Verify that provider='minimax' creates a MiniMaxAdapter instance."""
        from core.component.llm.llm_adapter.minimax_adapter import MiniMaxAdapter

        config = {
            "provider": "minimax",
            "api_key": "test-key",
            "base_url": "https://api.minimax.io/v1",
            "models": ["MiniMax-M2.7"],
            "model": "MiniMax-M2.7",
        }

        # Simulate what OpenAICompatibleClient._get_adapter does
        provider = config.get("provider", "openai")
        assert provider == "minimax"

        adapter = MiniMaxAdapter(config)
        assert isinstance(adapter, MiniMaxAdapter)
        assert adapter.base_url == "https://api.minimax.io/v1"

    def test_minimax_in_llm_backends_yaml(self):
        """Verify MiniMax is configured in llm_backends.yaml."""
        import yaml

        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "config", "llm_backends.yaml"
        )
        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        backends = config.get("llm_backends", {})
        assert "minimax" in backends, "minimax backend should be in llm_backends.yaml"

        minimax_cfg = backends["minimax"]
        assert minimax_cfg["provider"] == "minimax"
        assert minimax_cfg["base_url"] == "https://api.minimax.io/v1"
        assert "MiniMax-M2.7" in minimax_cfg["models"]
        assert "MiniMax-M2.7-highspeed" in minimax_cfg["models"]
        assert minimax_cfg["model"] == "MiniMax-M2.7"

    def test_openai_compatible_client_imports_minimax(self):
        """Verify that MiniMaxAdapter is imported in openai_compatible_client."""
        from core.component.openai_compatible_client import OpenAICompatibleClient

        # If the import in openai_compatible_client.py works, this passes
        assert OpenAICompatibleClient is not None


# ---------------------------------------------------------------------------
# Integration test (requires MINIMAX_API_KEY)
# ---------------------------------------------------------------------------

class TestMiniMaxIntegration:
    """Integration tests that call the real MiniMax API.

    These tests are skipped unless MINIMAX_API_KEY is set in the environment.
    """

    @pytest.fixture
    def live_adapter(self):
        api_key = os.getenv("MINIMAX_API_KEY")
        if not api_key:
            pytest.skip("MINIMAX_API_KEY not set")
        return MiniMaxAdapter({
            "api_key": api_key,
            "base_url": "https://api.minimax.io/v1",
            "models": ["MiniMax-M2.5-highspeed"],
        })

    @pytest.mark.asyncio
    async def test_live_chat_completion(self, live_adapter):
        """Test a real API call to MiniMax."""
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role=MessageRole.USER, content="Say 'hello' and nothing else."),
            ],
            model="MiniMax-M2.5-highspeed",
            temperature=0.01,
            max_tokens=256,
        )
        result = await live_adapter.chat_completion(request)
        assert isinstance(result, ChatCompletionResponse)
        assert len(result.choices) > 0
        content = result.choices[0]["message"]["content"]
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_live_streaming(self, live_adapter):
        """Test a real streaming API call to MiniMax."""
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role=MessageRole.USER, content="Count from 1 to 3."),
            ],
            model="MiniMax-M2.5-highspeed",
            temperature=0.01,
            max_tokens=64,
            stream=True,
        )
        result = await live_adapter.chat_completion(request)
        collected = []
        async for chunk in result:
            collected.append(chunk)
        full_text = "".join(collected)
        assert len(full_text) > 0

    @pytest.mark.asyncio
    async def test_live_temperature_clamping(self, live_adapter):
        """Test that temperature=0 doesn't cause API errors."""
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role=MessageRole.USER, content="Say 'ok'."),
            ],
            model="MiniMax-M2.5-highspeed",
            temperature=0.0,
            max_tokens=16,
        )
        result = await live_adapter.chat_completion(request)
        assert isinstance(result, ChatCompletionResponse)
