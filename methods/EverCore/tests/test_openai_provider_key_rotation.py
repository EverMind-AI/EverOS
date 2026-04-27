"""OpenAIProvider key rotation integration tests (mock HTTP)."""

import pytest
import aiohttp
from unittest.mock import AsyncMock, patch

from memory_layer.llm.openai_provider import OpenAIProvider
from memory_layer.llm.protocol import LLMError
from memory_layer.llm.api_key_rotator import ApiKeyRotator


@pytest.fixture(autouse=True)
def _reset_shared_rotator():
    """Ensure each test starts with a clean singleton state."""
    ApiKeyRotator._shared = None
    yield
    ApiKeyRotator._shared = None


def _success_body(content: str = "hello") -> dict:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _error_body(message: str = "rate limited") -> dict:
    return {"error": {"message": message}}


class TestKeyRotationOn429:
    """429: rotate key immediately, no sleep."""

    @pytest.mark.asyncio
    async def test_429_then_success_with_next_key(self) -> None:
        provider = OpenAIProvider(
            api_key="key-a,key-b,key-c", base_url="https://fake.api", model="test-model"
        )

        responses = [(429, _error_body()), (200, _success_body("ok"))]
        call_keys: list[str] = []

        async def capture_do_request(data: dict, api_key: str) -> tuple[int, dict]:
            call_keys.append(api_key)
            return responses.pop(0)

        provider._do_request = capture_do_request

        result = await provider.generate("test")
        assert result == "ok"
        assert len(call_keys) == 2
        assert call_keys[0] != call_keys[1]

    @pytest.mark.asyncio
    async def test_all_keys_429_raises_error(self) -> None:
        provider = OpenAIProvider(
            api_key="key-a,key-b,key-c", base_url="https://fake.api", model="test-model"
        )

        async def always_429(data: dict, api_key: str) -> tuple[int, dict]:
            return 429, _error_body("rate limited")

        provider._do_request = always_429

        with pytest.raises(LLMError, match="3 keys exhausted"):
            await provider.generate("test")

    @pytest.mark.asyncio
    async def test_429_does_not_sleep(self) -> None:
        provider = OpenAIProvider(
            api_key="key-a,key-b", base_url="https://fake.api", model="test-model"
        )

        call_count = 0

        async def mock_request(data: dict, api_key: str) -> tuple[int, dict]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 429, _error_body()
            return 200, _success_body()

        provider._do_request = mock_request

        with patch(
            "memory_layer.llm.openai_provider.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            await provider.generate("test")
            mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_429_counter_not_reset_by_5xx(self) -> None:
        """429 -> 5xx -> 429: consecutive_rate_limits does not reset on 5xx."""
        provider = OpenAIProvider(
            api_key="key-a,key-b", base_url="https://fake.api", model="test-model"
        )

        responses = [
            (429, _error_body("rate limited")),
            (502, _error_body("bad gateway")),
            (429, _error_body("rate limited")),
        ]
        idx = 0

        async def mock_request(data: dict, api_key: str) -> tuple[int, dict]:
            nonlocal idx
            resp = responses[idx]
            idx += 1
            return resp

        provider._do_request = mock_request

        with patch(
            "memory_layer.llm.openai_provider.asyncio.sleep", new_callable=AsyncMock
        ):
            with pytest.raises(LLMError, match="2 keys exhausted"):
                await provider.generate("test")


class TestKeyRotationOn429And5xxInterleaved:
    """429/5xx interleaved: sleep only on 5xx, not on 429."""

    @pytest.mark.asyncio
    async def test_429_then_5xx_sleeps_only_on_5xx(self) -> None:
        """429 -> 502 -> 200: sleep called exactly once (on 502 only)."""
        provider = OpenAIProvider(
            api_key="key-a,key-b,key-c", base_url="https://fake.api", model="test-model"
        )

        responses = [
            (429, _error_body("rate limited")),
            (502, _error_body("bad gateway")),
            (200, _success_body("ok")),
        ]

        async def mock_request(data: dict, api_key: str) -> tuple[int, dict]:
            return responses.pop(0)

        provider._do_request = mock_request

        with patch(
            "memory_layer.llm.openai_provider.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            result = await provider.generate("test")
            assert result == "ok"
            mock_sleep.assert_called_once()  # only on 502, not on 429


class TestRequestLevelErrors:
    """400/404/422: no retry, raise immediately."""

    @pytest.mark.asyncio
    async def test_400_raises_immediately_no_retry(self) -> None:
        provider = OpenAIProvider(
            api_key="key-a,key-b", base_url="https://fake.api", model="test-model"
        )

        call_count = 0

        async def mock_request(data: dict, api_key: str) -> tuple[int, dict]:
            nonlocal call_count
            call_count += 1
            return 400, _error_body("bad request")

        provider._do_request = mock_request

        with pytest.raises(LLMError, match="HTTP Error 400"):
            await provider.generate("test")
        assert call_count == 1  # no retry


class TestNetworkErrors:
    """aiohttp.ClientError: retry up to max attempts."""

    @pytest.mark.asyncio
    async def test_client_error_retries_then_raises(self) -> None:
        provider = OpenAIProvider(
            api_key="key-a", base_url="https://fake.api", model="test-model"
        )

        async def always_fail(data: dict, api_key: str) -> tuple[int, dict]:
            raise aiohttp.ClientError("connection reset")

        provider._do_request = always_fail

        with pytest.raises(LLMError, match="Request failed"):
            await provider.generate("test")

    @pytest.mark.asyncio
    async def test_client_error_then_success(self) -> None:
        provider = OpenAIProvider(
            api_key="key-a", base_url="https://fake.api", model="test-model"
        )

        call_count = 0

        async def fail_then_ok(data: dict, api_key: str) -> tuple[int, dict]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise aiohttp.ClientError("timeout")
            return 200, _success_body("recovered")

        provider._do_request = fail_then_ok

        result = await provider.generate("test")
        assert result == "recovered"
        assert call_count == 2


class TestKeyRotationOn5xx:
    """5xx: sleep then retry."""

    @pytest.mark.asyncio
    async def test_5xx_retries_with_sleep(self) -> None:
        provider = OpenAIProvider(
            api_key="key-a,key-b", base_url="https://fake.api", model="test-model"
        )

        call_count = 0

        async def mock_request(data: dict, api_key: str) -> tuple[int, dict]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 502, _error_body("bad gateway")
            return 200, _success_body()

        provider._do_request = mock_request

        with patch(
            "memory_layer.llm.openai_provider.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            result = await provider.generate("test")
            assert result == "hello"
            mock_sleep.assert_called_once()


class TestRetryStartsFromNextKey:
    """Retries start from the key AFTER the failed one, cycling through all."""

    @pytest.mark.asyncio
    async def test_retry_uses_rotation_sequence(self) -> None:
        """All attempts follow the rotation: rotation[0], [1], [2], all distinct."""
        provider = OpenAIProvider(
            api_key="key-a,key-b,key-c", base_url="https://fake.api", model="test-model"
        )

        call_keys: list[str] = []

        async def capture_request(data: dict, api_key: str) -> tuple[int, dict]:
            call_keys.append(api_key)
            return 429, _error_body("rate limited")

        provider._do_request = capture_request

        with pytest.raises(LLMError, match="keys exhausted"):
            await provider.generate("test")
        # 3 keys, 3 attempts, all distinct, in rotation order
        assert len(call_keys) == 3
        assert len(set(call_keys)) == 3
        first = call_keys[0]
        keys = ["key-a", "key-b", "key-c"]
        first_idx = keys.index(first)
        assert call_keys[1] == keys[(first_idx + 1) % 3]
        assert call_keys[2] == keys[(first_idx + 2) % 3]

    @pytest.mark.asyncio
    async def test_keys_repeat_after_full_cycle(self) -> None:
        """With 2 keys and 5 retries (5xx), keys alternate without adjacent repeats."""
        provider = OpenAIProvider(
            api_key="key-a,key-b", base_url="https://fake.api", model="test-model"
        )

        call_keys: list[str] = []

        async def capture_request(data: dict, api_key: str) -> tuple[int, dict]:
            call_keys.append(api_key)
            return 502, _error_body("bad gateway")

        provider._do_request = capture_request

        with patch(
            "memory_layer.llm.openai_provider.asyncio.sleep", new_callable=AsyncMock
        ):
            with pytest.raises(LLMError, match="after 5 retries"):
                await provider.generate("test")
        assert len(call_keys) == 5
        # Adjacent keys always differ
        for i in range(len(call_keys) - 1):
            assert call_keys[i] != call_keys[i + 1]


class TestSingleKeyBackwardCompat:
    """Single key: behavior unchanged."""

    @pytest.mark.asyncio
    async def test_single_key_works_normally(self) -> None:
        provider = OpenAIProvider(
            api_key="single-key", base_url="https://fake.api", model="test-model"
        )

        async def mock_request(data: dict, api_key: str) -> tuple[int, dict]:
            assert api_key == "single-key"
            return 200, _success_body("single key response")

        provider._do_request = mock_request

        result = await provider.generate("test")
        assert result == "single key response"
