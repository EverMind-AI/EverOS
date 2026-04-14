"""LLM Prometheus metrics integration tests (mock HTTP)."""

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


def _error_body(message: str = "error") -> dict:
    return {"error": {"message": message}}


METRICS_PATCH = "memory_layer.llm.openai_provider.record_llm_request"


class TestMetricsOnSuccess:
    """HTTP 200: record status=success."""

    @pytest.mark.asyncio
    async def test_success_records_metric(self) -> None:
        provider = OpenAIProvider(
            api_key="key-a", base_url="https://fake.api", model="test-model"
        )

        async def mock_request(data: dict, api_key: str) -> tuple[int, dict]:
            return 200, _success_body("ok")

        provider._do_request = mock_request

        with patch(METRICS_PATCH) as mock_record:
            await provider.generate("test")
            mock_record.assert_called_once_with("test-model", "success")


class TestMetricsOnRateLimit:
    """HTTP 429 (all keys exhausted): record status=rate_limit."""

    @pytest.mark.asyncio
    async def test_429_all_keys_exhausted_records_rate_limit(self) -> None:
        provider = OpenAIProvider(
            api_key="key-a,key-b", base_url="https://fake.api", model="test-model"
        )

        async def always_429(data: dict, api_key: str) -> tuple[int, dict]:
            return 429, _error_body("rate limited")

        provider._do_request = always_429

        with patch(METRICS_PATCH) as mock_record:
            with pytest.raises(LLMError, match="keys exhausted"):
                await provider.generate("test")
            mock_record.assert_called_once_with("test-model", "rate_limit")


class TestMetricsOnKeyError:
    """HTTP 401/402/403 (all keys exhausted): record status=key_error."""

    @pytest.mark.asyncio
    async def test_401_all_keys_exhausted_records_key_error(self) -> None:
        provider = OpenAIProvider(
            api_key="key-a", base_url="https://fake.api", model="test-model"
        )

        async def always_401(data: dict, api_key: str) -> tuple[int, dict]:
            return 401, _error_body("unauthorized")

        provider._do_request = always_401

        with patch(METRICS_PATCH) as mock_record:
            with pytest.raises(LLMError, match="keys exhausted"):
                await provider.generate("test")
            mock_record.assert_called_once_with("test-model", "key_error")


class TestMetricsOnServerError:
    """HTTP 5xx (after max retries): record status=server_error."""

    @pytest.mark.asyncio
    async def test_5xx_exhausted_records_server_error(self) -> None:
        provider = OpenAIProvider(
            api_key="key-a", base_url="https://fake.api", model="test-model"
        )

        async def always_502(data: dict, api_key: str) -> tuple[int, dict]:
            return 502, _error_body("bad gateway")

        provider._do_request = always_502

        with (
            patch(METRICS_PATCH) as mock_record,
            patch(
                "memory_layer.llm.openai_provider.asyncio.sleep", new_callable=AsyncMock
            ),
        ):
            with pytest.raises(LLMError, match="after 5 retries"):
                await provider.generate("test")
            mock_record.assert_called_once_with("test-model", "server_error")


class TestMetricsOnClientError:
    """Network errors (after max retries): record status=client_error."""

    @pytest.mark.asyncio
    async def test_network_error_records_client_error(self) -> None:
        provider = OpenAIProvider(
            api_key="key-a", base_url="https://fake.api", model="test-model"
        )

        async def always_fail(data: dict, api_key: str) -> tuple[int, dict]:
            raise aiohttp.ClientError("connection reset")

        provider._do_request = always_fail

        with patch(METRICS_PATCH) as mock_record:
            with pytest.raises(LLMError, match="Request failed"):
                await provider.generate("test")
            mock_record.assert_called_once_with("test-model", "client_error")


class TestMetricsOnRequestError:
    """HTTP 400/404/422: record status=request_error."""

    @pytest.mark.asyncio
    async def test_400_records_request_error(self) -> None:
        provider = OpenAIProvider(
            api_key="key-a", base_url="https://fake.api", model="test-model"
        )

        async def always_400(data: dict, api_key: str) -> tuple[int, dict]:
            return 400, _error_body("bad request")

        provider._do_request = always_400

        with patch(METRICS_PATCH) as mock_record:
            with pytest.raises(LLMError, match="HTTP Error 400"):
                await provider.generate("test")
            mock_record.assert_called_once_with("test-model", "request_error")


class TestMetricsNotRecordedOnRetry:
    """Metrics only recorded on final outcome, not intermediate retries."""

    @pytest.mark.asyncio
    async def test_429_then_success_records_only_success(self) -> None:
        """429 followed by 200: only 'success' is recorded."""
        provider = OpenAIProvider(
            api_key="key-a,key-b", base_url="https://fake.api", model="test-model"
        )

        responses = [(429, _error_body()), (200, _success_body("ok"))]

        async def mock_request(data: dict, api_key: str) -> tuple[int, dict]:
            return responses.pop(0)

        provider._do_request = mock_request

        with patch(METRICS_PATCH) as mock_record:
            await provider.generate("test")
            mock_record.assert_called_once_with("test-model", "success")

    @pytest.mark.asyncio
    async def test_5xx_then_success_records_only_success(self) -> None:
        """502 followed by 200: only 'success' is recorded."""
        provider = OpenAIProvider(
            api_key="key-a", base_url="https://fake.api", model="test-model"
        )

        responses = [(502, _error_body()), (200, _success_body("ok"))]

        async def mock_request(data: dict, api_key: str) -> tuple[int, dict]:
            return responses.pop(0)

        provider._do_request = mock_request

        with (
            patch(METRICS_PATCH) as mock_record,
            patch(
                "memory_layer.llm.openai_provider.asyncio.sleep", new_callable=AsyncMock
            ),
        ):
            await provider.generate("test")
            mock_record.assert_called_once_with("test-model", "success")
