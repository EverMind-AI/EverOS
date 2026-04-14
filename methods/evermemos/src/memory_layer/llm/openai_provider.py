"""
OpenAI-compatible LLM provider implementation.

This provider uses a caller-supplied API key and base URL.
"""

import asyncio
import json
import os
import random
import time

import aiohttp

from core.component.token_usage_collector import TokenUsageCollector
from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger
from memory_layer.llm.api_key_rotator import ApiKeyRotator
from memory_layer.llm.llm_metrics import record_llm_request
from memory_layer.llm.protocol import LLMProvider, LLMError

logger = get_logger(__name__)

_MAX_RETRIES = 5


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible LLM provider.

    Sends requests to any OpenAI-compatible endpoint (OpenRouter, OpenAI, etc.)
    with automatic multi-key rotation and differentiated retry strategies.

    Args:
        model: Model name (e.g. "gpt-4.1-mini", "qwen/qwen3-235b-a22b-2507").
        api_key: API key(s), comma-separated for multi-key rotation.
        base_url: API base URL.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        enable_stats: Enable per-call usage statistics.
        provider_type: Provider identifier ("openai" or "openrouter").
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",  # skip-sensitive-check
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = 100 * 1024,
        enable_stats: bool = False,
        provider_type: str | None = None,
        **kwargs,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_stats = enable_stats
        self.provider_type = (
            provider_type or "openrouter"  # skip-sensitive-check
        ).lower()
        self._key_rotator = (
            ApiKeyRotator.get_or_create(api_key) if api_key else ApiKeyRotator([""])
        )
        self.base_url = base_url

        self._validate_model_whitelist(self.provider_type, model)

        if self.enable_stats:
            self.current_call_stats = None

    @staticmethod
    def _validate_model_whitelist(provider_type: str, model: str) -> None:
        """Validate model against the provider's whitelist from environment variable."""
        env_key = f"{provider_type.upper()}_WHITE_LIST"
        raw = os.getenv(env_key, "").strip()
        if not raw:
            return
        allowed_models = {m.strip() for m in raw.split(",") if m.strip()}
        if not allowed_models:
            return
        if model not in allowed_models:
            raise ValueError(
                f"Provider '{provider_type}' only supports: {', '.join(sorted(allowed_models))}. Got: '{model}'."
            )

    @staticmethod
    def _resolve_openrouter_provider() -> dict | None:
        """Parse LLM_OPENROUTER_PROVIDER env var into an OpenRouter provider dict."""
        raw = os.getenv("LLM_OPENROUTER_PROVIDER", "default")  # skip-sensitive-check
        if raw == "default":
            return None
        provider_list = [p.strip() for p in raw.split(",")]
        return {"order": provider_list, "allow_fallbacks": False}

    @staticmethod
    def _extract_error_message(response_data: dict, status_code: int) -> str:
        """Extract a human-readable error message from an error response body."""
        return response_data.get("error", {}).get("message", f"HTTP {status_code}")

    def _build_request_data(
        self,
        prompt: str,
        temperature: float | None,
        max_tokens: int | None,
        response_format: dict | None,
    ) -> dict:
        """Build the JSON payload for the chat completions request."""
        data: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else self.temperature,
            "provider": self._resolve_openrouter_provider(),
            "response_format": response_format,
        }
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        elif self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens
        return data

    async def _do_request(self, data: dict, api_key: str) -> tuple[int, dict]:
        """Execute a single HTTP POST to the chat completions endpoint.

        Returns:
            (status_code, parsed_response_body)
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.base_url}/chat/completions", json=data, headers=headers
            ) as response:
                raw = await response.read()
                try:
                    response_data = json.loads(raw.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Non-JSON response (e.g. Cloudflare HTML error page)
                    return response.status, {
                        "error": {"message": raw[:500].decode(errors="replace")}
                    }
                return response.status, response_data

    def _report_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Report token usage to the global TokenUsageCollector (best-effort)."""
        try:
            collector = get_bean_by_type(TokenUsageCollector)
            collector.add(self.model, prompt_tokens, completion_tokens, call_type="llm")
        except Exception:
            pass

    def _log_completion_metrics(self, response_data: dict, duration: float) -> None:
        """Log finish reason, duration, and token usage for a completed request."""
        finish_reason = response_data.get("choices", [{}])[0].get("finish_reason", "")
        if finish_reason == "stop":
            logger.debug("[OpenAI-%s] Finish reason: %s", self.model, finish_reason)
        else:
            logger.warning("[OpenAI-%s] Finish reason: %s", self.model, finish_reason)

        usage = response_data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        logger.debug("[OpenAI-%s] Duration: %.2fs", self.model, duration)
        if duration > 30:
            logger.warning("[OpenAI-%s] Duration too long: %.2fs", self.model, duration)
        logger.debug(
            "[OpenAI-%s] Tokens: %s prompt, %s completion, %s total",
            self.model,
            format(prompt_tokens, ","),
            format(completion_tokens, ","),
            format(usage.get("total_tokens", 0), ","),
        )

        self._report_token_usage(prompt_tokens, completion_tokens)

    def _handle_success(self, response_data: dict, start_time: float) -> str:
        """Process a successful (HTTP 200) response: log metrics, report usage, return text."""
        duration = time.perf_counter() - start_time
        self._log_completion_metrics(response_data, duration)

        if self.enable_stats:
            usage = response_data.get("usage", {})
            self.current_call_stats = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "duration": duration,
                "timestamp": time.time(),
            }

        return response_data["choices"][0]["message"]["content"]

    def _handle_key_error(
        self, status_code: int, error_msg: str, consecutive_failures: int
    ) -> int:
        """Handle key-level errors (401/402/403/429): rotate key, raise if all exhausted."""
        consecutive_failures += 1
        if consecutive_failures >= self._key_rotator.size:
            metric_status = "rate_limit" if status_code == 429 else "key_error"
            record_llm_request(self.model, metric_status)
            raise LLMError(
                f"HTTP {status_code}: {error_msg} "
                f"(all {self._key_rotator.size} keys exhausted)"
            )
        logger.warning(
            "[OpenAI-%s] Key error %d, rotating key (%d/%d exhausted)",
            self.model,
            status_code,
            consecutive_failures,
            self._key_rotator.size,
        )
        return consecutive_failures

    async def _handle_server_error(
        self, status_code: int, error_msg: str, retry_num: int
    ) -> None:
        """Handle 5xx server error: sleep and retry, or raise on final attempt."""
        if retry_num < _MAX_RETRIES - 1:
            logger.warning(
                "[OpenAI-%s] Server error %d, retry %d/%d",
                self.model,
                status_code,
                retry_num + 1,
                _MAX_RETRIES,
            )
            await asyncio.sleep(random.randint(5, 20))
            return
        record_llm_request(self.model, "server_error")
        raise LLMError(
            f"HTTP Error {status_code}: {error_msg} (after {_MAX_RETRIES} retries)"
        )

    async def _execute_with_retry(self, data: dict, start_time: float) -> str:
        """Retry loop: key-level errors rotate key, 5xx backs off with sleep."""
        consecutive_key_failures = 0
        key_rotation = self._key_rotator.get_rotation()

        for retry_num in range(_MAX_RETRIES):
            current_key = key_rotation[retry_num % len(key_rotation)]
            try:
                status_code, response_data = await self._do_request(data, current_key)
            except aiohttp.ClientError as exc:
                logger.error("aiohttp.ClientError: %s", exc)
                if retry_num == _MAX_RETRIES - 1:
                    record_llm_request(self.model, "client_error")
                    raise LLMError(f"Request failed: {exc}") from exc
                continue
            except LLMError:
                raise
            except Exception as exc:
                logger.error("Unexpected error: %s", exc)
                if retry_num == _MAX_RETRIES - 1:
                    record_llm_request(self.model, "client_error")
                    raise LLMError(f"Request failed: {exc}") from exc
                continue

            if status_code == 200:
                record_llm_request(self.model, "success")
                return self._handle_success(response_data, start_time)

            error_msg = self._extract_error_message(response_data, status_code)
            logger.error("[OpenAI-%s] HTTP %d: %s", self.model, status_code, error_msg)

            # Key-level errors: rotate key immediately, no sleep.
            # - 401 Unauthorized: invalid/missing key
            # - 402 Payment Required: key quota exhausted
            # - 403 Forbidden: key lacks permission
            # - 429 Too Many Requests: key rate-limited
            if status_code in (401, 402, 403, 429):
                consecutive_key_failures = self._handle_key_error(
                    status_code, error_msg, consecutive_key_failures
                )
                continue

            # 5xx: sleep then retry (key rotates per retry_rotation sequence)
            if status_code in (500, 502, 503, 504):
                await self._handle_server_error(status_code, error_msg, retry_num)
                continue

            # Request-level errors (400, 404, 422, etc.): not key-related, no retry
            record_llm_request(self.model, "request_error")
            raise LLMError(f"HTTP Error {status_code}: {error_msg}")

        record_llm_request(self.model, "client_error")
        raise LLMError(f"Request failed after {_MAX_RETRIES} retries")

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
        response_format: dict | None = None,
    ) -> str:
        """Generate a response for the given prompt."""
        start_time = time.perf_counter()
        data = self._build_request_data(
            prompt, temperature, max_tokens, response_format
        )
        return await self._execute_with_retry(data, start_time)

    async def test_connection(self) -> bool:
        """Test the connection to the API endpoint."""
        try:
            logger.info("\U0001f517 [OpenAI-%s] Testing API connection...", self.model)
            test_response = await self.generate("Hello", temperature=0.1)
            success = len(test_response) > 0
            if success:
                logger.info(
                    "\u2705 [OpenAI-%s] API connection test succeeded", self.model
                )
            else:
                logger.error(
                    "\u274c [OpenAI-%s] API connection test failed: Empty response",
                    self.model,
                )
            return success
        except Exception as e:
            logger.error(
                "\u274c [OpenAI-%s] API connection test failed: %s", self.model, e
            )
            return False

    def get_current_call_stats(self) -> dict | None:
        """Return per-call statistics if stats collection is enabled."""
        if self.enable_stats:
            return self.current_call_stats
        return None

    def __repr__(self) -> str:
        return (
            "OpenAIProvider("
            f"provider_type={self.provider_type}, model={self.model}, "
            f"base_url={self.base_url}, keys={self._key_rotator.size}"
            ")"
        )
