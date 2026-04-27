import asyncio
import os
from typing import Dict, Any, List, Optional, AsyncGenerator, Union

from core.di.decorators import component
from core.observation.logger import get_logger
from core.component.config_provider import ConfigProvider

from core.component.llm.llm_adapter.message import ChatMessage
from core.component.llm.llm_adapter.completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from core.component.llm.llm_adapter.llm_backend_adapter import LLMBackendAdapter
from core.component.llm.llm_adapter.openai_adapter import OpenAIAdapter
from core.component.llm.llm_adapter.anthropic_adapter import AnthropicAdapter
from core.component.llm.llm_adapter.gemini_adapter import GeminiAdapter

logger = get_logger(__name__)


@component(name="openai_compatible_client", primary=True)
class OpenAICompatibleClient:
    """
    OpenAI-compatible API client.
    This client acts as a facade, managing multiple LLM backend adapters.
    """

    def __init__(self, config_provider: ConfigProvider):
        """
        Initialize the client.
        Args:
            config_provider: Configuration provider used to load llm_backends.yaml.
        """
        self.config_provider = config_provider
        self._adapters: Dict[str, LLMBackendAdapter] = {}
        self._config: Dict[str, Any] = self.config_provider.get_config("llm_backends")
        self._init_locks: Dict[str, asyncio.Lock] = {}  # One lock per backend
        self._lock_creation_lock = asyncio.Lock()  # Lock for creating locks

    async def _get_adapter(self, backend_name: str) -> LLMBackendAdapter:
        """
        Asynchronously initialize and retrieve the adapter for the specified backend on demand.
        Uses locks to ensure concurrency safety and avoid duplicate initialization.
        """
        # If adapter already exists, return directly
        if backend_name in self._adapters:
            return self._adapters[backend_name]

        # Ensure each backend has a corresponding lock
        async with self._lock_creation_lock:
            if backend_name not in self._init_locks:
                self._init_locks[backend_name] = asyncio.Lock()

        # Use backend-specific lock to ensure concurrency safety
        async with self._init_locks[backend_name]:
            # Re-check, as it might have been initialized by another coroutine while waiting for the lock
            if backend_name in self._adapters:
                return self._adapters[backend_name]

            llm_backends = self._config.get("llm_backends", {})
            if backend_name not in llm_backends:
                raise ValueError(
                    f"Backend '{backend_name}' not found in configuration."
                )

            backend_config = llm_backends[backend_name]
            provider = backend_config.get("provider", "openai")

            try:
                adapter: LLMBackendAdapter
                if provider in ["openai", "azure", "custom", "ollama"]:
                    adapter = OpenAIAdapter(backend_config)
                elif provider == "anthropic":
                    adapter = AnthropicAdapter(backend_config)
                elif provider == "gemini":
                    adapter = GeminiAdapter(backend_config)
                else:
                    raise ValueError(f"Unsupported provider type: {provider}")

                self._adapters[backend_name] = adapter
                return adapter
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize adapter for backend '{backend_name}': {e}"
                ) from e

    def _get_param_with_priority(
        self,
        param_name: str,
        passed_value: Any,
        default_settings: dict,
        backend_config: dict,
    ) -> Any:
        """
        Get parameter priority: passed value > backend_config > default_settings
        """
        if passed_value is not None:
            return passed_value
        if backend_config.get(param_name) is not None:
            return backend_config.get(param_name)
        if default_settings.get(param_name) is not None:
            return default_settings[param_name]
        return None

    async def chat_completion(
        self,
        messages: List[ChatMessage],
        backend: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        thinking_budget: Optional[
            int
        ] = None,  # Add support for thinking_budget parameter
        stream: bool = False,
    ) -> Union[ChatCompletionResponse, AsyncGenerator[str, None]]:
        """
        Perform chat completion.
        Args:
            messages: List of chat messages
            backend: Backend name, use default backend if not specified
            ... other params
            thinking_budget: Thinking budget (used to support think functionality)
            stream: Whether to stream the response
        Returns:
            Chat completion response or streaming generator
        """
        # Select backend
        backend_name = backend or self._config.get("default_backend", "openai")
        default_settings = self._config.get("default_settings", {})
        backend_config = self._config.get("llm_backends", {}).get(backend_name, {})

        # Unified parameter priority handling
        final_params = {}
        param_definitions = {
            # backend_config has default value below
            "model": model,
            # default_settings has default value below
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "thinking_budget": thinking_budget,
        }
        for name, value in param_definitions.items():
            final_params[name] = self._get_param_with_priority(
                name, value, default_settings, backend_config
            )

        # Assemble request
        request = ChatCompletionRequest(
            messages=messages, stream=stream, **final_params
        )

        adapter = await self._get_adapter(backend_name)
        return await adapter.chat_completion(request)

    def chat_completion_sync(
        self,
        messages: List[ChatMessage],
        backend: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stream: bool = False,
    ) -> Union[ChatCompletionResponse, AsyncGenerator[str, None]]:
        """
        Synchronous version of performing chat completion.

        Note: This method no longer supports synchronous calls because certain LLM adapters (e.g., Gemini)
        internally bind to the event loop, and creating new threads and event loops causes issues.

        Please use the asynchronous version chat_completion() method.
        """
        raise NotImplementedError(
            "Synchronous version of chat completion is no longer supported because certain LLM adapters (e.g., Gemini) "
            "internally bind to the event loop. Please use the asynchronous version chat_completion() method."
        )

    def get_available_backends(self) -> List[str]:
        """Get list of available backends"""
        return list(self._config.get("llm_backends", {}).keys())

    async def get_available_models(self, backend: Optional[str] = None) -> List[str]:
        """Get list of available models for the specified backend"""
        backend_name = backend or self._config.get("default_backend", "openai")
        try:
            adapter = await self._get_adapter(backend_name)
            return adapter.get_available_models()
        except (ValueError, RuntimeError):
            return []

    def get_available_models_sync(self, backend: Optional[str] = None) -> List[str]:
        """
        Synchronous version of getting available models for the specified backend.

        Note: This method no longer supports synchronous calls because certain LLM adapters (e.g., Gemini)
        internally bind to the event loop, and creating new threads and event loops causes issues.

        Please use the asynchronous version get_available_models() method.
        """
        raise NotImplementedError(
            "Synchronous version of model retrieval is no longer supported because certain LLM adapters (e.g., Gemini) "
            "internally bind to the event loop. Please use the asynchronous version get_available_models() method."
        )

    def get_backend_info(self, backend: str) -> Optional[Dict[str, Any]]:
        """Get backend information, hiding sensitive data"""
        config = self._config.get("llm_backends", {}).get(backend)
        if config:
            safe_config = config.copy()
            if "api_key" in safe_config:
                safe_config["api_key"] = (
                    f"***{safe_config['api_key'][-4:]}"
                    if len(safe_config.get('api_key', '')) > 4
                    else "***"
                )
            return safe_config
        return None

    def reload_config(self):
        """Reload configuration and clear existing adapter instances and locks"""
        self._config = self.config_provider.get_config("llm_backends")
        self._adapters.clear()
        self._init_locks.clear()

    async def close(self):
        """Close HTTP client connections for all adapters"""
        for adapter in self._adapters.values():
            if hasattr(adapter, 'close'):
                await adapter.close()  # type: ignore

    def close_sync(self):
        """
        Synchronous version of closing HTTP client connections for all adapters.

        Note: This method no longer supports synchronous calls because certain LLM adapters (e.g., Gemini)
        internally bind to the event loop, and creating new threads and event loops causes issues.

        Please use the asynchronous version close() method.
        """
        raise NotImplementedError(
            "Synchronous version of close operation is no longer supported because certain LLM adapters (e.g., Gemini) "
            "internally bind to the event loop. Please use the asynchronous version close() method."
        )
