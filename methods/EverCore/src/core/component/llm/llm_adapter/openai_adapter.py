from typing import Dict, Any, List, Union, AsyncGenerator
import os
import openai
from core.component.llm.llm_adapter.completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from core.component.llm.llm_adapter.llm_backend_adapter import LLMBackendAdapter
from core.constants.errors import ErrorMessage
from core.di.utils import get_bean_by_type
from core.component.token_usage_collector import TokenUsageCollector


class OpenAIAdapter(LLMBackendAdapter):
    """OpenAI API adapter (implemented based on the official openai package)"""

    def __init__(self, config: Dict[str, Any]):
        # Save configuration
        self.config = config
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.base_url = config.get("base_url") or os.getenv("OPENAI_BASE_URL")
        self.timeout = config.get("timeout", 600)

        if not self.api_key:
            raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

        # Instantiate openai async client
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> Union[ChatCompletionResponse, AsyncGenerator[str, None]]:
        """
        Perform chat completion, supporting both streaming and non-streaming modes.
        """
        if not request.model:
            raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

        params = request.to_dict()
        # The request `to_dict` method already filters for None values, but we can be explicit here for clarity
        # for what the openai client expects.
        client_params = {
            "model": params.get("model"),
            "messages": params.get("messages"),
            "temperature": params.get("temperature"),
            "max_tokens": params.get("max_tokens"),
            "top_p": params.get("top_p"),
            "frequency_penalty": params.get("frequency_penalty"),
            "presence_penalty": params.get("presence_penalty"),
            "stream": params.get("stream", False),
        }
        # Remove None values to avoid openai errors
        final_params = {k: v for k, v in client_params.items() if v is not None}

        try:
            if final_params.get("stream"):
                # Streaming response, return async generator
                # Enable usage reporting in the final streaming chunk
                final_params["stream_options"] = {"include_usage": True}

                async def stream_gen():
                    usage_data = None
                    try:
                        response_stream = await self.client.chat.completions.create(
                            **final_params
                        )
                        async for chunk in response_stream:
                            # Final chunk carries usage data (no choices)
                            if hasattr(chunk, 'usage') and chunk.usage:
                                usage_data = chunk.usage
                            if chunk.choices:
                                content = getattr(
                                    chunk.choices[0].delta, "content", None
                                )
                                if content:
                                    yield content
                    finally:
                        # Report usage even if client disconnects mid-stream
                        if usage_data:
                            try:
                                collector = get_bean_by_type(TokenUsageCollector)
                                collector.add(
                                    final_params.get("model", "unknown"),
                                    usage_data.prompt_tokens or 0,
                                    usage_data.completion_tokens or 0,
                                    call_type="llm",
                                )
                            except Exception:
                                pass

                return stream_gen()
            else:
                # Non-streaming response
                response = await self.client.chat.completions.create(**final_params)
                # Report token usage
                if hasattr(response, 'usage') and response.usage:
                    try:
                        collector = get_bean_by_type(TokenUsageCollector)
                        collector.add(
                            final_params.get("model", "unknown"),
                            response.usage.prompt_tokens or 0,
                            response.usage.completion_tokens or 0,
                            call_type="llm",
                        )
                    except Exception:
                        pass
                return ChatCompletionResponse.from_dict(response.model_dump())
        except Exception as e:
            raise RuntimeError(f"OpenAI chat completion request failed: {e}")

    def get_available_models(self) -> List[str]:
        """Get available model list (can be extended to call openai model list API)"""
        return self.config.get("models", [])
