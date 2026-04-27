import asyncio
import time
import logging
from typing import Dict, Any, List, Union, AsyncGenerator
import os
from google.genai.client import Client
from core.di.decorators import service
from google.genai.types import (
    GenerateContentConfig,
    ContentDict,
    HarmCategory,
    HarmBlockThreshold,
)
from google.genai.types import ThinkingConfig
from core.component.llm.llm_adapter.completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from core.component.llm.llm_adapter.message import MessageRole
from core.component.llm.llm_adapter.llm_backend_adapter import LLMBackendAdapter

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from core.constants.errors import ErrorMessage
from core.di.utils import get_bean_by_type
from core.component.token_usage_collector import TokenUsageCollector

logger = logging.getLogger(__name__)


class GeminiAdapter(LLMBackendAdapter):
    """Google Gemini API adapter"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("api_key") or os.getenv("GEMINI_API_KEY")
        self.max_retries = config.get("max_retries", 3)

        if not self.api_key:
            raise ValueError(ErrorMessage.CONFIGURATION_MISSING.value)

        # Use the new google.genai API
        self.client = Client(api_key=self.api_key)
        self.model_name = self.config.get("default_model", "gemini-2.5-flash")

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> Union[ChatCompletionResponse, AsyncGenerator[str, None]]:
        """Perform chat completion (convert to Gemini format)"""
        if not request.model:
            request.model = self.model_name

        contents = self._convert_messages_to_gemini_format(request.messages)

        # Build GenerationConfig
        generation_config_params = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_output_tokens": request.max_tokens,
        }

        # If thinking_budget parameter is provided, create ThinkingConfig
        thinking_config = None
        if request.thinking_budget is not None:
            thinking_config = ThinkingConfig(thinking_budget=request.thinking_budget)
            generation_config_params["thinking_config"] = thinking_config

        generation_config = GenerateContentConfig(**generation_config_params)

        for attempt in range(self.max_retries):
            try:
                if request.stream:
                    return self._stream_chat_completion(
                        contents=contents, generation_config=generation_config
                    )
                else:
                    response = await self.client.aio.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=generation_config,
                    )
                    return self._convert_gemini_response(response, request.model)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"An unexpected error occurred in GeminiAdapter: {e}"
                    ) from e
                await asyncio.sleep(2**attempt)

        raise RuntimeError(
            "Gemini chat completion request failed after multiple retries."
        )

    def _convert_messages_to_gemini_format(
        self, messages: List[Dict[str, Any]]
    ) -> List[ContentDict]:
        """Convert message list to Gemini format"""
        contents = []
        for msg in messages:
            if type(msg) == HumanMessage:
                contents.append(ContentDict(role="user", parts=[{"text": msg.content}]))
            elif type(msg) == AIMessage:
                contents.append(
                    ContentDict(role="model", parts=[{"text": msg.content}])
                )
            elif type(msg) == SystemMessage:
                contents.append(
                    ContentDict(role="model", parts=[{"text": msg.content}])
                )
            elif MessageRole(msg.role) == MessageRole.SYSTEM:
                contents.append(
                    ContentDict(role="model", parts=[{"text": msg.content}])
                )
            elif MessageRole(msg.role) == MessageRole.USER:
                contents.append(ContentDict(role="user", parts=[{"text": msg.content}]))
            elif MessageRole(msg.role) == MessageRole.ASSISTANT:
                contents.append(
                    ContentDict(role="model", parts=[{"text": msg.content}])
                )
        return contents

    def _convert_gemini_response(self, response, model: str) -> ChatCompletionResponse:
        """Convert Gemini response to OpenAI format"""
        # Extract token usage from Gemini's usage_metadata
        usage = {}
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            prompt_tokens = (
                getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
            )
            completion_tokens = (
                getattr(response.usage_metadata, 'candidates_token_count', 0) or 0
            )
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0)
                or 0,
            }
            # Report token usage
            try:
                collector = get_bean_by_type(TokenUsageCollector)
                collector.add(model, prompt_tokens, completion_tokens, call_type="llm")
            except Exception:
                pass

        result = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",  # Gemini does not provide ID, we generate one
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response.text},
                    "finish_reason": "stop",  # Gemini API v1 does not directly provide finish_reason
                }
            ],
            usage=usage,
        )

        # Attach the original Gemini response object to the result for use by the audit system
        result._original_gemini_response = response

        return result

    async def _stream_chat_completion(
        self, contents: List[ContentDict], generation_config: GenerateContentConfig
    ) -> AsyncGenerator[str, None]:
        """Streamed chat completion"""
        last_chunk = None
        try:
            response_stream = await self.client.aio.models.generate_content_stream(
                model=self.model_name, contents=contents, config=generation_config
            )
            async for chunk in response_stream:
                last_chunk = chunk
                if chunk.text:
                    yield chunk.text
        finally:
            # Report usage even if client disconnects mid-stream
            if (
                last_chunk
                and hasattr(last_chunk, 'usage_metadata')
                and last_chunk.usage_metadata
            ):
                try:
                    collector = get_bean_by_type(TokenUsageCollector)
                    collector.add(
                        self.model_name,
                        getattr(last_chunk.usage_metadata, 'prompt_token_count', 0)
                        or 0,
                        getattr(last_chunk.usage_metadata, 'candidates_token_count', 0)
                        or 0,
                        call_type="llm",
                    )
                except Exception:
                    pass

    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return self.config.get("models", [])

    async def close(self):
        """Close client (not required by Gemini library)"""
        pass
