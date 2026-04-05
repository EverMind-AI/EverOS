import re
from typing import Dict, Any, List, Union, AsyncGenerator
import os
import openai
from core.component.llm.llm_adapter.completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from core.component.llm.llm_adapter.llm_backend_adapter import LLMBackendAdapter
from core.constants.errors import ErrorMessage


class MiniMaxAdapter(LLMBackendAdapter):
    """MiniMax API adapter using OpenAI-compatible interface.

    MiniMax provides an OpenAI-compatible API at https://api.minimax.io/v1.
    This adapter handles MiniMax-specific behaviors:
    - Temperature clamping to [0.01, 1.0] range
    - Stripping <think>...</think> tags from reasoning model responses
    - Auto-detection of MINIMAX_API_KEY environment variable
    """

    # MiniMax API requires temperature in (0.0, 1.0] for most models,
    # but temperature=0 is now accepted. We clamp to [0.01, 1.0] for safety
    # with older model versions.
    MIN_TEMPERATURE = 0.01
    MAX_TEMPERATURE = 1.0

    # Pattern to strip thinking tags from reasoning model output
    _THINK_TAG_PATTERN = re.compile(
        r"<think>.*?</think>\s*", flags=re.DOTALL
    )

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("api_key") or os.getenv("MINIMAX_API_KEY")
        self.base_url = config.get(
            "base_url", "https://api.minimax.io/v1"
        )
        self.timeout = config.get("timeout", 600)

        if not self.api_key:
            raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    @classmethod
    def _clamp_temperature(cls, temperature: float | None) -> float | None:
        """Clamp temperature to MiniMax's accepted range."""
        if temperature is None:
            return None
        return max(cls.MIN_TEMPERATURE, min(cls.MAX_TEMPERATURE, temperature))

    @classmethod
    def _strip_think_tags(cls, text: str) -> str:
        """Strip <think>...</think> blocks from model output."""
        return cls._THINK_TAG_PATTERN.sub("", text).strip()

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> Union[ChatCompletionResponse, AsyncGenerator[str, None]]:
        """Perform chat completion via MiniMax OpenAI-compatible API."""
        if not request.model:
            raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

        params = request.to_dict()
        client_params = {
            "model": params.get("model"),
            "messages": params.get("messages"),
            "temperature": self._clamp_temperature(params.get("temperature")),
            "max_tokens": params.get("max_tokens"),
            "top_p": params.get("top_p"),
            "frequency_penalty": params.get("frequency_penalty"),
            "presence_penalty": params.get("presence_penalty"),
            "stream": params.get("stream", False),
        }
        final_params = {k: v for k, v in client_params.items() if v is not None}

        try:
            if final_params.get("stream"):
                async def stream_gen():
                    response_stream = await self.client.chat.completions.create(
                        **final_params
                    )
                    async for chunk in response_stream:
                        content = getattr(
                            chunk.choices[0].delta, "content", None
                        )
                        if content:
                            yield self._strip_think_tags(content)

                return stream_gen()
            else:
                response = await self.client.chat.completions.create(
                    **final_params
                )
                resp_dict = response.model_dump()
                # Strip think tags from non-streaming response
                for choice in resp_dict.get("choices", []):
                    msg = choice.get("message", {})
                    if msg.get("content"):
                        msg["content"] = self._strip_think_tags(msg["content"])
                return ChatCompletionResponse.from_dict(resp_dict)
        except Exception as e:
            raise RuntimeError(
                f"MiniMax chat completion request failed: {e}"
            )

    def get_available_models(self) -> List[str]:
        """Get available MiniMax model list."""
        return self.config.get("models", [])
