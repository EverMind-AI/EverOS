from abc import ABC, abstractmethod
from typing import Union, AsyncGenerator, List
from core.component.llm.llm_adapter.completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)


class LLMBackendAdapter(ABC):
    """Abstract base class for LLM backend adapter"""

    @abstractmethod
    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> Union[ChatCompletionResponse, AsyncGenerator[str, None]]:
        """Perform chat completion"""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        pass
