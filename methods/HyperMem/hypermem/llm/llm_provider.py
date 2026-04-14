import os
from typing import Optional
from .openai_provider import OpenAIProvider

class LLMProvider:
    def __init__(self, provider_type: str, **kwargs):
        self.provider_type = provider_type
        if provider_type == "openai":
            self.provider = OpenAIProvider(**kwargs)
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}. Supported types: 'openai'")

    async def generate(self, prompt: str, temperature: float | None = None, extra_body: dict | None = None, response_format: dict | None = None) -> str:
        return await self.provider.generate(prompt, temperature,self.provider.max_tokens, extra_body, response_format)

    def get_accumulated_stats(self) -> Optional[dict]:
        """Get accumulated statistics"""
        if hasattr(self.provider, 'get_accumulated_stats'):
            return self.provider.get_accumulated_stats()
        return None
    
    def reset_accumulated_stats(self) -> None:
        """Reset accumulated statistics"""
        if hasattr(self.provider, 'reset_accumulated_stats'):
            self.provider.reset_accumulated_stats()

