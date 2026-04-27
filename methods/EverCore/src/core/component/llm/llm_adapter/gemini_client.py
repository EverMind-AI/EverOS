import asyncio
import os
from typing import Dict, Any, List, Union, AsyncGenerator, Optional
from google.genai.client import Client
from google.genai.types import GenerateContentConfig, ContentDict
from google.genai.types import ThinkingConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from core.di.decorators import component
from core.component.config_provider import ConfigProvider
from core.constants.errors import ErrorMessage


@component(name="gemini_client", primary=True)
class GeminiClient:
    """Google Gemini API client - directly returns raw response"""

    def __init__(self, config_provider: ConfigProvider):
        """
        Initialize Gemini client

        Args:
            config_provider: Configuration provider, used to load llm_backends configuration
        """
        self.config_provider = config_provider
        self._llm_config: Dict[str, Any] = self.config_provider.get_config(
            "llm_backends"
        )

        # Get Gemini backend configuration
        gemini_backends = self._llm_config.get("llm_backends", {})
        if "gemini" not in gemini_backends:
            raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

        self._config = gemini_backends["gemini"]

        # Get API key, priority: configuration file > environment variable
        self.api_key = self._config.get("api_key") or os.getenv("GEMINI_API_KEY")
        self.default_model = self._config.get("default_model") or self._config.get(
            "model", "gemini-2.5-flash"
        )
        self.max_retries = self._config.get("max_retries", 3)

        if not self.api_key:
            raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

        # Use the new google.genai API
        self.client = Client(api_key=self.api_key)

    async def generate_content(
        self,
        messages: Union[List[Dict[str, Any]], List[BaseMessage], str],
        model: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None,
        thinking_budget: Optional[int] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_mime_type: Optional[str] = None,
    ) -> Union[Any, AsyncGenerator[str, None]]:
        """
        Generate content - directly return Gemini raw response

        Args:
            messages: Message list, supports multiple formats:
                     - List[Dict]: Standard message format [{"role": "user", "content": "..."}]
                     - List[BaseMessage]: LangChain message objects
                     - str: Single text message
            model: Model name, use default model if None
            temperature: Temperature parameter
            top_p: top_p parameter
            max_tokens: Maximum output token count
            thinking_budget: Thinking budget (supported only by certain models)
            stream: Whether to stream output
            tools: Tool list, used for function calling and grounding
            response_mime_type: Response MIME type, e.g., "application/json"

        Returns:
            If stream=False, return Gemini raw response object
            If stream=True, return async generator
        """
        if not model:
            model = self.default_model

        # Convert message format
        contents = self._convert_messages_to_gemini_format(messages)

        # Build GenerationConfig
        generation_config_params = {"temperature": temperature, "top_p": top_p}

        if max_tokens is not None:
            generation_config_params["max_output_tokens"] = max_tokens

        # If thinking_budget parameter is provided, create ThinkingConfig
        if thinking_budget is not None:
            thinking_config = ThinkingConfig(thinking_budget=thinking_budget)
            generation_config_params["thinking_config"] = thinking_config

        # Support response MIME type
        if response_mime_type is not None:
            generation_config_params["response_mime_type"] = response_mime_type

        # Support tools (tools should be in config)
        if tools is not None:
            generation_config_params["tools"] = tools

        generation_config = GenerateContentConfig(**generation_config_params)

        for attempt in range(self.max_retries):
            try:
                if stream:
                    return self._stream_generate_content(
                        model=model,
                        contents=contents,
                        generation_config=generation_config,
                    )
                else:
                    # Directly return Gemini raw response (tools already in config)
                    response = await self.client.aio.models.generate_content(
                        model=model, contents=contents, config=generation_config
                    )
                    return response
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"An unexpected error occurred in GeminiClient: {e}"
                    ) from e
                await asyncio.sleep(2**attempt)

        raise RuntimeError("Gemini content generation failed after multiple retries.")

    def _convert_messages_to_gemini_format(
        self, messages: Union[List[Dict[str, Any]], List[BaseMessage], str]
    ) -> List[ContentDict]:
        """
        Convert message list to Gemini format - compatible with multiple input formats

        Args:
            messages: Supports the following formats:
                     - str: Single text message, automatically converted to user role
                     - List[Dict]: Standard message format [{"role": "user", "content": "..."}]
                     - List[BaseMessage]: List of LangChain message objects

        Returns:
            List[ContentDict]: Message list in Gemini API format
        """
        contents = []

        # Handle string input
        if isinstance(messages, str):
            contents.append(ContentDict(role="user", parts=[{"text": messages}]))
            return contents

        # Handle list input
        if not isinstance(messages, list):
            raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

        for msg in messages:
            # Handle LangChain message objects
            if isinstance(msg, BaseMessage):
                if isinstance(msg, HumanMessage):
                    contents.append(
                        ContentDict(role="user", parts=[{"text": msg.content}])
                    )
                elif isinstance(msg, AIMessage):
                    contents.append(
                        ContentDict(role="model", parts=[{"text": msg.content}])
                    )
                elif isinstance(msg, SystemMessage):
                    # Gemini handles system messages as model role
                    contents.append(
                        ContentDict(role="model", parts=[{"text": msg.content}])
                    )
                else:
                    # For other message types, try to get content attribute
                    content = getattr(msg, 'content', str(msg))
                    contents.append(ContentDict(role="user", parts=[{"text": content}]))
                continue

            # Handle dictionary format messages
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")

                # Handle possible nested content structure
                if isinstance(content, list):
                    # If content is a list, extract text parts
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
                            elif "text" in part:
                                text_parts.append(part["text"])
                        else:
                            text_parts.append(str(part))
                    content = " ".join(text_parts)

                # Convert role mapping
                gemini_role = self._map_role_to_gemini(role)
                contents.append(
                    ContentDict(role=gemini_role, parts=[{"text": str(content)}])
                )
                continue

            # Handle other types, try to convert to string
            try:
                # Check if it has role and content attributes
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    role = getattr(msg, 'role')
                    content = getattr(msg, 'content')
                    gemini_role = self._map_role_to_gemini(role)
                    contents.append(
                        ContentDict(role=gemini_role, parts=[{"text": str(content)}])
                    )
                else:
                    # Treat as user message
                    contents.append(
                        ContentDict(role="user", parts=[{"text": str(msg)}])
                    )
            except Exception:
                # Final fallback
                contents.append(ContentDict(role="user", parts=[{"text": str(msg)}]))

        return contents

    def _map_role_to_gemini(self, role: str) -> str:
        """
        Map standard roles to Gemini format

        Args:
            role: Original role name

        Returns:
            str: Gemini-formatted role name
        """
        role_lower = str(role).lower()

        if role_lower in ["user", "human"]:
            return "user"
        elif role_lower in ["assistant", "ai", "model", "bot"]:
            return "model"
        elif role_lower in ["system"]:
            # Gemini handles system messages as model role
            return "model"
        else:
            # Default as user message
            return "user"

    async def _stream_generate_content(
        self,
        model: str,
        contents: List[ContentDict],
        generation_config: GenerateContentConfig,
    ) -> AsyncGenerator[str, None]:
        """Stream content generation"""
        # tools already passed in generation_config
        response_stream = await self.client.aio.models.generate_content_stream(
            model=model, contents=contents, config=generation_config
        )
        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    def reload_config(self):
        """Reload configuration"""
        self._llm_config = self.config_provider.get_config("llm_backends")

        # Get Gemini backend configuration
        gemini_backends = self._llm_config.get("llm_backends", {})
        if "gemini" not in gemini_backends:
            raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

        self._config = gemini_backends["gemini"]

        # Update configuration
        self.api_key = self._config.get("api_key") or os.getenv("GEMINI_API_KEY")
        self.default_model = self._config.get("default_model") or self._config.get(
            "model", "gemini-2.5-flash"
        )
        self.max_retries = self._config.get("max_retries", 3)

        if not self.api_key:
            raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

        # Recreate client
        self.client = Client(api_key=self.api_key)

    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return self._config.get("models", [])

    async def close(self):
        """Close client (Gemini library does not require this)"""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
