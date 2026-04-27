import os
from memory_layer.llm.openai_provider import OpenAIProvider
from core.observation.stage_timer import timed

# Constants for default provider and model settings
DEFAULT_PROVIDER_NAME = "default"

# for default settings
DEFAULT_LLM_MODEL = "openai/gpt-4.1-mini"

# for default & scenes default settings
DEFAULT_LLM_TEMPERATURE = 0.3
DEFAULT_LLM_MAX_TOKENS = 16384


def _normalize_provider(provider_type: str | None) -> str:
    return provider_type.lower()


def resolve_provider_env(
    provider_type: str,
    api_key: str | None = None,
    base_url: str | None = None,
    use_legacy_default: bool = False,
) -> tuple[str | None, str | None]:
    provider_upper = _normalize_provider(provider_type).upper()

    if not api_key:
        api_key = os.getenv(f"{provider_upper}_API_KEY")
    if not api_key and use_legacy_default:
        api_key = os.getenv("LLM_API_KEY")

    if not base_url:
        base_url = os.getenv(f"{provider_upper}_BASE_URL")
    if not base_url and use_legacy_default:
        base_url = os.getenv("LLM_BASE_URL")

    return api_key, base_url


def build_default_provider() -> "LLMProvider":
    """Build the default LLM provider from environment variables."""
    return LLMProvider(
        provider_type=DEFAULT_PROVIDER_NAME,
        model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
        temperature=float(os.getenv("LLM_TEMPERATURE", str(DEFAULT_LLM_TEMPERATURE))),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", str(DEFAULT_LLM_MAX_TOKENS))),
    )


class LLMProvider:
    def __init__(self, provider_type: str, **kwargs):
        provider_type = _normalize_provider(provider_type)
        use_legacy_default = provider_type == DEFAULT_PROVIDER_NAME
        if use_legacy_default:
            provider_type = _normalize_provider(os.getenv("LLM_PROVIDER"))
        self.provider_type = provider_type
        api_key, base_url = resolve_provider_env(
            provider_type,
            api_key=kwargs.pop("api_key", None),
            base_url=kwargs.pop("base_url", None),
            use_legacy_default=use_legacy_default,
        )

        # Validate required configuration
        provider_upper = provider_type.upper()
        if not api_key:
            raise ValueError(
                f"No API key found for provider '{provider_type}'. "
                f"Please set {provider_upper}_API_KEY environment variable."
            )
        if not base_url:
            raise ValueError(
                f"No base URL found for provider '{provider_type}'. "
                f"Please set {provider_upper}_BASE_URL environment variable."
            )

        self.provider = OpenAIProvider(
            provider_type=provider_type, api_key=api_key, base_url=base_url, **kwargs
        )
        # TODO: add other providers

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
        response_format: dict | None = None,
    ) -> str:
        with timed("call_llm"):
            return await self.provider.generate(
                prompt, temperature, max_tokens, extra_body, response_format
            )
