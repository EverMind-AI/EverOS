"""
Multi-language prompt module.

Use get_prompt_by() to dynamically fetch prompts by name and language.
Default language is controlled by MEMORY_LANGUAGE env var (default: 'en').

Example:
    from memory_layer.prompts import get_prompt_by
    
    prompt = get_prompt_by("EPISODE_GENERATION_PROMPT")  # default language
    prompt = get_prompt_by("EPISODE_GENERATION_PROMPT", language="zh")  # specific language
"""

from typing import Any, Optional, Callable

from common_utils.language_utils import (
    get_prompt_language,
    is_supported_language,
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
)

# ============================================================================
# Prompt Registry - maps prompt names to module paths
# Format: {prompt_name: {language: (module_path, is_function)}}
# ============================================================================#
# TODO: Optimize prompt registration method (avoid using module paths)
_PROMPT_REGISTRY = {
    # Conversation
    "CONV_BOUNDARY_DETECTION_PROMPT": {
        "en": ("memory_layer.prompts.en.conv_prompts", False),
        "zh": ("memory_layer.prompts.zh.conv_prompts", False),
    },
    "CONV_BATCH_BOUNDARY_DETECTION_PROMPT": {
        "en": ("memory_layer.prompts.en.conv_prompts", False),
        "zh": ("memory_layer.prompts.zh.conv_prompts", False),
    },
    "CONV_SUMMARY_PROMPT": {
        "en": ("memory_layer.prompts.en.conv_prompts", False),
        "zh": ("memory_layer.prompts.zh.conv_prompts", False),
    },
    # Episode
    "EPISODE_GENERATION_PROMPT": {
        "en": ("memory_layer.prompts.en.episode_mem_prompts", False),
        "zh": ("memory_layer.prompts.zh.episode_mem_prompts", False),
    },
    "GROUP_EPISODE_GENERATION_PROMPT": {
        "en": ("memory_layer.prompts.en.episode_mem_prompts", False),
        "zh": ("memory_layer.prompts.zh.episode_mem_prompts", False),
    },
    "DEFAULT_CUSTOM_INSTRUCTIONS": {
        "en": ("memory_layer.prompts.en.episode_mem_prompts", False),
        "zh": ("memory_layer.prompts.zh.episode_mem_prompts", False),
    },
    # Profile (Explicit information + Implicit traits)
    "PROFILE_UPDATE_PROMPT": {
        "en": ("memory_layer.prompts.en.profile_prompts", False),
        "zh": ("memory_layer.prompts.zh.profile_prompts", False),
    },
    "TEAM_PROFILE_UPDATE_PROMPT": {
        "en": ("memory_layer.prompts.en.profile_prompts", False),
        "zh": ("memory_layer.prompts.zh.profile_prompts", False),
    },
    "PROFILE_INITIAL_EXTRACTION_PROMPT": {
        "en": ("memory_layer.prompts.en.profile_prompts", False),
        "zh": ("memory_layer.prompts.zh.profile_prompts", False),
    },
    "PROFILE_COMPACT_PROMPT": {
        "en": ("memory_layer.prompts.en.profile_prompts", False),
        "zh": ("memory_layer.prompts.zh.profile_prompts", False),
    },
    # Foresight
    "FORESIGHT_GENERATION_PROMPT": {
        "en": ("memory_layer.prompts.en.foresight_prompts", False),
        "zh": ("memory_layer.prompts.zh.foresight_prompts", False),
    },
    # Atomic Fact
    "ATOMIC_FACT_PROMPT": {
        "en": ("memory_layer.prompts.en.atomic_fact_prompts", False),
        "zh": ("memory_layer.prompts.zh.atomic_fact_prompts", False),
    },
    # Agent Memory
    "AGENT_TOOL_PRE_COMPRESS_PROMPT": {
        "en": ("memory_layer.prompts.en.agent_prompts", False),
        "zh": ("memory_layer.prompts.zh.agent_prompts", False),
    },
    "AGENT_CASE_FILTER_PROMPT": {
        "en": ("memory_layer.prompts.en.agent_prompts", False),
        "zh": ("memory_layer.prompts.zh.agent_prompts", False),
    },
    "AGENT_CASE_COMPRESS_PROMPT": {
        "en": ("memory_layer.prompts.en.agent_prompts", False),
        "zh": ("memory_layer.prompts.zh.agent_prompts", False),
    },
    "AGENT_SKILL_SUCCESS_EXTRACT_PROMPT": {
        "en": ("memory_layer.prompts.en.agent_prompts", False),
        "zh": ("memory_layer.prompts.zh.agent_prompts", False),
    },
    "AGENT_SKILL_FAILURE_EXTRACT_PROMPT": {
        "en": ("memory_layer.prompts.en.agent_prompts", False),
        "zh": ("memory_layer.prompts.zh.agent_prompts", False),
    },
    "AGENT_SKILL_RELEVANCE_VERIFY_PROMPT": {
        "en": ("memory_layer.prompts.en.agent_prompts", False),
        "zh": ("memory_layer.prompts.zh.agent_prompts", False),
    },
    "AGENT_SKILL_MATURITY_SCORE_PROMPT": {
        "en": ("memory_layer.prompts.en.agent_prompts", False),
        "zh": ("memory_layer.prompts.zh.agent_prompts", False),
    },
    # Clustering
    "AGENT_CLUSTER_LLM_ASSIGN_PROMPT": {
        "en": ("memory_layer.prompts.en.agent_prompts", False),
        "zh": ("memory_layer.prompts.zh.agent_prompts", False),
    },
    # Skill relevance verification
    "AGENT_SKILL_RELEVANCE_VERIFY_PROMPT": {
        "en": ("memory_layer.prompts.en.agent_prompts", False),
        "zh": ("memory_layer.prompts.zh.agent_prompts", False),
    },
}


# ============================================================================
# PromptManager - Dynamic prompt loader with caching
# ============================================================================


class PromptManager:
    """Prompt manager for dynamic multi-language prompt loading."""

    def __init__(self):
        self._module_cache: dict[str, Any] = {}

    def _load_module(self, module_path: str) -> Any:
        """Load module dynamically with caching."""
        if module_path not in self._module_cache:
            import importlib

            self._module_cache[module_path] = importlib.import_module(module_path)
        return self._module_cache[module_path]

    def get_prompt(self, prompt_name: str, language: Optional[str] = None) -> Any:
        """Get prompt by name and language.

        Args:
            prompt_name: Prompt name (e.g. "EPISODE_GENERATION_PROMPT")
            language: Language code ("en" or "zh"). Defaults to MEMORY_LANGUAGE env var.

        Returns:
            Prompt string or function.

        Raises:
            ValueError: If prompt name or language is invalid.
        """
        if language is None:
            language = get_prompt_language()
        language = language.lower()

        if prompt_name not in _PROMPT_REGISTRY:
            raise ValueError(
                f"Unknown prompt: {prompt_name}. Available: {list(_PROMPT_REGISTRY.keys())}"
            )

        prompt_info = _PROMPT_REGISTRY[prompt_name]
        if language not in prompt_info:
            raise ValueError(
                f"Language '{language}' not supported for '{prompt_name}'. Available: {list(prompt_info.keys())}"
            )

        module_path, _ = prompt_info[language]
        module = self._load_module(module_path)
        return getattr(module, prompt_name)

    def list_prompts(self) -> list[str]:
        """List all available prompt names."""
        return list(_PROMPT_REGISTRY.keys())

    def get_supported_languages(self, prompt_name: str) -> list[str]:
        """Get supported languages for a prompt."""
        if prompt_name not in _PROMPT_REGISTRY:
            return []
        return list(_PROMPT_REGISTRY[prompt_name].keys())


# Global PromptManager instance
_prompt_manager = PromptManager()


def get_prompt_by(prompt_name: str, language: Optional[str] = None) -> Any:
    """Get prompt by name and language (convenience function).

    Args:
        prompt_name: Prompt name (e.g. "EPISODE_GENERATION_PROMPT")
        language: Language code ("en" or "zh"). Defaults to MEMORY_LANGUAGE env var.

    Returns:
        Prompt string or function.

    Raises:
        ValueError: If prompt name or language is invalid.
    """
    return _prompt_manager.get_prompt(prompt_name, language)


# ============================================================================
# Exported constants (for backward compatibility)
# ============================================================================

CURRENT_LANGUAGE = get_prompt_language()
MEMORY_LANGUAGE = CURRENT_LANGUAGE


def get_current_language() -> str:
    """Get current language setting."""
    return get_prompt_language()
