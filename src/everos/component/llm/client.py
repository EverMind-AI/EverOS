"""Process-wide LLM client accessor.

Lazy singleton — first call reads settings and builds the algo LLM
client; subsequent calls return the cached instance. Raises
:class:`LLMNotConfiguredError` when no credentials are present so
misconfiguration surfaces at app startup (via the LLM lifespan
provider) instead of silently failing per-request downstream.
"""

from __future__ import annotations

from everalgo.llm import build_client
from everalgo.llm.config import LLMConfig
from everalgo.llm.protocols import LLMClient

from everos.config import load_settings
from everos.core.observability.logging import get_logger

logger = get_logger(__name__)


class LLMNotConfiguredError(RuntimeError):
    """Raised when ``settings.llm`` is missing ``api_key`` or ``base_url``."""


_llm_client: LLMClient | None = None
_multimodal_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Return the singleton algo LLM client.

    Raises:
        LLMNotConfiguredError: When ``settings.llm.api_key`` or
            ``settings.llm.base_url`` is unset.
    """
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    llm_cfg = load_settings().llm
    api_key = (
        llm_cfg.api_key.get_secret_value() if llm_cfg.api_key is not None else None
    )
    if not api_key or not llm_cfg.base_url:
        raise LLMNotConfiguredError(
            "LLM is required; set EVEROS_LLM__API_KEY + EVEROS_LLM__BASE_URL"
        )
    _llm_client = build_client(
        LLMConfig(
            model=llm_cfg.model,
            api_key=api_key,
            base_url=llm_cfg.base_url,
        )
    )
    logger.info("llm_client_built", model=llm_cfg.model)
    return _llm_client


def get_multimodal_llm_client() -> LLMClient:
    """Return the singleton multimodal LLM client (for everalgo.parser).

    Reads the flat ``[multimodal]`` config — kept separate from the main
    ``[llm]`` so parsing can target a vision/audio-capable endpoint.

    Raises:
        LLMNotConfiguredError: When ``settings.multimodal.api_key`` or
            ``settings.multimodal.base_url`` is unset.
    """
    global _multimodal_client
    if _multimodal_client is not None:
        return _multimodal_client

    cfg = load_settings().multimodal
    api_key = cfg.api_key.get_secret_value() if cfg.api_key is not None else None
    if not api_key or not cfg.base_url:
        raise LLMNotConfiguredError(
            "Multimodal LLM is required for parsing; set "
            "EVEROS_MULTIMODAL__API_KEY + EVEROS_MULTIMODAL__BASE_URL"
        )
    _multimodal_client = build_client(
        LLMConfig(
            model=cfg.model,
            api_key=api_key,
            base_url=cfg.base_url,
        )
    )
    logger.info("multimodal_llm_client_built", model=cfg.model)
    return _multimodal_client
