"""Factory for building a rerank provider from :class:`RerankSettings`.

The ``provider`` field on :class:`RerankSettings` selects which concrete
implementation to build:

    - ``"deepinfra"``  → :class:`DeepInfraRerankProvider`
    - ``"vllm"``       → :class:`VllmRerankProvider`
    - ``"dashscope"``  → :class:`DashScopeRerankProvider`

Adding a new provider = one match arm here + one new file under
:mod:`everos.component.rerank`.
"""

from __future__ import annotations

from everos.config import RerankSettings

from .dashscope_provider import DashScopeRerankProvider
from .deepinfra_provider import DeepInfraRerankProvider
from .protocol import RerankProvider
from .vllm_provider import VllmRerankProvider


def build_rerank_provider(settings: RerankSettings) -> RerankProvider:
    """Build a rerank provider from settings.

    Args:
        settings: The :class:`RerankSettings` slice from
            :func:`everos.config.load_settings`.

    Returns:
        A :class:`RerankProvider` ready to call ``rerank``.

    Raises:
        ValueError: If ``model`` or ``base_url`` is unset, or if
            ``provider`` does not match a known implementation.
            ``api_key`` is required for ``deepinfra``; optional (empty
            string) for ``vllm`` self-hosted endpoints.
    """
    if not settings.model:
        raise ValueError(
            "Rerank model is not configured "
            "(set EVEROS_RERANK__MODEL or [rerank] model in user toml)"
        )
    if not settings.base_url:
        raise ValueError(
            "Rerank base_url is not configured (set EVEROS_RERANK__BASE_URL)"
        )
    api_key = settings.api_key.get_secret_value() if settings.api_key else ""

    if settings.provider == "deepinfra":
        if not api_key:
            raise ValueError(
                "DeepInfra rerank api_key is not configured "
                "(set EVEROS_RERANK__API_KEY)"
            )
        return DeepInfraRerankProvider(
            model=settings.model,
            api_key=api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
            batch_size=settings.batch_size,
            max_concurrent=settings.max_concurrent,
        )
    if settings.provider == "vllm":
        return VllmRerankProvider(
            model=settings.model,
            api_key=api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
            batch_size=settings.batch_size,
            max_concurrent=settings.max_concurrent,
        )
    if settings.provider == "dashscope":
        if not api_key:
            raise ValueError(
                "DashScope rerank api_key is not configured "
                "(set EVEROS_RERANK__API_KEY)"
            )
        return DashScopeRerankProvider(
            model=settings.model,
            api_key=api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
            batch_size=settings.batch_size,
            max_concurrent=settings.max_concurrent,
        )
    raise ValueError(f"unknown rerank provider: {settings.provider!r}")
