"""Factory for building a rerank provider from :class:`RerankSettings`.

The ``provider`` field on :class:`RerankSettings` selects which concrete
implementation to build:

    - ``"deepinfra"`` → :class:`DeepInfraRerankProvider`
    - ``"vllm"``      → :class:`VllmRerankProvider`
    - ``"dashscope"`` → :class:`DashScopeRerankProvider`
      (Aliyun Bailian ``gte-rerank-v2``)

Adding a new provider = one match arm here + one new file under
:mod:`everos.component.rerank`.
"""

from __future__ import annotations

from urllib.parse import urlparse

from everos.config import RerankSettings
from everos.core.observability.logging import get_logger

from .dashscope_provider import DashScopeRerankProvider
from .deepinfra_provider import DeepInfraRerankProvider
from .protocol import RerankProvider
from .vllm_provider import VllmRerankProvider

logger = get_logger(__name__)

# host substring → provider. Ordered most-specific first; matched against
# the ``base_url`` host so a Bailian / DeepInfra URL routes to the right
# request-shape without the operator also having to set ``provider``.
_PROVIDER_HOST_HINTS: tuple[tuple[str, str], ...] = (
    ("dashscope.aliyuncs.com", "dashscope"),
    ("deepinfra.com", "deepinfra"),
)

# Fallback when the host matches no hint and ``provider`` is unset. Keeps
# the historical default so existing configs without ``provider`` are
# unaffected.
_DEFAULT_PROVIDER = "deepinfra"


def _infer_provider(base_url: str) -> str | None:
    """Infer the rerank provider from the ``base_url`` host, or ``None``."""
    host = (urlparse(base_url).hostname or "").lower()
    if not host:
        return None
    for needle, provider in _PROVIDER_HOST_HINTS:
        if host == needle or host.endswith(f".{needle}"):
            return provider
    return None


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
            ``api_key`` is required for ``deepinfra`` and ``dashscope``;
            optional (empty string) for ``vllm`` self-hosted endpoints.
            ``dashscope`` currently supports ``gte-rerank-v2`` only.

    Notes:
        When ``settings.provider`` is ``None`` the provider is inferred
        from the ``base_url`` host (see :data:`_PROVIDER_HOST_HINTS`),
        falling back to ``"deepinfra"`` for unrecognized hosts.
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

    provider = settings.provider
    if provider is None:
        inferred = _infer_provider(settings.base_url)
        provider = inferred or _DEFAULT_PROVIDER
        logger.info(
            "rerank_provider_inferred",
            provider=provider,
            inferred=inferred is not None,
            base_url=settings.base_url,
        )

    if provider == "deepinfra":
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
    if provider == "vllm":
        return VllmRerankProvider(
            model=settings.model,
            api_key=api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
            batch_size=settings.batch_size,
            max_concurrent=settings.max_concurrent,
        )
    if provider == "dashscope":
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
    raise ValueError(f"unknown rerank provider: {provider!r}")
