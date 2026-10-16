"""Factory for building an embedding provider from :class:`EmbeddingSettings`."""

from __future__ import annotations

from everos.component.utils.config_hints import missing_config_error
from everos.config import EmbeddingSettings

from .openai_provider import OpenAIEmbeddingProvider
from .protocol import EmbeddingProvider


def build_embedding_provider(
    settings: EmbeddingSettings,
    *,
    dim: int | None = None,
) -> EmbeddingProvider:
    """Build an OpenAI-compatible embedding provider from settings.

    Args:
        settings: The :class:`EmbeddingSettings` slice from
            :func:`everos.config.load_settings`.
        dim: Target vector dimension. ``None`` (the default) follows
            ``settings.dim``, which is what the index columns are built
            from. Pass a deliberately large value to defeat the
            client-side truncation in
            :class:`~.openai_provider.OpenAIEmbeddingProvider` and observe
            the model's real output length — the startup probe does
            exactly that, since the truncation makes a longer-than-``dim``
            vector impossible to see through ``embed``.

    Returns:
        An :class:`EmbeddingProvider` ready to call ``embed`` /
        ``embed_batch``.

    Raises:
        ValueError: If ``model``, ``api_key`` or ``base_url`` is unset.
    """
    if not settings.model:
        raise ValueError(missing_config_error("Embedding model", "embedding"))
    if not settings.api_key or not settings.api_key.get_secret_value():
        raise ValueError(missing_config_error("Embedding api_key", "embedding"))
    if not settings.base_url:
        raise ValueError(missing_config_error("Embedding base_url", "embedding"))
    return OpenAIEmbeddingProvider(
        model=settings.model,
        api_key=settings.api_key.get_secret_value(),
        base_url=settings.base_url,
        dim=settings.dim if dim is None else dim,
        dimensions=settings.dimensions,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
        batch_size=settings.batch_size,
        max_concurrent=settings.max_concurrent,
        query_max_concurrent=settings.query_max_concurrent,
    )
