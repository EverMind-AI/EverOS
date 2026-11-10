"""Factories for building embedding providers from settings."""

from __future__ import annotations

from everos.config import (
    EmbeddingSettings,
    VisionEmbeddingSettings,
    missing_config_error,
)

from .openai_provider import OpenAIEmbeddingProvider
from .protocol import EmbeddingProvider
from .vision_protocol import VisionEmbeddingProvider
from .vision_provider import OpenAIVisionEmbeddingProvider


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


def build_vision_embedding_provider(
    settings: VisionEmbeddingSettings,
) -> VisionEmbeddingProvider:
    """Build the picture-side provider from ``[vision_embedding]``.

    Args:
        settings: The :class:`VisionEmbeddingSettings` slice.

    Returns:
        A :class:`VisionEmbeddingProvider` ready to call ``embed_image`` /
        ``embed_images`` / ``embed_query``.

    Raises:
        ValueError: If ``model``, ``api_key`` or ``base_url`` is unset. A
            deployment without picture embeddings must not call this at all
            (the bootstrap checks ``model`` first), so a clear error here
            means the section was half-filled.
    """
    if not settings.model:
        raise ValueError(
            missing_config_error("Vision embedding model", "vision_embedding")
        )
    if not settings.api_key or not settings.api_key.get_secret_value():
        raise ValueError(
            missing_config_error("Vision embedding api_key", "vision_embedding")
        )
    if not settings.base_url:
        raise ValueError(
            missing_config_error("Vision embedding base_url", "vision_embedding")
        )
    return OpenAIVisionEmbeddingProvider(
        model=settings.model,
        api_key=settings.api_key.get_secret_value(),
        base_url=settings.base_url,
        dim=settings.dim,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
        batch_size=settings.batch_size,
        max_concurrent=settings.max_concurrent,
        query_max_concurrent=settings.query_max_concurrent,
    )
