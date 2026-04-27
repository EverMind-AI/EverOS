"""
Vectorize Service - Hybrid Implementation with Automatic Fallback

This is the main vectorization service with built-in resilience.
Implements a hybrid strategy with flexible provider selection and automatic fallback.

Usage:
    from agentic_layer.vectorize_service import get_vectorize_service
    
    service = get_vectorize_service()
    embedding = await service.get_embedding("Hello world")  # Auto-fallback
"""

import logging
import os
import time
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
import numpy as np

from core.observation.stage_timer import timed

from core.di.decorators import service

from agentic_layer.vectorize_interface import (
    VectorizeServiceInterface,
    VectorizeError,
    UsageInfo,
)
from agentic_layer.vectorize_vllm import VllmVectorizeService, VllmVectorizeConfig
from agentic_layer.vectorize_deepinfra import (
    DeepInfraVectorizeService,
    DeepInfraVectorizeConfig,
)
from agentic_layer.metrics.vectorize_metrics import (
    record_vectorize_request,
    record_vectorize_fallback,
    record_vectorize_error,
)

logger = logging.getLogger(__name__)


@dataclass
class HybridVectorizeConfig:
    """Configuration for hybrid vectorize service with fallback"""

    # Provider types
    primary_provider: str = "vllm"  # vllm or deepinfra
    fallback_provider: str = "deepinfra"  # vllm, deepinfra, or none

    # Primary service config
    primary_api_key: str = ""
    primary_base_url: str = ""

    # Fallback service config
    fallback_api_key: str = ""
    fallback_base_url: str = ""

    # Shared model configuration
    model: str = "Qwen/Qwen3-Embedding-4B"

    # Common settings
    timeout: int = 30
    max_retries: int = 3
    batch_size: int = 10
    max_concurrent_requests: int = 5
    encoding_format: str = "float"
    dimensions: int = 1024

    # Fallback behavior
    enable_fallback: bool = True
    max_primary_failures: int = 3

    # Runtime state (failure tracking)
    _primary_failure_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        """Load hybrid service configuration from environment"""
        # Read provider types
        self.primary_provider = os.getenv("VECTORIZE_PROVIDER", self.primary_provider)
        self.fallback_provider = os.getenv(
            "VECTORIZE_FALLBACK_PROVIDER", self.fallback_provider
        )

        # Read primary service config
        self.primary_api_key = os.getenv("VECTORIZE_API_KEY", self.primary_api_key)
        self.primary_base_url = os.getenv("VECTORIZE_BASE_URL", self.primary_base_url)

        # Read fallback service config
        self.fallback_api_key = os.getenv(
            "VECTORIZE_FALLBACK_API_KEY", self.fallback_api_key
        )
        self.fallback_base_url = os.getenv(
            "VECTORIZE_FALLBACK_BASE_URL", self.fallback_base_url
        )

        # Read shared model configuration
        self.model = os.getenv("VECTORIZE_MODEL", self.model)

        # Read common settings
        self.timeout = int(os.getenv("VECTORIZE_TIMEOUT", str(self.timeout)))
        self.max_retries = int(
            os.getenv("VECTORIZE_MAX_RETRIES", str(self.max_retries))
        )
        self.batch_size = int(os.getenv("VECTORIZE_BATCH_SIZE", str(self.batch_size)))
        self.max_concurrent_requests = int(
            os.getenv("VECTORIZE_MAX_CONCURRENT", str(self.max_concurrent_requests))
        )
        self.encoding_format = os.getenv(
            "VECTORIZE_ENCODING_FORMAT", self.encoding_format
        )
        self.dimensions = int(os.getenv("VECTORIZE_DIMENSIONS", str(self.dimensions)))

        # Fallback behavior
        # Enable fallback only if:
        # 1. fallback_provider is not "none"
        # 2. fallback_base_url is configured
        # 3. fallback_api_key is configured (or not required for vllm)
        self.enable_fallback = (
            self.fallback_provider.lower() != "none"
            and bool(self.fallback_base_url)
            and (
                self.fallback_provider.lower() == "vllm"  # vllm doesn't require API key
                or bool(self.fallback_api_key)  # deepinfra requires API key
            )
        )
        self.max_primary_failures = int(
            os.getenv("VECTORIZE_MAX_PRIMARY_FAILURES", str(self.max_primary_failures))
        )


def _create_service_from_config(
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
    max_retries: int,
    batch_size: int,
    max_concurrent: int,
    encoding_format: str,
    dimensions: int,
) -> VectorizeServiceInterface:
    """
    Factory function to create a vectorize service based on provider type

    Args:
        provider: Provider type (vllm or deepinfra)
        api_key: API key for the service
        base_url: Base URL for the service
        model: Model name
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        batch_size: Batch size for requests
        max_concurrent: Maximum concurrent requests
        encoding_format: Encoding format for embeddings
        dimensions: Vector dimensions

    Returns:
        VectorizeServiceInterface: The created service instance
    """
    if provider.lower() == "vllm":
        config = VllmVectorizeConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            batch_size=batch_size,
            max_concurrent_requests=max_concurrent,
            encoding_format=encoding_format,
            dimensions=dimensions,
        )
        return VllmVectorizeService(config)
    elif provider.lower() == "deepinfra":
        config = DeepInfraVectorizeConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            batch_size=batch_size,
            max_concurrent_requests=max_concurrent,
            encoding_format=encoding_format,
            dimensions=dimensions,
        )
        return DeepInfraVectorizeService(config)
    else:
        raise VectorizeError(f"Unsupported provider: {provider}")


class HybridVectorizeService(VectorizeServiceInterface):
    """
    Hybrid Vectorization Service with Automatic Fallback

    This service implements a dual-strategy approach:
    1. Implements VectorizeServiceInterface with full API
    2. Primary: Configurable provider (vllm or deepinfra)
    3. Secondary: Configurable fallback provider
    4. Automatic failover on errors with failure tracking
    5. All method calls transparently use fallback logic

    Strategy Benefits:
    - Cost optimization: ~95% savings with vllm self-deployed service
    - High availability: Automatic failover ensures reliability
    - Zero downtime: Continues working during vllm service maintenance

    Usage:
        service = HybridVectorizeService()
        embedding = await service.get_embedding("Hello")  # Auto-fallback built-in
    """

    def __init__(self, config: Optional[HybridVectorizeConfig] = None):
        if config is None:
            config = HybridVectorizeConfig()

        self.config = config

        # Create primary service based on provider type
        self.primary_service = _create_service_from_config(
            provider=config.primary_provider,
            api_key=config.primary_api_key,
            base_url=config.primary_base_url,
            model=config.model,  # Use shared model
            timeout=config.timeout,
            max_retries=config.max_retries,
            batch_size=config.batch_size,
            max_concurrent=config.max_concurrent_requests,
            encoding_format=config.encoding_format,
            dimensions=config.dimensions,
        )

        # Create fallback service if enabled
        self.fallback_service = None
        if config.enable_fallback:
            self.fallback_service = _create_service_from_config(
                provider=config.fallback_provider,
                api_key=config.fallback_api_key,
                base_url=config.fallback_base_url,
                model=config.model,  # Use shared model
                timeout=config.timeout,
                max_retries=config.max_retries,
                batch_size=config.batch_size,
                max_concurrent=config.max_concurrent_requests,
                encoding_format=config.encoding_format,
                dimensions=config.dimensions,
            )

        logger.info(
            f"Initialized HybridVectorizeService | "
            f"primary={config.primary_provider} | "
            f"fallback={config.fallback_provider} | "
            f"fallback_enabled={config.enable_fallback} | "
            f"max_failures={config.max_primary_failures}"
        )

    def get_service(self) -> VectorizeServiceInterface:
        """
        Get the primary service (for advanced usage)

        Returns:
            VectorizeServiceInterface: The primary service

        Note: Prefer using proxy methods directly for automatic fallback
        """
        return self.primary_service

    # Implement VectorizeServiceInterface methods with automatic fallback

    async def get_embedding(
        self, text: str, instruction: Optional[str] = None, is_query: bool = False
    ) -> np.ndarray:
        """Get embedding for a single text with automatic fallback"""
        with timed("compute_embedding"):
            return await self.execute_with_fallback(
                "get_embedding",
                lambda: self.primary_service.get_embedding(text, instruction, is_query),
                lambda: (
                    self.fallback_service.get_embedding(text, instruction, is_query)
                    if self.fallback_service
                    else None
                ),
                batch_size=1,
            )

    async def get_embedding_with_usage(
        self, text: str, instruction: Optional[str] = None, is_query: bool = False
    ) -> Tuple[np.ndarray, Optional[UsageInfo]]:
        """Get embedding with usage information with automatic fallback"""
        return await self.execute_with_fallback(
            "get_embedding_with_usage",
            lambda: self.primary_service.get_embedding_with_usage(
                text, instruction, is_query
            ),
            lambda: (
                self.fallback_service.get_embedding_with_usage(
                    text, instruction, is_query
                )
                if self.fallback_service
                else None
            ),
            batch_size=1,
        )

    async def get_embeddings(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        is_query: bool = False,
    ) -> List[np.ndarray]:
        """Get embeddings for multiple texts with automatic fallback"""
        with timed("compute_embeddings"):
            return await self.execute_with_fallback(
                "get_embeddings",
                lambda: self.primary_service.get_embeddings(
                    texts, instruction, is_query
                ),
                lambda: (
                    self.fallback_service.get_embeddings(texts, instruction, is_query)
                    if self.fallback_service
                    else None
                ),
                batch_size=len(texts),
            )

    async def get_embeddings_batch(
        self,
        text_batches: List[List[str]],
        instruction: Optional[str] = None,
        is_query: bool = False,
    ) -> List[List[np.ndarray]]:
        """Get embeddings for multiple batches with automatic fallback"""
        total_texts = sum(len(batch) for batch in text_batches)
        with timed("compute_embeddings"):
            return await self.execute_with_fallback(
                "get_embeddings_batch",
                lambda: self.primary_service.get_embeddings_batch(
                    text_batches, instruction, is_query
                ),
                lambda: (
                    self.fallback_service.get_embeddings_batch(
                        text_batches, instruction, is_query
                    )
                    if self.fallback_service
                    else None
                ),
                batch_size=total_texts,
            )

    def get_model_name(self) -> str:
        """Get the current model name (from primary service)"""
        return self.primary_service.get_model_name()

    async def execute_with_fallback(
        self, operation_name: str, primary_func, fallback_func, batch_size: int = 1
    ):
        """
        Execute operation with automatic fallback logic

        Args:
            operation_name: Name of the operation for logging
            primary_func: Function to call on primary service
            fallback_func: Function to call on fallback service (or None if no fallback)
            batch_size: Number of texts being processed (for metrics)

        Returns:
            Result from primary or fallback service

        Raises:
            VectorizeError: If both services fail
        """
        start_time = time.perf_counter()

        # Try primary service first
        try:
            result = await primary_func()
            duration = time.perf_counter() - start_time

            # Record success metrics
            record_vectorize_request(
                provider=self.config.primary_provider,
                operation=operation_name,
                status='success',
                duration_seconds=duration,
                batch_size=batch_size,
            )

            # Reset failure count on success
            self.config._primary_failure_count = 0
            return result

        except Exception as primary_error:
            primary_duration = time.perf_counter() - start_time

            # Increment failure count
            self.config._primary_failure_count += 1

            # Determine error type
            error_type = self._classify_error(primary_error)

            # Record error metrics
            record_vectorize_error(
                provider=self.config.primary_provider,
                operation=operation_name,
                error_type=error_type,
            )

            logger.warning(
                f"Primary service ({self.config.primary_provider}) {operation_name} failed "
                f"(count: {self.config._primary_failure_count}): {primary_error}"
            )

            # Check if fallback is enabled
            if not self.config.enable_fallback or fallback_func is None:
                # Record failed request (no fallback)
                record_vectorize_request(
                    provider=self.config.primary_provider,
                    operation=operation_name,
                    status='error',
                    duration_seconds=primary_duration,
                    batch_size=batch_size,
                )
                logger.error("Fallback disabled or not configured, re-raising error")
                raise VectorizeError(
                    f"Primary service failed and fallback is disabled: {primary_error}"
                )

            # Determine fallback reason
            fallback_reason = error_type
            if self.config._primary_failure_count >= self.config.max_primary_failures:
                fallback_reason = 'max_failures_exceeded'
                logger.warning(
                    f"⚠️ Primary service exceeded max failures ({self.config.max_primary_failures}), "
                    f"using {self.config.fallback_provider} fallback"
                )

            # Record fallback event
            record_vectorize_fallback(
                primary_provider=self.config.primary_provider,
                fallback_provider=self.config.fallback_provider,
                reason=fallback_reason,
            )

            # Try fallback service
            fallback_start = time.perf_counter()
            try:
                logger.info(
                    f"🔄 Falling back to {self.config.fallback_provider} for {operation_name}"
                )
                result = await fallback_func()
                fallback_duration = time.perf_counter() - fallback_start

                # Record fallback success metrics
                record_vectorize_request(
                    provider=self.config.fallback_provider,
                    operation=operation_name,
                    status='fallback',
                    duration_seconds=fallback_duration,
                    batch_size=batch_size,
                )

                return result

            except Exception as fallback_error:
                fallback_duration = time.perf_counter() - fallback_start

                # Record fallback error
                record_vectorize_error(
                    provider=self.config.fallback_provider,
                    operation=operation_name,
                    error_type=self._classify_error(fallback_error),
                )
                record_vectorize_request(
                    provider=self.config.fallback_provider,
                    operation=operation_name,
                    status='error',
                    duration_seconds=fallback_duration,
                    batch_size=batch_size,
                )

                logger.error(f"❌ Fallback also failed: {fallback_error}")
                raise VectorizeError(
                    f"Both primary and fallback services failed. "
                    f"Primary ({self.config.primary_provider}): {primary_error}, "
                    f"Fallback ({self.config.fallback_provider}): {fallback_error}"
                )

    def _classify_error(self, error: Exception) -> str:
        """Classify error type for metrics"""
        error_str = str(error).lower()
        if 'timeout' in error_str or 'timed out' in error_str:
            return 'timeout'
        elif 'rate' in error_str and 'limit' in error_str:
            return 'rate_limit'
        elif 'validation' in error_str or 'invalid' in error_str:
            return 'validation_error'
        elif isinstance(error, VectorizeError):
            return 'api_error'
        else:
            return 'unknown'

    def get_failure_count(self) -> int:
        """Get current primary service failure count"""
        return self.config._primary_failure_count

    def reset_failure_count(self):
        """Reset failure count (useful for health check recovery)"""
        self.config._primary_failure_count = 0
        logger.info("Reset primary service failure count to 0")

    async def close(self):
        """Close all services"""
        await self.primary_service.close()
        if self.fallback_service:
            await self.fallback_service.close()


# Global service instance (lazy initialization)
_service_instance: Optional[HybridVectorizeService] = None


def get_hybrid_service() -> HybridVectorizeService:
    """
    Get the global hybrid service instance (singleton)

    Returns:
        HybridVectorizeService: The global hybrid service instance
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = HybridVectorizeService()
    return _service_instance


# Main entry point - registered with DI container
@service(name="vectorize_service", primary=True)
def get_vectorize_service() -> VectorizeServiceInterface:
    """
    Get the vectorization service (main entry point)

    Returns the hybrid service which implements VectorizeServiceInterface.
    All method calls automatically go through fallback logic.

    Returns:
        VectorizeServiceInterface: The hybrid service with automatic fallback

    Example:
        ```python
        from agentic_layer.vectorize_service import get_vectorize_service

        service = get_vectorize_service()  # Returns hybrid service with fallback
        embedding = await service.get_embedding("Hello world")  # Auto-fallback
        embeddings = await service.get_embeddings(["Text 1", "Text 2"])  # Auto-fallback
        await service.close()
        ```
    """
    return (
        get_hybrid_service()
    )  # Return hybrid service (implements VectorizeServiceInterface)


# Export public API
__all__ = ["get_vectorize_service"]
