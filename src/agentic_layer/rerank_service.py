"""
Rerank Service - Hybrid Implementation with Automatic Fallback

This is the main reranking service with built-in resilience.
Implements a hybrid strategy with flexible provider selection and automatic fallback.

Usage:
    from agentic_layer.rerank_service import get_rerank_service
    
    service = get_rerank_service()
    result = await service.rerank_memories(query, hits, top_k)
"""

import logging
import os
import time
from typing import Optional, Any, List, Dict
from dataclasses import dataclass, field

from core.di import service

from agentic_layer.rerank_interface import RerankServiceInterface, RerankError
from agentic_layer.rerank_vllm import VllmRerankService, VllmRerankConfig
from agentic_layer.rerank_deepinfra import DeepInfraRerankService, DeepInfraRerankConfig
from agentic_layer.metrics.rerank_metrics import (
    record_rerank_request,
    record_rerank_fallback,
    record_rerank_error,
)

logger = logging.getLogger(__name__)


@dataclass
class HybridRerankConfig:
    """Configuration for hybrid rerank service with fallback"""

    # Provider types
    primary_provider: str = "vllm"  # vllm, openai, or deepinfra
    fallback_provider: str = "deepinfra"  # vllm, openai, deepinfra, or none

    # Primary service config
    primary_api_key: str = ""
    primary_base_url: str = ""

    # Fallback service config
    fallback_api_key: str = ""
    fallback_base_url: str = ""

    # Shared model configuration
    model: str = "Qwen/Qwen3-Reranker-4B"

    # Common settings
    timeout: int = 30
    max_retries: int = 3
    batch_size: int = 10
    max_concurrent_requests: int = 5

    # Fallback behavior
    enable_fallback: bool = True
    max_primary_failures: int = 3

    # Runtime state (failure tracking)
    _primary_failure_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        """Load hybrid service configuration from environment"""
        # Read provider types
        self.primary_provider = os.getenv("RERANK_PROVIDER", self.primary_provider)
        self.fallback_provider = os.getenv(
            "RERANK_FALLBACK_PROVIDER", self.fallback_provider
        )

        # Read primary service config
        self.primary_api_key = os.getenv("RERANK_API_KEY", self.primary_api_key)
        self.primary_base_url = os.getenv("RERANK_BASE_URL", self.primary_base_url)

        # Read fallback service config
        self.fallback_api_key = os.getenv(
            "RERANK_FALLBACK_API_KEY", self.fallback_api_key
        )
        self.fallback_base_url = os.getenv(
            "RERANK_FALLBACK_BASE_URL", self.fallback_base_url
        )

        # Read shared model configuration
        self.model = os.getenv("RERANK_MODEL", self.model)

        # Read common settings
        self.timeout = int(os.getenv("RERANK_TIMEOUT", str(self.timeout)))
        self.max_retries = int(os.getenv("RERANK_MAX_RETRIES", str(self.max_retries)))
        self.batch_size = int(os.getenv("RERANK_BATCH_SIZE", str(self.batch_size)))
        self.max_concurrent_requests = int(
            os.getenv("RERANK_MAX_CONCURRENT", str(self.max_concurrent_requests))
        )

        # Fallback behavior
        # Enable fallback only if:
        # 1. fallback_provider is not "none"
        # 2. fallback_base_url is configured
        # 3. fallback_api_key is configured (or not required for vllm)
        self.enable_fallback = (
            self.fallback_provider.lower() != "none"
            and bool(self.fallback_base_url)
            and (
                self.fallback_provider.lower() in ("vllm", "openai")  # vllm/openai don't require API key
                or bool(self.fallback_api_key)  # deepinfra requires API key
            )
        )
        self.max_primary_failures = int(
            os.getenv("RERANK_MAX_PRIMARY_FAILURES", str(self.max_primary_failures))
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
) -> RerankServiceInterface:
    """
    Factory function to create a rerank service based on provider type

    Args:
        provider: Provider type (vllm, openai, or deepinfra)
        api_key: API key for the service
        base_url: Base URL for the service
        model: Model name
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        batch_size: Batch size for requests
        max_concurrent: Maximum concurrent requests

    Returns:
        RerankServiceInterface: The created service instance
    """
    if provider.lower() == "vllm":
        config = VllmRerankConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            batch_size=batch_size,
            max_concurrent_requests=max_concurrent,
        )
        return VllmRerankService(config)
    elif provider.lower() == "openai":
        # Generic OpenAI-compatible hosted API (e.g. DashScope, Cohere)
        # Reuses VllmRerankService but skips Qwen3 special token formatting
        config = VllmRerankConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            batch_size=batch_size,
            max_concurrent_requests=max_concurrent,
            skip_formatting=True,
        )
        return VllmRerankService(config)
    elif provider.lower() == "deepinfra":
        config = DeepInfraRerankConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            batch_size=batch_size,
            max_concurrent_requests=max_concurrent,
        )
        return DeepInfraRerankService(config)
    else:
        raise RerankError(f"Unsupported provider: {provider}")


class HybridRerankService(RerankServiceInterface):
    """
    Hybrid Reranking Service with Automatic Fallback

    This service implements a dual-strategy approach:
    1. Implements RerankServiceInterface with full API
    2. Primary: Configurable provider (vllm or deepinfra)
    3. Secondary: Configurable fallback provider
    4. Automatic failover on errors with failure tracking
    5. All method calls transparently use fallback logic

    Strategy Benefits:
    - Cost optimization: ~95% savings with vllm self-deployed service
    - High availability: Automatic failover ensures reliability
    - Zero downtime: Continues working during vllm service maintenance

    Usage:
        service = HybridRerankService()
        result = await service.rerank_memories(query, hits, top_k)  # Auto-fallback built-in
    """

    def __init__(self, config: Optional[HybridRerankConfig] = None):
        if config is None:
            config = HybridRerankConfig()

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
            )

        logger.info(
            f"Initialized HybridRerankService | "
            f"primary={config.primary_provider} | "
            f"fallback={config.fallback_provider} | "
            f"fallback_enabled={config.enable_fallback} | "
            f"max_failures={config.max_primary_failures}"
        )

    def get_service(self) -> RerankServiceInterface:
        """
        Get the primary service (for advanced usage)

        Returns:
            RerankServiceInterface: The primary service

        Note: Prefer using hybrid service methods directly for automatic fallback
        """
        return self.primary_service

    async def rerank_memories(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank memories with automatic fallback"""
        start_time = time.perf_counter()
        documents_count = len(hits)

        try:
            result = await self.execute_with_fallback(
                "rerank_memories",
                lambda: self.primary_service.rerank_memories(
                    query, hits, top_k, instruction
                ),
                lambda: (
                    self.fallback_service.rerank_memories(
                        query, hits, top_k, instruction
                    )
                    if self.fallback_service
                    else None
                ),
            )

            # Record success metrics
            duration = time.perf_counter() - start_time
            record_rerank_request(
                provider=self.config.primary_provider,
                status='success',
                duration_seconds=duration,
                documents_count=documents_count,
            )

            return result

        except Exception as e:
            # Record error metrics (fallback failure is recorded in execute_with_fallback)
            duration = time.perf_counter() - start_time
            record_rerank_request(
                provider=self.config.primary_provider,
                status='error',
                duration_seconds=duration,
                documents_count=documents_count,
            )
            raise

    def get_model_name(self) -> str:
        """Get the current model name (from primary service)"""
        return self.primary_service.get_model_name()

    async def rerank_documents(
        self, query: str, documents: List[str], instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Rerank raw documents (low-level API) with fallback support

        Args:
            query: Query text
            documents: List of document strings to rerank
            instruction: Optional reranking instruction

        Returns:
            Dict with 'results' key containing list of {index, score, rank}
        """
        return await self.execute_with_fallback(
            "rerank_documents",
            lambda: self.primary_service.rerank_documents(
                query, documents, instruction
            ),
            lambda: (
                self.fallback_service.rerank_documents(query, documents, instruction)
                if self.fallback_service
                else None
            ),
        )

    async def execute_with_fallback(
        self, operation_name: str, primary_func, fallback_func
    ):
        """
        Execute operation with automatic fallback logic

        Args:
            operation_name: Name of the operation for logging
            primary_func: Function to call on primary service
            fallback_func: Function to call on fallback service (or None if no fallback)

        Returns:
            Result from primary or fallback service

        Raises:
            RerankError: If both services fail
        """
        # Try primary service first
        try:
            result = await primary_func()
            # Reset failure count on success
            self.config._primary_failure_count = 0
            return result

        except Exception as primary_error:
            # Increment failure count
            self.config._primary_failure_count += 1

            logger.warning(
                f"Primary service ({self.config.primary_provider}) {operation_name} failed "
                f"(count: {self.config._primary_failure_count}): {primary_error}"
            )

            # Record primary error
            error_type = self._classify_error(primary_error)
            record_rerank_error(
                provider=self.config.primary_provider, error_type=error_type
            )

            # Check if fallback is enabled
            if not self.config.enable_fallback or fallback_func is None:
                logger.error("Fallback disabled or not configured, re-raising error")
                raise RerankError(
                    f"Primary service failed and fallback is disabled: {primary_error}"
                )

            # Determine fallback reason
            fallback_reason = 'error'
            if self.config._primary_failure_count >= self.config.max_primary_failures:
                fallback_reason = 'max_failures_exceeded'
                logger.warning(
                    f"⚠️ Primary service exceeded max failures ({self.config.max_primary_failures}), "
                    f"using {self.config.fallback_provider} fallback"
                )

            # Try fallback service
            try:
                logger.info(
                    f"🔄 Falling back to {self.config.fallback_provider} for {operation_name}"
                )

                # Record fallback event
                record_rerank_fallback(
                    primary_provider=self.config.primary_provider,
                    fallback_provider=self.config.fallback_provider,
                    reason=fallback_reason,
                )

                result = await fallback_func()
                return result

            except Exception as fallback_error:
                logger.error(f"❌ Fallback also failed: {fallback_error}")

                # Record fallback error
                fallback_error_type = self._classify_error(fallback_error)
                record_rerank_error(
                    provider=self.config.fallback_provider,
                    error_type=fallback_error_type,
                )

                raise RerankError(
                    f"Both primary and fallback services failed. "
                    f"Primary ({self.config.primary_provider}): {primary_error}, "
                    f"Fallback ({self.config.fallback_provider}): {fallback_error}"
                )

    def _classify_error(self, error: Exception) -> str:
        """Classify error type for metrics"""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        if 'timeout' in error_str or 'timeout' in error_type:
            return 'timeout'
        elif 'rate' in error_str and 'limit' in error_str:
            return 'rate_limit'
        elif 'validation' in error_str or 'invalid' in error_str:
            return 'validation_error'
        elif 'connection' in error_str or 'connect' in error_type:
            return 'connection_error'
        elif 'api' in error_str or 'http' in error_str:
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
_service_instance: Optional[HybridRerankService] = None


def get_hybrid_service() -> HybridRerankService:
    """
    Get the global hybrid service instance (singleton)

    Returns:
        HybridRerankService: The global hybrid service instance
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = HybridRerankService()
    return _service_instance


# Main entry point - registered with DI container
@service(name="rerank_service", primary=True)
def get_rerank_service() -> RerankServiceInterface:
    """
    Get the reranking service (main entry point)

    Returns the hybrid service which implements RerankServiceInterface.
    All method calls automatically go through fallback logic.

    Returns:
        RerankServiceInterface: The hybrid service with automatic fallback

    Example:
        ```python
        from agentic_layer.rerank_service import get_rerank_service

        service = get_rerank_service()  # Returns hybrid service with fallback
        result = await service.rerank_memories(query, hits, top_k)  # Auto-fallback
        await service.close()
        ```
    """
    return (
        get_hybrid_service()
    )  # Return hybrid service (implements RerankServiceInterface)


# Export public API
__all__ = ["get_rerank_service"]
