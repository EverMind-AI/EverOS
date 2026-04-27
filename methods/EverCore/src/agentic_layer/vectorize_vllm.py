"""
vLLM (Self-Deployed) Vectorize Service Implementation

This module provides vectorization service for self-deployed embedding servers,
such as vLLM, Ollama, or other OpenAI-compatible endpoints.
"""

import os
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

from agentic_layer.vectorize_base import BaseVectorizeService

logger = logging.getLogger(__name__)


@dataclass
class VllmVectorizeConfig:
    """Configuration for vLLM self-deployed vectorization service"""

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"  # Many self-deployed services don't require API key
    model: str = "Qwen/Qwen3-Embedding-4B"
    timeout: int = 30
    max_retries: int = 3
    batch_size: int = 10
    max_concurrent_requests: int = 5
    encoding_format: str = "float"
    dimensions: int = 1024  # Client-side truncation target


class VllmVectorizeService(BaseVectorizeService):
    """
    vLLM self-deployed embedding service implementation
    
    Supports:
    - vLLM (https://github.com/vllm-project/vllm)
    - Any OpenAI-compatible embedding endpoint
    """

    def __init__(self, config: Optional[VllmVectorizeConfig] = None):
        if config is None:
            config = VllmVectorizeConfig()
        super().__init__(config)

    def _get_config_params(self) -> Tuple[str, str, str]:
        """Return (api_key, base_url, model) for logging"""
        return self.config.api_key, self.config.base_url, self.config.model

    def _should_pass_dimensions(self) -> bool:
        """vLLM services don't support dimensions parameter"""
        return False

    def _should_truncate_client_side(self) -> bool:
        """vLLM services need client-side truncation"""
        return True

