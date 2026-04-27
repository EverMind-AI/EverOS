"""
DeepInfra Vectorize Service Implementation

Commercial API implementation for DeepInfra embedding service
"""

import os
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

from agentic_layer.vectorize_base import BaseVectorizeService

logger = logging.getLogger(__name__)


@dataclass
class DeepInfraVectorizeConfig:
    """DeepInfra Vectorize configuration"""

    api_key: str = ""
    base_url: str = "https://api.deepinfra.com/v1/openai"
    model: str = "Qwen/Qwen3-Embedding-4B"
    timeout: int = 30
    max_retries: int = 3
    batch_size: int = 10
    max_concurrent_requests: int = 5
    encoding_format: str = "float"
    dimensions: int = 1024


class DeepInfraVectorizeService(BaseVectorizeService):
    """
    DeepInfra embedding service implementation
    Uses DeepInfra's commercial API for text embeddings
    """

    def __init__(self, config: Optional[DeepInfraVectorizeConfig] = None):
        if config is None:
            config = DeepInfraVectorizeConfig()
        super().__init__(config)

    def _get_config_params(self) -> Tuple[str, str, str]:
        """Return (api_key, base_url, model) for logging"""
        return self.config.api_key, self.config.base_url, self.config.model

    def _should_pass_dimensions(self) -> bool:
        """DeepInfra supports dimensions parameter"""
        return True

    def _should_truncate_client_side(self) -> bool:
        """DeepInfra handles truncation server-side"""
        return False
