"""
Base Vectorize Service Implementation

Provides common functionality for embedding services using OpenAI-compatible APIs.
"""

import asyncio
import logging
from typing import List, Optional, Tuple
from abc import abstractmethod
import numpy as np
from openai import AsyncOpenAI

from agentic_layer.vectorize_interface import (
    VectorizeServiceInterface,
    VectorizeError,
    UsageInfo,
)
from core.di.utils import get_bean_by_type
from core.component.token_usage_collector import TokenUsageCollector

logger = logging.getLogger(__name__)


class BaseVectorizeService(VectorizeServiceInterface):
    """
    Base class for OpenAI-compatible embedding services

    Subclasses only need to implement:
    - _get_config_params(): return (api_key, base_url, model)
    - _should_pass_dimensions(): return True/False
    - _should_truncate_client_side(): return True/False
    """

    def __init__(self, config):
        self.config = config
        self.client: Optional[AsyncOpenAI] = None
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)

        api_key, base_url, model = self._get_config_params()
        logger.info(
            f"Initialized {self.__class__.__name__} | model={model} | base_url={base_url}"
        )

    @abstractmethod
    def _get_config_params(self) -> Tuple[str, str, str]:
        """Return (api_key, base_url, model) for logging"""
        pass

    @abstractmethod
    def _should_pass_dimensions(self) -> bool:
        """Whether to pass dimensions parameter to API"""
        pass

    @abstractmethod
    def _should_truncate_client_side(self) -> bool:
        """Whether to truncate embeddings on client side"""
        pass

    async def __aenter__(self):
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _ensure_client(self):
        """Ensure OpenAI client is initialized"""
        if self.client is None:
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )

    async def close(self):
        """Close the client connection"""
        if self.client:
            await self.client.close()
            self.client = None

    async def _make_request(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        is_query: bool = False,
    ):
        """Make embedding request to API"""
        await self._ensure_client()
        if not self.config.model:
            raise VectorizeError("Embedding model is not configured.")

        # Format texts with instruction if needed
        if is_query:
            default_instruction = (
                "Given a search query, retrieve relevant passages that answer the query"
            )
            final_instruction = (
                instruction if instruction is not None else default_instruction
            )
            formatted_texts = [
                f"Instruct: {final_instruction}\nQuery: {text}" for text in texts
            ]
        else:
            formatted_texts = texts

        async with self._semaphore:
            for attempt in range(self.config.max_retries):
                try:
                    request_kwargs = {
                        "model": self.config.model,
                        "input": formatted_texts,
                        "encoding_format": self.config.encoding_format,
                    }

                    # Add dimensions parameter if supported
                    if self._should_pass_dimensions() and self.config.dimensions > 0:
                        request_kwargs["dimensions"] = self.config.dimensions

                    response = await self.client.embeddings.create(**request_kwargs)

                    # Report embedding call to token usage collector
                    try:
                        collector = get_bean_by_type(TokenUsageCollector)
                        prompt_tokens = (
                            getattr(response.usage, 'prompt_tokens', 0) or 0
                            if response.usage
                            else 0
                        )
                        collector.add(
                            self.config.model, prompt_tokens, 0, call_type="embedding"
                        )
                    except Exception:
                        pass

                    return response

                except Exception as e:
                    error_msg = str(e)
                    logger.error(
                        f"{self.__class__.__name__} API error (attempt {attempt + 1}/{self.config.max_retries}): {error_msg}"
                    )

                    # Log detailed error for debugging
                    if "Connection" in error_msg or "timeout" in error_msg.lower():
                        logger.warning(
                            f"Network issue connecting to {self.config.base_url}: {error_msg}"
                        )

                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    else:
                        raise VectorizeError(
                            f"{self.__class__.__name__} API request failed after {self.config.max_retries} attempts: {error_msg}"
                        )

    def _parse_embeddings_response(self, response) -> List[np.ndarray]:
        """Parse embeddings from API response"""
        if not response.data:
            raise VectorizeError("Invalid API response: missing data")

        embeddings = []
        for item in response.data:
            emb = np.array(item.embedding, dtype=np.float32)

            # Client-side truncation if needed
            if self._should_truncate_client_side():
                if (
                    self.config.dimensions
                    and self.config.dimensions > 0
                    and len(emb) > self.config.dimensions
                ):
                    logger.debug(
                        f"Client-side truncation: {len(emb)}D → {self.config.dimensions}D"
                    )
                    emb = emb[: self.config.dimensions]

            embeddings.append(emb)
        return embeddings

    async def get_embedding(
        self, text: str, instruction: Optional[str] = None, is_query: bool = False
    ) -> np.ndarray:
        """Get embedding for a single text"""
        response = await self._make_request([text], instruction, is_query)
        if not response.data:
            raise VectorizeError("Invalid API response: missing data")
        return np.array(self._parse_embeddings_response(response)[0], dtype=np.float32)

    async def get_embedding_with_usage(
        self, text: str, instruction: Optional[str] = None, is_query: bool = False
    ) -> Tuple[np.ndarray, Optional[UsageInfo]]:
        """Get embedding with usage information"""
        response = await self._make_request([text], instruction, is_query)
        if not response.data:
            raise VectorizeError("Invalid API response: missing data")

        embeddings = self._parse_embeddings_response(response)
        embedding = np.array(embeddings[0], dtype=np.float32)
        usage_info = (
            UsageInfo.from_openai_usage(response.usage) if response.usage else None
        )
        return embedding, usage_info

    async def get_embeddings(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        is_query: bool = False,
    ) -> List[np.ndarray]:
        """Get embeddings for multiple texts"""
        if not texts:
            return []

        if len(texts) <= self.config.batch_size:
            response = await self._make_request(texts, instruction, is_query)
            return self._parse_embeddings_response(response)

        embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i : i + self.config.batch_size]
            response = await self._make_request(batch_texts, instruction, is_query)
            embeddings.extend(self._parse_embeddings_response(response))
            if i + self.config.batch_size < len(texts):
                await asyncio.sleep(0.1)
        return embeddings

    async def get_embeddings_batch(
        self,
        text_batches: List[List[str]],
        instruction: Optional[str] = None,
        is_query: bool = False,
    ) -> List[List[np.ndarray]]:
        """Get embeddings for multiple batches"""
        tasks = [
            self.get_embeddings(batch, instruction, is_query) for batch in text_batches
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        embeddings_batches = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing batch {i}: {result}")
                embeddings_batches.append([])
            else:
                embeddings_batches.append(result)
        return embeddings_batches

    def get_model_name(self) -> str:
        """Get the current model name"""
        return self.config.model
