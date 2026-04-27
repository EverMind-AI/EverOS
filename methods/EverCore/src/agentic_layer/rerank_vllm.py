"""
vLLM (Self-Deployed) Rerank Service Implementation

Reranking service for self-deployed vLLM or similar OpenAI-compatible services.
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from agentic_layer.rerank_interface import (
    RerankServiceInterface,
    RerankError,
    extract_text_from_hit,
)
from core.di.utils import get_bean_by_type
from core.component.token_usage_collector import TokenUsageCollector

logger = logging.getLogger(__name__)


@dataclass
class VllmRerankConfig:
    """vLLM rerank service configuration"""

    api_key: str = "EMPTY"
    base_url: str = "http://localhost:12000/v1/rerank"
    model: str = "Qwen/Qwen3-Reranker-4B"  # skip-sensitive-check
    timeout: int = 3
    max_retries: int = 2
    batch_size: int = 10
    max_concurrent_requests: int = 5


class VllmRerankService(RerankServiceInterface):
    """vLLM reranking service implementation"""

    def __init__(self, config: Optional[VllmRerankConfig] = None):
        if config is None:
            config = VllmRerankConfig()

        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        logger.info(
            f"Initialized VllmRerankService | url={config.base_url} | model={config.model}"  # skip-sensitive-check
        )

    async def _ensure_session(self):
        """Ensure HTTP session is created"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            headers = {"Content-Type": "application/json"}
            if self.config.api_key and self.config.api_key != "EMPTY":
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)

    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()

    def _format_rerank_texts(
        self, query: str, documents: List[str], instruction: Optional[str] = None
    ):
        """
        Format rerank request texts (Qwen-Reranker official format)

        Reference: https://docs.vllm.ai/en/v0.9.2/examples/offline_inference/qwen3_reranker.html
        """
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        # Use vLLM official default instruction for optimal performance
        instruction = (
            instruction
            or "Given a question and a passage, determine if the passage contains information relevant to answering the question."
        )

        formatted_query = f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
        formatted_docs = [f"<Document>: {doc}{suffix}" for doc in documents]

        return [formatted_query], formatted_docs

    async def _send_rerank_request_batch(
        self,
        query: str,
        documents: List[str],
        start_index: int,
        instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send rerank request batch to vLLM rerank API (OpenAI-compatible format)"""
        await self._ensure_session()

        # Format texts using Qwen-Reranker official format
        queries, formatted_docs = self._format_rerank_texts(
            query, documents, instruction
        )

        url = self.config.base_url
        # Use OpenAI-compatible rerank API format with formatted texts
        request_data = {
            "model": self.config.model,
            "query": queries[0] if queries else query,  # Use formatted query
            "documents": formatted_docs,  # Use formatted documents
        }

        async with self._semaphore:
            for attempt in range(self.config.max_retries):
                try:
                    async with self.session.post(url, json=request_data) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result
                        else:
                            error_text = await response.text()
                            logger.warning(
                                f"vLLM rerank API error (status {response.status}, attempt {attempt + 1}/{self.config.max_retries}): {error_text}"
                            )
                            if attempt < self.config.max_retries - 1:
                                await asyncio.sleep(2**attempt)
                                continue
                            else:
                                raise RerankError(
                                    f"Rerank request failed after {self.config.max_retries} attempts: {error_text}"
                                )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"vLLM rerank timeout (attempt {attempt + 1}/{self.config.max_retries}), timeout={self.config.timeout}s"
                    )
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    else:
                        raise RerankError(
                            f"Rerank request timed out after {self.config.max_retries} attempts"
                        )
                except aiohttp.ClientError as e:
                    logger.warning(
                        f"vLLM rerank client error (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                    )
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    else:
                        raise RerankError(
                            f"Rerank request failed after {self.config.max_retries} attempts: {e}"
                        )
                except Exception as e:
                    logger.error(f"Unexpected error in vLLM rerank request: {e}")
                    raise RerankError(f"Unexpected rerank error: {e}")

    async def rerank_documents(
        self, query: str, documents: List[str], instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Rerank raw documents (low-level API)

        Args:
            query: Query text
            documents: List of document strings to rerank
            instruction: Optional reranking instruction

        Returns:
            Dict with 'results' key containing list of {index, score, rank}
        """
        if not documents:
            return {"results": []}

        batch_size = self.config.batch_size
        if batch_size <= 0:
            batch_size = 10

        batches = [
            documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
        ]

        batch_tasks = []
        for i, batch in enumerate(batches):
            start_index = i * batch_size
            batch_tasks.append(
                self._send_rerank_request_batch(query, batch, start_index, instruction)
            )

        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        all_scores = []
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Rerank batch {i} failed: {result}")
                batch_len = len(batches[i])
                all_scores.extend([-100.0] * batch_len)
                continue

            # vLLM returns {"results": [{"index": ..., "relevance_score": ...}, ...]}
            results = result.get("results", [])
            results_sorted = sorted(results, key=lambda x: x.get("index", 0))
            for r in results_sorted:
                all_scores.append(r.get("relevance_score", 0.0))

        # Report rerank call to token usage collector
        try:
            collector = get_bean_by_type(TokenUsageCollector)
            collector.add(self.config.model, 0, 0, call_type="rerank")
        except Exception:
            pass

        # Convert to same format as DeepInfra
        return self._convert_response_format(all_scores, len(documents))

    def _convert_response_format(
        self, scores: List[float], num_documents: int
    ) -> Dict[str, Any]:
        """Convert scores to standard format (same as DeepInfra)"""
        if len(scores) < num_documents:
            scores.extend([0.0] * (num_documents - len(scores)))
        scores = scores[:num_documents]

        indexed_scores = [(i, score) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (original_index, score) in enumerate(indexed_scores):
            results.append({"index": original_index, "score": score, "rank": rank})

        return {"results": results}

    async def rerank_memories(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank memories using vLLM reranking service

        Args:
            query: Search query
            hits: List of memory hits to rerank
            top_k: Return top K results (optional)
            instruction: Optional instruction for reranking

        Returns:
            List of reranked memory hits, sorted by relevance score
        """
        if not hits:
            return []

        # Extract text content from hits for reranking
        documents = []
        for hit in hits:
            text = extract_text_from_hit(hit)
            documents.append(text)

        if not documents:
            return []

        # Send rerank request
        try:
            result = await self._send_rerank_request_batch(
                query=query, documents=documents, start_index=0, instruction=instruction
            )

            # Parse results (OpenAI-compatible format)
            if "results" not in result:
                raise RerankError(
                    f"Invalid rerank response format: missing 'results' key"
                )

            # Create score mapping
            score_map = {}
            for item in result["results"]:
                index = item.get("index")
                score = item.get("relevance_score", 0.0)
                if index is not None:
                    score_map[index] = score

            # Create reranked hits with updated scores
            reranked_hits = []
            for i, hit in enumerate(hits):
                if i in score_map:
                    hit_copy = hit.copy()
                    hit_copy['score'] = score_map[i]  # Update score
                    reranked_hits.append(hit_copy)

            # Sort by rerank score (descending)
            reranked_hits.sort(key=lambda x: x.get('score', 0.0), reverse=True)

            # Apply top_k if specified
            if top_k is not None and top_k > 0:
                reranked_hits = reranked_hits[:top_k]

            # Log results
            if reranked_hits:
                top_scores = [f"{h.get('score', 0):.4f}" for h in reranked_hits[:3]]
                logger.info(
                    f"Reranked {len(hits)} hits -> {len(reranked_hits)} results, "
                    f"top scores: {top_scores}"
                )

            return reranked_hits

        except Exception as e:
            logger.error(f"Error in rerank_memories: {e}")
            # Re-raise exception to allow HybridRerankService fallback
            raise

    def get_model_name(self) -> str:
        """Get the current model name"""
        return self.config.model
