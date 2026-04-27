"""
DeepInfra Rerank Service Implementation

Reranking service using DeepInfra commercial API.
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
class DeepInfraRerankConfig:
    """DeepInfra rerank service configuration"""

    api_key: str = ""  # skip-sensitive-check
    base_url: str = "https://api.deepinfra.com/v1/inference"
    model: str = "Qwen/Qwen3-Reranker-4B"
    timeout: int = 30
    max_retries: int = 3
    batch_size: int = 10
    max_concurrent_requests: int = 5


class DeepInfraRerankService(RerankServiceInterface):
    """DeepInfra reranking service implementation"""

    def __init__(self, config: Optional[DeepInfraRerankConfig] = None):
        if config is None:
            config = DeepInfraRerankConfig()

        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        logger.info(f"Initialized DeepInfraRerankService | model={config.model}")

    async def _ensure_session(self):
        """Ensure HTTP session is created"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
            )

    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()

    def _format_rerank_texts(
        self, query: str, documents: List[str], instruction: Optional[str] = None
    ):
        """Format rerank request texts (Qwen-Reranker format)"""
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
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
        """Send rerank request batch to DeepInfra API"""
        await self._ensure_session()

        # Format texts
        queries, formatted_docs = self._format_rerank_texts(
            query, documents, instruction
        )

        url = self.config.base_url
        if not url.endswith(self.config.model):
            url = f"{url}/{self.config.model}"

        request_data = {"queries": queries, "documents": formatted_docs}

        async with self._semaphore:
            for attempt in range(self.config.max_retries):
                try:
                    async with self.session.post(url, json=request_data) as response:
                        if response.status == 200:
                            json_body = await response.json()
                            return self._parse_response(json_body)
                        else:
                            error_text = await response.text()
                            logger.error(
                                f"DeepInfra rerank API error {response.status}: {error_text}"
                            )
                            if attempt < self.config.max_retries - 1:
                                await asyncio.sleep(2**attempt)
                                continue
                            raise RerankError(
                                f"API failed: {response.status} - {error_text}"
                            )
                except Exception as e:
                    logger.error(f"DeepInfra rerank exception: {e}")
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    raise RerankError(f"Exception: {e}")

    def _parse_response(self, json_body: Dict[str, Any]) -> Dict[str, Any]:
        """Parse DeepInfra API response"""
        scores = []
        if "results" in json_body:
            results = json_body["results"]
            results.sort(key=lambda x: x.get("index", 0))
            scores = [item.get("relevance_score", 0.0) for item in results]
        elif "scores" in json_body:
            scores = json_body["scores"]

        return {
            "scores": scores,
            "input_tokens": json_body.get("usage", {}).get("prompt_tokens", 0),
            "request_id": json_body.get("id"),
        }

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

        # Split into batches
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
        total_input_tokens = 0
        last_response = None

        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Rerank batch {i} failed: {result}")
                batch_len = len(batches[i])
                all_scores.extend([-100.0] * batch_len)
                continue

            scores = result.get("scores", [])
            all_scores.extend(scores)
            total_input_tokens += result.get("input_tokens", 0)
            last_response = result

        # Report rerank token usage
        try:
            collector = get_bean_by_type(TokenUsageCollector)
            collector.add(self.config.model, total_input_tokens, 0, call_type="rerank")
        except Exception:
            pass

        combined_response = {
            "scores": all_scores,
            "input_tokens": total_input_tokens,
            "request_id": last_response.get("request_id") if last_response else None,
        }
        return self._convert_response_format(combined_response, len(documents))

    def _convert_response_format(
        self, combined_response: Dict[str, Any], num_documents: int
    ) -> Dict[str, Any]:
        """Convert response to standard format"""
        scores = combined_response.get("scores", [])
        if len(scores) < num_documents:
            scores.extend([0.0] * (num_documents - len(scores)))
        scores = scores[:num_documents]

        indexed_scores = [(i, score) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (original_index, score) in enumerate(indexed_scores):
            results.append({"index": original_index, "score": score, "rank": rank})

        return {
            "results": results,
            "input_tokens": combined_response.get("input_tokens", 0),
            "request_id": combined_response.get("request_id"),
        }

    async def rerank_memories(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank memories using DeepInfra API

        Args:
            query: Query text
            hits: List of memory hits to rerank
            top_k: Return top K results (optional)
            instruction: Optional reranking instruction

        Returns:
            List of reranked memory hits, sorted by relevance score
        """
        if not hits:
            return []

        # Extract text content from hits for reranking
        all_texts = []
        for hit in hits:
            text = extract_text_from_hit(hit)
            all_texts.append(text)

        if not all_texts:
            return []

        # Call reranking API
        try:
            logger.debug(
                f"Starting reranking, query text: {query}, number of texts: {len(all_texts)}"
            )
            rerank_result = await self.rerank_documents(query, all_texts, instruction)

            if "results" not in rerank_result:
                raise RerankError("Invalid rerank API response: missing results field")

            # Parse reranking results
            results_meta = rerank_result.get("results", [])

            # Reorganize hits according to reranked order
            reranked_hits = []
            for item in results_meta:
                original_idx = item.get("index", 0)
                score = item.get("score", 0.0)
                if 0 <= original_idx < len(hits):
                    hit = hits[original_idx].copy()
                    hit['score'] = score  # Unified score field
                    reranked_hits.append(hit)

            # If top_k is specified, return only the top_k results
            if top_k is not None and top_k > 0:
                reranked_hits = reranked_hits[:top_k]

            # Print top 3 result scores for debugging
            if reranked_hits:
                top_scores = [f"{h.get('score', 0):.4f}" for h in reranked_hits[:3]]
                logger.info(
                    f"Reranking completed: {len(reranked_hits)} results, top scores: {top_scores}"
                )
            return reranked_hits

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # If reranking fails, return original results (sorted by original score)
            sorted_hits = sorted(hits, key=lambda x: x.get('score', 0), reverse=True)
            if top_k is not None and top_k > 0:
                sorted_hits = sorted_hits[:top_k]
            return sorted_hits

    def get_model_name(self) -> str:
        """Get the current model name"""
        return self.config.model
