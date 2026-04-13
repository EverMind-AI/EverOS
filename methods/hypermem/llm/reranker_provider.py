"""
Qwen3-Reranker provider via vLLM HTTP completions API.

Parameters aligned with official usage:
- logprobs=20 (official uses 20, not 5)
- default logprob=-10 when token not found (official uses -10, not -100)
- score = true_score / (true_score + false_score)

Reference: https://huggingface.co/Qwen/Qwen3-Reranker-4B
"""

from typing import List
import requests
import math
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time


class RerankerProvider:
    def __init__(self, base_url: str, model_name: str, timeout: int = 120, max_retries: int = 10):
        self.base_url = base_url
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries

        # Create a session with retry mechanism
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def rerank(self, queries: List[str], docs: List[str], instruction: str = None) -> List[float]:
        """Rerank documents against queries using Qwen3-Reranker via completions API + logprobs.

        Aligned with official vLLM usage from HuggingFace model card.
        """
        if 'Qwen3' not in self.model_name:
            raise ValueError(f"Model {self.model_name} is not supported, only Qwen3-Reranker series models is supported")

        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        if instruction is None:
            instruction = "Given a user's question and a text passage, determine if the passage contains specific information that directly answers the question. A relevant passage should provide a clear and precise answer, not just be on the same topic."

        # Build prompts
        prompts = []
        for query, doc in zip(queries, docs):
            prompt = f"{prefix}<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}{suffix}"
            prompts.append(prompt)

        # Use completions API with logprobs (official: logprobs=20)
        completions_url = self.base_url.rstrip('/') + '/v1/completions'

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    completions_url,
                    json={
                        "model": self.model_name,
                        "prompt": prompts,
                        "max_tokens": 1,
                        "logprobs": 20,  # Official uses 20
                        "temperature": 0,
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()

                scores = []
                for choice in result['choices']:
                    top_logprobs = choice.get('logprobs', {}).get('top_logprobs', [{}])[0]
                    # Official default: -10 when token not found
                    yes_logprob = top_logprobs.get('yes', -10)
                    no_logprob = top_logprobs.get('no', -10)
                    yes_prob = math.exp(yes_logprob)
                    no_prob = math.exp(no_logprob)
                    score = yes_prob / (yes_prob + no_prob) if (yes_prob + no_prob) > 0 else 0
                    scores.append(score)
                return scores

            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.ChunkedEncodingError) as e:
                last_exception = e
                wait_time = 2 ** attempt
                print(f"    [!] Reranker connection failed (attempt {attempt + 1}/{self.max_retries}), retrying in {wait_time}s: {str(e)[:50]}")
                time.sleep(wait_time)
            except Exception as e:
                raise e

        raise ConnectionError(f"Reranker service connection failed after {self.max_retries} retries: {str(last_exception)}")
