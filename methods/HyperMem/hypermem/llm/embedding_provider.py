from typing import List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import time

class EmbeddingProvider:
    def __init__(self, base_url: str, model_name: str, timeout: int = 120, max_retries: int = 5):
        self.base_url = base_url
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Create a session with retry mechanism
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,  # 1, 2, 4, 8, 16 seconds wait
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def cosine_similarity(self, query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """
        Calculates cosine similarity between a query vector and multiple document vectors.

        Args:
            query_vec: A 1D numpy array for the query.
            doc_vecs: A 2D numpy array where each row is a document vector.

        Returns:
            A 1D numpy array of cosine similarity scores.
        """
        # Calculate dot product
        dot_product = np.dot(doc_vecs, query_vec)

        # Calculate norms
        query_norm = np.linalg.norm(query_vec)
        doc_norms = np.linalg.norm(doc_vecs, axis=1)

        # Calculate cosine similarity, handling potential division by zero
        denominator = query_norm * doc_norms
        # Replace 0s in denominator with a small number to avoid division by zero
        denominator[denominator == 0] = 1e-9
        
        similarity_scores = dot_product / denominator
        
        return similarity_scores
        
    def embed(self, texts: List[str]) -> List[List[float]]:
        if 'Qwen3' not in self.model_name:
            raise ValueError(f"Model {self.model_name} is not supported, only Qwen3-Embedding series models is supported")

        # Manual retry logic (handles connection interruption and other exceptions)
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    self.base_url, 
                    json={"input": texts, "model": self.model_name},
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()
                vectors = [item['embedding'] for item in result['data']]
                return vectors
            except (requests.exceptions.ConnectionError, 
                    requests.exceptions.Timeout,
                    requests.exceptions.ChunkedEncodingError) as e:
                last_exception = e
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                print(f"    [!] Connection failed (attempt {attempt + 1}/{self.max_retries}), retrying in {wait_time}s: {str(e)[:50]}")
                time.sleep(wait_time)
            except Exception as e:
                # Raise other errors directly
                raise e
        
        # All retries exhausted
        raise last_exception

if __name__ == "__main__":
    inputs = [
        "Tom moved here from his hometown last month",
        "Frank's hometown is Switzerland",
    ]

    reranker = EmbeddingProvider(base_url="http://0.0.0.0:11000/v1/embeddings", model_name="Qwen3-Embedding-4B")
    print(reranker.embed(inputs))