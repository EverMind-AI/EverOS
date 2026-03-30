import os
from dotenv import load_dotenv

load_dotenv()


class ExperimentConfig:
    experiment_name: str = "locomo_evaluation"
    datase_path: str = "data/locomo10.json"
    use_emb: bool = True
    use_reranker: bool = True
    use_agentic_retrieval: bool = True
    use_multi_query: bool = True
    num_conv: int = 10

    # MemCell extraction feature switches
    enable_foresight_extraction: bool = False
    enable_clustering: bool = True
    enable_profile_extraction: bool = False

    # Clustering configuration
    cluster_similarity_threshold: float = 0.65
    cluster_max_time_gap_days: float = 7.0

    # Profile configuration
    profile_scenario: str = "assistant"  # group_chat or assistant
    profile_min_confidence: float = 0.6
    profile_min_memcells: int = 1

    # Retrieval mode: 'agentic' or 'lightweight'
    # - agentic: Multi-round retrieval with LLM guidance, high quality but slower
    # - lightweight: Fast retrieval without LLM, faster but slightly lower quality
    retrieval_mode: str = "agentic"  # 'agentic' | 'lightweight'

    # Retrieval configuration
    use_hybrid_search: bool = True  # Use hybrid retrieval (Embedding + BM25 + RRF)
    emb_recall_top_n: int = 40
    reranker_top_n: int = 20

    # Lightweight retrieval parameters (only effective when retrieval_mode='lightweight')
    # lightweight_search_mode: controls which search method to use in lightweight mode
    # - "bm25_only": Only use BM25 search (fast, lexical matching)
    # - "hybrid": BM25 + Embedding + RRF fusion (balanced)
    # - "emb_only": Only use Embedding search (semantic matching)
    lightweight_search_mode: str = "bm25_only"  # 'bm25_only' | 'hybrid' | 'emb_only'
    lightweight_bm25_top_n: int = 50
    lightweight_emb_top_n: int = 50
    lightweight_final_top_n: int = 20

    # Hybrid search parameters (only effective when use_hybrid_search=True)
    hybrid_emb_candidates: int = 50
    hybrid_bm25_candidates: int = 50
    hybrid_rrf_k: int = 40

    # Multi-query retrieval parameters (only effective when use_multi_query=True)
    multi_query_num: int = 3
    multi_query_top_n: int = 50

    # Reranker optimization parameters (high performance configuration)
    reranker_batch_size: int = 20
    reranker_max_retries: int = 3
    reranker_retry_delay: float = 0.8  # Retry interval with exponential backoff
    reranker_timeout: float = 60.0
    reranker_fallback_threshold: float = (
        0.3  # Fall back to original ranking when success rate below threshold
    )
    reranker_concurrent_batches: int = 5

    reranker_instruction: str = (
        "Determine if the passage contains specific facts, entities (names, dates, locations), "
        "or details that directly answer the question."
    )

    # Stage4 parameter: select top-k from event_ids to build context
    response_top_k: int = 10
    
    llm_service: str = "openai"  # openai, vllm
    llm_config: dict = {
        "openai": {
            "llm_provider": "openai",
            "model": "openai/gpt-4.1-mini",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": os.getenv("LLM_API_KEY"),
            "temperature": 0.3,
            "max_tokens": 16384,
        },
        "vllm": {
            "llm_provider": "openai",
            "model": "Qwen3-30B",
            "base_url": "http://0.0.0.0:8000/v1",
            "api_key": "123",
            "temperature": 0,
            "max_tokens": 16384,
        },
    }

    max_retries: int = 5
    max_concurrent_requests: int = 10
