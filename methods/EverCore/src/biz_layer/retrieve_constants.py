"""
Memory Retrieval Configuration Constants

This module defines constants related to memory retrieval behavior.
These constants can be configured via environment variables.
"""

import os

# ===================
# Retrieve Configuration (configurable via .env)
# ===================

# === Unlimited Mode Configuration (top_k=-1) ===
# Maximum return count when in unlimited mode
# Configurable via TOPK_LIMIT env var, default: 100
DEFAULT_TOPK_LIMIT = int(os.getenv("TOPK_LIMIT", "100"))

# Magic constant: Maximum group_ids array length
# Maximum number of group_ids that can be queried in a single search request
MAX_GROUP_IDS_COUNT = 50

# === Recall Multiplier (Normal Mode: top_k > 0) ===
# When top_k > 0, actual recall count = top_k * RECALL_MULTIPLIER
# Purpose: Provide a larger candidate pool for rerank to improve final result quality
# Configurable via RECALL_MULTIPLIER env var, default: 2
DEFAULT_RECALL_MULTIPLIER = int(os.getenv("RECALL_MULTIPLIER", "2"))

# === Threshold Configuration (Unlimited Mode Only) ===
# Milvus COSINE similarity threshold (range 0-1), effective in unlimited mode
# When top_k=-1 and radius is not specified, use this threshold to filter results
# Configurable via MILVUS_SIMILARITY_THRESHOLD env var, default: 0.6
DEFAULT_MILVUS_SIMILARITY_THRESHOLD = float(
    os.getenv("MILVUS_SIMILARITY_THRESHOLD", "0.6")
)

# Rerank score threshold (range depends on reranker model, typically 0-1)
# Applied after rerank in unlimited mode to filter low-quality results
# Configurable via RERANK_SCORE_THRESHOLD env var, default: 0.6
DEFAULT_RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.6"))

# Note: ES BM25 threshold is not set for now, as BM25 scores have no standardized range
# Will be determined after data validation

# === Agent Memory Retrieval Thresholds ===
# Milvus COSINE similarity radius for agent_memory hybrid retrieval
# Filters out low-similarity vectors at Milvus search level
# Configurable via AGENT_MEMORY_MILVUS_RADIUS env var, default: 0.1
AGENT_MEMORY_MILVUS_RADIUS = float(os.getenv("AGENT_MEMORY_MILVUS_RADIUS", "0.1"))

# ES minimum_should_match for agent_memory BM25 keyword search
# Percentage of query terms that must match (e.g., "10%" means 1 of 10 terms)
# Supports ES syntax: integer (2), negative (-2), percentage ("10%")
# Configurable via AGENT_MEMORY_ES_MIN_SHOULD_MATCH env var, default: "10%"
AGENT_MEMORY_ES_MIN_SHOULD_MATCH = os.getenv("AGENT_MEMORY_ES_MIN_SHOULD_MATCH", "10%")

# Export all constants
__all__ = [
    "DEFAULT_TOPK_LIMIT",
    "MAX_GROUP_IDS_COUNT",
    "DEFAULT_RECALL_MULTIPLIER",
    "DEFAULT_MILVUS_SIMILARITY_THRESHOLD",
    "DEFAULT_RERANK_SCORE_THRESHOLD",
    "AGENT_MEMORY_MILVUS_RADIUS",
    "AGENT_MEMORY_ES_MIN_SHOULD_MATCH",
]
