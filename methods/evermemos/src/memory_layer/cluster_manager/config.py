"""Configuration for ClusterManager."""

from dataclasses import dataclass


@dataclass
class ClusterManagerConfig:
    """Configuration for ClusterManager.

    Attributes:
        similarity_threshold: Minimum cosine similarity to join existing cluster (0.0-1.0)
        max_time_gap_days: Maximum time gap in days to link to existing cluster
        enable_persistence: Whether to persist mem scene state to disk
        persist_dir: Directory for mem scene state persistence (required if enable_persistence=True)
        clustering_algorithm: Algorithm to use ('centroid' or 'nearest')
        llm_top_k_clusters: Number of candidate clusters pre-filtered by embedding for LLM
        llm_max_context_per_cluster: Max recent items per cluster in LLM context
        llm_skip_threshold: Skip LLM if top-1 embedding similarity exceeds this
    """

    similarity_threshold: float = 0.65
    max_time_gap_days: float = 7.0
    enable_persistence: bool = False
    persist_dir: str = None
    clustering_algorithm: str = "centroid"  # 'centroid' or 'nearest'
    # LLM clustering: number of candidate clusters pre-filtered by embedding similarity
    llm_top_k_clusters: int = 30
    # LLM clustering: max recent items per cluster to include in LLM context
    llm_max_context_per_cluster: int = 5
    # LLM clustering: if top-1 embedding similarity exceeds this threshold,
    # skip LLM and assign directly (set to 1.0 to always use LLM)
    llm_skip_threshold: float = 0.85

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be in [0.0, 1.0], got {self.similarity_threshold}"
            )

        if self.max_time_gap_days < 0:
            raise ValueError(
                f"max_time_gap_days must be >= 0, got {self.max_time_gap_days}"
            )

        if self.enable_persistence and not self.persist_dir:
            raise ValueError("persist_dir is required when enable_persistence=True")

        if self.clustering_algorithm not in ("centroid", "nearest"):
            raise ValueError(
                f"clustering_algorithm must be 'centroid' or 'nearest', got {self.clustering_algorithm}"
            )

    @property
    def max_time_gap_seconds(self) -> float:
        """Get max time gap in seconds."""
        return self.max_time_gap_days * 24 * 60 * 60
