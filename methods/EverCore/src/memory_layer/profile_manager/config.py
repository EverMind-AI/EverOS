"""Configuration for ProfileManager."""

from dataclasses import dataclass


@dataclass
class ProfileManagerConfig:
    """Configuration for ProfileManager.

    Attributes:
        min_confidence: Minimum confidence threshold for value discrimination (0.0-1.0)
        enable_versioning: Whether to keep profile version history
        auto_extract: Whether to automatically extract profiles on cluster updates
        batch_size: Maximum memcells per batch for profile extraction
        max_retries: Maximum retry attempts for failed profile extractions
    """

    min_confidence: float = 0.6
    enable_versioning: bool = True
    auto_extract: bool = True
    batch_size: int = 50
    max_retries: int = 3

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence must be in [0.0, 1.0], got {self.min_confidence}"
            )

        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
