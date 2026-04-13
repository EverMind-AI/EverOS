"""
HyperMem Extractors

This module contains extractors for building hypergraph memory structures:
- EpisodeExtractor: Base class for episode boundary detection
- ConvEpisodeExtractor: Conversation-specific episode extraction
- FactExtractor: Extract facts from topics
- TopicExtractor: Extract topics from episodes
- HypergraphExtractor: Build complete hypergraph from extraction results
"""

from hypermem.extractors.episode_extractor import (
    EpisodeExtractor,
    EpisodeExtractRequest,
    RawData,
    StatusResult,
)
from hypermem.extractors.conv_episode_extractor import (
    ConvEpisodeExtractor,
    ConvEpisodeExtractRequest,
    BoundaryDetectionResult,
)
from hypermem.extractors.fact_extractor import (
    FactExtractor,
    FactExtractResult,
    FactHyperedgeExtractResult,
)
from hypermem.extractors.topic_extractor import (
    TopicExtractor,
    TopicExtractRequest,
    TopicExtractResult,
    EpisodeHyperedgeExtractResult,
)
from hypermem.extractors.hypergraph_extractor import HypergraphExtractor

__all__ = [
    "EpisodeExtractor",
    "EpisodeExtractRequest",
    "RawData",
    "StatusResult",
    "ConvEpisodeExtractor",
    "ConvEpisodeExtractRequest",
    "BoundaryDetectionResult",
    "FactExtractor",
    "FactExtractResult",
    "FactHyperedgeExtractResult",
    "TopicExtractor",
    "TopicExtractRequest",
    "TopicExtractResult",
    "EpisodeHyperedgeExtractResult",
    "HypergraphExtractor",
]
