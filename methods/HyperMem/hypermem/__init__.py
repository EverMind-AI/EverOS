"""
HyperMem: Hypergraph-based Memory System for Long-term Conversational QA
"""

from hypermem.types import Episode, Fact, Topic, RawDataType
from hypermem.structure import (
    Hypergraph,
    FactNode,
    EpisodeNode,
    TopicNode,
    FactHyperedge,
    EpisodeHyperedge,
    FactRole,
    EpisodeRole,
)

__version__ = "0.1.0"
__all__ = [
    # Types
    "Episode",
    "Fact",
    "Topic",
    "RawDataType",
    # Structure
    "Hypergraph",
    "FactNode",
    "EpisodeNode",
    "TopicNode",
    "FactHyperedge",
    "EpisodeHyperedge",
    "FactRole",
    "EpisodeRole",
]
