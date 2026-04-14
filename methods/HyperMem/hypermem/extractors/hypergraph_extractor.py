"""
Hypergraph Builder - Hypergraph Extractor

Features:
- Build a complete three-layer hypergraph in one pass from various extractor results
- Convert Fact, Episode, and Topic into hypergraph nodes
- Connect hyperedges into the hypergraph, establishing bidirectional link relationships

Workflow:
1. Extract Episodes using ConvEpisodeExtractor
2. Extract TopicExtractResult and EpisodeHyperedgeExtractResult using TopicExtractor
3. Extract FactExtractResult and FactHyperedgeExtractResult using FactExtractor
4. Use HypergraphExtractor to build all the above results into a complete hypergraph in one pass

Example:
    extractor = HypergraphExtractor()

    hypergraph = extractor.build_hypergraph(
        episodes=[episode1, episode2],
        fact_results=[fact_result1],
        fact_hyperedge_results=[fact_hyperedge1, fact_hyperedge2],
        topic_extract_result=topic_result,
        episode_hyperedge_results=[episode_hyperedge1]
    )
"""

from typing import List, Optional, Dict, Any
from hypermem.utils.logger import get_logger
from hypermem.types import Episode, Topic, Fact
from hypermem.structure import (
    Hypergraph,
    FactNode,
    EpisodeNode,
    TopicNode
)
from hypermem.extractors.fact_extractor import (
    FactExtractResult,
    FactHyperedgeExtractResult
)
from hypermem.extractors.topic_extractor import (
    TopicExtractResult,
    EpisodeHyperedgeExtractResult
)

logger = get_logger(__name__)


class HypergraphExtractor:
    """
    Hypergraph Builder

    Responsible for building a complete three-layer hypergraph structure in one pass
    from the results of various extractors
    """

    def __init__(self):
        """Initialize the hypergraph builder"""
        pass
    
    def build_hypergraph(
        self,
        episodes: List[Episode],
        fact_results: List[FactExtractResult],
        fact_hyperedge_results: List[FactHyperedgeExtractResult],
        topic_extract_result: Optional[TopicExtractResult],
        episode_hyperedge_results: List[EpisodeHyperedgeExtractResult]
    ) -> Hypergraph:
        """
        Build the complete three-layer hypergraph (L1-L2-L3) in one pass

        Args:
            episodes: List of Episodes
            fact_results: List of fact extraction results (one per topic)
            fact_hyperedge_results: List of fact hyperedge extraction results
            topic_extract_result: Topic extraction result
            episode_hyperedge_results: List of Episode hyperedge extraction results

        Returns:
            Complete three-layer hypergraph
        """
        logger.info("[HypergraphExtractor] Starting to build complete three-layer hypergraph")
        
        hypergraph = Hypergraph()
        
        # ==================== Step 1: Build L2 Layer - Episode Nodes ====================
        for episode in episodes:
            episode_id = episode.event_id

            episode_node = EpisodeNode.from_episode(
                episode=episode,
                fact_hyperedge_id="",
                hyperedge={}
            )

            hypergraph.episodes[episode_id] = episode_node
            logger.debug(f"  L2 Layer: Added Episode node {episode_id}")

        logger.info(f"  L2 Layer: Added {len(episodes)} Episode nodes")
        
        # ==================== Step 2: Build L3 Layer - Topic Nodes ====================
        topics = []
        if topic_extract_result:
            topics = topic_extract_result.topics
        
        # Build episode hyperedge mapping
        episode_hyperedge_map = {}  # topic_id -> EpisodeHyperedge
        for result in episode_hyperedge_results:
            if result.episode_hyperedge:
                topic_id_val = result.topic_id
                episode_hyperedge_map[topic_id_val] = result.episode_hyperedge
        
        # Add topic nodes and episode hyperedges
        for topic in topics:
            topic_id_val = topic.topic_id

            episode_hyperedge = episode_hyperedge_map.get(topic_id_val)

            if not episode_hyperedge:
                logger.warning(f"  Topic {topic_id_val}: No episode hyperedge, skipping")
                continue

            # Add episode hyperedge
            hypergraph.add_hyperedge(
                layer="episode",
                hyperedge_id=episode_hyperedge.id,
                relation=episode_hyperedge.relation,
                weights=episode_hyperedge.weights,
                topic_node_id=episode_hyperedge.topic_node_id,
                created_at=episode_hyperedge.created_at,
                coherence_score=episode_hyperedge.coherence_score
            )

            # Create TopicNode
            topic_node = TopicNode.from_topic(
                topic=topic,
                episode_hyperedge_id=episode_hyperedge.id
            )

            hypergraph.topics[topic_id_val] = topic_node
            logger.debug(f"  L3 Layer: Added topic node {topic_id_val}")

        logger.info(f"  L3 Layer: Added {len(topics)} topic nodes")
        
        # ==================== Step 3: Build L1 Layer - Fact Nodes ====================
        # Build fact hyperedge mapping (the same episode may have multiple hyperedges that need merging)
        fact_hyperedge_map = {}  # episode_id -> FactHyperedge (after merging)
        for result in fact_hyperedge_results:
            if result.fact_hyperedge:
                episode_id = result.episode_id
                new_he = result.fact_hyperedge

                if episode_id in fact_hyperedge_map:
                    # Merge relation and weights from multiple hyperedges
                    existing_he = fact_hyperedge_map[episode_id]
                    existing_he.relation.update(new_he.relation)
                    if existing_he.weights and new_he.weights:
                        existing_he.weights.update(new_he.weights)
                    elif new_he.weights:
                        existing_he.weights = new_he.weights
                else:
                    fact_hyperedge_map[episode_id] = new_he
        
        # Collect all facts
        all_facts = []
        for result in fact_results:
            all_facts.extend(result.facts)

        # Add fact nodes
        for fact in all_facts:
            fact_id = fact.fact_id

            # Build hyperedge mapping {hyperedge_id: role}
            hyperedge_dict = {}
            for episode_id in fact.episode_ids:
                fact_hyperedge = fact_hyperedge_map.get(episode_id)
                if fact_hyperedge:
                    role = fact_hyperedge.relation.get(fact_id, "detail")
                    hyperedge_dict[fact_hyperedge.id] = role

            # Create FactNode
            fact_node = FactNode.from_fact(
                fact=fact,
                hyperedge=hyperedge_dict
            )

            hypergraph.facts[fact_id] = fact_node
            logger.debug(f"  L1 Layer: Added fact node {fact_id}")

        logger.info(f"  L1 Layer: Added {len(all_facts)} fact nodes")
        
        # Add fact hyperedges
        for episode_id, fact_hyperedge in fact_hyperedge_map.items():
            hypergraph.fact_hyperedges[fact_hyperedge.id] = fact_hyperedge

            # Update episode's fact_hyperedge_id
            if episode_id in hypergraph.episodes:
                hypergraph.episodes[episode_id].fact_hyperedge_id = fact_hyperedge.id
        
        logger.info(f"  L1 Layer: Added {len(fact_hyperedge_map)} fact hyperedges")

        # ==================== Step 4: Validate Hypergraph Structure ====================
        validation_errors = hypergraph.validate_bidirectional_links()
        if validation_errors:
            logger.warning("[HypergraphExtractor] Hypergraph bidirectional link validation failed:")
            for error_type, error_list in validation_errors.items():
                logger.warning(f"  {error_type}:")
                for error in error_list:
                    logger.warning(f"    - {error}")
        else:
            logger.info("[HypergraphExtractor] Hypergraph bidirectional link validation passed")
        
        logger.info(f"[HypergraphExtractor] Complete three-layer hypergraph built - {hypergraph.get_stats()}")
        
        return hypergraph
    
    def get_hypergraph_summary(self, hypergraph: Hypergraph) -> Dict[str, Any]:
        """
        Get summary information of the hypergraph

        Args:
            hypergraph: Hypergraph instance

        Returns:
            Dictionary containing statistics
        """
        stats = hypergraph.get_stats()
        
        # Compute average degrees
        fact_degrees = [
            hypergraph.get_node_degree("fact", node_id)
            for node_id in hypergraph.facts.keys()
        ]
        episode_degrees = [
            hypergraph.get_node_degree("episode", node_id)
            for node_id in hypergraph.episodes.keys()
        ]
        
        fact_hyperedge_degrees = [
            hypergraph.get_hyperedge_degree("fact", hyperedge_id)
            for hyperedge_id in hypergraph.fact_hyperedges.keys()
        ]
        episode_hyperedge_degrees = [
            hypergraph.get_hyperedge_degree("episode", hyperedge_id)
            for hyperedge_id in hypergraph.episode_hyperedges.keys()
        ]

        summary = {
            "stats": stats,
            "avg_fact_degree": sum(fact_degrees) / len(fact_degrees) if fact_degrees else 0,
            "avg_episode_degree": sum(episode_degrees) / len(episode_degrees) if episode_degrees else 0,
            "avg_fact_hyperedge_degree": sum(fact_hyperedge_degrees) / len(fact_hyperedge_degrees) if fact_hyperedge_degrees else 0,
            "avg_episode_hyperedge_degree": sum(episode_hyperedge_degrees) / len(episode_hyperedge_degrees) if episode_hyperedge_degrees else 0,
        }
        
        return summary
    
    def print_hypergraph_structure(self, hypergraph: Hypergraph):
        """
        Print summary of the hypergraph structure (for debugging)

        Args:
            hypergraph: Hypergraph instance
        """
        logger.info("=" * 80)
        logger.info("Hypergraph Structure Summary")
        logger.info("=" * 80)
        
        summary = self.get_hypergraph_summary(hypergraph)
        
        logger.info(f"Node Statistics:")
        logger.info(f"  L1 Layer (Facts): {summary['stats']['facts']} nodes")
        logger.info(f"  L2 Layer (Episodes): {summary['stats']['episodes']} nodes")
        logger.info(f"  L3 Layer (Topics): {summary['stats']['topics']} nodes")

        logger.info(f"\nHyperedge Statistics:")
        logger.info(f"  L1 Layer (Fact Hyperedges): {summary['stats']['fact_hyperedges']}")
        logger.info(f"  L2 Layer (Episode Hyperedges): {summary['stats']['episode_hyperedges']}")

        logger.info(f"\nAverage Degrees:")
        logger.info(f"  Fact node average degree: {summary['avg_fact_degree']:.2f}")
        logger.info(f"  Episode node average degree: {summary['avg_episode_degree']:.2f}")
        logger.info(f"  Fact hyperedge average degree: {summary['avg_fact_hyperedge_degree']:.2f}")
        logger.info(f"  Episode hyperedge average degree: {summary['avg_episode_hyperedge_degree']:.2f}")
        
        logger.info("=" * 80)
