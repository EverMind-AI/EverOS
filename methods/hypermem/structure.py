"""
Three-Layer Hypergraph Structure Definition

- L1: FactNode / FactHyperedge — facts extracted from episodes
- L2: EpisodeNode / EpisodeHyperedge — conversation episodes
- L3: TopicNode — generalized topic information
"""

import numpy as np
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

from .types import Fact, Episode, Topic, RawDataType


# ==================== Role Type Enumerations ====================

class FactRole(str, Enum):
    """Fact role types in an episode."""
    CORE = "core"
    CONTEXT = "context"
    DETAIL = "detail"
    TEMPORAL = "temporal"
    CAUSAL = "causal"



class EpisodeRole(str, Enum):
    """Episode role types in a topic."""
    INITIATING = "initiating"     # Initiating event
    DEVELOPING = "developing"     # Developing event
    CLIMAX = "climax"            # Climax event
    CONCLUDING = "concluding"     # Concluding event
    RECURRING = "recurring"       # Recurring pattern
    BACKGROUND = "background"     # Background event
    KEY_MOMENT = "key_moment"    # Key moment
    TRANSITION = "transition"     # Transition event



# ==================== Node and Hyperedge Data Types ====================

class FactNode(BaseModel):
    """L1 layer Fact node."""
    id: str = Field(..., description="Fact unique identifier (fact_id)")
    content: str = Field(..., description="Fact content")
    episode_ids: List[str] = Field(default_factory=list, description="List of episode IDs that compose this fact")
    topic_id: str = Field(default="", description="Source topic ID")
    
    confidence: float = Field(default=0.8, description="Confidence (0.0-1.0)")
    temporal: Optional[str] = Field(default=None, description="Time information, format: 'relative time (absolute time)'")
    spatial: Optional[str] = Field(default=None, description="Location information")
    keywords: List[str] = Field(default_factory=list, description="Keywords")
    query_patterns: List[str] = Field(default_factory=list, description="Query patterns that can be answered")
    timestamp: Optional[datetime] = Field(default=None, description="Timestamp")
    
    hyperedge: Dict[str, str] = Field(
        default_factory=dict, 
        description="Hyperedge ID to role mapping (hyperedge_id -> role)"
    )
    
    @classmethod
    def from_fact(cls, fact: Fact, hyperedge: Optional[Dict[str, str]] = None) -> 'FactNode':
        """Create FactNode from Fact data class"""
        return cls(
            id=fact.fact_id,
            content=fact.content,
            episode_ids=fact.episode_ids or [],
            topic_id=fact.topic_id or "",
            confidence=fact.confidence,
            temporal=fact.temporal,
            spatial=fact.spatial,
            keywords=fact.keywords or [],
            query_patterns=fact.query_patterns or [],
            timestamp=fact.timestamp,
            hyperedge=hyperedge or {}
        )
    
    def to_fact(self) -> Fact:
        """Convert to Fact data class"""
        return Fact(
            fact_id=self.id,
            content=self.content,
            episode_ids=self.episode_ids,
            topic_id=self.topic_id,
            confidence=self.confidence,
            temporal=self.temporal,
            spatial=self.spatial,
            keywords=self.keywords,
            query_patterns=self.query_patterns,
            timestamp=self.timestamp
        )
    
    def to_text(self) -> str:
        parts = [self.content]
        if self.temporal:
            parts.append(f"Time: {self.temporal}")
        if self.spatial:
            parts.append(f"Location: {self.spatial}")
        if len(parts) > 1:
            return f"{parts[0]} ({'; '.join(parts[1:])})"
        return self.content



class FactHyperedge(BaseModel):
    """L1 layer Fact hyperedge, connecting multiple fact nodes to the same episode."""
    id: str = Field(..., description="Hyperedge unique identifier")
    
    relation: Dict[str, str] = Field(
        default_factory=dict, 
        description="Fact ID to role mapping (fact_id -> role)"
    )
    
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Importance weight for each fact (fact_id -> weight)"
    )
    
    episode_node_id: str = Field(
        default="", 
        description="Corresponding episode node ID (bidirectional link)"
    )
    
    created_at: Optional[datetime] = Field(default=None, description="Hyperedge creation time")
    extraction_confidence: float = Field(default=0.8, description="Extraction confidence (0.0-1.0)")



class EpisodeNode(BaseModel):
    """
    Episode node in L2 layer.
    
    Stores complete episode information
    This class extends the Episode dataclass, adding hypergraph-specific fields
    Field order is consistent with Episode
    """
    # Node identifier
    id: str = Field(..., description="Node unique identifier, corresponds to event_id")
    
    # Core episode fields (aligned with Episode)
    user_id_list: List[str] = Field(default_factory=list, description="Involved user ID list")
    original_data: List[Dict[str, Any]] = Field(default_factory=list, description="Original data")
    timestamp: Optional[datetime] = Field(default=None, description="Event timestamp")
    summary: Optional[str] = Field(default=None, description="Episode summary")
    
    # Optional episode fields (aligned with Episode)
    participants: Optional[List[str]] = Field(default=None, description="Participant list")
    type: Optional[RawDataType] = Field(default=None, description="Raw data type")
    keywords: Optional[List[str]] = Field(default=None, description="Keywords extracted from episode")
    subject: Optional[str] = Field(default=None, description="Episode subject")
    episode_description: Optional[str] = Field(default=None, description="Episode memory description")
    
    # Hypergraph structure fields
    hyperedge: Dict[str, str] = Field(
        default_factory=dict, 
        description="Hyperedge ID to role mapping (hyperedge_id -> role)"
    )
    fact_hyperedge_id: str = Field(
        default="", 
        description="Corresponding fact hyperedge ID (bidirectional link)"
    )
    
    @classmethod
    def from_episode(cls, episode: Episode, fact_hyperedge_id: str = "", hyperedge: Optional[Dict[str, str]] = None) -> 'EpisodeNode':
        """Create EpisodeNode from Episode data class"""
        return cls(
            id=episode.event_id,
            user_id_list=episode.user_id_list,
            original_data=episode.original_data,
            timestamp=episode.timestamp,
            summary=episode.summary,
            participants=episode.participants,
            type=episode.type,
            keywords=episode.keywords,
            subject=episode.subject,
            episode_description=episode.episode_description,
            hyperedge=hyperedge or {},
            fact_hyperedge_id=fact_hyperedge_id
        )
    
    def to_episode(self) -> Episode:
        """Convert to Episode data class"""
        return Episode(
            event_id=self.id,
            user_id_list=self.user_id_list,
            original_data=self.original_data,
            timestamp=self.timestamp if self.timestamp else datetime.now(),
            summary=self.summary if self.summary else "",
            participants=self.participants,
            type=self.type,
            keywords=self.keywords,
            subject=self.subject,
            episode_description=self.episode_description
        )



class EpisodeHyperedge(BaseModel):
    """
    Episode hyperedge in L2 layer.
    
    Connects multiple episode nodes to the same topic
    Describes each episode's role and importance in the topic
    
    Responsibilities:
    - Store episode role (initiating/developing/climax etc)
    - Store episode weight (importance)
    - Do not store specific semantic content (stored by nodes)
    - Do not store binary relationships between nodes
    """
    # Hyperedge identifier
    id: str = Field(..., description="Hyperedge unique identifier")
    
    # Node role mapping
    relation: Dict[str, str] = Field(
        default_factory=dict, 
        description="""
        Episode ID to role mapping (episode_id -> role)
        Role types refer to EpisodeRole enum:
        - "initiating": Topic initiating event
        - "developing": Topic developing event
        - "climax": Topic climax event
        - "concluding": Topic concluding event
        - "recurring": Recurring pattern event
        - "background": Background event
        - "key_moment": Key moment
        - "transition": Transition event
        """
    )
    
    # Node weights
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="""
        Importance weight of each episode in the topic (episode_id -> weight)
        Range: 0.0-1.0, higher value means more important
        Used for sorting and filtering during retrieval
        """
    )
    
    # Cross-layer connection
    topic_node_id: str = Field(
        default="", 
        description="Corresponding topic node ID (bidirectional link)"
    )
    
    # Hyperedge metadata
    created_at: Optional[datetime] = Field(
        default=None, 
        description="Hyperedge creation time"
    )
    
    coherence_score: float = Field(
        default=0.8,
        description="Topic coherence score, indicates how closely these episodes form a topic (0.0-1.0)"
    )



class TopicNode(BaseModel):
    """
    Topic node in L3 layer.
    
    Stores generalized topic information
    This class extends the Topic dataclass, adding hypergraph-specific fields
    Field order is consistent with Topic
    Provides conversion methods between Topic and TopicNode
    """
    # Node identifier
    id: str = Field(..., description="Node unique identifier, corresponds to topic_id")
    
    # Core topic fields (aligned with Topic)
    title: str = Field(..., description="Topic title/theme")
    summary: str = Field(..., description="Topic summary")
    episode_ids: List[str] = Field(default_factory=list, description="List of episode IDs that compose this topic")
    timestamp: Optional[datetime] = Field(default=None, description="Topic creation time (time of last episode)")
    user_id_list: List[str] = Field(default_factory=list, description="Involved user ID list")
    
    # Optional topic fields (aligned with Topic)
    participants: Optional[List[str]] = Field(default=None, description="Participant list")
    keywords: Optional[List[str]] = Field(default=None, description="Keywords describing this topic")
    
    # Hypergraph structure fields
    episode_hyperedge_id: str = Field(
        default="", 
        description="Corresponding episode hyperedge ID (bidirectional link)"
    )
    
    @classmethod
    def from_topic(cls, topic: Topic, episode_hyperedge_id: str = "") -> 'TopicNode':
        """Create TopicNode from Topic data class"""
        return cls(
            id=topic.topic_id,
            title=topic.title,
            summary=topic.summary,
            episode_ids=topic.episode_ids,
            timestamp=topic.timestamp,
            user_id_list=topic.user_id_list,
            participants=topic.participants,
            keywords=topic.keywords,
            episode_hyperedge_id=episode_hyperedge_id
        )
    
    def to_topic(self) -> Topic:
        """Convert to Topic data class"""
        return Topic(
            topic_id=self.id,
            title=self.title,
            summary=self.summary,
            episode_ids=self.episode_ids,
            timestamp=self.timestamp if self.timestamp else datetime.now(),
            user_id_list=self.user_id_list,
            participants=self.participants,
            keywords=self.keywords
        )



# ==================== Hypergraph Container Class ====================

class Hypergraph(BaseModel):
    """
    Complete three-layer hypergraph structure container
    
    L1: Fact layer - fact nodes and hyperedges
    L2: Episode layer - episode nodes and hyperedges
    L3: Topic layer - topic nodes
    """
    # L1 layer: Fact layer
    facts: Dict[str, FactNode] = Field(
        default_factory=dict, 
        description="Fact node dictionary"
    )
    fact_hyperedges: Dict[str, FactHyperedge] = Field(
        default_factory=dict, 
        description="Fact hyperedge dictionary"
    )
    
    # L2 layer: Episode layer
    episodes: Dict[str, EpisodeNode] = Field(
        default_factory=dict, 
        description="Episode node dictionary, keyed by episode ID"
    )
    episode_hyperedges: Dict[str, EpisodeHyperedge] = Field(
        default_factory=dict, 
        description="Episode hyperedge dictionary, keyed by hyperedge ID"
    )
    
    # L3 layer: Topic layer
    topics: Dict[str, TopicNode] = Field(
        default_factory=dict, 
        description="Topic node dictionary, keyed by topic ID"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'facts': {k: v.model_dump(mode='json') for k, v in self.facts.items()},
            'fact_hyperedges': {k: v.model_dump(mode='json') for k, v in self.fact_hyperedges.items()},
            'episodes': {k: v.model_dump(mode='json') for k, v in self.episodes.items()},
            'episode_hyperedges': {k: v.model_dump(mode='json') for k, v in self.episode_hyperedges.items()},
            'topics': {k: v.model_dump(mode='json') for k, v in self.topics.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Hypergraph':
        return cls(
            facts={k: FactNode(**v) for k, v in data.get('facts', {}).items()},
            fact_hyperedges={k: FactHyperedge(**v) for k, v in data.get('fact_hyperedges', {}).items()},
            episodes={k: EpisodeNode(**v) for k, v in data.get('episodes', {}).items()},
            episode_hyperedges={k: EpisodeHyperedge(**v) for k, v in data.get('episode_hyperedges', {}).items()},
            topics={k: TopicNode(**v) for k, v in data.get('topics', {}).items()}
        )
    
    def get_stats(self) -> Dict[str, int]:
        return {
            'facts': len(self.facts),
            'fact_hyperedges': len(self.fact_hyperedges),
            'episodes': len(self.episodes),
            'episode_hyperedges': len(self.episode_hyperedges),
            'topics': len(self.topics)
        }

    def add_node(self, layer: str, node_id: str, **kwargs):
        """
        Add node to specified layer
        
        Args:
            layer: Layer type ("fact", "episode", "topic")
            node_id: Node ID
        """
        if layer == "fact":
            self.facts[node_id] = FactNode(
                id=node_id,
                content=kwargs.get("content", ""),
                episode_ids=kwargs.get("episode_ids", []),
                topic_id=kwargs.get("topic_id", ""),
                confidence=kwargs.get("confidence", 0.8),
                temporal=kwargs.get("temporal"),
                spatial=kwargs.get("spatial"),
                keywords=kwargs.get("keywords", []),
                query_patterns=kwargs.get("query_patterns", []),
                timestamp=kwargs.get("timestamp"),
                hyperedge=kwargs.get("hyperedge", {})
            )

        elif layer == "episode":
            self.episodes[node_id] = EpisodeNode(
                id=node_id,
                user_id_list=kwargs.get("user_id_list", []),
                original_data=kwargs.get("original_data", []),
                timestamp=kwargs.get("timestamp", None),
                summary=kwargs.get("summary", ""),
                participants=kwargs.get("participants", None),
                type=kwargs.get("type", None),
                keywords=kwargs.get("keywords", None),
                subject=kwargs.get("subject", None),
                episode_description=kwargs.get("episode_description", None),
                hyperedge=kwargs.get("hyperedge", {}),
                fact_hyperedge_id=kwargs.get("fact_hyperedge_id", "")
            )

        elif layer == "topic":
            self.topics[node_id] = TopicNode(
                id=node_id,
                title=kwargs.get("title", ""),
                summary=kwargs.get("summary", ""),
                episode_ids=kwargs.get("episode_ids", []),
                timestamp=kwargs.get("timestamp", None),
                user_id_list=kwargs.get("user_id_list", []),
                participants=kwargs.get("participants", None),
                keywords=kwargs.get("keywords", None),
                episode_hyperedge_id=kwargs.get("episode_hyperedge_id", "")
            )
            
        else:
            raise ValueError(f"Invalid layer: {layer}. Must be 'fact', 'episode', or 'topic'")
    
    def add_hyperedge(self, layer: str, hyperedge_id: str, **kwargs):
        """
        Add hyperedge to specified layer and update connected nodes' adjacency lists
        
        Args:
            layer: Layer type ("fact", "episode")
            hyperedge_id: Hyperedge ID
        """
        if layer == "fact":
            self.fact_hyperedges[hyperedge_id] = FactHyperedge(
                id=hyperedge_id,
                relation=kwargs.get("relation", {}),
                weights=kwargs.get("weights", None),
                episode_node_id=kwargs.get("episode_node_id", ""),
                created_at=kwargs.get("created_at", None),
                extraction_confidence=kwargs.get("extraction_confidence", 0.8)
            )
            for node_id, role in kwargs.get("relation", {}).items():
                if node_id in self.facts:
                    self.facts[node_id].hyperedge[hyperedge_id] = role
                
        elif layer == "episode":
            self.episode_hyperedges[hyperedge_id] = EpisodeHyperedge(
                id=hyperedge_id,
                relation=kwargs.get("relation", {}),
                weights=kwargs.get("weights", None),
                topic_node_id=kwargs.get("topic_node_id", ""),
                created_at=kwargs.get("created_at", None),
                coherence_score=kwargs.get("coherence_score", 0.8)
            )
            for node_id, role in kwargs.get("relation", {}).items():
                if node_id in self.episodes:
                    self.episodes[node_id].hyperedge[hyperedge_id] = role

        else:
            raise ValueError(f"Invalid layer: {layer}. Must be 'fact' or 'episode'")
    
    def get_node(self, layer: str, node_id: str) -> Dict[str, Any]:
        if layer == "fact":
            node = self.facts.get(node_id)
            return node.model_dump() if node else {}
        elif layer == "episode":
            node = self.episodes.get(node_id)
            return node.model_dump() if node else {}
        elif layer == "topic":
            node = self.topics.get(node_id)
            return node.model_dump() if node else {}
        else:
            raise ValueError(f"Invalid layer: {layer}. Must be 'fact', 'episode', or 'topic'")
    
    def get_hyperedge(self, layer: str, hyperedge_id: str) -> Dict[str, Any]:
        if layer == "fact":
            hyperedge = self.fact_hyperedges.get(hyperedge_id)
            return hyperedge.model_dump() if hyperedge else {}
        elif layer == "episode":
            hyperedge = self.episode_hyperedges.get(hyperedge_id)
            return hyperedge.model_dump() if hyperedge else {}
        else:
            raise ValueError(f"Invalid layer: {layer}. Must be 'fact' or 'episode'")
    
    def get_node_degree(self, layer: str, node_id: str) -> int:
        if layer == "fact" and node_id in self.facts:
            return len(self.facts[node_id].hyperedge)
        elif layer == "episode" and node_id in self.episodes:
            return len(self.episodes[node_id].hyperedge)
        elif layer == "topic":
            return 0
        else:
            return 0
    
    def get_hyperedge_degree(self, layer: str, hyperedge_id: str) -> int:
        if layer == "fact":
            hyperedge = self.fact_hyperedges.get(hyperedge_id)
            return len(hyperedge.relation) if hyperedge else 0
        elif layer == "episode":
            hyperedge = self.episode_hyperedges.get(hyperedge_id)
            return len(hyperedge.relation) if hyperedge else 0
        else:
            return 0
    
    def validate_bidirectional_links(self) -> Dict[str, List[str]]:
        """Validate consistency of all bidirectional links in the hypergraph"""
        errors = {
            'fact_node_to_hyperedge': [],
            'fact_hyperedge_to_node': [],
            'fact_hyperedge_to_episode': [],
            'episode_to_fact_hyperedge': [],
            'episode_node_to_hyperedge': [],
            'episode_hyperedge_to_node': [],
            'episode_hyperedge_to_topic': [],
            'topic_to_episode_hyperedge': []
        }
        
        # L1: FactNode -> FactHyperedge
        for fact_id, fact in self.facts.items():
            for hyperedge_id, role in fact.hyperedge.items():
                if hyperedge_id not in self.fact_hyperedges:
                    errors['fact_node_to_hyperedge'].append(
                        f"Fact '{fact_id}' references non-existent hyperedge '{hyperedge_id}'"
                    )
                elif fact_id not in self.fact_hyperedges[hyperedge_id].relation:
                    errors['fact_node_to_hyperedge'].append(
                        f"Fact '{fact_id}' references hyperedge '{hyperedge_id}', but hyperedge doesn't reference back"
                    )
                elif self.fact_hyperedges[hyperedge_id].relation[fact_id] != role:
                    errors['fact_node_to_hyperedge'].append(
                        f"Fact '{fact_id}' has role '{role}' in hyperedge '{hyperedge_id}', but hyperedge has role '{self.fact_hyperedges[hyperedge_id].relation[fact_id]}'"
                    )
        
        # L1: FactHyperedge -> FactNode
        for hyperedge_id, hyperedge in self.fact_hyperedges.items():
            for node_id, role in hyperedge.relation.items():
                if node_id not in self.facts:
                    errors['fact_hyperedge_to_node'].append(
                        f"Fact hyperedge '{hyperedge_id}' references non-existent fact '{node_id}'"
                    )
                elif hyperedge_id not in self.facts[node_id].hyperedge:
                    errors['fact_hyperedge_to_node'].append(
                        f"Fact hyperedge '{hyperedge_id}' references fact '{node_id}', but fact doesn't reference back"
                    )
                elif self.facts[node_id].hyperedge[hyperedge_id] != role:
                    errors['fact_hyperedge_to_node'].append(
                        f"Fact hyperedge '{hyperedge_id}' has role '{role}' for fact '{node_id}', but fact has role '{self.facts[node_id].hyperedge[hyperedge_id]}'"
                    )
        
        # L1-L2: FactHyperedge -> EpisodeNode
        for hyperedge_id, hyperedge in self.fact_hyperedges.items():
            if hyperedge.episode_node_id:
                if hyperedge.episode_node_id not in self.episodes:
                    errors['fact_hyperedge_to_episode'].append(
                        f"Fact hyperedge '{hyperedge_id}' references non-existent episode '{hyperedge.episode_node_id}'"
                    )
                elif self.episodes[hyperedge.episode_node_id].fact_hyperedge_id != hyperedge_id:
                    errors['fact_hyperedge_to_episode'].append(
                        f"Fact hyperedge '{hyperedge_id}' references episode '{hyperedge.episode_node_id}', but episode references hyperedge '{self.episodes[hyperedge.episode_node_id].fact_hyperedge_id}'"
                    )
        
        # L1-L2: EpisodeNode -> FactHyperedge
        for episode_id, episode in self.episodes.items():
            if episode.fact_hyperedge_id:
                if episode.fact_hyperedge_id not in self.fact_hyperedges:
                    errors['episode_to_fact_hyperedge'].append(
                        f"Episode '{episode_id}' references non-existent fact hyperedge '{episode.fact_hyperedge_id}'"
                    )
                elif self.fact_hyperedges[episode.fact_hyperedge_id].episode_node_id != episode_id:
                    errors['episode_to_fact_hyperedge'].append(
                        f"Episode '{episode_id}' references fact hyperedge '{episode.fact_hyperedge_id}', but hyperedge references episode '{self.fact_hyperedges[episode.fact_hyperedge_id].episode_node_id}'"
                    )
        
        # L2: EpisodeNode -> EpisodeHyperedge
        for episode_id, episode in self.episodes.items():
            for hyperedge_id, role in episode.hyperedge.items():
                if hyperedge_id not in self.episode_hyperedges:
                    errors['episode_node_to_hyperedge'].append(
                        f"Episode '{episode_id}' references non-existent hyperedge '{hyperedge_id}'"
                    )
                elif episode_id not in self.episode_hyperedges[hyperedge_id].relation:
                    errors['episode_node_to_hyperedge'].append(
                        f"Episode '{episode_id}' references hyperedge '{hyperedge_id}', but hyperedge doesn't reference back"
                    )
                elif self.episode_hyperedges[hyperedge_id].relation[episode_id] != role:
                    errors['episode_node_to_hyperedge'].append(
                        f"Episode '{episode_id}' has role '{role}' in hyperedge '{hyperedge_id}', but hyperedge has role '{self.episode_hyperedges[hyperedge_id].relation[episode_id]}'"
                    )
        
        # L2: EpisodeHyperedge -> EpisodeNode
        for hyperedge_id, hyperedge in self.episode_hyperedges.items():
            for node_id, role in hyperedge.relation.items():
                if node_id not in self.episodes:
                    errors['episode_hyperedge_to_node'].append(
                        f"Episode hyperedge '{hyperedge_id}' references non-existent episode '{node_id}'"
                    )
                elif hyperedge_id not in self.episodes[node_id].hyperedge:
                    errors['episode_hyperedge_to_node'].append(
                        f"Episode hyperedge '{hyperedge_id}' references episode '{node_id}', but episode doesn't reference back"
                    )
                elif self.episodes[node_id].hyperedge[hyperedge_id] != role:
                    errors['episode_hyperedge_to_node'].append(
                        f"Episode hyperedge '{hyperedge_id}' has role '{role}' for episode '{node_id}', but episode has role '{self.episodes[node_id].hyperedge[hyperedge_id]}'"
                    )
        
        # L2-L3: EpisodeHyperedge -> TopicNode
        for hyperedge_id, hyperedge in self.episode_hyperedges.items():
            if hyperedge.topic_node_id:
                if hyperedge.topic_node_id not in self.topics:
                    errors['episode_hyperedge_to_topic'].append(
                        f"Episode hyperedge '{hyperedge_id}' references non-existent topic '{hyperedge.topic_node_id}'"
                    )
                elif self.topics[hyperedge.topic_node_id].episode_hyperedge_id != hyperedge_id:
                    errors['episode_hyperedge_to_topic'].append(
                        f"Episode hyperedge '{hyperedge_id}' references topic '{hyperedge.topic_node_id}', but topic references hyperedge '{self.topics[hyperedge.topic_node_id].episode_hyperedge_id}'"
                    )
        
        # L2-L3: TopicNode -> EpisodeHyperedge
        for topic_id, topic in self.topics.items():
            if topic.episode_hyperedge_id:
                if topic.episode_hyperedge_id not in self.episode_hyperedges:
                    errors['topic_to_episode_hyperedge'].append(
                        f"Topic '{topic_id}' references non-existent episode hyperedge '{topic.episode_hyperedge_id}'"
                    )
                elif self.episode_hyperedges[topic.episode_hyperedge_id].topic_node_id != topic_id:
                    errors['topic_to_episode_hyperedge'].append(
                        f"Topic '{topic_id}' references episode hyperedge '{topic.episode_hyperedge_id}', but hyperedge references topic '{self.episode_hyperedges[topic.episode_hyperedge_id].topic_node_id}'"
                    )
        
        errors = {k: v for k, v in errors.items() if v}
        return errors
    

# ==================== Hypergraph Embedding Container Class ====================

class HypergraphEmbedding(BaseModel):
    """Complete three-layer hypergraph embedding container"""
    model_config = {"arbitrary_types_allowed": True}
    
    # L1 layer: Fact layer embeddings
    facts: Dict[str, np.ndarray] = Field(default_factory=dict)
    fact_hyperedges: Dict[str, np.ndarray] = Field(default_factory=dict)
    
    # L2 layer: Episode layer embeddings
    episodes: Dict[str, np.ndarray] = Field(default_factory=dict)
    episode_hyperedges: Dict[str, np.ndarray] = Field(default_factory=dict)
    
    # L3 layer: Topic layer embeddings
    topics: Dict[str, np.ndarray] = Field(default_factory=dict)
    
    def get_stats(self) -> Dict[str, int]:
        return {
            'facts': len(self.facts),
            'fact_hyperedges': len(self.fact_hyperedges),
            'episodes': len(self.episodes),
            'episode_hyperedges': len(self.episode_hyperedges),
            'topics': len(self.topics)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        return convert_numpy({
            'facts': self.facts,
            'fact_hyperedges': self.fact_hyperedges,
            'episodes': self.episodes,
            'episode_hyperedges': self.episode_hyperedges,
            'topics': self.topics
        })
    
    def add_embedding(self, layer: str, node_id: str, embedding: np.ndarray):
        if layer == "fact":
            self.facts[node_id] = embedding
        elif layer == "fact_hyperedge":
            self.fact_hyperedges[node_id] = embedding
        elif layer == "episode":
            self.episodes[node_id] = embedding
        elif layer == "episode_hyperedge":
            self.episode_hyperedges[node_id] = embedding
        elif layer == "topic":
            self.topics[node_id] = embedding
        else:
            raise ValueError(f"Invalid layer: {layer}. Must be 'fact', 'episode', 'topic', 'fact_hyperedge', or 'episode_hyperedge'")

    def get_embedding(self, layer: str, node_id: str) -> np.ndarray:
        if layer == "fact":
            return self.facts.get(node_id, np.array([]))
        elif layer == "fact_hyperedge":
            return self.fact_hyperedges.get(node_id, np.array([]))
        elif layer == "episode":
            return self.episodes.get(node_id, np.array([]))
        elif layer == "episode_hyperedge":
            return self.episode_hyperedges.get(node_id, np.array([]))
        elif layer == "topic":
            return self.topics.get(node_id, np.array([]))
        else:
            raise ValueError(f"Invalid layer: {layer}. Must be 'fact', 'episode', 'topic', 'fact_hyperedge', or 'episode_hyperedge'")
    
    
def _print_hypergraph(hypergraph):
    print("  L1 Layer - Fact Layer:")
    print("    Fact Nodes:")
    for fact_id, fact_data in hypergraph.facts.items():
        print(f"      {fact_id}: {fact_data}")
    print("    Fact Hyperedges:")
    for hyperedge_id, hyperedge_data in hypergraph.fact_hyperedges.items():
        print(f"      {hyperedge_id}: {hyperedge_data}")
    
    print("  L2 Layer - Episode Layer:")
    print("    Episode Nodes:")
    for episode_id, episode_data in hypergraph.episodes.items():
        print(f"      {episode_id}: {episode_data}")
    print("    Episode Hyperedges:")
    for hyperedge_id, hyperedge_data in hypergraph.episode_hyperedges.items():
        print(f"      {hyperedge_id}: {hyperedge_data}")
    
    print("  L3 Layer - Topic Layer:")
    print("    Topic Nodes:")
    for topic_id, topic_data in hypergraph.topics.items():
        print(f"      {topic_id}: {topic_data}")


def _test():
    print("=== Testing Hypergraph Basic Functionality ===")
    
    hypergraph = Hypergraph()
    
    print("\n1. Adding nodes and hyperedges...")
    
    # Add fact node
    hypergraph.add_node("fact", "fact_1", 
                       content="John works in New York",
                       episode_ids=["episode_1"],
                       topic_id="topic_1",
                       confidence=0.9,
                       temporal="current",
                       keywords=["work", "location"],
                       query_patterns=["Where does John work?"],
                       timestamp=datetime.now())
    
    # Add fact hyperedge
    hypergraph.add_hyperedge("fact", "fh_1", 
                           relation={"fact_1": "core"},
                           weights={"fact_1": 1.0},
                           episode_node_id="episode_1",
                           created_at=datetime.now(),
                           extraction_confidence=0.9)
    
    # Add episode node
    hypergraph.add_node("episode", "episode_1", 
                       user_id_list=["user_1"],
                       original_data=[{"content": "John works in New York"}],
                       timestamp=datetime.now(),
                       summary="Work information",
                       fact_hyperedge_id="fh_1")
    
    # Add episode hyperedge
    hypergraph.add_hyperedge("episode", "eh_1", 
                           relation={"episode_1": "key_moment"},
                           weights={"episode_1": 1.0},
                           topic_node_id="topic_1",
                           created_at=datetime.now(),
                           coherence_score=0.85)
    
    # Add topic node
    hypergraph.add_node("topic", "topic_1", 
                       title="Work Topic",
                       summary="John works in New York",
                       episode_ids=["episode_1"],
                       timestamp=datetime.now(),
                       user_id_list=["user_1"],
                       episode_hyperedge_id="eh_1")
    
    print("Added 1 fact node, 1 fact hyperedge, 1 episode node, 1 episode hyperedge, 1 topic node")
    print(f"Statistics: {hypergraph.get_stats()}")
    print("Complete hypergraph after addition:")
    _print_hypergraph(hypergraph)
    
    print("\n2. Verifying adjacency lists...")
    
    fact_1_data = hypergraph.get_node("fact", "fact_1")
    print(f"Fact 1 hyperedge adjacency: {fact_1_data.get('hyperedge', {})}")
    print(f"Fact 1 text: {hypergraph.facts['fact_1'].to_text()}")
    
    fh_1_data = hypergraph.get_hyperedge("fact", "fh_1")
    print(f"Fact hyperedge 1 relation: {fh_1_data.get('relation', {})}")
    
    episode_1_data = hypergraph.get_node("episode", "episode_1")
    print(f"Episode 1 hyperedge adjacency: {episode_1_data.get('hyperedge', {})}")
    
    print("\n3. Testing degrees...")
    print(f"Fact 1 degree: {hypergraph.get_node_degree('fact', 'fact_1')}")
    print(f"Episode 1 degree: {hypergraph.get_node_degree('episode', 'episode_1')}")
    print(f"Topic 1 degree: {hypergraph.get_node_degree('topic', 'topic_1')}")
    print(f"Fact hyperedge 1 degree: {hypergraph.get_hyperedge_degree('fact', 'fh_1')}")
    print(f"Episode hyperedge 1 degree: {hypergraph.get_hyperedge_degree('episode', 'eh_1')}")
    
    print("\n4. Validating bidirectional links...")
    validation_errors = hypergraph.validate_bidirectional_links()
    if validation_errors:
        print("Found bidirectional link errors:")
        for error_type, error_list in validation_errors.items():
            print(f"  {error_type}:")
            for error in error_list:
                print(f"    - {error}")
    else:
        print("✓ All bidirectional links validation passed")
    
    print("\n5. Testing error detection (intentionally breaking a bidirectional link)...")
    hypergraph.facts["fact_1"].hyperedge["non_existent_hyperedge"] = "test_role"
    
    validation_errors = hypergraph.validate_bidirectional_links()
    if validation_errors:
        print("✓ Successfully detected bidirectional link errors:")
        for error_type, error_list in validation_errors.items():
            print(f"  {error_type}:")
            for error in error_list:
                print(f"    - {error}")
    else:
        print("✗ Failed to detect bidirectional link errors")
    
    hypergraph.facts["fact_1"].hyperedge.pop("non_existent_hyperedge", None)
    
    print("\n6. Testing role and weight functionality...")
    
    hypergraph.add_node("fact", "fact_2", 
                       content="John started working in 2020",
                       episode_ids=["episode_1"],
                       topic_id="topic_1",
                       confidence=0.8,
                       temporal="2020",
                       keywords=["work", "time"],
                       timestamp=datetime.now())
    
    hypergraph.add_node("fact", "fact_3", 
                       content="New York is a big city",
                       episode_ids=["episode_1"],
                       topic_id="topic_1",
                       confidence=0.5,
                       keywords=["location", "attribute"],
                       timestamp=datetime.now())
    
    hypergraph.add_hyperedge("fact", "fh_2", 
                           relation={
                               "fact_1": "core",
                               "fact_2": "temporal",
                               "fact_3": "context"
                           },
                           weights={
                               "fact_1": 1.0,
                               "fact_2": 0.7,
                               "fact_3": 0.3
                           },
                           episode_node_id="episode_1",
                           created_at=datetime.now(),
                           extraction_confidence=0.85)
    
    fh_2 = hypergraph.fact_hyperedges.get("fh_2")
    if fh_2:
        print(f"\nFact Hyperedge fh_2:")
        print(f"  All relations: {fh_2.relation}")
        print(f"  All weights: {fh_2.weights}")
        
        core_facts = [fid for fid, role in fh_2.relation.items() if role == "core"]
        print(f"  Core facts: {core_facts}")
        
        if fh_2.weights:
            sorted_facts = sorted(fh_2.weights.items(), key=lambda x: x[1], reverse=True)
            print(f"  Facts by importance: {sorted_facts}")
    
    print("\n=== Test Completed Successfully! ===")


if __name__ == "__main__":
    _test()
