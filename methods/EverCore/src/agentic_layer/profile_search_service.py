"""
Profile Search Service

Provides vector search for user profile items in Milvus.
Returns profile items directly without reranking.

Profile items include:
- explicit_info: User's explicit information (category + description)
- implicit_trait: Inferred user traits (trait_name + description)
"""

import os
from typing import List, Dict, Any, Optional
import time

from core.di import get_bean_by_type
from core.di.decorators import service
from core.observation.logger import get_logger
from agentic_layer.vectorize_service import get_vectorize_service
from infra_layer.adapters.out.search.repository.user_profile_milvus_repository import (
    UserProfileMilvusRepository,
)

logger = get_logger(__name__)

# Configuration from environment variables
PROFILE_RECALL_THRESHOLD = float(os.getenv("PROFILE_RECALL_THRESHOLD", "0.5"))
PROFILE_DEFAULT_TOPK = int(os.getenv("PROFILE_DEFAULT_TOPK", "30"))


def parse_embed_text(embed_text: str, item_type: str) -> Dict[str, str]:
    """
    Parse embed_text to extract category/trait_name and description.
    
    Format:
    - explicit_info: "category: description"
    - implicit_trait: "trait_name: description. basis" or "trait_name: description"
    
    Args:
        embed_text: The embedded text string
        item_type: "explicit_info" or "implicit_trait"
    
    Returns:
        Dict with parsed fields
    """
    result = {}
    
    if not embed_text:
        return {"category": "", "description": ""} if item_type == "explicit_info" else {"trait_name": "", "description": ""}
    
    # Split by first colon
    parts = embed_text.split(":", 1)
    
    if len(parts) == 2:
        key = parts[0].strip()
        value = parts[1].strip()
        
        if item_type == "explicit_info":
            result["category"] = key
            result["description"] = value
        else:  # implicit_trait
            result["trait_name"] = key
            # For implicit_trait, the description may contain ". basis"
            # We just use everything after the colon as description
            result["description"] = value
    else:
        # Fallback: use entire text as description
        if item_type == "explicit_info":
            result["category"] = ""
            result["description"] = embed_text
        else:
            result["trait_name"] = ""
            result["description"] = embed_text
    
    return result


@service(name="profile_search_service", primary=True)
class ProfileSearchService:
    """
    Profile Search Service
    
    Searches user profile items in Milvus using vector similarity.
    No reranking step - directly returns Milvus results with score threshold.
    """
    
    def __init__(
        self,
        milvus_repo: Optional[UserProfileMilvusRepository] = None,
    ):
        """Initialize service
        
        Args:
            milvus_repo: User profile Milvus repository (auto-injected if None)
        """
        self._milvus_repo = milvus_repo
    
    @property
    def milvus_repo(self) -> UserProfileMilvusRepository:
        """Lazy load Milvus repository"""
        if self._milvus_repo is None:
            self._milvus_repo = get_bean_by_type(UserProfileMilvusRepository)
        return self._milvus_repo
    
    async def search_profiles(
        self,
        query: str,
        user_id: str,
        group_id: str,
        top_k: int = PROFILE_DEFAULT_TOPK,
        score_threshold: float = PROFILE_RECALL_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Search profile items by query text
        
        Args:
            query: Search query text
            user_id: User ID filter
            group_id: Group ID filter
            top_k: Maximum number of results
            score_threshold: Minimum similarity score (0.0-1.0)
        
        Returns:
            Dict with:
            - profiles: List of profile items
            - metadata: Search metadata (latency, count, etc.)
        """
        start_time = time.perf_counter()
        
        result = {
            "profiles": [],
            "metadata": {
                "profile_count": 0,
                "latency_ms": 0,
            }
        }
        
        if not query:
            logger.warning("Empty query for profile search")
            return result
        
        try:
            # Step 1: Generate query embedding
            vectorize_service = get_vectorize_service()
            query_vector = await vectorize_service.get_embedding(query)
            
            if query_vector is None or len(query_vector) == 0:
                logger.warning("Failed to generate query embedding")
                return result
            
            # Step 2: Search Milvus (recall with threshold)
            # Recall more candidates, then filter by threshold
            recall_limit = top_k * 2 if top_k > 0 else PROFILE_DEFAULT_TOPK * 2
            
            logger.info(
                f"🔍 Profile search params: user_id={user_id}, group_id={group_id}, "
                f"top_k={top_k}, recall_limit={recall_limit}, score_threshold={score_threshold}"
            )
            
            milvus_results = await self.milvus_repo.vector_search(
                query_vector=query_vector,
                user_id=user_id,
                group_id=group_id,
                limit=recall_limit,
                score_threshold=score_threshold,
            )
            
            logger.info(
                f"✅ Milvus returned {len(milvus_results)} results, will take top {top_k}"
            )
            
            # Step 3: Process results - parse embed_text and format output
            profiles = []
            for item in milvus_results[:top_k]:
                item_type = item.get("item_type", "")
                embed_text = item.get("embed_text", "")
                
                # Parse embed_text to get category/trait_name and description
                parsed = parse_embed_text(embed_text, item_type)
                
                profile_item = {
                    "item_type": item_type,
                    "score": round(item.get("score", 0.0), 4),
                }
                
                # Add parsed fields based on item_type
                if item_type == "explicit_info":
                    profile_item["category"] = parsed.get("category", "")
                    profile_item["description"] = parsed.get("description", "")
                else:  # implicit_trait
                    profile_item["trait_name"] = parsed.get("trait_name", "")
                    profile_item["description"] = parsed.get("description", "")
                
                profiles.append(profile_item)
            
            # Calculate latency
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            
            result["profiles"] = profiles
            result["metadata"]["profile_count"] = len(profiles)
            result["metadata"]["latency_ms"] = latency_ms
            
            # Log profile scores for debugging
            if profiles:
                scores_str = ", ".join([f"{p['score']:.4f}" for p in profiles])
                logger.info(
                    f"📊 Profile scores: [{scores_str}]"
                )
            
            logger.info(
                "Profile search completed: user_id=%s, group_id=%s, "
                "query='%s', found=%d, latency=%dms",
                user_id,
                group_id,
                query[:50] if query else "",
                len(profiles),
                latency_ms,
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Profile search failed: user_id=%s, group_id=%s, error=%s",
                user_id,
                group_id,
                e,
                exc_info=True,
            )
            result["metadata"]["latency_ms"] = int((time.perf_counter() - start_time) * 1000)
            return result


def get_profile_search_service() -> ProfileSearchService:
    """Get ProfileSearchService from DI container"""
    return get_bean_by_type(ProfileSearchService)

