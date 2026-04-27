"""
Profile Life Indexer

Handles indexing of UserProfile data into Milvus for vector search.

Indexing Strategy: Delete-then-Insert with Redis Distributed Lock
- When a profile is created or updated, delete all existing items for the user
- Then rebuild the full index with fresh embeddings
- Uses Redis distributed lock to prevent concurrent indexing of same user+group

This ensures:
1. No stale data in the index
2. Index always reflects the current profile state
3. Handles item deletions and reorderings correctly
4. Prevents race conditions in distributed environment
"""

from typing import List, Dict, Any, Optional

from core.di import get_bean_by_type
from core.di.decorators import service
from core.observation.logger import get_logger
from core.lock.redis_distributed_lock import distributed_lock
from agentic_layer.vectorize_service import get_vectorize_service
from infra_layer.adapters.out.search.repository.user_profile_milvus_repository import (
    UserProfileMilvusRepository,
)
from infra_layer.adapters.out.search.milvus.converter.user_profile_milvus_converter import (
    UserProfileMilvusConverter,
)
from infra_layer.adapters.out.persistence.document.memory.user_profile import (
    UserProfile as MongoUserProfile,
)
from api_specs.memory_types import ProfileMemory

logger = get_logger(__name__)


@service(name="profile_indexer", primary=True)
class ProfileIndexer:
    """
    Profile Life Indexer Service

    Responsible for:
    1. Splitting ProfileMemory into individual items
    2. Generating embeddings for each item
    3. Storing items in Milvus with delete-then-insert strategy
    """

    def __init__(self, milvus_repo: Optional[UserProfileMilvusRepository] = None):
        """Initialize indexer

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

    async def index_profile(
        self, user_id: str, group_id: str, profile: ProfileMemory, doc_id: str = ""
    ) -> Dict[str, int]:
        """
        Index a user profile into Milvus

        Strategy: Delete-then-Insert with Redis Distributed Lock
        1. Acquire distributed lock for user_id + group_id (prevent concurrent indexing across instances)
        2. Delete all existing items for user_id + group_id
        3. Split profile into individual items
        4. Generate embeddings for all items
        5. Batch insert into Milvus
        6. Release lock automatically (context manager)

        Args:
            user_id: User ID
            group_id: Group ID
            profile: ProfileMemory object containing explicit_info and implicit_traits
            doc_id: MongoDB document ID for generating unique Milvus entity IDs.
                    If empty, will query MongoDB to get the actual document ID.

        Returns:
            Dict with indexing statistics:
            - deleted_count: Number of deleted items
            - explicit_count: Number of explicit_info items indexed
            - implicit_count: Number of implicit_trait items indexed
            - total_count: Total items indexed
        """
        # Create distributed lock key for this user+group
        lock_resource = f"profile_index:{user_id}:{group_id}"

        # Acquire Redis distributed lock
        # timeout: 30s (enough for vectorization + insertion)
        # blocking_timeout: 40s (wait for previous task to complete)
        async with distributed_lock(
            resource=lock_resource, timeout=30.0, blocking_timeout=40.0
        ) as acquired:
            if not acquired:
                logger.error(
                    "[ProfileIndexer] Failed to acquire distributed lock: user_id=%s, group_id=%s",
                    user_id,
                    group_id,
                )
                return {
                    "deleted_count": 0,
                    "explicit_count": 0,
                    "implicit_count": 0,
                    "total_count": 0,
                }

            logger.debug(
                "[ProfileIndexer] Acquired distributed lock for user_id=%s, group_id=%s",
                user_id,
                group_id,
            )

            stats = {
                "deleted_count": 0,
                "explicit_count": 0,
                "implicit_count": 0,
                "total_count": 0,
            }

            try:
                logger.info(
                    "[ProfileIndexer] Starting index for user_id=%s, group_id=%s",
                    user_id,
                    group_id,
                )

                # Step 1: Delete existing items
                deleted_count = await self.milvus_repo.delete_by_user_group(
                    user_id=user_id, group_id=group_id
                )
                stats["deleted_count"] = deleted_count
                logger.info("[ProfileIndexer] Deleted %d existing items", deleted_count)

                # Step 2: Fetch the real MongoDB document for converter.
                # The converter needs source_doc.id to generate unique Milvus
                # entity IDs (e.g. "6650a1b2..._e0"). Without a real id, all
                # entities get IDs like "_e0", "_i1", causing primary-key
                # collisions across different users' profiles.
                if doc_id:
                    source_doc = await MongoUserProfile.get(doc_id)
                else:
                    source_doc = await MongoUserProfile.find_one(
                        MongoUserProfile.user_id == user_id,
                        MongoUserProfile.group_id == group_id,
                    )

                if not source_doc or not source_doc.id:
                    logger.error(
                        "[ProfileIndexer] Cannot find MongoDB profile doc: user_id=%s, group_id=%s, doc_id=%s",
                        user_id, group_id, doc_id,
                    )
                    return stats

                # Use the MongoDB document's profile_data directly.
                # Do NOT overwrite with profile.to_dict() — the profile object
                # may have been constructed via ProfileMemory.from_dict()
                # which only reads explicit_info/implicit_traits and discards
                # V1-format fields (hard_skills, soft_skills, etc.).
                entities = UserProfileMilvusConverter.from_mongo(source_doc)

                if not entities:
                    logger.info("[ProfileIndexer] No items to index")
                    return stats

                # Step 3: Generate embeddings
                texts = [entity["embed_text"] for entity in entities]
                vectors = await self._generate_embeddings(texts)

                if len(vectors) != len(entities):
                    logger.error(
                        "[ProfileIndexer] Embedding count mismatch: expected %d, got %d",
                        len(entities),
                        len(vectors),
                    )
                    return stats

                # Step 4: Add vectors to entities
                valid_entities = []
                for entity, vector in zip(entities, vectors):
                    if vector is not None and len(vector) > 0:
                        entity["vector"] = vector
                        valid_entities.append(entity)

                # Step 5: Batch insert
                if valid_entities:
                    await self.milvus_repo.insert_batch(
                        entities=valid_entities, flush=True
                    )
                    stats["total_count"] = len(valid_entities)

                    # Count by type
                    for entity in valid_entities:
                        if entity["item_type"] == "explicit_info":
                            stats["explicit_count"] += 1
                        elif entity["item_type"] == "implicit_trait":
                            stats["implicit_count"] += 1

                logger.info(
                    "[ProfileIndexer] ✅ Indexing completed: deleted=%d, explicit=%d, implicit=%d, total=%d",
                    stats["deleted_count"],
                    stats["explicit_count"],
                    stats["implicit_count"],
                    stats["total_count"],
                )

                return stats

            except Exception as e:
                logger.error(
                    "[ProfileIndexer] ❌ Failed to index profile: user_id=%s, group_id=%s, error=%s",
                    user_id,
                    group_id,
                    e,
                    exc_info=True,
                )
                return stats

    async def delete_profile_index(self, user_id: str, group_id: str) -> int:
        """
        Delete all indexed items for a user profile

        Args:
            user_id: User ID
            group_id: Group ID

        Returns:
            Number of deleted items
        """
        try:
            deleted_count = await self.milvus_repo.delete_by_user_group(
                user_id=user_id, group_id=group_id
            )
            logger.info(
                "[ProfileIndexer] Deleted profile index: user_id=%s, group_id=%s, count=%d",
                user_id,
                group_id,
                deleted_count,
            )
            return deleted_count
        except Exception as e:
            logger.error("[ProfileIndexer] Failed to delete profile index: %s", e)
            return 0

    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts using VectorizeService

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        try:
            vectorize_service = get_vectorize_service()

            # Batch embed all texts
            vectors = await vectorize_service.get_embeddings(texts)

            logger.debug("[ProfileIndexer] Generated %d embeddings", len(vectors))

            return vectors

        except Exception as e:
            logger.error("[ProfileIndexer] Failed to generate embeddings: %s", e)
            return []


# Convenience function for external calls
async def index_user_profile(
    user_id: str, group_id: str, profile: ProfileMemory, doc_id: str = ""
) -> Dict[str, int]:
    """
    Index a user profile into Milvus

    Convenience function that gets the indexer service and calls index_profile.

    Args:
        user_id: User ID
        group_id: Group ID
        profile: ProfileMemory object
        doc_id: MongoDB document ID for generating unique Milvus entity IDs

    Returns:
        Indexing statistics
    """
    try:
        indexer = get_bean_by_type(ProfileIndexer)
        return await indexer.index_profile(user_id, group_id, profile, doc_id=doc_id)
    except Exception as e:
        logger.error("[ProfileIndexer] Failed to get indexer service: %s", e)
        return {
            "deleted_count": 0,
            "explicit_count": 0,
            "implicit_count": 0,
            "total_count": 0,
        }
