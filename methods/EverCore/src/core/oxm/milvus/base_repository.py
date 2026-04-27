"""
Milvus Base Repository Class

Provides common basic operations, all Milvus repositories should inherit from this class to obtain unified operation support.
"""

from abc import ABC
from typing import Optional, TypeVar, Generic, Type, List, Any
from pymilvus.orm.mutation import MutationResult
from core.oxm.milvus.milvus_collection_base import MilvusCollectionBase
from core.oxm.milvus.async_collection import AsyncCollection
from core.observation.logger import get_logger
from core.di.utils import get_bean

logger = get_logger(__name__)

# Generic type variable
T = TypeVar('T', bound=MilvusCollectionBase)


class BaseMilvusRepository(ABC, Generic[T]):
    """
    Milvus Base Repository Class

    Provides common basic operations, all Milvus repositories should inherit from this class.

    Features:
    - Asynchronous Milvus collection management
    - Basic CRUD operation templates
    - Unified error handling and logging
    - Collection management
    """

    def __init__(self, model: Type[T]):
        """
        Initialize base repository

        Args:
            model: Milvus collection model class
        """
        self.model = model
        self.model_name = model.__name__
        self.collection: Optional[AsyncCollection] = model.async_collection()
        self.schema = model._SCHEMA
        self.all_output_fields = [field.name for field in self.schema.fields]

    # ==================== Basic CRUD Operations ====================

    async def insert(self, entity: T, flush: bool = False) -> str:
        """
        Insert new entity

        Args:
            entity: Entity instance
            flush: Whether to flush immediately

        Returns:
            str: Inserted entity ID
        """
        try:
            entity_id = await self.collection.insert(entity)
            if flush:
                await self.collection.flush()
            logger.debug(
                "✅ Insert entity successful [%s]: %s", self.model_name, entity_id
            )
            return entity_id
        except Exception as e:
            logger.error("❌ Insert entity failed [%s]: %s", self.model_name, e)
            raise

    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """
        Get entity by ID

        Args:
            entity_id: Entity ID

        Returns:
            Entity instance or None
        """
        try:
            # Get all fields of the collection
            # Use query to search
            results = await self.collection.query(
                expr=f'id == "{entity_id}"',
                output_fields=self.all_output_fields,
                limit=1,
            )
            return results[0] if results else None
        except Exception as e:
            logger.error("❌ Failed to get entity by ID [%s]: %s", self.model_name, e)
            return None

    async def upsert(self, entity: T, flush: bool = False) -> str:
        """
        Update or insert entity

        Args:
            entity: Entity instance
            flush: Whether to flush immediately

        Returns:
            str: Entity ID
        """
        try:
            entity_id = await self.collection.upsert(entity)
            if flush:
                await self.collection.flush()
            logger.debug(
                "✅ Upsert entity successful [%s]: %s", self.model_name, entity_id
            )
            return entity_id
        except Exception as e:
            logger.error("❌ Upsert entity failed [%s]: %s", self.model_name, e)
            raise

    async def delete_by_id(self, entity_id: str, flush: bool = False) -> bool:
        """
        Delete entity by ID

        Args:
            entity_id: Entity ID
            flush: Whether to flush immediately

        Returns:
            bool: Return True if deletion is successful
        """
        try:
            result = await self.collection.delete(expr=f'id == "{entity_id}"')
            success = result.delete_count > 0

            if flush and success:
                await self.collection.flush()
            if success:
                logger.debug(
                    "✅ Delete entity successful [%s]: %s", self.model_name, entity_id
                )
            return success
        except Exception as e:
            logger.error("❌ Delete entity failed [%s]: %s", self.model_name, e)
            return False

    # ==================== Batch Operations ====================

    async def insert_batch(
        self, entities: List[T], flush: bool = False
    ) -> MutationResult:
        """
        Insert entities in batch

        Args:
            entities: List of entities
            flush: Whether to flush immediately

        Returns:
            MutationResult from Milvus
        """
        try:
            result = await self.collection.insert(entities)
            if flush:
                await self.collection.flush()
            logger.debug(
                "✅ Batch insert entities successful [%s]: %d records",
                self.model_name,
                len(entities),
            )
            return result
        except Exception as e:
            logger.error("❌ Batch insert entities failed [%s]: %s", self.model_name, e)
            raise

    # ==================== Collection Operations ====================

    async def flush(self) -> bool:
        """
        Flush collection

        Returns:
            bool: Return True if flush is successful
        """
        try:
            await self.collection.flush()
            logger.debug("✅ Flush collection successful [%s]", self.model_name)
            return True
        except Exception as e:
            logger.error("❌ Flush collection failed [%s]: %s", self.model_name, e)
            return False

    async def load(self) -> bool:
        """
        Load collection into memory

        Returns:
            bool: Return True if load is successful
        """
        try:
            await self.collection.load()
            logger.debug("✅ Load collection successful [%s]", self.model_name)
            return True
        except Exception as e:
            logger.error("❌ Load collection failed [%s]: %s", self.model_name, e)
            return False

    # ==================== Helper Methods ====================

    def get_model_name(self) -> str:
        """
        Get model name

        Returns:
            str: Model class name
        """
        return self.model_name
