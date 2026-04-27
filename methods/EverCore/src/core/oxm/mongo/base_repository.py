"""
MongoDB Base Repository Class

Base repository class based on Beanie ODM, providing generic transaction management and basic CRUD operations.
All MongoDB repositories should inherit from this base class to obtain unified transaction support.
"""

from abc import ABC
from contextlib import asynccontextmanager
from typing import Optional, TypeVar, Generic, Type, Union, List
from beanie import PydanticObjectId
from pymongo.asynchronous.client_session import AsyncClientSession
from core.observation.logger import get_logger
from core.oxm.mongo.document_base import DocumentBase

logger = get_logger(__name__)

# Generic type variable
T = TypeVar('T', bound=DocumentBase)


class BaseRepository(ABC, Generic[T]):
    """
    MongoDB Base Repository Class

    Provides generic transaction management and basic operations; all MongoDB repositories should inherit from this class.

    Features:
    - Transaction context manager
    - Session management
    - Basic CRUD operation templates
    - Unified error handling and logging
    """

    def __init__(self, model: Type[T]):
        """
        Initialize base repository

        Args:
            model: Beanie document model class
        """
        self.model = model
        self.model_name = model.__name__

    # ==================== Transaction Management ====================

    @asynccontextmanager
    async def transaction(self):
        """
        Transaction context manager

        Usage:
            async with repository.transaction() as session:
                await repository.create(document, session=session)
                await repository.update(another_document, session=session)
                # Automatically commits or rolls back

        Yields:
            AsyncClientSession: MongoDB session object
        """
        client = self.model.get_pymongo_client()
        async with await client.start_session() as session:
            async with session.start_transaction():
                try:
                    logger.info("🔄 Starting MongoDB transaction [%s]", self.model_name)
                    yield session
                    logger.info(
                        "✅ MongoDB transaction committed successfully [%s]",
                        self.model_name,
                    )
                except Exception as e:
                    logger.error(
                        "❌ MongoDB transaction rolled back [%s]: %s",
                        self.model_name,
                        e,
                    )
                    raise

    async def start_session(self) -> AsyncClientSession:
        """
        Start a new session (without transaction)

        Returns:
            AsyncClientSession: MongoDB session object

        Note:
            The session must be manually closed after use:
            session = await repository.start_session()
            try:
                # Use session
                pass
            finally:
                await session.end_session()
        """
        client = self.model.get_pymongo_client()
        session = await client.start_session()
        logger.info("🔄 Created MongoDB session [%s]", self.model_name)
        return session

    # ==================== Basic CRUD Template Methods ====================

    async def create(
        self, document: T, session: Optional[AsyncClientSession] = None
    ) -> T:
        """
        Create a new document

        Args:
            document: Document instance
            session: Optional MongoDB session, used for transaction support

        Returns:
            Created document instance
        """
        try:
            await document.insert(session=session)
            logger.info(
                "✅ Document created successfully [%s]: %s",
                self.model_name,
                getattr(document, 'id', 'unknown'),
            )
            return document
        except Exception as e:
            logger.error("❌ Failed to create document [%s]: %s", self.model_name, e)
            raise

    async def get_by_id(self, object_id: Union[str, PydanticObjectId]) -> Optional[T]:
        """
        Get document by ObjectId

        Args:
            object_id: MongoDB ObjectId

        Returns:
            Document instance or None
        """
        try:
            if isinstance(object_id, str):
                object_id = PydanticObjectId(object_id)
            return await self.model.get(object_id)
        except Exception as e:
            logger.error("❌ Failed to get document by ID [%s]: %s", self.model_name, e)
            return None

    async def update(
        self, document: T, session: Optional[AsyncClientSession] = None
    ) -> T:
        """
        Update document

        Args:
            document: Document instance to update
            session: Optional MongoDB session, used for transaction support

        Returns:
            Updated document instance
        """
        try:
            await document.save(session=session)
            logger.info(
                "✅ Document updated successfully [%s]: %s",
                self.model_name,
                getattr(document, 'id', 'unknown'),
            )
            return document
        except Exception as e:
            logger.error("❌ Failed to update document [%s]: %s", self.model_name, e)
            raise

    async def delete_by_id(
        self,
        object_id: Union[str, PydanticObjectId],
        session: Optional[AsyncClientSession] = None,
    ) -> bool:
        """
        Delete document by ObjectId

        Args:
            object_id: MongoDB ObjectId
            session: Optional MongoDB session, used for transaction support

        Returns:
            Returns True if deletion succeeds, otherwise False
        """
        try:
            document = await self.get_by_id(object_id)
            if document:
                await document.delete(session=session)
                logger.info(
                    "✅ Document deleted successfully [%s]: %s",
                    self.model_name,
                    object_id,
                )
                return True
            return False
        except Exception as e:
            logger.error("❌ Failed to delete document [%s]: %s", self.model_name, e)
            return False

    async def delete(
        self, document: T, session: Optional[AsyncClientSession] = None
    ) -> bool:
        """
        Delete document instance

        Args:
            document: Document instance to delete
            session: Optional MongoDB session, used for transaction support

        Returns:
            Returns True if deletion succeeds, otherwise False
        """
        try:
            await document.delete(session=session)
            logger.info(
                "✅ Document deleted successfully [%s]: %s",
                self.model_name,
                getattr(document, 'id', 'unknown'),
            )
            return True
        except Exception as e:
            logger.error("❌ Failed to delete document [%s]: %s", self.model_name, e)
            return False

    # ==================== Batch Operations ====================

    async def create_batch(
        self, documents: List[T], session: Optional[AsyncClientSession] = None
    ) -> List[T]:
        """
        Batch create documents

        Args:
            documents: List of documents
            session: Optional MongoDB session, used for transaction support

        Returns:
            List of successfully created documents
        """
        try:
            # Beanie's insert_many does not automatically update the id attribute of input objects
            # We need to manually retrieve inserted_ids from the returned InsertManyResult and set them
            result = await self.model.insert_many(documents, session=session)
            # Set the _id generated by MongoDB back to the id attribute of each document object
            for doc, inserted_id in zip(documents, result.inserted_ids):
                doc.id = inserted_id
            logger.info(
                "✅ Batch document creation successful [%s]: %d records",
                self.model_name,
                len(documents),
            )
            return documents
        except Exception as e:
            logger.error(
                "❌ Failed to batch create documents [%s]: %s", self.model_name, e
            )
            raise

    # ==================== Query Methods ====================

    async def find_by_query(
        self,
        query: dict,
        skip: int = 0,
        limit: int = 20,
        sort: list = None,
        projection_model=None,
    ):
        """
        Generic query with pagination and optional projection.

        Args:
            query: MongoDB filter dict
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            sort: List of (field, direction) tuples
            projection_model: Optional Pydantic model for field projection (e.g. exclude vector)

        Returns:
            Tuple of (documents list, total count)
        """
        import asyncio

        try:
            find_kwargs = {"skip": skip, "limit": limit}
            if sort:
                find_kwargs["sort"] = sort
            if projection_model:
                find_kwargs["projection_model"] = projection_model

            find_query = self.model.find_many(query, **find_kwargs)
            count_query = self.model.find_many(query).count()

            docs, total_count = await asyncio.gather(find_query.to_list(), count_query)

            return docs, total_count
        except Exception as e:
            logger.error("❌ Failed to query documents [%s]: %s", self.model_name, e)
            raise

    # ==================== Counting Methods ====================

    async def count_all(self, filter_query: Optional[dict] = None) -> int:
        """
        Count documents with optimized strategy

        When no filter is provided, uses estimated_document_count() which is
        extremely fast (milliseconds) as it reads from collection metadata.
        When a filter is provided, uses count_documents() for accurate results.

        Args:
            filter_query: Optional filter conditions. If None, uses fast estimation.

        Returns:
            Total number of documents (estimated if no filter, exact if with filter)
        """
        try:
            if filter_query is None or filter_query == {}:
                # No filter: use estimated_document_count() for speed (metadata-based)
                collection = self.model.get_pymongo_collection()
                count = await collection.estimated_document_count()
                logger.info(
                    "✅ Fast estimated count [%s]: %d records", self.model_name, count
                )
            else:
                # With filter: use count_documents() for accuracy
                count = await self.model.find(filter_query).count()
                logger.info(
                    "✅ Exact count with filter [%s]: %d records",
                    self.model_name,
                    count,
                )
            return count
        except Exception as e:
            logger.error("❌ Failed to count documents [%s]: %s", self.model_name, e)
            return 0

    async def exists_by_id(self, object_id: Union[str, PydanticObjectId]) -> bool:
        """
        Check if document exists

        Args:
            object_id: MongoDB ObjectId

        Returns:
            Returns True if exists, otherwise False
        """
        try:
            if isinstance(object_id, str):
                object_id = PydanticObjectId(object_id)
            document = await self.model.get(object_id)
            return document is not None
        except Exception:
            return False

    # ==================== Helper Methods ====================

    def get_model_name(self) -> str:
        """
        Get model name

        Returns:
            Model class name
        """
        return self.model_name

    def get_collection_name(self) -> str:
        """
        Get collection name

        Returns:
            MongoDB collection name
        """
        return self.model.get_collection_name()


# Export
__all__ = ["BaseRepository"]
