"""
Elasticsearch Base Repository Class

Base repository class based on elasticsearch-dsl, providing common basic CRUD operations.
All Elasticsearch repositories should inherit from this base class to obtain unified operation support.
"""

from abc import ABC
from typing import Optional, TypeVar, Generic, Type, List, Dict, Any
from elasticsearch import AsyncElasticsearch
from core.oxm.es.doc_base import DocBase
from core.observation.logger import get_logger

logger = get_logger(__name__)

# Generic type variable
T = TypeVar('T', bound=DocBase)


class BaseRepository(ABC, Generic[T]):
    """
    Elasticsearch Base Repository Class

    Provides common basic operations; all Elasticsearch repositories should inherit from this class.

    Features:
    - Async Elasticsearch client management
    - Basic CRUD operation templates
    - Unified error handling and logging
    - Index management
    """

    def __init__(self, model: Type[T]):
        """
        Initialize base repository

        Args:
            model: Elasticsearch document model class
        """
        self.model = model
        self.model_name = model.__name__

    # ==================== Client Management ====================

    async def get_client(self) -> AsyncElasticsearch:
        """
        Get Elasticsearch async client

        Returns:
            AsyncElasticsearch: Async client instance
        """
        return self.model.get_connection()

    def get_index_name(self) -> str:
        """
        Get index name

        Delegates to the model class's get_index_name method to ensure consistent index name retrieval logic.

        Returns:
            str: Index alias
        """
        return self.model.get_index_name()

    # ==================== Basic CRUD Template Methods ====================

    async def create(self, document: T, refresh: bool = False) -> T:
        """
        Create a new document

        Args:
            document: Document instance
            refresh: Whether to refresh the index immediately

        Returns:
            Created document instance
        """
        try:
            client = await self.get_client()
            await document.save(using=client, refresh=refresh)
            return document
        except Exception as e:
            logger.error("❌ Failed to create document [%s]: %s", self.model_name, e)
            raise

    async def get_by_id(self, doc_id: str) -> Optional[T]:
        """
        Get document by document ID

        Args:
            doc_id: Document ID

        Returns:
            Document instance or None
        """
        try:
            client = await self.get_client()
            return await self.model.get(id=doc_id, using=client)
        except Exception as e:
            logger.error("❌ Failed to get document by ID [%s]: %s", self.model_name, e)
            return None

    async def update(self, document: T, refresh: bool = False) -> T:
        """
        Update document

        Args:
            document: Document instance to update
            refresh: Whether to refresh the index immediately

        Returns:
            Updated document instance
        """
        try:
            client = await self.get_client()
            await document.save(using=client, refresh=refresh)
            doc_id = getattr(getattr(document, 'meta', None), 'id', 'unknown')
            logger.debug(
                "✅ Document updated successfully [%s]: %s", self.model_name, doc_id
            )
            return document
        except Exception as e:
            logger.error("❌ Failed to update document [%s]: %s", self.model_name, e)
            raise

    async def delete_by_id(self, doc_id: str, refresh: bool = False) -> bool:
        """
        Delete document by document ID

        Args:
            doc_id: Document ID
            refresh: Whether to refresh the index immediately

        Returns:
            Returns True if deletion succeeds, otherwise False
        """
        try:
            document = await self.get_by_id(doc_id)
            if document:
                client = await self.get_client()
                await document.delete(using=client, refresh=refresh)
                logger.debug(
                    "✅ Document deleted successfully [%s]: %s", self.model_name, doc_id
                )
                return True
            return False
        except Exception as e:
            logger.error("❌ Failed to delete document [%s]: %s", self.model_name, e)
            return False

    async def delete(self, document: T, refresh: bool = False) -> bool:
        """
        Delete document instance

        Args:
            document: Document instance to delete
            refresh: Whether to refresh the index immediately

        Returns:
            Returns True if deletion succeeds, otherwise False
        """
        try:
            client = await self.get_client()
            await document.delete(using=client, refresh=refresh)
            logger.debug(
                "✅ Document deleted successfully [%s]: %s",
                self.model_name,
                getattr(document, 'meta', {}).get('id', 'unknown'),
            )
            return True
        except Exception as e:
            logger.error("❌ Failed to delete document [%s]: %s", self.model_name, e)
            return False

    # ==================== Batch Operations ====================

    async def create_batch(self, documents: List[T], refresh: bool = False) -> List[T]:
        """
        Batch create documents

        Args:
            documents: List of documents
            refresh: Whether to refresh the index immediately

        Returns:
            List of successfully created documents
        """
        try:
            client = await self.get_client()
            index_name = self.get_index_name()

            # Build bulk operations
            actions = []
            for doc in documents:
                action = {"_index": index_name, "_source": doc.to_dict()}
                actions.append(action)

            # Execute bulk operation
            from elasticsearch.helpers import async_bulk

            await async_bulk(client, actions, refresh=refresh)

            logger.debug(
                "✅ Batch document creation succeeded [%s]: %d records",
                self.model_name,
                len(documents),
            )
            return documents
        except Exception as e:
            logger.error(
                "❌ Failed to batch create documents [%s]: %s", self.model_name, e
            )
            raise

    # ==================== Search Methods ====================

    async def search(
        self, query: Dict[str, Any], size: int = 10, from_: int = 0
    ) -> Dict[str, Any]:
        """
        Execute search query

        Args:
            query: Elasticsearch query DSL
            size: Number of results to return
            from_: Pagination starting position

        Returns:
            Search results
        """
        try:
            client = await self.get_client()
            index_name = self.get_index_name()

            response = await client.search(
                index=index_name, body={"query": query, "size": size, "from": from_}
            )

            logger.debug(
                "✅ Search executed successfully [%s]: Found %d results",
                self.model_name,
                response.get('hits', {}).get('total', {}).get('value', 0),
            )
            return response
        except Exception as e:
            logger.error("❌ Failed to execute search [%s]: %s", self.model_name, e)
            raise

    async def match_all(self, size: int = 10, from_: int = 0) -> List[T]:
        """
        Get all documents

        Args:
            size: Number of results to return
            from_: Pagination starting position

        Returns:
            List of documents
        """
        try:
            response = await self.search(
                query={"match_all": {}}, size=size, from_=from_
            )

            documents = []
            for hit in response.get('hits', {}).get('hits', []):
                doc = self.model.from_dict(hit['_source'])
                doc.meta.id = hit['_id']
                documents.append(doc)

            return documents
        except Exception as e:
            logger.error("❌ Failed to get all documents [%s]: %s", self.model_name, e)
            return []

    # ==================== Statistics Methods ====================

    async def exists_by_id(self, doc_id: str) -> bool:
        """
        Check if document exists

        Args:
            doc_id: Document ID

        Returns:
            Returns True if exists, otherwise False
        """
        try:
            client = await self.get_client()
            index_name = self.get_index_name()

            response = await client.exists(index=index_name, id=doc_id)
            return response
        except Exception:
            return False

    # ==================== Index Management ====================

    async def refresh_index(self) -> bool:
        """
        Manually refresh index

        Uses connection.indices.refresh(index=index_name) to manually refresh the index,
        ensuring newly written data is immediately searchable.

        Returns:
            Returns True if refresh succeeds, otherwise False
        """
        try:
            client = await self.get_client()
            index_name = self.get_index_name()

            await client.indices.refresh(index=index_name)
            logger.debug(
                "✅ Manual index refresh succeeded [%s]: %s",
                self.model_name,
                index_name,
            )
            return True

        except (ConnectionError, TimeoutError) as e:
            logger.error("❌ Manual index refresh failed [%s]: %s", self.model_name, e)
            return False
        except Exception as e:
            logger.error(
                "❌ Manual index refresh failed (unknown error) [%s]: %s",
                self.model_name,
                e,
            )
            return False

    async def create_index(self) -> bool:
        """
        Create index

        Returns:
            Returns True if creation succeeds, otherwise False
        """
        try:
            client = await self.get_client()

            # Use document class's init method to create index
            index_name = self.model.dest()

            await self.model.init(index=index_name, using=client)

            # Create alias
            alias = self.get_index_name()
            await client.indices.update_aliases(
                body={
                    "actions": [
                        {
                            "add": {
                                "index": index_name,
                                "alias": alias,
                                "is_write_index": True,
                            }
                        }
                    ]
                }
            )

            logger.debug(
                "✅ Index creation succeeded [%s]: %s -> %s",
                self.model_name,
                index_name,
                alias,
            )
            return True
        except Exception as e:
            logger.error("❌ Index creation failed [%s]: %s", self.model_name, e)
            return False

    async def delete_index(self) -> bool:
        """
        Delete index

        Returns:
            Returns True if deletion succeeds, otherwise False
        """
        try:
            client = await self.get_client()
            index_name = self.get_index_name()

            await client.indices.delete(index=index_name)
            logger.debug(
                "✅ Index deletion succeeded [%s]: %s", self.model_name, index_name
            )
            return True
        except Exception as e:
            logger.error("❌ Index deletion failed [%s]: %s", self.model_name, e)
            return False

    async def index_exists(self) -> bool:
        """
        Check if index exists

        Returns:
            Returns True if exists, otherwise False
        """
        try:
            client = await self.get_client()
            index_name = self.get_index_name()

            return await client.indices.exists(index=index_name)
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
        Get index name (compatible with MongoDB repository interface)

        Returns:
            Elasticsearch index name
        """
        return self.get_index_name()


# Export
__all__ = ["BaseRepository"]
