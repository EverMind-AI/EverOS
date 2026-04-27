"""
MongoDB Document Base With Soft Delete

Base document class with soft delete functionality, providing complete soft delete support.
"""

from datetime import datetime
from beanie.odm.enums import SortDirection
from beanie.odm.bulk import BulkWriter
from beanie.odm.actions import ActionDirections
from beanie import DeleteRules
from pydantic import Field, BaseModel
from typing import List, Optional, Any, Mapping, Union, Tuple, Dict, Type
from pymongo.asynchronous.client_session import AsyncClientSession
from pymongo.results import UpdateResult, DeleteResult

from common_utils.datetime_utils import get_now_with_timezone
from core.oxm.mongo.document_base import DocumentBase


class DocumentBaseWithSoftDelete(DocumentBase):
    """
    Base document class with soft delete functionality

    Inherits from DocumentBase, integrating complete soft delete capabilities:
    - Provides full soft delete capability (self-implemented, not relying on beanie's DocumentWithSoftDelete)
    - Supports timezone-aware datetime handling (from DocumentBase)
    - Supports database binding configuration (from DocumentBase)
    - Supports audit field handling during bulk insert (from DocumentBase)
    - **Extended deletion audit fields: deleted_by (deleter) and deleted_id (uniqueness trick)**
    - **Complete bulk soft delete support**

    Soft delete field descriptions:
        - deleted_at: deletion timestamp
        - deleted_by: identifier of the deletion operator
        - deleted_id: deletion identifier ID, used for unique index trick
          * When not deleted: deleted_id = 0 (all undeleted documents share this value)
          * When deleted: deleted_id = hash value of the document's _id
          * Advantage: can create unique index on (business field + deleted_id), achieving:
            - Only one record allowed for the same business key when undeleted
            - Multiple historical records allowed for the same business key after deletion
            - New records with the same business key can be inserted after soft deletion

    Core methods:
        Instance methods:
        - delete(): soft delete current document
        - restore(): restore a deleted document
        - hard_delete(): hard delete current document (physical deletion)
        - is_deleted(): check if document is deleted

        Class methods (queries):
        - find_one(): find one document (automatically filters out deleted ones)
        - find_many(): find multiple documents (automatically filters out deleted ones)
        - hard_find_one(): hard find one document (including deleted ones)
        - hard_find_many(): hard find multiple documents (including deleted ones)

        Class methods (bulk operations):
        - delete_many(): bulk soft delete
        - restore_many(): bulk restore
        - hard_delete_many(): bulk hard delete

        Utility methods (for native pymongo API):
        - apply_soft_delete_filter(): apply soft delete filter condition to query
        - get_soft_delete_filter(): get pure soft delete filter condition

    Important notes:
        ⚠️ Do not use Model.find().delete_many(), it performs hard deletion!
        Use Model.delete_many(filter) to perform bulk soft deletion.

    Usage example:
        from pydantic import Field

        class MyDocument(DocumentBaseWithSoftDelete, AuditBase):
            email: str
            name: str

            class Settings:
                bind_database = "my_database"
                collection = "my_collection"
                # Unique index: only one record allowed for the same email when undeleted
                indexes = [
                    [("email", 1), ("deleted_id", 1)],  # composite unique index
                ]

        # Single soft delete
        doc = await MyDocument.find_one({"email": "test@example.com"})
        await doc.delete(deleted_by="admin")  # soft delete

        # Bulk soft delete
        result = await MyDocument.delete_many(
            {"status": "inactive"},
            deleted_by="system"
        )

        # Restore single document
        doc = await MyDocument.hard_find_one({"email": "test@example.com"})
        if doc and doc.is_deleted():
            await doc.restore()

        # Bulk hard delete (use with caution!)
        result = await MyDocument.hard_delete_many({"is_test": True})

        # Apply soft delete filter when using native pymongo API
        filter_dict = MyDocument.apply_soft_delete_filter({"status": "active"})
        result = await MyDocument.get_pymongo_collection().find(filter_dict).to_list(100)
    """

    # Soft delete related fields
    deleted_at: Optional[datetime] = Field(
        default=None, description="Soft deletion timestamp"
    )
    deleted_by: Optional[str] = Field(default=None, description="Deletion operator")
    deleted_id: int = Field(
        default=0, description="Deletion identifier ID, used for unique index trick"
    )

    def is_deleted(self) -> bool:
        """
        Check if the document has been soft deleted

        Returns:
            bool: Returns True if document is deleted, otherwise False
        """
        return self.deleted_at is not None

    @classmethod
    def apply_soft_delete_filter(
        cls,
        filter_query: Optional[Mapping[str, Any]] = None,
        include_deleted: bool = False,
    ) -> Dict[str, Any]:
        """
        Apply soft delete filter condition to query filter

        This is a utility method used to manually apply soft delete filtering when directly using get_pymongo_collection().
        If deleted_at condition is already present in filter_query, it remains unchanged.
        If not present and include_deleted=False, adds deleted_at=None condition.

        Args:
            filter_query: Original query filter condition (optional)
            include_deleted: Whether to include deleted documents, default False

        Returns:
            Dict[str, Any]: Query condition with soft delete filtering applied

        Example:
            # Scenario 1: Automatically filter out deleted when using native pymongo API
            filter_dict = User.apply_soft_delete_filter({"status": "active"})
            result = await User.get_pymongo_collection().find(filter_dict).to_list(100)

            # Scenario 2: Need to include deleted documents
            filter_dict = User.apply_soft_delete_filter(
                {"status": "active"},
                include_deleted=True
            )
            result = await User.get_pymongo_collection().find(filter_dict).to_list(100)

            # Scenario 3: Empty filter condition, only query undeleted
            filter_dict = User.apply_soft_delete_filter()
            result = await User.get_pymongo_collection().find(filter_dict).to_list(100)

            # Scenario 4: Using aggregation pipeline
            match_stage = {"$match": User.apply_soft_delete_filter({"age": {"$gt": 18}})}
            pipeline = [match_stage, {"$group": {"_id": "$city", "count": {"$sum": 1}}}]
            result = await User.get_pymongo_collection().aggregate(pipeline).to_list(100)
        """
        # If no filter condition is provided, create empty dictionary
        if filter_query is None:
            result_filter = {}
        else:
            # Copy original filter condition to avoid modifying original object
            result_filter = dict(filter_query)

        # If not including deleted documents, and deleted_at field is not in filter
        if not include_deleted and "deleted_at" not in result_filter:
            result_filter["deleted_at"] = None

        return result_filter

    @classmethod
    def get_soft_delete_filter(cls, include_deleted: bool = False) -> Dict[str, Any]:
        """
        Get default soft delete filter condition

        This is a simplified utility method that returns pure soft delete filter condition.

        Args:
            include_deleted: Whether to include deleted documents, default False

        Returns:
            Dict[str, Any]: Soft delete filter condition, returns empty dictionary if include_deleted=True

        Example:
            # Only get filter condition for undeleted
            soft_delete_filter = User.get_soft_delete_filter()
            # Returns: {"deleted_at": None}

            # Get filter condition including deleted (actually returns empty dictionary)
            all_filter = User.get_soft_delete_filter(include_deleted=True)
            # Returns: {}

            # Merge with other conditions
            my_filter = {"status": "active", **User.get_soft_delete_filter()}
            result = await User.get_pymongo_collection().find(my_filter).to_list(100)
        """
        if include_deleted:
            return {}
        return {"deleted_at": None}

    @classmethod
    async def count(cls, include_deleted: bool = False, **kwargs) -> int:
        """
        Count documents in the collection, with soft delete awareness

        By default, excludes soft-deleted documents (deleted_at is not None).
        Set include_deleted=True to count all documents including deleted ones.

        Performance notes:
            - When include_deleted=True: uses estimated_document_count() (fast, ~1ms)
            - When include_deleted=False: uses count_documents() with filter (requires deleted_at index)

        Args:
            include_deleted: Whether to include soft-deleted documents, default False

        Returns:
            int: Document count

        Example:
            # Count only non-deleted documents (default)
            active_count = await User.count()

            # Count all documents including deleted
            total_count = await User.count(include_deleted=True)
        """
        collection = cls.get_pymongo_collection()

        if include_deleted:
            # Fast estimation for all documents
            return await collection.estimated_document_count()
        else:
            # Precise count excluding soft-deleted documents
            # Requires index on deleted_at field for performance
            return await collection.count_documents({"deleted_at": None})

    async def delete(
        self,
        session: Optional[AsyncClientSession] = None,
        bulk_writer: Optional[Any] = None,
        link_rule: Optional[Any] = None,
        skip_actions: Optional[List[Any]] = None,
        deleted_by: Optional[str] = None,
        **pymongo_kwargs: Any,
    ) -> Optional[Any]:
        """
        Soft delete current document (override parent method to support deleted_by)

        ⚠️ If document has already been soft deleted, this method returns directly without modifying audit fields.
        ⚠️ Directly uses PyMongo's update_one method, completely bypassing Beanie's save mechanism.

        Args:
            session: MongoDB session (beanie parameter)
            bulk_writer: Bulk writer (beanie parameter)
            link_rule: Link rule (beanie parameter)
            skip_actions: Skipped actions (beanie parameter)
            deleted_by: Deletion operator identifier (optional, extended parameter of this class)
            **pymongo_kwargs: Other pymongo parameters

        Returns:
            None (soft delete does not return DeleteResult)

        Example:
            doc = await MyDocument.find_one({"name": "test"})
            await doc.delete(deleted_by="admin")
        """
        # Check if already soft deleted, avoid repeated deletion that would damage audit records
        if self.is_deleted():
            return None

        now = get_now_with_timezone()

        # Set deleted_id to string hash value of document ID
        # If id is ObjectId, convert to string then take hash
        deleted_id_value = 0
        if self.id:
            # Convert ObjectId to integer (absolute value of hash)
            deleted_id_value = abs(hash(str(self.id)))

        # Directly use PyMongo's update_one to update database, completely bypassing Beanie
        await self.get_pymongo_collection().update_one(
            {"_id": self.id},
            {
                "$set": {
                    "deleted_at": now,
                    "deleted_by": deleted_by,
                    "deleted_id": deleted_id_value,
                }
            },
            session=session,
        )

        # Update current object state to maintain consistency
        self.deleted_at = now
        self.deleted_by = deleted_by
        self.deleted_id = deleted_id_value

        return None

    async def restore(self, session: Optional[AsyncClientSession] = None) -> None:
        """
        Restore a single soft-deleted document

        Clears the soft delete mark of the current document, restoring it to normal state.

        ⚠️ If document is not soft deleted, this method returns directly without any operation.
        ⚠️ Directly uses PyMongo's update_one method, completely bypassing Beanie's save mechanism.

        Example:
            # Find deleted document (using hard_find_one can query including deleted ones)
            doc = await MyDocument.hard_find_one(
                {"email": "user@example.com", "deleted_at": {"$ne": None}}
            )

            # Restore document
            if doc and doc.is_deleted():
                await doc.restore()
        """
        # If document is not deleted, return directly
        if not self.is_deleted():
            return

        # Directly use PyMongo's update_one to update database, completely bypassing Beanie
        await self.get_pymongo_collection().update_one(
            {"_id": self.id},
            {"$set": {"deleted_at": None, "deleted_by": None, "deleted_id": 0}},
            session=session,
        )

        # Update current object state to maintain consistency
        self.deleted_at = None
        self.deleted_by = None
        self.deleted_id = 0

    async def hard_delete(
        self,
        session: Optional[AsyncClientSession] = None,
        bulk_writer: Optional[BulkWriter] = None,
        link_rule: DeleteRules = DeleteRules.DO_NOTHING,
        skip_actions: Optional[List[Union[ActionDirections, str]]] = None,
        **pymongo_kwargs: Any,
    ) -> Optional[DeleteResult]:
        """
        Hard delete current document (physical deletion)

        ⚠️ Warning: This operation is irreversible! Use with caution.

        Calls parent class's delete method to perform actual physical deletion.

        Args:
            session: MongoDB session
            bulk_writer: Bulk writer
            link_rule: Link rule
            skip_actions: Skipped actions
            **pymongo_kwargs: Other pymongo parameters

        Returns:
            Optional[DeleteResult]: Deletion result

        Example:
            doc = await MyDocument.find_one({"name": "test"})
            await doc.hard_delete()  # Permanently delete
        """
        return await super().delete(
            session=session,
            bulk_writer=bulk_writer,
            link_rule=link_rule,
            skip_actions=skip_actions,
            **pymongo_kwargs,
        )

    @classmethod
    async def delete_many(
        cls,
        filter_query: Mapping[str, Any],
        deleted_by: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
        **pymongo_kwargs: Any,
    ) -> UpdateResult:
        """
        Bulk soft delete documents (default bulk deletion method)

        Marks matching documents as deleted instead of physically deleting them.
        This supplements the missing functionality in beanie DocumentWithSoftDelete.

        ⚠️ Note:
        - deleted_id is set to microsecond-level timestamp during bulk deletion
        - Automatically filters out already soft-deleted documents to avoid repeated deletion that would damage audit records

        Args:
            filter_query: MongoDB query filter condition
            deleted_by: Deletion operator identifier (optional)
            session: Optional MongoDB session, for transaction support
            **pymongo_kwargs: Other parameters passed to PyMongo

        Returns:
            UpdateResult: Update result containing number of matched and modified documents

        Example:
            # Bulk soft delete
            result = await User.delete_many(
                {"is_active": False},
                deleted_by="admin"
            )
            print(f"Soft deleted {result.modified_count} documents")

            # Transactional soft delete using session
            async with await client.start_session() as session:
                await User.delete_many(
                    {"status": "expired"},
                    deleted_by="system",
                    session=session
                )
        """
        # Set deletion timestamp
        now = get_now_with_timezone()

        # Note: Handling strategy for deleted_id during bulk deletion
        # Since bulk operations cannot efficiently set hash value of each document's _id, we use timestamp here
        # For scenarios requiring strict uniqueness constraints, it is recommended to:
        # 1. First query all matching documents
        # 2. Call doc.delete() method individually
        # Or implement more complex bulk deletion logic at application layer

        update_doc = {
            "deleted_at": now,
            "deleted_by": deleted_by,
            # Use microsecond-level timestamp as deleted_id, providing certain uniqueness
            # For scenarios requiring strict uniqueness, recommend using individual deletion or custom implementation
            "deleted_id": int(now.timestamp() * 1000000),  # microsecond-level timestamp
        }

        # Apply soft delete filter: only delete documents not already soft deleted, to avoid repeated deletion that would damage audit records
        final_filter = cls.apply_soft_delete_filter(filter_query, include_deleted=False)

        return await cls.get_pymongo_collection().update_many(
            final_filter, {"$set": update_doc}, session=session, **pymongo_kwargs
        )

    @classmethod
    async def restore_many(
        cls,
        filter_query: Mapping[str, Any],
        session: Optional[AsyncClientSession] = None,
        **pymongo_kwargs: Any,
    ) -> UpdateResult:
        """
        Bulk restore soft-deleted documents

        Restores matching deleted documents (clears all soft delete marker fields).

        ⚠️ Automatically only restores documents that have been soft deleted; undeleted documents will not be modified.

        Args:
            filter_query: MongoDB query filter condition
            session: Optional MongoDB session, for transaction support
            **pymongo_kwargs: Other parameters passed to PyMongo

        Returns:
            UpdateResult: Update result containing number of matched and modified documents

        Example:
            # Restore specific user
            result = await User.restore_many({"email": "user@example.com"})

            # Restore all documents deleted yesterday
            from datetime import timedelta
            from common_utils.datetime_utils import get_now_with_timezone
            yesterday = get_now_with_timezone() - timedelta(days=1)
            result = await User.restore_many(
                {"deleted_at": {"$gte": yesterday}}
            )
        """
        # Apply deleted filter: only restore documents that have been soft deleted
        final_filter = cls.apply_soft_delete_filter(filter_query, include_deleted=True)
        # Manually add condition that deleted_at is not None to ensure only deleted documents are restored
        if "deleted_at" not in final_filter:
            final_filter["deleted_at"] = {"$ne": None}

        # Perform bulk update operation to clear all soft delete markers
        return await cls.get_pymongo_collection().update_many(
            final_filter,
            {"$set": {"deleted_at": None, "deleted_by": None, "deleted_id": 0}},
            session=session,
            **pymongo_kwargs,
        )

    @classmethod
    async def hard_delete_many(
        cls,
        filter_query: Mapping[str, Any],
        session: Optional[AsyncClientSession] = None,
        **pymongo_kwargs: Any,
    ):
        """
        Bulk hard delete documents (physical deletion)

        ⚠️ Warning: This operation is irreversible! Use with caution.

        For bulk hard deletion, native approach can also be used:
            await Model.find(query).delete_many()

        Args:
            filter_query: MongoDB query filter condition
            session: Optional MongoDB session, for transaction support
            **pymongo_kwargs: Other parameters passed to PyMongo

        Returns:
            DeleteResult: Deletion result

        Example:
            # Permanently delete all test data
            result = await User.hard_delete_many({"is_test": True})
        """
        return await cls.get_pymongo_collection().delete_many(
            filter_query, session=session, **pymongo_kwargs
        )

    @classmethod
    def hard_find_many(  # type: ignore
        cls,
        *args: Union[Mapping[Any, Any], bool],
        projection_model: Optional[Type[BaseModel]] = None,
        skip: Optional[int] = None,
        limit: Optional[int] = None,
        sort: Union[None, str, List[Tuple[str, SortDirection]]] = None,
        session: Optional[AsyncClientSession] = None,
        ignore_cache: bool = True,
        fetch_links: bool = False,
        with_children: bool = False,
        lazy_parse: bool = False,
        nesting_depth: Optional[int] = None,
        nesting_depths_per_field: Optional[Dict[str, int]] = None,
        **pymongo_kwargs: Any,
    ):
        """
        Hard find multiple documents (including those soft deleted)

        Unlike find_many(), this method does not filter out deleted documents.
        Used in scenarios requiring viewing history or restoring deleted documents.
        Naming is consistent with hard_delete.

        Args:
            *args: Query conditions
            projection_model: Projection model
            skip: Number of documents to skip
            limit: Limit on number of documents returned
            sort: Sorting rule
            session: MongoDB session
            ignore_cache: Whether to ignore cache
            fetch_links: Whether to fetch linked documents
            with_children: Whether to include children
            lazy_parse: Whether to parse lazily
            nesting_depth: Nesting depth
            nesting_depths_per_field: Nesting depth per field
            **pymongo_kwargs: Other pymongo parameters

        Returns:
            FindMany query object

        Example:
            # Find all users including deleted ones
            all_users = await User.hard_find_many({"email": "test@example.com"}).to_list()

            # Find deleted documents
            deleted_users = await User.hard_find_many(
                {"deleted_at": {"$ne": None}}
            ).to_list()
        """
        args = cls._add_class_id_filter(args, with_children)
        return cls._find_many_query_class(document_model=cls).find_many(
            *args,
            sort=sort,
            skip=skip,
            limit=limit,
            projection_model=projection_model,
            session=session,
            ignore_cache=ignore_cache,
            fetch_links=fetch_links,
            lazy_parse=lazy_parse,
            nesting_depth=nesting_depth,
            nesting_depths_per_field=nesting_depths_per_field,
            **pymongo_kwargs,
        )

    @classmethod
    def find_many_in_all(cls, *args, **kwargs):
        """
        Deprecated: Please use hard_find_many() instead

        Kept for backward compatibility, recommended to use hard_find_many().
        """
        return cls.hard_find_many(*args, **kwargs)

    @classmethod
    def find_many(  # type: ignore
        cls,
        *args: Union[Mapping[Any, Any], bool],
        projection_model: Optional[Type[BaseModel]] = None,
        skip: Optional[int] = None,
        limit: Optional[int] = None,
        sort: Union[None, str, List[Tuple[str, SortDirection]]] = None,
        session: Optional[AsyncClientSession] = None,
        ignore_cache: bool = False,
        fetch_links: bool = False,
        with_children: bool = False,
        lazy_parse: bool = False,
        nesting_depth: Optional[int] = None,
        nesting_depths_per_field: Optional[Dict[str, int]] = None,
        **pymongo_kwargs: Any,
    ):
        """
        Find multiple documents (automatically filters out soft deleted ones)

        This method overrides parent's find_many, automatically adding deleted_at = None filter condition.
        Only returns documents not soft deleted.

        Use hard_find_many() if you need to query including deleted documents.

        Args:
            *args: Query conditions
            projection_model: Projection model
            skip: Number of documents to skip
            limit: Limit on number of documents returned
            sort: Sorting rule
            session: MongoDB session
            ignore_cache: Whether to ignore cache
            fetch_links: Whether to fetch linked documents
            with_children: Whether to include children
            lazy_parse: Whether to parse lazily
            nesting_depth: Nesting depth
            nesting_depths_per_field: Nesting depth per field
            **pymongo_kwargs: Other pymongo parameters

        Returns:
            FindMany query object

        Example:
            # Only find undeleted users
            active_users = await User.find_many({"status": "active"}).to_list()
        """
        # Add deleted_at = None filter condition
        args = cls._add_class_id_filter(args, with_children) + ({"deleted_at": None},)
        return cls._find_many_query_class(document_model=cls).find_many(
            *args,
            sort=sort,
            skip=skip,
            limit=limit,
            projection_model=projection_model,
            session=session,
            ignore_cache=ignore_cache,
            fetch_links=fetch_links,
            lazy_parse=lazy_parse,
            nesting_depth=nesting_depth,
            nesting_depths_per_field=nesting_depths_per_field,
            **pymongo_kwargs,
        )

    @classmethod
    def hard_find_one(  # type: ignore
        cls,
        *args: Union[Mapping[Any, Any], bool],
        projection_model: Optional[Type[BaseModel]] = None,
        session: Optional[AsyncClientSession] = None,
        ignore_cache: bool = True,
        fetch_links: bool = False,
        with_children: bool = False,
        nesting_depth: Optional[int] = None,
        nesting_depths_per_field: Optional[Dict[str, int]] = None,
        **pymongo_kwargs: Any,
    ):
        """
        Hard find single document (including those soft deleted)

        Unlike find_one(), this method does not filter out deleted documents.
        Used in scenarios requiring viewing history or restoring deleted documents.
        Naming is consistent with hard_delete.

        Args:
            *args: Query conditions
            projection_model: Projection model
            session: MongoDB session
            ignore_cache: Whether to ignore cache
            fetch_links: Whether to fetch linked documents
            with_children: Whether to include children
            nesting_depth: Nesting depth
            nesting_depths_per_field: Nesting depth per field
            **pymongo_kwargs: Other pymongo parameters

        Returns:
            FindOne query object

        Example:
            # Find user including deleted ones
            user = await User.hard_find_one({"email": "test@example.com"})

            # Find deleted user and restore
            deleted_user = await User.hard_find_one(
                {"email": "test@example.com", "deleted_at": {"$ne": None}}
            )
            if deleted_user:
                await deleted_user.restore()
        """
        args = cls._add_class_id_filter(args, with_children)
        return cls._find_one_query_class(document_model=cls).find_one(
            *args,
            projection_model=projection_model,
            session=session,
            ignore_cache=ignore_cache,
            fetch_links=fetch_links,
            nesting_depth=nesting_depth,
            nesting_depths_per_field=nesting_depths_per_field,
            **pymongo_kwargs,
        )

    @classmethod
    def find_one(  # type: ignore
        cls,
        *args: Union[Mapping[Any, Any], bool],
        projection_model: Optional[Type[BaseModel]] = None,
        session: Optional[AsyncClientSession] = None,
        ignore_cache: bool = True,
        fetch_links: bool = False,
        with_children: bool = False,
        nesting_depth: Optional[int] = None,
        nesting_depths_per_field: Optional[Dict[str, int]] = None,
        **pymongo_kwargs: Any,
    ):
        """
        Find single document (automatically filters out soft deleted ones)

        This method overrides parent's find_one, automatically adding deleted_at = None filter condition.
        Only returns documents not soft deleted.

        Use hard_find_one() if you need to query including deleted documents.

        Args:
            *args: Query conditions
            projection_model: Projection model
            session: MongoDB session
            ignore_cache: Whether to ignore cache
            fetch_links: Whether to fetch linked documents
            with_children: Whether to include children
            nesting_depth: Nesting depth
            nesting_depths_per_field: Nesting depth per field
            **pymongo_kwargs: Other pymongo parameters

        Returns:
            FindOne query object

        Example:
            # Find undeleted user
            user = await User.find_one({"email": "test@example.com"})
        """
        # Add deleted_at = None filter condition
        args = cls._add_class_id_filter(args, with_children) + ({"deleted_at": None},)
        return cls._find_one_query_class(document_model=cls).find_one(
            *args,
            projection_model=projection_model,
            session=session,
            ignore_cache=ignore_cache,
            fetch_links=fetch_links,
            nesting_depth=nesting_depth,
            nesting_depths_per_field=nesting_depths_per_field,
            **pymongo_kwargs,
        )

    class Settings:
        """Document settings"""

        # Common document configurations can be set here
        # For example: indexes, validation rules, etc


__all__ = ["DocumentBaseWithSoftDelete"]
