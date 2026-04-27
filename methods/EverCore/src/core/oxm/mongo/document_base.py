"""
MongoDB Document Base Class

Base document class based on Beanie ODM, providing common foundational document functionality.
"""

from datetime import datetime
from common_utils.datetime_utils import to_timezone
from beanie import Document, WriteRules
from pydantic import model_validator, BaseModel
from typing import Self, List, Optional
from pymongo.asynchronous.client_session import AsyncClientSession
from pymongo.results import InsertManyResult

from core.oxm.mongo.audit_base import AuditBase

MAX_RECURSION_DEPTH = 4
DEFAULT_DATABASE = "default"


class DocumentBase(Document):
    """
    Document base class

    Base document class based on Beanie Document, providing common foundational document functionality.
    """

    @classmethod
    def get_bind_database(cls) -> str | None:
        """
        Read the bound database name (read-only).

        Only reads from `Settings.bind_database`, no runtime modification allowed.
        Subclasses can bind by overriding the class variable `bind_database` in the internal `Settings`:

        class MyDoc(DocumentBase):
            class Settings:
                bind_database = "my_db"
        """
        settings = getattr(cls, "Settings", None)
        if settings is not None:
            return getattr(settings, "bind_database", DEFAULT_DATABASE)
        return DEFAULT_DATABASE

    def _recursive_datetime_check(self, obj, path: str = "", depth: int = 0):
        """
        Recursively check and convert all datetime objects to default timezone

        Args:
            obj: Object to check
            path: Current object path (for debugging)
            depth: Current recursion depth

        Returns:
            Converted object
        """
        # Control maximum recursion depth
        if depth >= MAX_RECURSION_DEPTH:
            return obj

        # Case 1: Object is datetime
        if isinstance(obj, datetime):
            if obj.tzinfo is None:
                # No timezone info, convert to default timezone; usually created within the process and passed as parameter
                return to_timezone(obj)
            else:
                # Return if read with timezone and it's the default timezone
                return obj

        # Case 2: Object is BaseModel
        if isinstance(obj, BaseModel):
            for field_name, value in obj:
                new_path = f"{path}.{field_name}" if path else field_name
                new_value = self._recursive_datetime_check(value, new_path, depth + 1)
                # Directly update value using __dict__ to avoid triggering validators
                obj.__dict__[field_name] = new_value
            return obj

        # Case 3: Object is list, tuple, or set (performance optimization)
        if isinstance(obj, (list, tuple, set)):
            # If collection is empty, return directly
            if not obj:
                return obj

            # List: only check the first element
            if isinstance(obj, list):
                first_item = obj[0]
                first_checked = self._recursive_datetime_check(
                    first_item, f"{path}[0]", depth + 2
                )

                # If the first element hasn't changed, assume the whole list doesn't need conversion
                if first_checked is first_item:
                    return obj

            # Set: check any one element (set is unordered, take the first one)
            elif isinstance(obj, set):
                sample_item = next(iter(obj))
                sample_checked = self._recursive_datetime_check(
                    sample_item, f"{path}[sample]", depth + 2
                )

                # If the sampled element hasn't changed, assume the whole set doesn't need conversion
                if sample_checked is sample_item:
                    return obj

            # Tuple: only check the first 3 elements
            elif isinstance(obj, tuple):
                # Check first 3 elements (or all if length < 3)
                check_count = min(3, len(obj))
                need_transform = False

                for idx in range(check_count):
                    item = obj[idx]
                    checked = self._recursive_datetime_check(
                        item, f"{path}[{idx}]", depth + 2
                    )
                    if checked is not item:
                        need_transform = True
                        break

                # If first 3 elements don't need conversion, assume the whole tuple doesn't need conversion
                if not need_transform:
                    return obj

            # Need to process all elements
            cls = type(obj)
            return cls(
                self._recursive_datetime_check(item, f"{path}[{i}]", depth + 2)
                for i, item in enumerate(obj)
            )

        # Case 4: Object is dictionary
        if isinstance(obj, dict):
            return {
                key: self._recursive_datetime_check(
                    value, f"{path}[{repr(key)}]", depth + 2
                )
                for key, value in obj.items()
            }

        return obj

    @model_validator(mode='after')
    def check_datetimes_are_aware(self) -> Self:
        """
        Recursively traverse all fields of the model to ensure any datetime object is 'aware' (contains timezone information).
        Maximum recursion depth is 3 to avoid potential issues.

        Returns:
            Self: Current object instance
        """
        for field_name, value in self:
            new_value = self._recursive_datetime_check(value, field_name, depth=0)
            if new_value is not value:  # Only update if value has changed

                # Directly update value using __dict__ to avoid triggering validators
                self.__dict__[field_name] = new_value
        return self

    @classmethod
    async def insert_many(
        cls,
        documents: List["DocumentBase"],
        session: Optional[AsyncClientSession] = None,
        link_rule: WriteRules = WriteRules.DO_NOTHING,
        **pymongo_kwargs,
    ) -> InsertManyResult:
        """
        Override bulk insert method, delegate audit logic to AuditBase

        As a technical entry point, check if the model inherits from AuditBase, if so, delegate audit field handling.
        This maintains responsibility cohesion: DocumentBase handles coordination, AuditBase handles audit logic.

        Args:
            documents: List of documents to insert
            session: Optional MongoDB session, used for transaction support
            link_rule: Write rule for linked documents
            **pymongo_kwargs: Other parameters passed to PyMongo

        Returns:
            InsertManyResult: Insert result, containing inserted_ids
        """
        # Check if model inherits from AuditBase, if so, delegate audit handling

        if issubclass(cls, AuditBase):
            # Delegate to AuditBase to handle audit fields
            AuditBase.prepare_for_insert_many(documents)

        # Call parent class's insert_many method
        return await super().insert_many(
            documents, session=session, link_rule=link_rule, **pymongo_kwargs
        )

    class Settings:
        """Document settings"""

        # Common document configurations can be set here
        # For example: indexes, validation rules, etc.

    @classmethod
    async def count(cls, **kwargs) -> int:
        """
        Count documents in the collection

        Uses estimated_document_count() for fast counting based on collection metadata.
        This is very fast (millisecond level) but returns an approximate count.

        Subclasses (e.g., DocumentBaseWithSoftDelete) may override this method
        to provide filtering capabilities (e.g., excluding soft-deleted documents).

        Returns:
            int: Estimated document count
        """
        collection = cls.get_pymongo_collection()
        return await collection.estimated_document_count()

    def __str__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}({self.id})"

    def __repr__(self) -> str:
        """Developer representation"""
        return f"{self.__class__.__name__}(id={self.id})"
