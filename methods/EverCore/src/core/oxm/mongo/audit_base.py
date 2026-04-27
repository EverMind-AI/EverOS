"""
MongoDB audit base class

Audit base class based on Beanie ODM, including common timestamp fields and automatic processing logic.
"""

from datetime import datetime
from typing import Optional, List, Any
from beanie import before_event, Insert, Replace, Save, Update
from pydantic import Field, BaseModel
from common_utils.datetime_utils import get_now_with_timezone


class AuditBase(BaseModel):
    """
    Audit base class

    Includes common timestamp fields and automatic processing logic

    Note:
    - For single insertion, @before_event(Insert) automatically triggers timestamp setting
    - For bulk insertion, DocumentBase.insert_many delegates to this class's prepare_for_insert_many method for handling
    """

    # System fields
    created_at: Optional[datetime] = Field(default=None, description="Creation time")
    updated_at: Optional[datetime] = Field(default=None, description="Update time")

    @before_event(Insert)
    async def set_created_at(self):
        """Set creation time before insertion"""
        now = get_now_with_timezone()
        self.created_at = now
        self.updated_at = now

    @before_event(Update, Replace, Save)
    async def set_updated_at(self):
        """Set update time before update, replace, or save"""
        self.updated_at = get_now_with_timezone()

    @classmethod
    def prepare_for_insert_many(cls, documents: List[Any]) -> None:
        """
        Prepare before bulk insertion: set audit timestamp fields for documents

        Since Beanie's @before_event(Insert) does not automatically trigger during bulk insertion,
        this method is called by DocumentBase.insert_many, responsible for setting audit fields before bulk insertion.

        Args:
            documents: List of documents to be inserted

        Note:
            This method is automatically called by DocumentBase.insert_many,
            developers typically do not need to call this method manually.
        """
        now = get_now_with_timezone()
        for doc in documents:
            # Only set time for audit fields with None value, avoiding overwriting existing values
            if hasattr(doc, 'created_at') and doc.created_at is None:
                doc.created_at = now
            if hasattr(doc, 'updated_at') and doc.updated_at is None:
                doc.updated_at = now


__all__ = ["AuditBase"]
