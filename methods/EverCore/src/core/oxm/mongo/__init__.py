"""
MongoDB OXM (Object-XML Mapping) Module

Provides document base classes and utilities for MongoDB operations.
"""

from core.oxm.mongo.document_base import DocumentBase
from core.oxm.mongo.document_base_with_soft_delete import DocumentBaseWithSoftDelete
from core.oxm.mongo.audit_base import AuditBase

__all__ = [
    "DocumentBase",
    "DocumentBaseWithSoftDelete",
    "AuditBase",
]
