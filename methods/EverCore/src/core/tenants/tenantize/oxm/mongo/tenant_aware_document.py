"""
Tenant-aware MongoDB Document Base Classes

Provides tenant_id field and index injection for MongoDB documents,
consistent with the ES (TenantAwareAsyncDocument) and Milvus
(TenantAwareMilvusCollectionWithSuffix) patterns.

The tenant_id field and index are ALWAYS present regardless of tenant mode,
keeping the schema consistent across all environments. In non-tenant mode
the field exists but is never populated (always None).

The actual tenant_id value is injected at runtime by
TenantCommandInterceptor (the _encrypter hook). This module ensures:
1. The field is declared in the Pydantic schema (so Beanie can
   deserialize it without dropping it)
2. A single-field index on tenant_id is created at startup
"""

from typing import Optional

from pydantic import Field
from pymongo import IndexModel, ASCENDING

from core.oxm.mongo.document_base import DocumentBase
from core.oxm.mongo.document_base_with_soft_delete import DocumentBaseWithSoftDelete
from core.tenants.tenant_constants import TENANT_ID_FIELD


def _inject_tenant_index(cls) -> None:
    """
    Append a tenant_id index to cls.Settings.indexes if not already present.

    This ensures every concrete document subclass gets a tenant_id index
    without requiring manual changes to each Settings class.
    """
    settings = getattr(cls, "Settings", None)
    if settings is None:
        return

    indexes = getattr(settings, "indexes", None)
    if indexes is None:
        settings.indexes = []
        indexes = settings.indexes

    # Check if tenant_id index already exists
    for idx in indexes:
        if isinstance(idx, IndexModel) and any(
            field_name == TENANT_ID_FIELD for field_name in idx.document.get("key", {})
        ):
            return

    # Prepend tenant_id index (most important for query performance)
    indexes.insert(
        0, IndexModel([(TENANT_ID_FIELD, ASCENDING)], name="idx_tenant_id", sparse=True)
    )


class TenantAwareDocumentBase(DocumentBase):
    """
    Tenant-aware MongoDB document base class.

    Adds an explicit tenant_id field so that:
    - Beanie can deserialize the field from MongoDB without dropping it
    - The field appears in the Pydantic schema for validation and serialization
    - A single-field index is auto-created at Beanie init time

    The runtime value injection is handled by TenantCommandInterceptor.
    """

    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier for multi-tenant isolation"
    )

    def __init_subclass__(cls, **kwargs) -> None:
        """Auto-inject tenant_id index into subclass Settings."""
        super().__init_subclass__(**kwargs)
        _inject_tenant_index(cls)


class TenantAwareDocumentBaseWithSoftDelete(DocumentBaseWithSoftDelete):
    """
    Tenant-aware MongoDB document base class with soft delete.

    Combines tenant_id field injection with full soft-delete capabilities.
    """

    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier for multi-tenant isolation"
    )

    def __init_subclass__(cls, **kwargs) -> None:
        """Auto-inject tenant_id index into subclass Settings."""
        super().__init_subclass__(**kwargs)
        _inject_tenant_index(cls)
