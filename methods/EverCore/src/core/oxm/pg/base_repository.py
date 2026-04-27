from abc import ABC, abstractmethod
from typing import List, Optional, TypeVar, Generic, Type
from core.oxm.pg.audit_base import get_auditable_model
from core.di.decorators import repository

# Define generic type
T = TypeVar('T', bound=get_auditable_model())


class BaseSoftDeleteRepository(Generic[T], ABC):
    """Base repository interface supporting soft delete - pure business interface without technical implementation details"""

    @abstractmethod
    async def add(self, entity: T) -> T:
        """Add new entity"""
        raise NotImplementedError

    @abstractmethod
    async def get(self, entity_id: int, include_deleted: bool = False) -> Optional[T]:
        """Get entity by ID (exclude deleted by default)"""
        raise NotImplementedError

    @abstractmethod
    async def get_all(self, include_deleted: bool = False) -> List[T]:
        """Get all entities (exclude deleted by default)"""
        raise NotImplementedError

    @abstractmethod
    async def update(self, entity: T) -> T:
        """Update entity"""
        raise NotImplementedError

    @abstractmethod
    async def delete(self, entity_id: int, deleted_by: str = "system") -> bool:
        """Soft delete entity"""
        raise NotImplementedError

    @abstractmethod
    async def restore(self, entity_id: int, restored_by: str = "system") -> bool:
        """Restore soft-deleted entity"""
        raise NotImplementedError

    @abstractmethod
    async def hard_delete(self, entity_id: int) -> bool:
        """Hard delete entity (use with caution!)"""
        raise NotImplementedError

    @abstractmethod
    async def count(self, include_deleted: bool = False) -> int:
        """Count entities (exclude deleted by default)"""
        raise NotImplementedError
