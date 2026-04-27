from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel
from sqlalchemy import Column, TIMESTAMP, event
from common_utils.datetime_utils import get_now_with_timezone
from core.context.context import get_current_user_info
from core.observation.logger import get_logger

logger = get_logger(__name__)


def get_auditable_model() -> SQLModel:
    """
    Get the base model class with audit capabilities

    This model includes audit fields that are automatically populated by event listeners:
    - created_at, updated_at: Automatically set timestamps by event listeners
    - created_by, updated_by: Automatically set the operating user by event listeners
    - deleted_at, deleted_by: Set during soft deletion by event listeners or business logic

    Returns:
        SQLModel: Base model class with audit capabilities
    """

    class AuditableModel(SQLModel):
        """Base model with audit information including creation and update details"""

        # Timestamp audit fields - automatically populated by event listeners
        created_at: Optional[datetime] = Field(
            default=None,
            sa_column=Column(TIMESTAMP(timezone=True), nullable=True),
            description="Creation time (automatically populated by event listener)",
        )

        updated_at: Optional[datetime] = Field(
            default=None,
            sa_column=Column(TIMESTAMP(timezone=True), nullable=True),
            description="Update time (automatically populated by event listener)",
        )

        deleted_at: Optional[datetime] = Field(
            default=None,
            sa_column=Column(TIMESTAMP(timezone=True), nullable=True),
            description="Deletion time (set during soft delete)",
        )

        # User audit fields - automatically populated by event listeners
        created_by: Optional[str] = Field(
            default=None,
            description="Creator (automatically populated by event listener)",
        )
        updated_by: Optional[str] = Field(
            default=None,
            description="Updater (automatically populated by event listener)",
        )
        deleted_by: Optional[str] = Field(
            default=None, description="Deleter (set during soft delete)"
        )

        def soft_delete(self, deleted_by: str):
            """Soft delete the record"""
            self.deleted_at = get_now_with_timezone()
            self.deleted_by = deleted_by

        def restore(self, restored_by: str = None):
            """Restore a soft-deleted record"""
            # restored_by parameter is kept for interface compatibility, but actually set by event listener
            _ = restored_by  # Avoid unused parameter warning
            self.deleted_at = None
            self.deleted_by = None

        @property
        def is_deleted(self) -> bool:
            """Check if the record has been soft-deleted"""
            return self.deleted_at is not None

    # Register event listeners
    @event.listens_for(AuditableModel, 'before_insert', propagate=True)
    def before_insert_listener(
        mapper, connection, target
    ):  # pylint: disable=unused-argument
        """Event listener before INSERT operation"""
        # Ignore unused parameters (required signature for SQLAlchemy event listeners)
        _ = mapper, connection

        current_time = get_now_with_timezone()
        current_user_id = _get_current_user_id()

        # Set creation time and creator
        if hasattr(target, 'created_at') and target.created_at is None:
            target.created_at = current_time

        if hasattr(target, 'created_by') and target.created_by is None:
            target.created_by = current_user_id or "system"

        # Set update time and updater
        if hasattr(target, 'updated_at') and target.updated_at is None:
            target.updated_at = current_time

        if hasattr(target, 'updated_by') and target.updated_by is None:
            target.updated_by = current_user_id or "system"

    @event.listens_for(AuditableModel, 'before_update', propagate=True)
    def before_update_listener(
        mapper, connection, target
    ):  # pylint: disable=unused-argument
        """Event listener before UPDATE operation"""
        # Ignore unused parameters (required signature for SQLAlchemy event listeners)
        _ = mapper, connection

        current_time = get_now_with_timezone()
        current_user_id = _get_current_user_id()

        # Set update time and updater
        if hasattr(target, 'updated_at'):
            target.updated_at = current_time

        # Only set updated_by if it's None, do not overwrite existing values (e.g., "system")
        if hasattr(target, 'updated_by') and target.updated_by is None:
            target.updated_by = current_user_id or "system"

        # Special handling for soft delete scenario
        if hasattr(target, 'deleted_at') and target.deleted_at is not None:
            # If deleted_at is set, it indicates a soft delete operation
            if hasattr(target, 'deleted_by') and target.deleted_by is None:
                target.deleted_by = current_user_id or "system"

    return AuditableModel


def _get_current_user_id() -> Optional[str]:
    """
    Get current user ID

    Returns:
        Optional[str]: Current user ID, returns None if not set
    """
    try:
        user_info = get_current_user_info()
        if user_info and 'user_id' in user_info:
            return str(user_info['user_id'])
    except Exception as e:  # pylint: disable=broad-except
        logger.debug("Failed to get current user information: %s", e)
    return None
