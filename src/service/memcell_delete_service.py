"""
MemCell Delete Service - Handle soft delete logic for MemCell

Provides multiple deletion methods:
- Delete by single event_id
- Batch delete by user_id
- Batch delete by group_id
"""

from typing import Optional
from core.di.decorators import component
from core.observation.logger import get_logger
from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
    MemCellRawRepository,
)

logger = get_logger(__name__)


@component("memcell_delete_service")
class MemCellDeleteService:
    """MemCell soft delete service"""

    def __init__(self, memcell_repository: MemCellRawRepository):
        """
        Initialize deletion service

        Args:
            memcell_repository: MemCell data repository
        """
        self.memcell_repository = memcell_repository
        logger.info("MemCellDeleteService initialized")

    async def delete_by_event_id(
        self, event_id: str, deleted_by: Optional[str] = None
    ) -> bool:
        """
        Soft delete a single MemCell by event_id

        Args:
            event_id: The event_id of MemCell
            deleted_by: Identifier of the deleter (optional)

        Returns:
            bool: Returns True if deletion succeeds, False if not found or already deleted

        Example:
            >>> service = MemCellDeleteService(repo)
            >>> success = await service.delete_by_event_id("507f1f77bcf86cd799439011", "admin")
        """
        logger.info(
            "Deleting MemCell by event_id: event_id=%s, deleted_by=%s",
            event_id,
            deleted_by,
        )

        try:
            result = await self.memcell_repository.delete_by_event_id(
                event_id=event_id, deleted_by=deleted_by
            )

            if result:
                logger.info(
                    "Successfully deleted MemCell: event_id=%s, deleted_by=%s",
                    event_id,
                    deleted_by,
                )
            else:
                logger.warning(
                    "MemCell not found or already deleted: event_id=%s", event_id
                )

            return result

        except Exception as e:
            logger.error(
                "Failed to delete MemCell by event_id: event_id=%s, error=%s",
                event_id,
                e,
                exc_info=True,
            )
            raise

    async def delete_by_user_id(
        self, user_id: str, deleted_by: Optional[str] = None
    ) -> int:
        """
        Batch soft delete all MemCells of a user by user_id

        Args:
            user_id: User ID
            deleted_by: Identifier of the deleter (optional)

        Returns:
            int: Number of deleted records

        Example:
            >>> service = MemCellDeleteService(repo)
            >>> count = await service.delete_by_user_id("user_123", "admin")
            >>> print(f"Deleted {count} records")
        """
        logger.info(
            "Deleting MemCells by user_id: user_id=%s, deleted_by=%s",
            user_id,
            deleted_by,
        )

        try:
            count = await self.memcell_repository.delete_by_user_id(
                user_id=user_id, deleted_by=deleted_by
            )

            logger.info(
                "Successfully deleted MemCells by user_id: user_id=%s, deleted_by=%s, count=%d",
                user_id,
                deleted_by,
                count,
            )

            return count

        except Exception as e:
            logger.error(
                "Failed to delete MemCells by user_id: user_id=%s, error=%s",
                user_id,
                e,
                exc_info=True,
            )
            raise

    async def delete_by_group_id(
        self, group_id: str, deleted_by: Optional[str] = None
    ) -> int:
        """
        Batch soft delete all MemCells of a group by group_id

        Args:
            group_id: Group ID
            deleted_by: Identifier of the deleter (optional)

        Returns:
            int: Number of deleted records

        Example:
            >>> service = MemCellDeleteService(repo)
            >>> count = await service.delete_by_group_id("group_456", "admin")
            >>> print(f"Deleted {count} records")
        """
        logger.info(
            "Deleting MemCells by group_id: group_id=%s, deleted_by=%s",
            group_id,
            deleted_by,
        )

        try:
            # Use repository's delete_many method
            from infra_layer.adapters.out.persistence.document.memory.memcell import (
                MemCell,
            )

            result = await MemCell.delete_many(
                {"group_id": group_id}, deleted_by=deleted_by
            )

            count = result.modified_count if result else 0

            logger.info(
                "Successfully deleted MemCells by group_id: group_id=%s, deleted_by=%s, count=%d",
                group_id,
                deleted_by,
                count,
            )

            return count

        except Exception as e:
            logger.error(
                "Failed to delete MemCells by group_id: group_id=%s, error=%s",
                group_id,
                e,
                exc_info=True,
            )
            raise

    async def _delete_milvus_vectors(
        self,
        user_id: Optional[str],
        group_id: Optional[str],
        event_id: Optional[str],
        filters_used: list,
    ) -> int:
        """同步清理 Milvus 向量索引，防止已删除记忆仍被召回。

        Args:
            user_id: 用户 ID（MAGIC_ALL 或 None 表示不用此维度过滤）
            group_id: 群组 ID（MAGIC_ALL 或 None 表示不用此维度过滤）
            event_id: 单条记忆 ID（MAGIC_ALL 或 None 表示不按此过滤）
            filters_used: 已使用的过滤条件列表（用于日志）

        Returns:
            删除的向量条数之和
        """
        import asyncio
        from core.oxm.constants import MAGIC_ALL
        from infra_layer.adapters.out.search.repository.episodic_memory_milvus_repository import (
            EpisodicMemoryMilvusRepository,
        )
        from infra_layer.adapters.out.search.repository.foresight_milvus_repository import (
            ForesightMilvusRepository,
        )
        from infra_layer.adapters.out.search.repository.event_log_milvus_repository import (
            EventLogMilvusRepository,
        )
        from core.di.context import get_bean_by_type

        episodic_repo = get_bean_by_type(EpisodicMemoryMilvusRepository)
        foresight_repo = get_bean_by_type(ForesightMilvusRepository)
        event_log_repo = get_bean_by_type(EventLogMilvusRepository)

        effective_user_id = user_id if (user_id and user_id != MAGIC_ALL) else None
        effective_group_id = group_id if (group_id and group_id != MAGIC_ALL) else None
        effective_event_id = event_id if (event_id and event_id != MAGIC_ALL) else None

        tasks = []

        # 按 user/group 批量删除向量
        if effective_user_id or effective_group_id:
            tasks.append(
                episodic_repo.delete_by_filters(
                    user_id=effective_user_id, group_id=effective_group_id
                )
            )
            tasks.append(
                foresight_repo.delete_by_filters(
                    user_id=effective_user_id, group_id=effective_group_id
                )
            )
            tasks.append(
                event_log_repo.delete_by_filters(
                    user_id=effective_user_id, group_id=effective_group_id
                )
            )

        # 按 event_id 精确删除单条向量
        if effective_event_id:
            tasks.append(episodic_repo.delete_by_event_id(effective_event_id))
            tasks.append(foresight_repo.delete_by_parent_id(effective_event_id))
            tasks.append(event_log_repo.delete_by_parent_id(effective_event_id))

        if not tasks:
            return 0

        results = await asyncio.gather(*tasks, return_exceptions=True)
        total = 0
        for r in results:
            if isinstance(r, Exception):
                logger.warning(
                    "Milvus vector deletion failed (non-fatal): filters=%s, error=%s",
                    filters_used,
                    r,
                )
            elif isinstance(r, int):
                total += r

        logger.info(
            "Milvus vector cleanup completed: filters=%s, deleted_vectors=%d",
            filters_used,
            total,
        )
        return total

    async def delete_by_combined_criteria(
        self,
        event_id: Optional[str] = None,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> dict:
        """
        Delete MemCell based on combined criteria (multiple conditions must all be satisfied)

        Args:
            event_id: The event_id of MemCell (one of the combined conditions)
            user_id: User ID (one of the combined conditions)
            group_id: Group ID (one of the combined conditions)

        Returns:
            dict: Dictionary containing deletion results
                - filters: List of filter conditions used
                - count: Number of deleted records
                - success: Whether the operation succeeded

        Example:
            >>> service = MemCellDeleteService(repo)
            >>> # Delete records of a specific user in a specific group
            >>> result = await service.delete_by_combined_criteria(
            ...     user_id="user_123",
            ...     group_id="group_456",
            ... )
            >>> print(result)
            {'filters': ['user_id', 'group_id'], 'count': 5, 'success': True}
        """
        from core.oxm.constants import MAGIC_ALL
        from infra_layer.adapters.out.persistence.document.memory.memcell import MemCell
        from infra_layer.adapters.out.persistence.document.memory.episodic_memory import (
            EpisodicMemory,
        )
        from infra_layer.adapters.out.persistence.document.memory.foresight_record import (
            ForesightRecord,
        )
        from infra_layer.adapters.out.persistence.document.memory.event_log_record import (
            EventLogRecord,
        )
        from infra_layer.adapters.out.persistence.document.memory.user_profile import (
            UserProfile,
        )
        from infra_layer.adapters.out.persistence.document.memory.global_user_profile import (
            GlobalUserProfile,
        )

        # Build filter conditions
        filter_dict = {}
        filters_used = []

        if event_id and event_id != MAGIC_ALL:
            from bson import ObjectId

            try:
                filter_dict["_id"] = ObjectId(event_id)
                filters_used.append("event_id")
            except Exception as e:
                logger.error("Invalid event_id format: %s, error: %s", event_id, e)
                return {
                    "filters": [],
                    "count": 0,
                    "success": False,
                    "error": f"Invalid event_id format: {event_id}",
                }

        if user_id and user_id != MAGIC_ALL:
            filter_dict["user_id"] = user_id
            filters_used.append("user_id")

        if group_id and group_id != MAGIC_ALL:
            filter_dict["group_id"] = group_id
            filters_used.append("group_id")

        # If no filter conditions are provided
        if not filter_dict:
            logger.warning("No deletion criteria provided (all are MAGIC_ALL)")
            return {
                "filters": [],
                "count": 0,
                "success": False,
                "error": "No deletion criteria provided",
            }

        logger.info(
            "Deleting memories with combined criteria: filters=%s", filters_used
        )

        def _count_deleted(result) -> int:
            if not result:
                return 0
            if hasattr(result, "deleted_count"):
                return int(result.deleted_count or 0)
            if hasattr(result, "modified_count"):
                return int(result.modified_count or 0)
            return 0

        try:
            total_count = 0

            # 1) 删除 MemCell
            memcell_result = await MemCell.delete_many(filter_dict)
            memcell_count = _count_deleted(memcell_result)
            total_count += memcell_count

            # 2) 删除 EpisodicMemory（按 user/group，或 event_id 关联）
            ep_filter = {}
            if user_id and user_id != MAGIC_ALL:
                ep_filter["user_id"] = user_id
            if group_id and group_id != MAGIC_ALL:
                ep_filter["group_id"] = group_id
            if event_id and event_id != MAGIC_ALL:
                ep_filter["event_id"] = event_id
            if ep_filter:
                ep_result = await EpisodicMemory.find(ep_filter).delete()
                total_count += _count_deleted(ep_result)

            # 3) 删除 ForesightRecord（按 user/group，或 parent_id 关联）
            foresight_filter = {}
            if user_id and user_id != MAGIC_ALL:
                foresight_filter["user_id"] = user_id
            if group_id and group_id != MAGIC_ALL:
                foresight_filter["group_id"] = group_id
            if event_id and event_id != MAGIC_ALL:
                foresight_filter["parent_id"] = event_id
            if foresight_filter:
                foresight_result = await ForesightRecord.find(foresight_filter).delete()
                total_count += _count_deleted(foresight_result)

            # 4) 删除 EventLogRecord（按 user/group，或 parent_id 关联）
            event_log_filter = {}
            if user_id and user_id != MAGIC_ALL:
                event_log_filter["user_id"] = user_id
            if group_id and group_id != MAGIC_ALL:
                event_log_filter["group_id"] = group_id
            if event_id and event_id != MAGIC_ALL:
                event_log_filter["parent_id"] = event_id
            if event_log_filter:
                event_log_result = await EventLogRecord.find(event_log_filter).delete()
                total_count += _count_deleted(event_log_result)

            # 5) 删除 UserProfile（按 user/group）
            profile_filter = {}
            if user_id and user_id != MAGIC_ALL:
                profile_filter["user_id"] = user_id
            if group_id and group_id != MAGIC_ALL:
                profile_filter["group_id"] = group_id
            if profile_filter:
                profile_result = await UserProfile.find(profile_filter).delete()
                total_count += _count_deleted(profile_result)

            # 6) 删除 GlobalUserProfile（仅按 user）
            if user_id and user_id != MAGIC_ALL:
                global_profile_result = await GlobalUserProfile.find(
                    {"user_id": user_id}
                ).delete()
                total_count += _count_deleted(global_profile_result)

            logger.info(
                "Successfully deleted memories: filters=%s, total_count=%d (memcell=%d)",
                filters_used,
                total_count,
                memcell_count,
            )

            # 7) 同步清理 Milvus 向量索引，防止已删除记忆仍被向量召回
            try:
                await self._delete_milvus_vectors(
                    user_id=user_id,
                    group_id=group_id,
                    event_id=event_id,
                    filters_used=filters_used,
                )
            except Exception as milvus_err:
                # Milvus 清理失败不影响主流程返回成功
                logger.warning(
                    "Milvus vector cleanup encountered an error (non-fatal): filters=%s, error=%s",
                    filters_used,
                    milvus_err,
                )

            return {"filters": filters_used, "count": total_count, "success": True}

        except Exception as e:
            logger.error(
                "Failed to delete MemCells with combined criteria: filters=%s, error=%s",
                filters_used,
                e,
                exc_info=True,
            )
            raise
