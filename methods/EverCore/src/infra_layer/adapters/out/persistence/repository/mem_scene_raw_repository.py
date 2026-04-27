"""
MemScene native CRUD repository

Mem scene data access layer based on Beanie ODM.
Provides ClusterStorage compatible interface (duck typing).
"""

from typing import Optional, Dict, Any
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository

from infra_layer.adapters.out.persistence.document.memory.mem_scene import MemScene

logger = get_logger(__name__)


@repository("mem_scene_raw_repository", primary=True)
class MemSceneRawRepository(BaseRepository[MemScene]):
    """
    MemScene native CRUD repository

    Provides ClusterStorage compatible interface:
    - save_mem_scene(group_id, state) -> bool
    - load_mem_scene(group_id) -> Optional[Dict]
    - get_cluster_assignments(group_id) -> Dict[str, str]
    - clear(group_id) -> bool
    """

    def __init__(self):
        super().__init__(MemScene)

    # ==================== ClusterStorage interface implementation ====================

    async def save_mem_scene(self, group_id: str, state: Dict[str, Any]) -> bool:
        result = await self.upsert_by_group_id(group_id, state)
        return result is not None

    async def load_mem_scene(self, group_id: str) -> Optional[Dict[str, Any]]:
        mem_scene = await self.get_by_group_id(group_id)
        if mem_scene is None:
            return None
        return mem_scene.model_dump(exclude={"id", "revision_id"})

    async def clear(self, group_id: Optional[str] = None) -> bool:
        if group_id is None:
            await self.delete_all()
        else:
            await self.delete_by_group_id(group_id)
        return True

    # ==================== Native CRUD methods ====================

    async def get_by_group_id(self, group_id: str) -> Optional[MemScene]:
        try:
            return await self.model.find_one(MemScene.group_id == group_id)
        except Exception as e:
            logger.error(
                f"Failed to retrieve mem scene: group_id={group_id}, error={e}"
            )
            return None

    async def upsert_by_group_id(
        self, group_id: str, state: Dict[str, Any]
    ) -> Optional[MemScene]:
        try:
            existing = await self.model.find_one(MemScene.group_id == group_id)

            if existing:
                for key, value in state.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                await existing.save()
                logger.debug(f"Updated mem scene: group_id={group_id}")
                return existing
            else:
                state["group_id"] = group_id
                mem_scene = MemScene(**state)
                await mem_scene.insert()
                logger.info(f"Created mem scene: group_id={group_id}")
                return mem_scene
        except Exception as e:
            logger.error(f"Failed to save mem scene: group_id={group_id}, error={e}")
            return None

    async def get_cluster_assignments(self, group_id: str) -> Dict[str, str]:
        try:
            mem_scene = await self.model.find_one(MemScene.group_id == group_id)
            if mem_scene is None:
                return {}
            # Derive eventid_to_cluster from memcell_info
            memcell_info = mem_scene.memcell_info or {}
            return {
                eid: info.get("memscene", "")
                for eid, info in memcell_info.items()
            }
        except Exception as e:
            logger.error(
                f"Failed to retrieve cluster assignments: group_id={group_id}, error={e}"
            )
            return {}

    async def delete_by_group_id(self, group_id: str) -> bool:
        try:
            mem_scene = await self.model.find_one(MemScene.group_id == group_id)
            if mem_scene:
                await mem_scene.delete()
                logger.info(f"Deleted mem scene: group_id={group_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete mem scene: group_id={group_id}, error={e}")
            return False

    async def delete_all(self) -> int:
        try:
            result = await self.model.delete_all()
            count = result.deleted_count if result else 0
            logger.info(f"Deleted all mem scenes: {count} items")
            return count
        except Exception as e:
            logger.error(f"Failed to delete all mem scenes: {e}")
            return 0
