"""
User Profile Milvus Converter

Converts MongoDB v1_user_profiles to Milvus v1_user_profile entities.
Splits profile into individual items (one per explicit_info / implicit_trait)
for per-item vector search.
"""

from typing import Dict, Any, List

from api_specs.memory_types import ScenarioType
from core.oxm.milvus.base_converter import BaseMilvusConverter
from core.observation.logger import get_logger
from core.oxm.mongo.mongo_utils import generate_object_id_str
from infra_layer.adapters.out.search.milvus.memory.user_profile_collection import (
    UserProfileCollection,
)
from infra_layer.adapters.out.persistence.document.memory.user_profile import (
    UserProfile as MongoUserProfile,
)

logger = get_logger(__name__)


class UserProfileMilvusConverter(BaseMilvusConverter[UserProfileCollection]):
    """
    User Profile Milvus Converter

    Converts a single MongoDB v1_user_profiles document into a **list** of
    Milvus entities — one entity per explicit_info item and one per
    implicit_trait.  Each entity carries an ``embed_text`` field used by
    ProfileIndexer to generate embeddings, and an ``item_type`` field
    used for statistics.
    """

    @classmethod
    def from_mongo(cls, source_doc: MongoUserProfile) -> List[Dict[str, Any]]:
        """
        Convert from MongoDB v1_user_profiles document to Milvus entities.

        Args:
            source_doc: MongoDB v1_user_profiles document instance

        Returns:
            List[Dict[str, Any]]: One Milvus entity dict per profile item.
        """
        if source_doc is None:
            raise ValueError("MongoDB document cannot be empty")

        try:
            profile_data: Dict[str, Any] = source_doc.profile_data or {}
            doc_id = str(source_doc.id) if source_doc.id else ""
            user_id = source_doc.user_id or ""
            group_id = source_doc.group_id or ""
            scenario = source_doc.scenario or ScenarioType.SOLO.value
            memcell_count = source_doc.memcell_count or 0

            entities: List[Dict[str, Any]] = []
            seq = 0

            def _make_entity(embed_text: str, item_type: str) -> Dict[str, Any]:
                nonlocal seq
                entity = {
                    "id": generate_object_id_str(),
                    "user_id": user_id,
                    "group_id": group_id,
                    "scenario": scenario,
                    "memcell_count": memcell_count,
                    "item_type": item_type,
                    "embed_text": embed_text,
                }
                seq += 1
                return entity

            # ProfileMemory format: hard_skills, soft_skills, personality, etc.
            # These fields contain [{value, evidences, ...}] items.
            _EXPLICIT_FIELDS = [
                ("hard_skills", "Hard Skill"),
                ("soft_skills", "Soft Skill"),
                ("work_responsibility", "Work Responsibility"),
                ("interests", "Interest"),
            ]
            _IMPLICIT_FIELDS = [
                ("personality", "Personality"),
                ("tendency", "Tendency"),
                ("way_of_decision_making", "Decision Making"),
                ("motivation_system", "Motivation"),
                ("fear_system", "Fear"),
                ("value_system", "Value"),
            ]

            for field_name, label in _EXPLICIT_FIELDS:
                for item in profile_data.get(field_name, []) or []:
                    value = item.get("value", "") if isinstance(item, dict) else str(item)
                    if not value:
                        continue
                    level = item.get("level", "") if isinstance(item, dict) else ""
                    embed_text = f"{label}: {value}" + (f" ({level})" if level else "")
                    entities.append(_make_entity(embed_text, "explicit_info"))

            for field_name, label in _IMPLICIT_FIELDS:
                for item in profile_data.get(field_name, []) or []:
                    value = item.get("value", "") if isinstance(item, dict) else str(item)
                    if not value:
                        continue
                    embed_text = f"{label}: {value}"
                    entities.append(_make_entity(embed_text, "implicit_trait"))

            # Legacy format: explicit_info[] and implicit_traits[]
            # Solo extractor produces {category, description} / {trait, description, basis}
            for item in profile_data.get("explicit_info", []) or []:
                if not isinstance(item, dict):
                    continue
                desc = item.get("description", "")
                if not desc:
                    continue
                category = item.get("category", "")
                embed_text = f"{category}: {desc}" if category else desc
                entities.append(_make_entity(embed_text, "explicit_info"))

            for item in profile_data.get("implicit_traits", []) or []:
                if not isinstance(item, dict):
                    continue
                desc = item.get("description", "")
                if not desc:
                    continue
                trait_name = item.get("trait") or item.get("trait_name", "")
                embed_text = f"{trait_name}: {desc}" if trait_name else desc
                if item.get("basis"):
                    embed_text += f". {item['basis']}"
                entities.append(_make_entity(embed_text, "implicit_trait"))

            # user_goal (single string)
            user_goal = profile_data.get("user_goal")
            if user_goal and isinstance(user_goal, str) and user_goal.strip():
                entities.append(_make_entity(f"Goal: {user_goal.strip()}", "explicit_info"))

            return entities

        except Exception as e:
            logger.error(
                "Failed to convert MongoDB UserProfile to Milvus entities: %s",
                e,
                exc_info=True,
            )
            raise
