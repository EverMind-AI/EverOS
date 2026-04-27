"""Tool function to clear all memory data

Can be imported by other test scripts or run independently
"""

import asyncio
import time
from typing import Dict, Any, List

from pymilvus import utility, Collection

from infra_layer.adapters.out.search.milvus.memory.episodic_memory_collection import (
    EpisodicMemoryCollection,
)
from infra_layer.adapters.out.search.milvus.memory.foresight_collection import (
    ForesightCollection,
)
from infra_layer.adapters.out.search.milvus.memory.atomic_fact_collection import (
    AtomicFactCollection,
)
from infra_layer.adapters.out.search.milvus.memory.agent_case_collection import (
    AgentCaseCollection,
)
from infra_layer.adapters.out.search.milvus.memory.agent_skill_collection import (
    AgentSkillCollection,
)
from infra_layer.adapters.out.search.elasticsearch.memory.episodic_memory import (
    EpisodicMemoryDoc,
)
from infra_layer.adapters.out.search.elasticsearch.memory.foresight import ForesightDoc
from infra_layer.adapters.out.search.elasticsearch.memory.atomic_fact import (
    AtomicFactDoc,
)
from infra_layer.adapters.out.search.elasticsearch.memory.agent_case import (
    AgentCaseDoc,
)
from infra_layer.adapters.out.search.elasticsearch.memory.agent_skill import (
    AgentSkillDoc,
)
from core.di import get_bean_by_type
from core.component.redis_provider import RedisProvider
from core.component.mongodb_client_factory import MongoDBClientFactory


async def _clear_mongodb(verbose: bool = True) -> Dict[str, Any]:
    """Delete all documents in MongoDB, keeping collections and indexes"""
    result: Dict[str, Any] = {
        "database": None,
        "collections": {},
        "deleted": {},
        "errors": [],
    }
    try:
        mongo_factory = get_bean_by_type(MongoDBClientFactory)
        client_wrapper = await mongo_factory.get_default_client()
        db = client_wrapper.database
        db_name = db.name
        result["database"] = db_name

        collection_names = await db.list_collection_names()
        for coll_name in collection_names:
            if coll_name.startswith("system."):
                continue
            collection = db[coll_name]
            count = await collection.count_documents({})
            if count == 0:
                continue
            # todo delete many without repository
            delete_result = await collection.delete_many({})
            deleted = delete_result.deleted_count if delete_result else 0
            result["collections"][coll_name] = count
            result["deleted"][coll_name] = deleted

        if verbose:
            total_deleted = sum(result["deleted"].values())
            print(
                f"      ✅ MongoDB '{db_name}': Deleted {total_deleted} documents ({len(result['deleted'])} collections)"
            )
    except Exception as exc:
        result["errors"].append(str(exc))
        if verbose:
            print(f"      ⚠️  MongoDB clear failed: {exc}")

    return result


def _get_milvus_row_count(name: str, coll: Collection) -> int:
    """Get Milvus real-time row count (priority: row_count > segment > num_entities)"""
    get_stats = getattr(utility, "get_collection_stats", None)
    if callable(get_stats):
        stats_info = get_stats(name)
        if isinstance(stats_info, dict):
            try:
                return int(stats_info.get("row_count", 0))
            except (ValueError, TypeError):
                pass

    try:
        segment_infos = utility.get_query_segment_info(name)
        if segment_infos:
            total = 0
            for seg in segment_infos:
                seg_rows = getattr(seg, "num_rows", None)
                if seg_rows is None:
                    seg_rows = getattr(seg, "row_count", 0)
                total += int(seg_rows or 0)
            return total
    except Exception:
        pass

    try:
        return int(coll.num_entities)
    except Exception:
        return 0


def _clear_milvus(
    verbose: bool = True, drop_collections: bool = False
) -> Dict[str, Any]:
    """Delete all vectors in Milvus collections

    Args:
        verbose: Whether to output logs
        drop_collections: Whether to drop physical collections and recreate (thorough clear)
    """
    stats: Dict[str, Any] = {"cleared": [], "errors": []}
    collection_classes = [
        EpisodicMemoryCollection,
        ForesightCollection,
        AtomicFactCollection,
        AgentCaseCollection,
        AgentSkillCollection,
    ]
    for cls in collection_classes:
        collection = cls()
        alias = collection.name
        try:
            related_collections: List[str] = []
            all_collections = utility.list_collections(using=collection.using)
            prefix = f"{alias}_"
            for real_name in all_collections:
                if real_name == alias or real_name.startswith(prefix):
                    related_collections.append(real_name)

            if not related_collections:
                continue

            if not drop_collections:
                for real_name in related_collections:
                    coll = Collection(name=real_name, using=collection.using)
                    coll.load()
                    before_count = coll.num_entities
                    if before_count == 0:
                        continue
                    coll.delete(expr="id != ''")
                    coll.flush()

            # Drop alias to prevent errors when dropping collection
            try:
                utility.drop_alias(alias, using=collection.using)
            except Exception:
                pass

            for real_name in related_collections:
                before_count = 0
                try:
                    coll = Collection(name=real_name, using=collection.using)
                    coll.load()
                    before_count = coll.num_entities
                except Exception:
                    before_count = 0

                utility.drop_collection(real_name, using=collection.using)
                stats["cleared"].append(
                    {"collection": real_name, "deleted": before_count, "dropped": True}
                )
                if verbose:
                    print(
                        f"      ✅ Milvus dropped collection {real_name} ({before_count} vectors)"
                    )

            # Clear class-level collection cache to ensure new instance is not old one
            cls._collection_instance = None

            # Recreate empty collection and associate alias
            try:
                collection.ensure_all()
            except Exception as ensure_exc:
                if verbose:
                    print(
                        f"      ⚠️  Recreate Milvus collection {alias} failed: {ensure_exc}"
                    )
        except Exception as exc:  # pylint: disable=broad-except
            stats["errors"].append(str(exc))
            if verbose:
                print(f"      ⚠️  Cannot clear Milvus collection {alias}: {exc}")

    return stats


async def _clear_elasticsearch(
    verbose: bool = True, rebuild_index: bool = False
) -> Dict[str, Any]:
    """Delete Elasticsearch documents related to memory, rebuild index if necessary"""
    stats: Dict[str, Any] = {"cleared": [], "errors": [], "recreated": False}
    try:
        # Get connection only
        es_client = EpisodicMemoryDoc.get_connection()

        alias_names = [
            EpisodicMemoryDoc.get_index_name(),
            ForesightDoc.get_index_name(),
            AtomicFactDoc.get_index_name(),
            AgentCaseDoc.get_index_name(),
            AgentSkillDoc.get_index_name(),
        ]

        if rebuild_index:
            for alias in alias_names:
                try:
                    existing = await es_client.indices.get_alias(
                        name=alias, ignore=[404]
                    )
                    if isinstance(existing, dict):
                        for index_name in existing.keys():
                            await es_client.indices.delete(
                                index=index_name, ignore=[400, 404]
                            )
                            stats["cleared"].append(
                                {"alias": alias, "deleted_index": index_name}
                            )
                            if verbose:
                                print(f"      ✅ Deleted index: {index_name}")
                except Exception as inner_exc:
                    stats["errors"].append(str(inner_exc))
                    if verbose:
                        print(f"      ⚠️ Delete index failed {alias}: {inner_exc}")
            for alias in alias_names:
                await es_client.indices.delete_alias(
                    index="*", name=alias, ignore=[404]
                )
            # Recreate indices using EsIndexInitializer
            from core.oxm.es.es_utils import EsIndexInitializer

            initializer = EsIndexInitializer()
            await initializer.initialize_indices(
                [EpisodicMemoryDoc, ForesightDoc, AtomicFactDoc, AgentCaseDoc, AgentSkillDoc]
            )
            stats["recreated"] = True
            if verbose:
                print("      ✅ Elasticsearch indices and aliases recreated")
            return stats

        for alias in alias_names:
            try:
                exists = await es_client.indices.exists_alias(name=alias)
                if not exists:
                    continue
                count_resp = await es_client.count(index=alias, query={"match_all": {}})
                total_docs = count_resp.get("count", 0)
                if total_docs == 0:
                    continue
                await es_client.delete_by_query(
                    index=alias,
                    query={"match_all": {}},
                    refresh=True,
                    conflicts="proceed",
                )
                stats["cleared"].append({"alias": alias, "deleted": total_docs})
                if verbose:
                    print(
                        f"      ✅ Elasticsearch {alias}: Deleted {total_docs} documents"
                    )
            except Exception as inner_exc:  # pylint: disable=broad-except
                stats["errors"].append(str(inner_exc))
                if verbose:
                    print(f"      ⚠️  Cannot clear ES {alias}: {inner_exc}")

    except Exception as exc:  # pylint: disable=broad-except
        stats["errors"].append(str(exc))
        if verbose:
            print(f"      ⚠️  Elasticsearch clear failed: {exc}")

    return stats


async def _clear_redis(verbose: bool = True) -> Dict[str, Any]:
    """Clear Redis current database"""
    stats: Dict[str, Any] = {}
    try:
        redis_provider = get_bean_by_type(RedisProvider)
        client = await redis_provider.get_client()
        await client.flushdb()
        stats["flushed_db"] = redis_provider.redis_db
        if verbose:
            print(f"      ✅ Redis DB {redis_provider.redis_db} flushed")
    except Exception as exc:  # pylint: disable=broad-except
        stats["error"] = str(exc)
        if verbose:
            print(f"      ⚠️  Redis clear failed: {exc}")
    return stats


async def clear_all_memories(
    verbose: bool = True, rebuild_es: bool = False, drop_milvus: bool = False
):
    """Clear all memory data (MongoDB, Milvus, Elasticsearch, Redis)

    Args:
        verbose: Whether to show detailed info
        rebuild_es: Whether to delete and rebuild Elasticsearch index (Default: False)
        drop_milvus: Whether to delete and rebuild Milvus physical collections (Default: False)
    """
    if verbose:
        print("\n🗑️  Clearing all memory data...")

    try:
        if verbose:
            print("   📦 Clearing MongoDB...")
        mongo_stats = await _clear_mongodb(verbose)

        if verbose:
            print("   🔍 Clearing Milvus...")
        milvus_stats = _clear_milvus(verbose, drop_collections=drop_milvus)

        if verbose:
            print("   🔎 Clearing Elasticsearch...")
        es_stats = await _clear_elasticsearch(verbose, rebuild_index=rebuild_es)

        if verbose:
            print("   💾 Clearing Redis...")
        redis_stats = await _clear_redis(verbose)

        if verbose:
            print("✅ All memory data cleared!\n")
            print("📊 Brief Statistics:")
            total_mongo_deleted = sum(mongo_stats.get("deleted", {}).values())
            print(
                f"   - MongoDB deleted docs: {total_mongo_deleted} (Database: {mongo_stats.get('database')})"
            )
            total_milvus_deleted = sum(
                item["deleted"] for item in milvus_stats.get("cleared", [])
            )
            print(f"   - Milvus deleted vectors: {total_milvus_deleted}")
            if es_stats.get("recreated"):
                print("   - Elasticsearch: Indices and aliases recreated")
            else:
                total_es_deleted = sum(
                    item["deleted"] for item in es_stats.get("cleared", [])
                )
                print(f"   - Elasticsearch deleted docs: {total_es_deleted}")
            print(f"   - Redis flushed DB: {redis_stats.get('flushed_db')}")

        return {
            "mongodb": mongo_stats,
            "milvus": milvus_stats,
            "elasticsearch": es_stats,
            "redis": redis_stats,
        }

    except Exception as e:
        print(f"❌ Error clearing data: {e}")
        import traceback

        traceback.print_exc()
        raise


async def main():
    """Entry function when running independently"""
    print("=" * 100)
    print("🗑️  Clear All Memory Data Tool")
    print("=" * 100)

    # Ensure bootstrap initialization for get_bean_by_type to work
    import sys
    import os
    from bootstrap import setup_project_context

    # Add project root to path
    sys.path.append(os.getcwd())

    await setup_project_context()

    result = await clear_all_memories(verbose=True, rebuild_es=False)

    print("\n📊 Clear Statistics:")
    mongo_total = sum(result["mongodb"].get("deleted", {}).values())
    print(f"   MongoDB deleted docs: {mongo_total}")
    milvus_total = sum(item["deleted"] for item in result["milvus"].get("cleared", []))
    print(f"   Milvus deleted vectors: {milvus_total}")
    if result["elasticsearch"].get("recreated"):
        print("   Elasticsearch: Indices and aliases recreated")
    else:
        es_total = sum(
            item["deleted"] for item in result["elasticsearch"].get("cleared", [])
        )
        print(f"   Elasticsearch deleted docs: {es_total}")
    print(f"   Redis flushed DB: {result['redis'].get('flushed_db')}")
    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
