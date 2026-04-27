"""
⚠️ DANGER: Completely reset all database structures ⚠️

This script performs destructive operations:
1. MongoDB: Drops the entire memsys database
2. Milvus: Drops all collections
3. Elasticsearch: Deletes all indices containing 'memsys' (old, new, aliases)
4. Redis: Flushes the database

Use cases:
- Environment data is messy and needs a clean slate
- Schema fixed, need to rebuild index structures

Must restart services after use to re-initialize structures.
"""

import asyncio
import sys
import os

# Ensure project modules can be imported
sys.path.append(os.getcwd())

from pymilvus import utility, connections
from core.di import get_bean_by_type
from core.component.redis_provider import RedisProvider
from core.component.mongodb_client_factory import MongoDBClientFactory
from core.component.elasticsearch_client_factory import ElasticsearchClientFactory
from bootstrap import setup_project_context


async def reset_mongodb():
    print("🔥 [MongoDB] Deleting database...")
    try:
        factory = get_bean_by_type(MongoDBClientFactory)
        client_wrapper = await factory.get_default_client()
        async_client = client_wrapper.client
        db_name = client_wrapper.database.name
        # Directly drop the entire database
        await async_client.drop_database(db_name)
        print(f"   ✅ Database deleted: {db_name}")
    except Exception as e:
        print(f"   ❌ MongoDB reset failed: {e}")


def reset_milvus():
    print("🔥 [Milvus] Dropping all collections...")
    try:
        # Connection config
        milvus_host = os.getenv('MILVUS_HOST', 'localhost')
        milvus_port = int(os.getenv('MILVUS_PORT', '19530'))
        connections.connect(host=milvus_host, port=milvus_port)

        collections = utility.list_collections()
        if not collections:
            print("   ⚪ No collections found")
            return

        for name in collections:
            utility.drop_collection(name)
            print(f"   ✅ Dropped collection: {name}")

    except Exception as e:
        print(f"   ❌ Milvus reset failed: {e}")


async def reset_elasticsearch():
    print("🔥 [Elasticsearch] Deleting all related indices...")
    try:
        factory = get_bean_by_type(ElasticsearchClientFactory)
        client_wrapper = await factory.register_default_client()
        es = client_wrapper.async_client

        # Delete all indices containing memsys
        target_pattern = "*memsys*"

        # 1. Get specific index list
        indices_resp = await es.cat.indices(index=target_pattern, format="json")

        if not indices_resp:
            print(f"   ⚪ No indices found matching '{target_pattern}'")
            return

        # 2. Extract index names
        index_names = [item['index'] for item in indices_resp]
        count = len(index_names)

        # 3. Explicitly delete these indices
        # Use comma-separated string or list
        await es.indices.delete(index=list(index_names), ignore=[404])
        print(f"   ✅ Deleted {count} indices: {', '.join(index_names[:3])}...")

    except Exception as e:
        print(f"   ❌ Elasticsearch reset failed: {e}")


async def reset_redis():
    print("🔥 [Redis] Flushing DB...")
    try:
        provider = get_bean_by_type(RedisProvider)
        client = await provider.get_client()
        await client.flushdb()
        print("   ✅ Redis flushed")
    except Exception as e:
        print(f"   ❌ Redis reset failed: {e}")


async def main():
    print("\n" + "=" * 60)
    print("🧨 Database Complete Reset Tool 🧨")
    print("=" * 60 + "\n")

    await setup_project_context()

    await reset_mongodb()
    reset_milvus()
    await reset_elasticsearch()
    await reset_redis()

    print("\n" + "=" * 60)
    print("✨ Reset complete! Please restart services immediately to rebuild index structures ✨")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Load .env
    from dotenv import load_dotenv

    load_dotenv()

    asyncio.run(main())
