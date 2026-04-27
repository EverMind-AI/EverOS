#!/usr/bin/env python3
"""
Database Data Viewer Tool

Usage:
    python debug_view_databases.py                    # View all databases overview
    python debug_view_databases.py --mongo            # View MongoDB only
    python debug_view_databases.py --milvus           # View Milvus only
    python debug_view_databases.py --es               # View Elasticsearch only
    python debug_view_databases.py --detail           # Show detailed data (including samples)
"""

import asyncio
import os
import sys
from common_utils.datetime_utils import get_now_with_timezone
from pymongo import AsyncMongoClient
from pymilvus import connections, Collection, utility
from elasticsearch import AsyncElasticsearch
import json


class Colors:
    """Terminal Colors"""

    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_section(title: str):
    """Print Section Title"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}\n")


def print_subsection(title: str):
    """Print Subsection Title"""
    print(f"\n{Colors.BOLD}{Colors.GREEN}--- {title} ---{Colors.END}")


def print_success(text: str):
    """Print Success Message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")


def print_warning(text: str):
    """Print Warning Message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")


def print_error(text: str):
    """Print Error Message"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")


async def check_mongodb(detail: bool = False):
    """Check MongoDB Data"""
    print_section("MongoDB Data")

    try:
        # Connection Config (No Auth)
        mongo_host = os.getenv('MONGO_HOST', 'localhost')
        mongo_port = int(os.getenv('MONGO_PORT', '27017'))

        client = AsyncMongoClient(f'mongodb://{mongo_host}:{mongo_port}')
        db = client['memsys']

        print_success(f"Connected to MongoDB: {mongo_host}:{mongo_port}")

        # Dynamically get all collections
        collection_names = await db.list_collection_names()
        collection_names.sort()

        if not collection_names:
            print_warning("No collections in database")
            return

        for collection_name in collection_names:
            # Skip system collections
            if collection_name.startswith("system."):
                continue

            print_subsection(f"Collection: {collection_name}")

            collection = db[collection_name]
            total = await collection.count_documents({})

            if total == 0:
                print_warning(f"No data")
                continue

            print(f"Total: {Colors.BOLD}{total}{Colors.END} items")

            # Count by user_id (if exists)
            try:
                pipeline = [
                    {'$group': {'_id': '$user_id', 'count': {'$sum': 1}}},
                    {'$sort': {'count': -1}},
                ]
                # Check for user_id field (simple sampling)
                sample = await collection.find_one()
                if sample and 'user_id' in sample:
                    cursor = await collection.aggregate(pipeline)
                    result = await cursor.to_list(length=None)

                    if result:
                        print("\nGroup by user_id:")
                        for item in result[:10]:  # Show top 10 only
                            user_id = item['_id'] if item['_id'] else '(Empty/Group)'
                            print(f"  - {user_id}: {item['count']} items")
            except Exception:
                pass  # Ignore aggregation errors

            # Show samples
            if detail:
                print("\nSample data:")
                cursor = collection.find().limit(2)
                async for doc in cursor:
                    # Remove overly long fields
                    doc.pop('_id', None)
                    doc.pop('vector', None)
                    doc.pop('embedding', None)
                    doc.pop('original_data', None)

                    # Limit field length
                    for key, value in doc.items():
                        if isinstance(value, str) and len(value) > 100:
                            doc[key] = value[:100] + '...'

                    print(
                        f"  {json.dumps(doc, ensure_ascii=False, indent=2, default=str)}"
                    )

        client.close()

    except Exception as e:
        print_error(f"MongoDB connection failed: {e}")


def _get_milvus_row_count(collection_name: str, collection: Collection) -> int:
    """
    Get real-time entity count from Milvus.

    Prioritize utility.get_collection_stats (if available),
    then try utility.get_query_segment_info, and finally fallback to num_entities.
    """
    get_stats = getattr(utility, "get_collection_stats", None)
    if callable(get_stats):
        stats_info = get_stats(collection_name)
        if isinstance(stats_info, dict):
            return int(stats_info.get("row_count", 0))

    # For some old versions, get_collection_stats is not available, so we fall back to summing the segment row count
    segment_infos = utility.get_query_segment_info(collection_name)
    if segment_infos:
        total_rows = 0
        for seg in segment_infos:
            num_rows = getattr(seg, "num_rows", None)
            if num_rows is None:
                num_rows = getattr(seg, "row_count", 0)
            total_rows += int(num_rows or 0)
        return total_rows

    # Final fallback: return num_entities (possibly containing deleted data)
    return collection.num_entities


def check_milvus(detail: bool = False):
    """Check Milvus Data"""
    print_section("Milvus Data")

    try:
        # Connection Config
        milvus_host = os.getenv('MILVUS_HOST', 'localhost')
        milvus_port = int(os.getenv('MILVUS_PORT', '19530'))

        connections.connect(host=milvus_host, port=milvus_port)
        print_success(f"Connected to Milvus: {milvus_host}:{milvus_port}")

        # Dynamically get all collections
        all_collections = utility.list_collections()
        all_collections.sort()

        if not all_collections:
            print_warning("No collections in Milvus")
            return

        for collection_name in all_collections:
            print_subsection(f"Collection: {collection_name}")

            collection = Collection(collection_name)
            collection.load()

            stats = _get_milvus_row_count(collection_name, collection)
            print(f"Current Entities: {Colors.BOLD}{stats}{Colors.END}")

            # Query samples
            if detail and stats > 0:
                print("\nSample Data:")

                # Original query logic depends on 'id' field, will fail if PK is not 'id'
                # We temporarily only keep counting, or only query samples for known collections
                known_pk_map = {
                    'episodic_memory_memsys': 'id',
                    'foresight_memsys': 'id',
                    'atomic_fact_memsys': 'id',
                }

                pk_field = known_pk_map.get(collection_name)
                if pk_field:
                    try:
                        results = collection.query(
                            expr=f"{pk_field} >= 0",
                            output_fields=[
                                "user_id",
                                "timestamp",
                                "episode",
                                "atomic_fact",
                                "content",
                                "foresight",
                            ],
                            limit=2,
                        )
                        for result in results:
                            for key, value in result.items():
                                if isinstance(value, str) and len(value) > 100:
                                    result[key] = value[:100] + '...'
                            print(
                                f"  {json.dumps(result, ensure_ascii=False, indent=2, default=str)}"
                            )
                    except Exception as e:
                        print(f"  Query sample failed: {e}")
                else:
                    print("  (No primary key configured, skipping sample query)")

        connections.disconnect("default")

    except Exception as e:
        print_error(f"Milvus connection failed: {e}")


async def check_elasticsearch(detail: bool = False):
    """Check Elasticsearch Data"""
    print_section("Elasticsearch Data")

    try:
        # Connection Config
        es_host = os.getenv('ES_HOSTS', 'http://localhost:19200')

        # Use HTTP direct query to avoid version compatibility issues
        import aiohttp

        async with aiohttp.ClientSession() as session:
            print_success(f"Connected to Elasticsearch: {es_host}")

            # Get all indices
            async with session.get(f"{es_host}/_cat/indices?format=json") as resp:
                if resp.status != 200:
                    print_error(f"Failed to get index list: {resp.status}")
                    return
                indices = await resp.json()

            # Show all indices (no filter)
            relevant_indices = indices

            if not relevant_indices:
                print_warning("No relevant indices found")
                return

            # Sort by index name
            relevant_indices.sort(key=lambda x: x['index'])

            for idx_info in relevant_indices:
                idx_name = idx_info['index']
                # Skip system indices
                if idx_name.startswith('.'):
                    continue

                doc_count = int(idx_info['docs.count'])

                print_subsection(f"{idx_name} ({doc_count} items)")

                if doc_count == 0:
                    print_warning("No data")
                    continue

                # Use aggregation query to count all types
                try:
                    aggs_body = {
                        "size": 0,
                        "aggs": {"types": {"terms": {"field": "type", "size": 50}}},
                    }
                    async with session.post(
                        f"{es_host}/{idx_name}/_search",
                        json=aggs_body,
                        headers={"Content-Type": "application/json"},
                    ) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            buckets = (
                                result.get('aggregations', {})
                                .get('types', {})
                                .get('buckets', [])
                            )
                            if buckets:
                                for bucket in buckets:
                                    print(
                                        f"  - type={bucket['key']}: {bucket['doc_count']} items"
                                    )
                            else:
                                print(f"  - (No type field classification)")
                except Exception as e:
                    print_warning(f"Aggregation statistics failed: {e}")

                # Show samples
                if detail and doc_count > 0:
                    print("\nSample data:")
                    try:
                        query_body = {"size": 2}
                        async with session.post(
                            f"{es_host}/{idx_name}/_search",
                            json=query_body,
                            headers={"Content-Type": "application/json"},
                        ) as resp:
                            if resp.status == 200:
                                result = await resp.json()

                                for hit in result['hits']['hits']:
                                    src = hit['_source']

                                    # Limit field length
                                    for key, value in src.items():
                                        if isinstance(value, str) and len(value) > 100:
                                            src[key] = value[:100] + '...'

                                    print(
                                        f"  {json.dumps(src, ensure_ascii=False, indent=2, default=str)}"
                                    )
                    except Exception as e:
                        print_warning(f"Query sample failed: {e}")

    except Exception as e:
        print_error(f"Elasticsearch connection failed: {e}")


async def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='View database data')
    parser.add_argument('--mongo', action='store_true', help='View MongoDB only')
    parser.add_argument('--milvus', action='store_true', help='View Milvus only')
    parser.add_argument('--es', action='store_true', help='View Elasticsearch only')
    parser.add_argument(
        '--detail', action='store_true', help='Show detailed data (including samples)'
    )

    args = parser.parse_args()

    # Load .env file (if exists)
    if os.path.exists('.env'):
        from dotenv import load_dotenv

        load_dotenv()

    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}Database Data Viewer Tool{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.END}")
    print(
        f"\n{Colors.CYAN}Time: {get_now_with_timezone().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}"
    )

    # If no database specified, view all
    if not (args.mongo or args.milvus or args.es):
        args.mongo = args.milvus = args.es = True

    if args.mongo:
        await check_mongodb(args.detail)

    if args.milvus:
        check_milvus(args.detail)

    if args.es:
        await check_elasticsearch(args.detail)

    print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}✅ Completed{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'='*80}{Colors.END}\n")


if __name__ == '__main__':
    asyncio.run(main())
