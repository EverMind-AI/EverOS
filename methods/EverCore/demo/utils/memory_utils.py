"""Shared Utility Module - For Memory Extraction and Chat System

This module provides common utility functions shared by extract_memory.py and chat_with_memory.py.

Key Features:
- MongoDB connection and initialization
- MemCell queries
- Time serialization tools

V4 Update:
- Removed custom retrieval strategies (using API in src)
- Retained basic utility functions
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from pymongo import AsyncMongoClient
from beanie import init_beanie

# Import document models from the project
from infra_layer.adapters.out.persistence.document.memory.memcell import (
    MemCell as DocMemCell,
)
from demo.config import MongoDBConfig

# Import language utility function from common_utils
from common_utils.language_utils import get_prompt_language


# ============================================================================
# MongoDB Tools
# ============================================================================


async def ensure_mongo_beanie_ready(mongo_config: MongoDBConfig) -> None:
    """Initialize MongoDB and Beanie Connection

    Args:
        mongo_config: MongoDB configuration object

    Raises:
        Exception: If connection fails
    """
    # Set environment variable for Beanie use
    os.environ["MONGODB_URI"] = mongo_config.uri

    # Create MongoDB client and test connection
    client = AsyncMongoClient(mongo_config.uri)
    try:
        await client.admin.command('ping')
        print(f"[MongoDB] ✅ Connected: {mongo_config.database}")
    except Exception as e:
        print(f"[MongoDB] ❌ Connection failed: {e}")
        raise

    # Initialize Beanie document models
    await init_beanie(
        database=client[mongo_config.database], document_models=[DocMemCell]
    )


async def query_all_groups_from_mongodb() -> List[Dict[str, Any]]:
    """Query all group IDs and their memory counts

    Uses aggregation pipeline to count MemCells per group.

    Returns:
        List of groups, format: [{"group_id": "xxx", "memcell_count": 76}, ...]
    """
    # Use aggregation pipeline to count memories per group
    pipeline = [
        {"$match": {"group_id": {"$ne": None}}},  # Filter records without group_id
        {"$group": {"_id": "$group_id", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},  # Sort by group_id
    ]

    # Get PyMongo AsyncCollection for aggregation
    # get_pymongo_collection() returns AsyncCollection in Beanie (async)
    collection = DocMemCell.get_pymongo_collection()
    cursor = await collection.aggregate(pipeline)
    results = await cursor.to_list(length=None)

    groups = []
    for result in results:
        groups.append({"group_id": result["_id"], "memcell_count": result["count"]})

    return groups


async def query_memcells_by_group_and_time(
    group_id: str, start_date: datetime, end_date: datetime
) -> List[DocMemCell]:
    """Query MemCells by Group and Time Range

    Args:
        group_id: Group ID
        start_date: Start date
        end_date: End date

    Returns:
        List of MemCell document objects
    """
    memcells = (
        await DocMemCell.find(
            {"group_id": group_id, "timestamp": {"$gte": start_date, "$lt": end_date}}
        )
        .sort("timestamp")
        .to_list()
    )

    return memcells


# ============================================================================
# Time Serialization Tools
# ============================================================================


def serialize_datetime(obj: Any) -> Any:
    """Recursively serialize datetime objects to ISO format strings

    Args:
        obj: Object to serialize (can be any type)

    Returns:
        Serialized object
    """
    # If already string, return directly (avoid processing already serialized timestamps)
    if isinstance(obj, str):
        return obj
    # Convert datetime object to ISO string
    elif isinstance(obj, datetime):
        return obj.isoformat()
    # Recursively process dict
    elif isinstance(obj, dict):
        return {k: serialize_datetime(v) for k, v in obj.items()}
    # Recursively process list
    elif isinstance(obj, list):
        return [serialize_datetime(item) for item in obj]
    # Process object (convert __dict__)
    elif hasattr(obj, '__dict__'):
        return serialize_datetime(obj.__dict__)
    # Return other types directly
    else:
        return obj
