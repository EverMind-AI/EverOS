"""
MongoDB ObjectId utility functions

Provides utility functions for ObjectId generation and conversion, usable without connecting to a database.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from bson.errors import InvalidId
from bson.objectid import ObjectId


def generate_object_id() -> Tuple[ObjectId, str, datetime]:
    """
    Generate a new MongoDB ObjectId (does not require database connection)

    Returns:
        Tuple[ObjectId, str, datetime]: Returns a tuple containing:
            - The ObjectId object itself
            - String representation of the ObjectId (suitable for API responses or frontend storage)
            - Timestamp when the ID was generated

    Example:
        >>> obj_id, id_str, gen_time = generate_object_id()
        >>> print(f"ObjectId object: {obj_id}")
        >>> print(f"String representation: {id_str}")
        >>> print(f"Generation time: {gen_time}")
    """
    new_id = ObjectId()
    return new_id, str(new_id), new_id.generation_time


def generate_object_id_str() -> str:
    """
    Generate a new MongoDB ObjectId and return its string representation

    Returns:
        str: String representation of the ObjectId (24-character hexadecimal string)

    Example:
        >>> id_str = generate_object_id_str()
        >>> print(f"ObjectId string: {id_str}")  # e.g.: "507f1f77bcf86cd799439011"
    """
    return str(ObjectId())


def build_id_filter(ids: List[str]) -> Optional[Dict[str, Any]]:
    """Build a MongoDB ``_id`` filter from a list of string IDs.

    Most IDs in this codebase are ObjectId-like strings, but some legacy
    datasets use raw string IDs.  This helper splits the input into the two
    groups and produces a single filter that matches either form, so the
    caller does not need to know which kind of IDs it is dealing with.

    Args:
        ids: List of document _id strings.  Empty / None entries are ignored.

    Returns:
        A MongoDB filter dict suitable for ``find(filter)``, or ``None`` when
        no usable IDs are provided.
    """
    if not ids:
        return None

    object_ids: List[ObjectId] = []
    raw_ids: List[str] = []
    for item_id in ids:
        if not item_id:
            continue
        try:
            object_ids.append(ObjectId(item_id))
        except (InvalidId, TypeError):
            raw_ids.append(item_id)

    clauses: List[Dict[str, Any]] = []
    if object_ids:
        clauses.append({"_id": {"$in": object_ids}})
    if raw_ids:
        clauses.append({"_id": {"$in": raw_ids}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$or": clauses}
