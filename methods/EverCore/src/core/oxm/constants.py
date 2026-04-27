"""
OXM (Object-XML/Document Mapping) Constants

This module defines magic constants used throughout the OXM layer.
"""

# Magic constant: Query filter bypass
# When used as user_id or group_id, indicates no filtering on that field
MAGIC_ALL = "__all__"

# Magic constant: Maximum fetch limit
# Maximum number of records that can be fetched in a single query
MAX_FETCH_LIMIT = 500

# Magic constant: Maximum retrieve limit
# Maximum number of records that can be retrieved in a single query
MAX_RETRIEVE_LIMIT = 500

# Export all constants
__all__ = [
    "MAGIC_ALL",
    "MAX_FETCH_LIMIT",
    "MAX_RETRIEVE_LIMIT",
]
