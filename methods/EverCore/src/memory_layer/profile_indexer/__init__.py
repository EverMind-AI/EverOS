"""
Profile Indexer Module

Provides indexing services for user profiles into vector databases (Milvus).
"""

from .profile_indexer import ProfileIndexer, index_user_profile

__all__ = [
    "ProfileIndexer",
    "index_user_profile",
]
