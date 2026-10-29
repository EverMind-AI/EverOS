"""Names of the physical Milvus columns EverOS derives from a logical field.

A logical index field maps to one or more Milvus columns: a datetime is stored
as epoch milliseconds under ``<name>_ms``, a dense vector carries a
``<name>__present`` flag (Milvus cannot store a null vector), a BM25 field
feeds a ``<name>__sparse`` column, and every collection gets the private
``__pk`` key plus — when it has no vector of its own — a dummy vector, since
Milvus requires one. The layout and the codec both speak this vocabulary.
"""

from __future__ import annotations

DUMMY_VECTOR_FIELD = "_everos_dummy_vector"
DUMMY_VECTOR_DIMENSION = 2
SPARSE_SUFFIX = "__sparse"
PRESENT_SUFFIX = "__present"
PK_FIELD = "__pk"
PK_SEPARATOR = "/"
PK_MAX_LENGTH = 65_535
"""Milvus has one scalar primary key per collection, so the declared
PRIMARY_KEY columns are joined into the private ``__pk`` column. No key
column can contain the separator: app_id / project_id / owner_id / skill
name are directory segments on disk and entry_id is ``<prefix>_<date>_<seq>``;
``compose_pk`` rejects a value that does."""


def datetime_storage_field(name: str) -> str:
    return f"{name}_ms"


def present_field(name: str) -> str:
    return f"{name}{PRESENT_SUFFIX}"


def sparse_field(name: str) -> str:
    return f"{name}{SPARSE_SUFFIX}"
