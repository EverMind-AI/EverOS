"""Milvus connection and collection management for the derived index."""

from __future__ import annotations

import re

from pymilvus import MilvusClient

from everos.config import MilvusSettings, load_settings
from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot

logger = get_logger(__name__)

_client: MilvusClient | None = None


class MilvusSchemaMismatchError(RuntimeError):
    """Raised when an existing Milvus collection does not match EverOS."""


def collection_name(table_name: str, settings: MilvusSettings | None = None) -> str:
    """Return the configured Milvus collection name for an EverOS table."""
    cfg = settings or load_settings().milvus
    prefix = _sanitize_name_part(cfg.collection_prefix)
    base = _sanitize_name_part(table_name)
    name = f"{prefix}_{base}" if prefix else base
    if not re.match(r"^[A-Za-z_]", name):
        name = f"_{name}"
    return name


async def get_client() -> MilvusClient:
    """Return the process-wide MilvusClient, creating it lazily."""
    global _client
    if _client is None:
        settings = load_settings().milvus
        uri = _resolve_uri(settings)
        token = _secret(settings.token)
        db_name = settings.db_name or ""
        _client = MilvusClient(uri=uri, token=token, db_name=db_name)
        logger.info(
            "milvus_connection_opened",
            uri=uri,
            db_name=db_name or None,
            consistency_level=settings.consistency_level,
        )
    return _client


async def dispose_connection() -> None:
    """Close the process-wide Milvus client."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
        logger.info("milvus_connection_closed")


def _resolve_uri(settings: MilvusSettings) -> str:
    if settings.uri:
        return settings.uri
    memory_root = MemoryRoot.default()
    memory_root.milvus_dir.mkdir(parents=True, exist_ok=True)
    return str(memory_root.milvus_db)


def _secret(value: object | None) -> str:
    if value is None:
        return ""
    getter = getattr(value, "get_secret_value", None)
    if callable(getter):
        return getter() or ""
    return str(value)


def _sanitize_name_part(value: str) -> str:
    clean = re.sub(r"\W+", "_", value.strip())
    return clean.strip("_")


__all__ = [
    "MilvusSchemaMismatchError",
    "collection_name",
    "dispose_connection",
    "get_client",
]
