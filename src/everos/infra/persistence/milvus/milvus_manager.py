"""Milvus connection and collection management for the derived index."""

from __future__ import annotations

import re

from pymilvus import MilvusClient

from everos.config import MilvusSettings, load_settings
from everos.core.errors import ConfigurationError
from everos.core.observability.logging import get_logger

logger = get_logger(__name__)

_client: MilvusClient | None = None


class MilvusSchemaMismatchError(RuntimeError):
    """Raised when an existing Milvus collection does not match EverOS."""


class MilvusConfigurationError(ConfigurationError):
    """Raised when the remote Milvus profile is incomplete or invalid."""


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
    from .repository import MilvusRepoBase

    MilvusRepoBase._reset_collection_cache()


def _resolve_uri(settings: MilvusSettings) -> str:
    uri = settings.uri.strip()
    if not uri:
        raise MilvusConfigurationError(
            "[index] backend = 'milvus' requires EVEROS_MILVUS__URI (or "
            "[milvus] uri) pointing to Milvus Server or Zilliz Cloud; "
            "embedded Milvus Lite is not supported"
        )
    scheme, separator, _rest = uri.partition("://")
    if not separator or scheme.lower() not in {"http", "https"}:
        raise MilvusConfigurationError(
            "[milvus] uri must be a remote http(s) endpoint for Milvus Server "
            "or Zilliz Cloud, not a local database path; embedded Milvus Lite "
            "is not supported"
        )
    return uri


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
    "MilvusConfigurationError",
    "MilvusSchemaMismatchError",
    "collection_name",
    "dispose_connection",
    "get_client",
]
