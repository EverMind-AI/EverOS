"""The Milvus connection of the derived index, and collection naming.

:class:`MilvusConnection` is built by the application bootstrap and handed to
the Milvus backend and its repositories; nothing reaches for a global client.
Building it opens nothing -- the client connects on first use.
"""

from __future__ import annotations

import re

from pymilvus import MilvusClient

from everos.config import MilvusSettings
from everos.core.errors import ConfigurationError, IndexSchemaMismatchError
from everos.core.observability.logging import get_logger

logger = get_logger(__name__)


class MilvusSchemaMismatchError(IndexSchemaMismatchError):
    """Raised when an existing Milvus collection does not match EverOS."""


class MilvusConfigurationError(ConfigurationError):
    """Raised when the remote Milvus profile is incomplete or invalid."""


def collection_name(table_name: str, settings: MilvusSettings) -> str:
    """Return the configured Milvus collection name for an EverOS table."""
    prefix = _sanitize_name_part(settings.collection_prefix)
    base = _sanitize_name_part(table_name)
    name = f"{prefix}_{base}" if prefix else base
    if not re.match(r"^[A-Za-z_]", name):
        name = f"_{name}"
    return name


class MilvusConnection:
    """The client of one remote Milvus deployment.

    Args:
        settings: The ``[milvus]`` section.
    """

    def __init__(self, settings: MilvusSettings) -> None:
        self._settings = settings
        self._client: MilvusClient | None = None

    @property
    def settings(self) -> MilvusSettings:
        """The ``[milvus]`` section this connection was built from."""
        return self._settings

    async def client(self) -> MilvusClient:
        """Return the client, creating it on first use."""
        if self._client is None:
            uri = _resolve_uri(self._settings)
            db_name = self._settings.db_name or ""
            self._client = MilvusClient(
                uri=uri, token=_secret(self._settings.token), db_name=db_name
            )
            logger.info(
                "milvus_connection_opened",
                uri=uri,
                db_name=db_name or None,
                consistency_level=self._settings.consistency_level,
            )
        return self._client

    async def close(self) -> None:
        """Close the client. Idempotent."""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("milvus_connection_closed")


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
    "MilvusConnection",
    "MilvusSchemaMismatchError",
    "collection_name",
]
