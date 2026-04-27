"""
Milvus Client Factory

Provides Milvus client connection functionality based on environment variables.
"""

import os
import asyncio
from typing import Optional, Dict
from hashlib import md5

from pymilvus import MilvusClient
from core.di.decorators import component
from core.observation.logger import get_logger

logger = get_logger(__name__)


def get_milvus_config(prefix: str = "") -> dict:
    """
    Get Milvus configuration from environment variables

    Args:
        prefix: Environment variable prefix, e.g., prefix="A" reads "A_MILVUS_HOST"
               If not provided, reads "MILVUS_HOST" etc.

    Environment variables:
    - {PREFIX_}MILVUS_HOST: Milvus host, default localhost
    - {PREFIX_}MILVUS_PORT: Milvus port, default 19530
    - {PREFIX_}MILVUS_USER: Username (optional)
    - {PREFIX_}MILVUS_PASSWORD: Password (optional)
    - {PREFIX_}MILVUS_DB_NAME: Database name (optional)

    Returns:
        dict: Configuration dictionary
    """

    def _env(name: str, default: Optional[str] = None) -> str:
        if prefix:
            prefix_upper = prefix.upper()
            key = f"{prefix_upper}_{name}"
        else:
            key = name
        return os.getenv(key, default) if default is not None else os.getenv(key, "")

    host = _env("MILVUS_HOST", "localhost")
    port = int(_env("MILVUS_PORT", "19530"))

    config = {
        "uri": f"{host}:{port}" if host.startswith("http") else f"http://{host}:{port}",
        "user": _env("MILVUS_USER"),
        "password": _env("MILVUS_PASSWORD"),
        "db_name": _env("MILVUS_DB_NAME"),
    }

    logger.info("Getting Milvus config [prefix=%s]:", prefix or "default")
    logger.info("  URI: %s", config["uri"])
    logger.info("  Auth: %s", "Basic" if config["user"] else "None")
    logger.info("  Database: %s", config["db_name"] or "default")

    return config


@component(name="milvus_client_factory", primary=True)
class MilvusClientFactory:
    """
    Milvus Client Factory

    Provides Milvus client caching and management functionality based on configuration
    """

    def __init__(self):
        """Initialize Milvus client factory"""
        self._clients: Dict[str, MilvusClient] = {}
        self._lock = asyncio.Lock()
        self._default_config = None
        logger.info("MilvusClientFactory initialized")

    def get_client(
        self, uri: str, user: str = "", password: str = "", db_name: str = "", **kwargs
    ) -> MilvusClient:
        """
        Get Milvus client instance

        Args:
            uri: Milvus connection address, e.g., "http://localhost:19530"
            user: Username (optional)
            password: Password (optional)
            db_name: Database name (optional)
            alias: Connection alias, default "default"
            **kwargs: Other connection parameters

        Returns:
            MilvusClient: Milvus client instance
        """
        alias = kwargs.get("alias", None)

        client = MilvusClient(
            uri=uri, user=user, password=password, db_name=db_name, **kwargs
        )

        # Cache client
        self._clients[alias] = client
        logger.info("Milvus client created and cached: %s (alias=%s)", uri, alias)

        return client

    def get_default_client(self) -> MilvusClient:
        """
        Get default Milvus client instance based on environment variable configuration

        Returns:
            MilvusClient: Milvus client instance
        """
        # Get or create default config
        if self._default_config is None:
            self._default_config = get_milvus_config()

        config = self._default_config
        return self.get_client(
            uri=config["uri"],
            user=config["user"],
            password=config["password"],
            db_name=config["db_name"],
            alias="default",  # Default client uses "default" as cache key
        )

    def get_named_client(self, name: str) -> MilvusClient:
        """
        Get Milvus client by name

        Convention: name is used as environment variable prefix, reading config from "{name}_MILVUS_XXX".
        For example, name="A" reads "A_MILVUS_HOST", "A_MILVUS_PORT", etc.

        Args:
            name: Prefix name (environment variable prefix)

        Returns:
            MilvusClient: Milvus client instance
        """
        if name.lower() == "default":
            return self.get_default_client()

        # Get config with prefix
        config = get_milvus_config(prefix=name)
        logger.info("📋 Loading named Milvus config [name=%s]: %s", name, config["uri"])

        return self.get_client(
            uri=config["uri"],
            user=config["user"],
            password=config["password"],
            db_name=config["db_name"],
            alias=name,  # Use name as cache key
        )

    def close_all_clients(self):
        """Close all client connections"""
        for _, client in self._clients.items():
            try:
                client.close()
            except Exception as e:
                logger.error("Error closing Milvus client: %s", e)

        self._clients.clear()
        logger.info("All Milvus clients closed")
