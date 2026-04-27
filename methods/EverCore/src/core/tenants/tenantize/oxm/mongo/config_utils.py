"""
MongoDB configuration utility functions

Provides common utility functions related to tenant-aware MongoDB configuration.
"""

import os
from typing import Optional, Dict, Any
from functools import lru_cache
from core.observation.logger import get_logger
from core.tenants.tenant_contextvar import get_current_tenant


logger = get_logger(__name__)

DEFAULT_DATABASE_NAME = "memsys"


def get_tenant_mongo_config() -> Optional[Dict[str, Any]]:
    """
    Get MongoDB configuration information for the current tenant

    Extract MongoDB-related configuration from the tenant's storage_info.
    If tenant configuration is incomplete or missing, supplement from environment variables (except for database).

    Returns:
        Optional[Dict[str, Any]]: MongoDB configuration dictionary, returns None if unable to obtain

    Fields possibly included in the configuration dictionary:
        - uri: MongoDB connection URI
        - host: MongoDB host address
        - port: MongoDB port
        - username: Username
        - password: Password
        - database: Database name (only obtained from tenant configuration, no fallback from environment variables)
        - Other connection parameters
    """
    tenant_info = get_current_tenant()
    if not tenant_info:
        logger.debug("⚠️ Unable to get tenant information, returning None")
        return None

    mongo_config = tenant_info.get_storage_info("mongodb")

    # Get environment variable configuration as fallback
    env_fallback_config = load_mongo_config_from_env()

    if not mongo_config:
        final_config = {
            "host": env_fallback_config.get("host", "localhost"),
            "port": env_fallback_config.get("port", 27017),
            "username": env_fallback_config.get("username"),
            "password": env_fallback_config.get("password"),
            "database": generate_tenant_database_name(DEFAULT_DATABASE_NAME),
        }
        logger.info(
            "✅ MongoDB information missing in tenant [%s] configuration, using environment variable configuration to complete: %s, database=%s",
            tenant_info.tenant_id,
            final_config.get("uri")
            or f"host={final_config.get('host')}:{final_config.get('port')}",
            final_config.get("database"),
        )
        return final_config

    # Compatibility logic: if tenant configuration is missing certain fields, supplement from environment variables (except database)
    # Prioritize using URI (complete connection string)
    if mongo_config.get("uri"):
        final_config = {
            "uri": mongo_config["uri"],
            # database: use if specified in tenant configuration, otherwise generate tenant-aware name
            "database": mongo_config.get("database")
            or generate_tenant_database_name(DEFAULT_DATABASE_NAME),
        }
    else:
        # Use separate connection parameters
        final_config = {
            "host": mongo_config.get("host")
            or env_fallback_config.get("host", "localhost"),
            "port": mongo_config.get("port") or env_fallback_config.get("port", 27017),
            "username": mongo_config.get("username")
            or env_fallback_config.get("username"),
            "password": mongo_config.get("password")
            or env_fallback_config.get("password"),
            # database: use if specified in tenant configuration, otherwise generate tenant-aware name
            "database": mongo_config.get("database")
            or generate_tenant_database_name(DEFAULT_DATABASE_NAME),
        }

    logger.debug(
        "✅ Retrieved MongoDB configuration from tenant [%s]: %s, database=%s",
        tenant_info.tenant_id,
        (
            "uri"
            if final_config.get("uri")
            else f"host={final_config.get('host')}:{final_config.get('port')}"
        ),
        final_config.get("database") or "(not specified)",
    )

    return final_config


def get_mongo_client_cache_key(config: Dict[str, Any]) -> str:
    """
    Generate cache key based on MongoDB configuration

    Generate a unique cache key based on connection parameters (host/port/username/password/uri),
    so that connection clients with the same configuration can reuse the same client instance.

    Args:
        config: MongoDB configuration dictionary

    Returns:
        str: Cache key
    """
    # Prioritize using URI to generate cache key
    uri = config.get("uri")
    if uri:
        # For URI, directly use it as the primary identifier
        # Note: URI may contain sensitive information, but this is just an in-memory cache key
        return f"uri:{uri}"

    # Use combination of host/port/username to generate cache key
    host = config.get("host", "localhost")
    port = config.get("port", 27017)
    username = config.get("username", "")

    # Do not include password in cache key (when passwords are the same, other parameters should also be the same)
    # Do not include database in cache key (the same client can access multiple databases)
    cache_key = f"host:{host}:port:{port}:user:{username}"

    return cache_key


def load_mongo_config_from_env() -> Dict[str, Any]:
    """
    Load MongoDB configuration from environment variables

    Read MONGODB_* environment variables, prioritize using MONGODB_URI.
    Used for loading configuration for fallback or default clients.

    Returns:
        Dict[str, Any]: Configuration dictionary containing connection information

    Environment variables:
        - MONGODB_URI: MongoDB connection URI (prioritized)
        - MONGODB_HOST: MongoDB host address (default: localhost)
        - MONGODB_PORT: MongoDB port (default: 27017)
        - MONGODB_USERNAME: Username (optional)
        - MONGODB_PASSWORD: Password (optional)
        - MONGODB_DATABASE: Database name (default: memsys)
    """
    # Prioritize using MONGODB_URI
    uri = os.getenv("MONGODB_URI")
    if uri:
        logger.info("📋 Loading configuration from environment variable MONGODB_URI")
        return {"uri": uri, "database": get_default_database_name()}

    # Read individual configuration items separately
    host = os.getenv("MONGODB_HOST", "localhost")
    port = int(os.getenv("MONGODB_PORT", "27017"))
    username = os.getenv("MONGODB_USERNAME")
    password = os.getenv("MONGODB_PASSWORD")
    database = get_default_database_name()

    logger.info(
        "📋 Loading configuration from environment variables: host=%s, port=%s, database=%s",
        host,
        port,
        database,
    )

    return {
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "database": database,
    }


@lru_cache(maxsize=1)
def get_default_database_name() -> str:
    """
    Get the default database name

    Read from environment variable MONGODB_DATABASE, return "memsys" if not set.

    Returns:
        str: Default database name
    """
    return os.getenv("MONGODB_DATABASE", DEFAULT_DATABASE_NAME)


def _base_prefixed_database_name(base_name: str) -> str:
    """Apply the base resource prefix to a database name (e.g., "memsys" → "b0001_memsys")."""
    from core.tenants.tenant_constants import get_base_resource_prefix

    return f"{get_base_resource_prefix()}_{base_name}"


def generate_tenant_database_name(base_name: str = "memsys") -> str:
    """
    Get tenant-aware database name.

    Resolution order:
    1. Tenant context exists → read from storage_info.mongodb.database
    2. No tenant context (e.g., startup) → base resource prefix + base_name

    Args:
        base_name: Base database name (e.g., "memsys")

    Returns:
        str: Resolved database name (e.g., "dev_memsys", "b0001_memsys")
    """
    try:
        tenant_info = get_current_tenant()
        if not tenant_info:
            return _base_prefixed_database_name(base_name)

        # Read database name from storage_info (set by routing layer)
        mongo_config = tenant_info.get_storage_info("mongodb")
        if mongo_config and mongo_config.get("database"):
            return mongo_config["database"]

        # No database configured in storage_info
        logger.warning(
            "Tenant [%s] storage_info has no mongodb.database configured, "
            "using base prefix. Configure database in routing layer.",
            tenant_info.tenant_id,
        )
        return _base_prefixed_database_name(base_name)

    except Exception as e:
        logger.warning("Failed to get tenant database name, using base prefix: %s", e)
        return _base_prefixed_database_name(base_name)
