"""
Elasticsearch tenant configuration utility functions

This module provides utility functions related to tenant Elasticsearch configuration, used to extract ES configuration from tenant information.
"""

import os
from typing import Optional, Dict, Any
from hashlib import md5

from core.observation.logger import get_logger
from core.tenants.tenant_contextvar import get_current_tenant


logger = get_logger(__name__)


def get_tenant_es_config() -> Optional[Dict[str, Any]]:
    """
    Get Elasticsearch configuration from the current tenant context

    Extract Elasticsearch-related configuration from the storage_info of tenant information.
    If tenant configuration is incomplete or missing, supplement it from environment variables.

    Configuration structure example:
    {
        "hosts": ["http://localhost:9200"],
        "username": "elastic",
        "password": "password",
        "api_key": None,
        "timeout": 120,
        "verify_certs": False
    }

    Returns:
        Elasticsearch configuration dictionary, return None if not exists

    Examples:
        >>> config = get_tenant_es_config()
        >>> if config:
        ...     print(f"ES Hosts: {config['hosts']}")
    """
    tenant_info = get_current_tenant()
    if not tenant_info:
        logger.debug(
            "⚠️ Tenant context is not set, unable to get tenant Elasticsearch configuration"
        )
        return None

    # Get ES configuration from tenant's storage_info
    # Support multiple configuration key names: elasticsearch, es_config, es
    es_config = tenant_info.get_storage_info("elasticsearch")
    if es_config is None:
        es_config = tenant_info.get_storage_info("es_config")
    if es_config is None:
        es_config = tenant_info.get_storage_info("es")

    # Get environment variable configuration as fallback
    env_fallback_config = load_es_config_from_env()

    if not es_config:
        # No ES information at all in tenant configuration, use environment variable configuration
        final_config = {
            "hosts": env_fallback_config.get("hosts", ["http://localhost:9200"]),
            "username": env_fallback_config.get("username"),
            "password": env_fallback_config.get("password"),
            "api_key": env_fallback_config.get("api_key"),
            "timeout": env_fallback_config.get("timeout", 120),
            "verify_certs": env_fallback_config.get("verify_certs", False),
        }
        logger.info(
            "✅ Elasticsearch information missing in tenant [%s] configuration, using environment variable configuration to complete: hosts=%s",
            tenant_info.tenant_id,
            final_config["hosts"],
        )
        return final_config

    # Compatibility logic: if tenant configuration is missing certain fields, supplement from environment variables
    # Handle multiple formats of hosts field
    tenant_hosts = es_config.get("hosts")
    if tenant_hosts is None:
        # Try to build from host/port
        tenant_host = es_config.get("host")
        tenant_port = es_config.get("port", 9200)
        if tenant_host:
            tenant_hosts = [f"http://{tenant_host}:{tenant_port}"]

    final_config = {
        "hosts": tenant_hosts
        or env_fallback_config.get("hosts", ["http://localhost:9200"]),
        "username": es_config.get("username") or env_fallback_config.get("username"),
        "password": es_config.get("password") or env_fallback_config.get("password"),
        "api_key": es_config.get("api_key") or env_fallback_config.get("api_key"),
        "timeout": es_config.get("timeout") or env_fallback_config.get("timeout", 120),
        "verify_certs": es_config.get(
            "verify_certs", env_fallback_config.get("verify_certs", False)
        ),
    }

    logger.debug(
        "✅ Retrieved Elasticsearch configuration from tenant [%s]: hosts=%s",
        tenant_info.tenant_id,
        final_config["hosts"],
    )

    return final_config


def get_es_connection_cache_key(config: Dict[str, Any]) -> str:
    """
    Generate cache key based on Elasticsearch connection configuration

    Use the hash value of connection parameters (hosts, authentication info) as the cache key.
    Also used as the alias for elasticsearch-dsl connections.

    Args:
        config: Elasticsearch connection configuration dictionary

    Returns:
        Cache key string (MD5 hash value)

    Examples:
        >>> config = {"hosts": ["http://localhost:9200"], "username": "elastic", "password": "pwd"}
        >>> cache_key = get_es_connection_cache_key(config)
    """
    # Handle hosts
    hosts = config.get("hosts", [])
    if isinstance(hosts, list):
        hosts_str = ",".join(sorted(hosts))
    else:
        hosts_str = str(hosts)

    # Handle authentication info
    auth_str = ""
    api_key = config.get("api_key")
    username = config.get("username")
    password = config.get("password")

    if api_key:
        # Use first 8 characters of api_key as identifier
        auth_str = f"api_key:{api_key[:8]}..."
    elif username and password:
        # Use md5 of username and password as identifier
        auth_str = f"basic:{username}:{md5(password.encode()).hexdigest()[:8]}"
    elif username:
        # When only username is present, use username only
        auth_str = f"basic:{username}"

    key_content = f"{hosts_str}:{auth_str}"
    return md5(key_content.encode()).hexdigest()[:16]


def load_es_config_from_env() -> Dict[str, Any]:
    """
    Load default Elasticsearch configuration from environment variables

    Read the following environment variables:
    - ES_HOSTS: Elasticsearch host list, comma-separated (takes precedence)
    - ES_HOST: Elasticsearch host address, default localhost
    - ES_PORT: Elasticsearch port, default 9200
    - ES_USERNAME: Username (optional)
    - ES_PASSWORD: Password (optional)
    - ES_API_KEY: API key (optional)
    - ES_TIMEOUT: Timeout in seconds, default 120
    - ES_VERIFY_CERTS: Whether to verify certificates, default false

    Returns:
        Elasticsearch configuration dictionary

    Examples:
        >>> config = load_es_config_from_env()
        >>> print(f"ES Hosts: {config['hosts']}")
    """
    # Get host information
    es_hosts_str = os.getenv("ES_HOSTS")
    if es_hosts_str:
        # ES_HOSTS already contains full URL (https://host:port), use directly
        es_hosts = [host.strip() for host in es_hosts_str.split(",")]
    else:
        # Fall back to single host configuration
        es_host = os.getenv("ES_HOST", "localhost")
        es_port = int(os.getenv("ES_PORT", "9200"))
        es_hosts = [f"http://{es_host}:{es_port}"]

    config = {
        "hosts": es_hosts,
        "username": os.getenv("ES_USERNAME"),
        "password": os.getenv("ES_PASSWORD"),
        "api_key": os.getenv("ES_API_KEY"),
        "timeout": int(os.getenv("ES_TIMEOUT", "120")),
        "verify_certs": os.getenv("ES_VERIFY_CERTS", "false").lower() == "true",
    }

    logger.debug(
        "Loaded default Elasticsearch configuration from environment variables: hosts=%s",
        config["hosts"],
    )

    return config


def _base_prefixed_index_name(original_name: str) -> str:
    """Apply the base resource prefix to an index name (e.g., "v1_memories" → "b0001_v1_memories")."""
    from core.tenants.tenant_constants import get_base_resource_prefix

    return f"{get_base_resource_prefix()}_{original_name}"


def get_tenant_aware_index_name(original_name: str) -> str:
    """
    Get tenant-aware index name.

    Resolution order:
    1. Tenant context exists → read from storage_info.elasticsearch.index_prefix
    2. No tenant context (e.g., startup) → base resource prefix + original_name

    Args:
        original_name: Original index name (e.g., "v1_atomic_fact_record")

    Returns:
        str: Resolved index name (e.g., "dev-v1_atomic_fact_record", "b0001-v1_atomic_fact_record")
    """
    try:
        tenant_info = get_current_tenant()
        if not tenant_info:
            return _base_prefixed_index_name(original_name)

        # Read index_prefix from storage_info (set by routing layer)
        es_config = tenant_info.get_storage_info("elasticsearch")
        if es_config is None:
            es_config = tenant_info.get_storage_info("es_config")
        if es_config is None:
            es_config = tenant_info.get_storage_info("es")

        if es_config and es_config.get("index_prefix"):
            return f"{es_config['index_prefix']}_{original_name}"

        # No index_prefix configured
        logger.warning(
            "Tenant [%s] storage_info has no elasticsearch.index_prefix configured, "
            "using base prefix. Configure index_prefix in routing layer.",
            tenant_info.tenant_id,
        )
        return _base_prefixed_index_name(original_name)

    except Exception as e:
        logger.warning(
            "Failed to get tenant-aware index name, using base prefix: %s", e
        )
        return _base_prefixed_index_name(original_name)
