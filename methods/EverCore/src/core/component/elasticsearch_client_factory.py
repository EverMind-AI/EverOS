"""
Elasticsearch Client Factory

Provides Elasticsearch client caching and management functionality based on environment variables.
"""

import os
import asyncio
from typing import Dict, Optional, List, Any
from hashlib import md5
from elasticsearch import AsyncElasticsearch
from elasticsearch.dsl.async_connections import connections as async_connections

from core.di.decorators import component
from core.observation.logger import get_logger

logger = get_logger(__name__)


def get_default_es_config() -> Dict[str, Any]:
    """
    Get default Elasticsearch configuration based on environment variables

    Environment variables:
    - ES_HOST: Elasticsearch host, default localhost
    - ES_PORT: Elasticsearch port, default 9200
    - ES_HOSTS: Elasticsearch host list, comma-separated, takes precedence over ES_HOST
    - ES_USERNAME: Username
    - ES_PASSWORD: Password
    - ES_API_KEY: API key
    - ES_TIMEOUT: Timeout (seconds), default 120

    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    # Get host information
    es_hosts_str = os.getenv("ES_HOSTS")
    if es_hosts_str:
        # ES_HOSTS already contains full URL (https://host:port), use directly
        es_hosts = [host.strip() for host in es_hosts_str.split(",")]
    else:
        # Fallback to single host configuration
        es_host = os.getenv("ES_HOST", "localhost")
        es_port = int(os.getenv("ES_PORT", "9200"))
        es_hosts = [f"http://{es_host}:{es_port}"]

    # Authentication information
    es_username = os.getenv("ES_USERNAME")
    es_password = os.getenv("ES_PASSWORD")
    es_api_key = os.getenv("ES_API_KEY")

    # Connection parameters
    es_timeout = int(os.getenv("ES_TIMEOUT", "120"))
    es_verify_certs = os.getenv("ES_VERIFY_CERTS", "false").lower() == "true"

    config = {
        "hosts": es_hosts,
        "timeout": es_timeout,
        "username": es_username,
        "password": es_password,
        "api_key": es_api_key,
        "verify_certs": es_verify_certs,
    }

    logger.info("Getting default Elasticsearch config:")
    logger.info("  Hosts: %s", es_hosts)
    logger.info("  Timeout: %s seconds", es_timeout)
    logger.info(
        "  Auth: %s", "API Key" if es_api_key else ("Basic" if es_username else "None")
    )

    return config


def get_cache_key(
    hosts: List[str],
    username: Optional[str] = None,
    password: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Generate cache key (also used as alias for elasticsearch-dsl connections)
    Generate unique identifier based on hosts and authentication info

    Args:
        hosts: Elasticsearch host list
        username: Username
        password: Password
        api_key: API key

    Returns:
        str: Cache key
    """
    hosts_str = ",".join(sorted(hosts))
    auth_str = ""
    if api_key:
        # Use first 8 characters of api_key as identifier
        auth_str = f"api_key:{api_key[:8]}..."
    elif username and password:
        # Use md5 of username and password as identifier
        auth_str = f"basic:{username}:{md5(password.encode()).hexdigest()[:8]}"
    elif username:
        # When only username is provided, use username only
        auth_str = f"basic:{username}"

    key_content = f"{hosts_str}:{auth_str}"
    return md5(key_content.encode()).hexdigest()


class ElasticsearchClientWrapper:
    """Elasticsearch client wrapper"""

    def __init__(self, async_client: AsyncElasticsearch, hosts: List[str]):
        self.async_client = async_client
        self.hosts = hosts

    async def test_connection(self) -> bool:
        """Test connection"""
        try:
            await self.async_client.ping()
            logger.info("✅ Elasticsearch connection test successful: %s", self.hosts)
            return True
        except Exception as e:
            logger.error(
                "❌ Elasticsearch connection test failed: %s, error: %s", self.hosts, e
            )
            return False

    async def close(self):
        """Close connection"""
        try:
            if self.async_client:
                await self.async_client.close()
            logger.info("🔌 Elasticsearch connection closed: %s", self.hosts)
        except Exception as e:
            logger.error("Error closing Elasticsearch connection: %s", e)


@component(name="elasticsearch_client_factory")
class ElasticsearchClientFactory:
    """
    Elasticsearch client factory
    ### AsyncElasticsearch is stateful, so the same instance can be used in multiple places ###

    Provides Elasticsearch client caching and management functionality based on configuration
    """

    def __init__(self):
        """Initialize Elasticsearch client factory"""
        self._clients: Dict[str, ElasticsearchClientWrapper] = {}
        self._lock = asyncio.Lock()
        self._default_config: Optional[Dict[str, Any]] = None
        self._default_client: Optional[ElasticsearchClientWrapper] = None
        logger.info("ElasticsearchClientFactory initialized")

    async def _create_client(
        self,
        hosts: List[str],
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 120,
        **kwargs,
    ) -> ElasticsearchClientWrapper:
        """
        Create Elasticsearch client instance

        Args:
            hosts: Elasticsearch host list
            username: Username
            password: Password
            api_key: API key
            timeout: Timeout (seconds)
            **kwargs: Other connection parameters

        Returns:
            ElasticsearchClientWrapper instance
        """
        # Build connection parameters
        conn_params = {
            "hosts": hosts,
            "timeout": timeout,
            "max_retries": 3,
            "retry_on_timeout": True,
            "verify_certs": False,  # Disable SSL certificate verification
            "ssl_show_warn": False,  # Disable SSL warnings
            **kwargs,
        }

        # Add authentication information
        if api_key:
            conn_params["api_key"] = api_key
        elif username and password:
            conn_params["basic_auth"] = (username, password)

        # Generate connection alias (used for elasticsearch-dsl connections management)
        alias = get_cache_key(hosts, username, password, api_key)

        # Create async client via async_connections.create_connection
        async_client = async_connections.create_connection(alias=alias, **conn_params)

        client_wrapper = ElasticsearchClientWrapper(async_client, hosts)

        logger.info("Created Elasticsearch client for %s with alias %s", hosts, alias)
        return client_wrapper

    async def _get_client(
        self,
        hosts: List[str],
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> ElasticsearchClientWrapper:
        """
        Get Elasticsearch client instance

        Args:
            hosts: Elasticsearch host list
            username: Username
            password: Password
            api_key: API key
            **kwargs: Other configuration parameters

        Returns:
            ElasticsearchClientWrapper instance
        """
        cache_key = get_cache_key(hosts, username, password, api_key)

        async with self._lock:
            # Check cache
            if cache_key in self._clients:
                logger.debug("Using cached Elasticsearch client for %s", hosts)
                return self._clients[cache_key]

            # Create new client instance
            logger.info("Creating new Elasticsearch client for %s", hosts)

            client_wrapper = await self._create_client(
                hosts=hosts,
                username=username,
                password=password,
                api_key=api_key,
                **kwargs,
            )

            self._clients[cache_key] = client_wrapper
            logger.info(
                "Elasticsearch client %s created and cached with key %s",
                hosts,
                cache_key,
            )

        return client_wrapper

    async def get_default_client(self) -> ElasticsearchClientWrapper:
        """
        Get default Elasticsearch client instance based on environment variable configuration
        Getting default client is not supported, direct calls to factory are prohibited

        Returns:
            ElasticsearchClientWrapper instance
        """
        raise NotImplementedError(
            "ElasticsearchClientFactory does not support get_default_client, use register_default_client instead"
        )

    async def register_default_client(self) -> ElasticsearchClientWrapper:
        """
        Register a default client

        Returns:
            ElasticsearchClientWrapper instance
        """
        # Get or create default configuration

        if self._default_client is not None:
            return self._default_client

        if self._default_config is None:
            self._default_config = get_default_es_config()

        config = self._default_config
        default_client = await self._get_client(
            hosts=config["hosts"],
            username=config.get("username"),
            password=config.get("password"),
            api_key=config.get("api_key"),
            timeout=config.get("timeout", 120),
        )

        # Register a default client
        async_connections.add_connection(
            alias="default", conn=default_client.async_client
        )
        self._default_client = default_client
        return default_client

    async def remove_client(
        self,
        hosts: List[str],
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> bool:
        """
        Remove specified client

        Args:
            hosts: Elasticsearch host list
            username: Username
            password: Password
            api_key: API key

        Returns:
            bool: Whether removal was successful
        """
        cache_key = get_cache_key(hosts, username, password, api_key)

        async with self._lock:
            if cache_key in self._clients:
                client_wrapper = self._clients[cache_key]
                try:
                    await client_wrapper.close()
                except Exception as e:
                    logger.error(
                        "Error closing Elasticsearch client during removal: %s", e
                    )

                del self._clients[cache_key]
                logger.info("Elasticsearch client %s removed from cache", hosts)
                return True
            else:
                logger.warning("Elasticsearch client %s not found in cache", hosts)
                return False

    async def close_all_clients(self) -> None:
        """Close all cached clients"""
        async with self._lock:
            for cache_key, client_wrapper in self._clients.items():
                try:
                    await client_wrapper.close()
                except Exception as e:
                    logger.error(
                        "Error closing Elasticsearch client %s: %s", cache_key, e
                    )

            self._clients.clear()
            logger.info("All Elasticsearch clients closed and cleared from cache")
