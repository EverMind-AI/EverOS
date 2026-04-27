"""
MongoDB Client Factory

Provides MongoDB client caching and management functionality based on configuration.
Supports reading configuration from environment variables and provides default client.
"""

import os
import asyncio
from abc import ABC, abstractmethod
import traceback
from typing import Dict, Optional, List
from urllib.parse import quote_plus
from pymongo import AsyncMongoClient
from beanie import init_beanie
from core.class_annotations.utils import get_annotation
from core.oxm.mongo.constant.annotations import ClassAnnotationKey, Toggle

from core.di.decorators import component
from core.observation.logger import get_logger
from common_utils.datetime_utils import timezone
from core.oxm.mongo.document_base import DEFAULT_DATABASE

logger = get_logger(__name__)


class MongoDBConfig:
    """MongoDB configuration class"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "memsys",
        uri: Optional[str] = None,
        uri_params: Optional[str] = None,
        **kwargs,
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.uri = uri
        self.uri_params = uri_params
        self.kwargs = kwargs

    def get_connection_string(self) -> str:
        """Get connection string and append unified URI parameters (if any)"""
        # Base URI
        if self.uri:
            base_uri = self.uri
        else:
            if self.username and self.password:
                # URL encode username and password
                encoded_username = quote_plus(self.username)
                encoded_password = quote_plus(self.password)
                base_uri = f"mongodb://{encoded_username}:{encoded_password}@{self.host}:{self.port}/{self.database}"
            else:
                base_uri = f"mongodb://{self.host}:{self.port}/{self.database}"

        # Append unified parameters
        uri_params: Optional[str] = self.uri_params
        if uri_params:
            separator = '&' if ('?' in base_uri) else '?'
            return f"{base_uri}{separator}{uri_params}"
        return base_uri

    def get_cache_key(self) -> str:
        """Get cache key

        Generate signature based only on basic info + unified URI parameter string to avoid reusing the same client for different parameters.
        """
        base = f"{self.host}:{self.port}:{self.database}:{self.username or 'anonymous'}"
        uri_params: Optional[str] = self.uri_params
        signature = uri_params.strip() if isinstance(uri_params, str) else ""
        return f"{base}:{signature}" if signature else base

    @classmethod
    def from_env(cls, prefix: str = "") -> 'MongoDBConfig':
        """
        Create configuration from environment variables.

        prefix rule: if prefix is provided, read variables in the format "{prefix}_XXX", otherwise read "XXX".
        For example: prefix="a" reads "A_MONGODB_URI", "A_MONGODB_HOST", etc.
        """

        def _env(name: str, default: Optional[str] = None) -> Optional[str]:
            if prefix == DEFAULT_DATABASE:
                key = name
            else:
                prefix_upper = prefix.upper()
                key = f"{prefix_upper}_{name}" if prefix else name
            return os.getenv(key, default) if default is not None else os.getenv(key)

        # Prioritize using MONGODB_URI
        uri = _env("MONGODB_URI")
        if uri:
            return cls(uri=uri, database=_env("MONGODB_DATABASE", "memsys"))

        # Read individual configuration items
        host = _env("MONGODB_HOST", "localhost")
        port = int(_env("MONGODB_PORT", "27017"))
        username = _env("MONGODB_USERNAME")
        password = _env("MONGODB_PASSWORD")
        database = _env("MONGODB_DATABASE", "memsys")
        uri_params = _env("MONGODB_URI_PARAMS", "")

        return cls(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
            uri_params=uri_params,
        )

    def __repr__(self) -> str:
        return f"MongoDBConfig(host={self.host}, port={self.port}, database={self.database})"


class MongoDBClientWrapper:
    """MongoDB client wrapper"""

    def __init__(self, client: AsyncMongoClient, config: MongoDBConfig):
        self.client = client
        self.config = config
        self.database = client[config.database]
        self._initialized = False
        self._document_models: List = []

    async def initialize_beanie(self, document_models: Optional[List] = None):
        """Initialize Beanie ODM"""
        if self._initialized:
            return

        if document_models:
            try:
                # Group models: writable group (requires indexes), read-only group (skip indexes)
                writable_models = []
                readonly_models = []
                for model in document_models:
                    readonly_flag = get_annotation(model, ClassAnnotationKey.READONLY)
                    if readonly_flag == Toggle.ENABLED:
                        readonly_models.append(model)
                    else:
                        writable_models.append(model)

                if writable_models and readonly_models:
                    # Multiple init_beanie calls seem fine in code, but potential issues with referencing both types may exist; currently no business case uses both modes in one DB, but future needs should be guarded against, hence the warning
                    raise ValueError("Writable and read-only groups cannot coexist")

                logger.info(
                    "Initializing Beanie ODM (writable group), database: %s, model count: %d",
                    self.config.database,
                    len(writable_models),
                )
                if writable_models:
                    model_info_list = []
                    for model in writable_models:
                        try:
                            collection_name = (
                                model.get_collection_name()
                                if hasattr(model, 'get_collection_name')
                                else "unknown"
                            )
                        except Exception:
                            collection_name = "unknown"
                        model_info_list.append(f"{model.__name__} -> {collection_name}")
                    logger.info(
                        "📋 Writable models to initialize: [%s]",
                        ", ".join(model_info_list),
                    )
                    await init_beanie(
                        database=self.database,
                        document_models=writable_models,
                        skip_indexes=False,
                    )

                logger.info(
                    "Initializing Beanie ODM (read-only group), database: %s, model count: %d",
                    self.config.database,
                    len(readonly_models),
                )
                if readonly_models:
                    model_info_list = []
                    for model in readonly_models:
                        try:
                            collection_name = (
                                model.get_collection_name()
                                if hasattr(model, 'get_collection_name')
                                else "unknown"
                            )
                        except Exception:
                            collection_name = "unknown"
                        model_info_list.append(f"{model.__name__} -> {collection_name}")
                    logger.info(
                        "📋 Readonly models to initialize: [%s]",
                        ", ".join(model_info_list),
                    )
                    await init_beanie(
                        database=self.database,
                        document_models=readonly_models,
                        skip_indexes=True,
                    )

                self._document_models = document_models
                self._initialized = True
                logger.info(
                    "✅ Beanie ODM initialized successfully, registered %d models",
                    len(document_models),
                )

                for model in document_models:
                    logger.info(
                        "📋 Registered model: database=%s, model=%s -> %s",
                        self.config.database,
                        model.__name__,
                        model.get_collection_name(),
                    )

            except Exception as e:
                logger.error("❌ Beanie initialization failed: %s", e)
                traceback.print_exc()
                raise

    async def test_connection(self) -> bool:
        """Test connection"""
        try:
            await self.client.admin.command('ping')
            logger.info("✅ MongoDB connection test successful: %s", self.config)
            return True
        except Exception as e:
            logger.error(
                "❌ MongoDB connection test failed: %s, error: %s", self.config, e
            )
            return False

    async def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            stats = {}
            collections = await self.database.list_collection_names()

            for collection_name in collections:
                try:
                    collection_stats = await self.database.command(
                        "collStats", collection_name
                    )
                    stats[collection_name] = {
                        "count": collection_stats.get("count", 0),
                        "size": collection_stats.get("size", 0),
                        "avgObjSize": collection_stats.get("avgObjSize", 0),
                        "storageSize": collection_stats.get("storageSize", 0),
                        "indexes": collection_stats.get("nindexes", 0),
                    }
                except Exception as e:
                    logger.warning(
                        "Failed to get collection %s stats: %s", collection_name, e
                    )

            return stats
        except Exception as e:
            logger.error("Failed to get stats: %s", e)
            return {}

    async def close(self):
        """Close connection"""
        if self.client:
            await self.client.close()
            logger.info("🔌 MongoDB connection closed: %s", self.config)

    @property
    def is_initialized(self) -> bool:
        """Check if Beanie is initialized"""
        return self._initialized


class MongoDBClientFactory(ABC):
    """MongoDB client factory interface"""

    @abstractmethod
    async def get_client(
        self, config: Optional[MongoDBConfig] = None, **connection_kwargs
    ) -> MongoDBClientWrapper:
        """
        Get MongoDB client

        Args:
            config: MongoDB configuration, use default if None
            **connection_kwargs: additional connection parameters

        Returns:
            MongoDBClientWrapper: MongoDB client wrapper
        """
        ...

    @abstractmethod
    async def get_default_client(self) -> MongoDBClientWrapper:
        """
        Get default MongoDB client

        Returns:
            MongoDBClientWrapper: default MongoDB client wrapper
        """
        ...

    @abstractmethod
    async def get_named_client(self, name: str) -> MongoDBClientWrapper:
        """
        Get MongoDB client by name.

        Convention: name as environment variable prefix, read configuration from "{name}_MONGODB_XXX".
        For example, when name="A", try to read "A_MONGODB_URI", "A_MONGODB_HOST", etc.

        Args:
            name: prefix name (i.e., environment variable prefix)

        Returns:
            MongoDBClientWrapper: MongoDB client wrapper
        """
        ...

    @abstractmethod
    async def create_client_with_config(
        self,
        host: str = "localhost",
        port: int = 27017,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "memsys",
        **kwargs,
    ) -> MongoDBClientWrapper:
        """
        Create client with specified configuration

        Args:
            host: MongoDB host
            port: MongoDB port
            username: username
            password: password
            database: database name
            **kwargs: other connection parameters

        Returns:
            MongoDBClientWrapper: MongoDB client wrapper
        """
        ...

    @abstractmethod
    async def close_client(self, config: Optional[MongoDBConfig] = None):
        """
        Close specified client

        Args:
            config: configuration, close default client if None
        """
        ...

    @abstractmethod
    async def close_all_clients(self):
        """Close all clients"""
        ...


@component(name="mongodb_client_factory")
class MongoDBClientFactoryImpl(MongoDBClientFactory):
    """MongoDB client factory implementation class"""

    def __init__(self):
        """Initialize factory"""
        self._clients: Dict[str, MongoDBClientWrapper] = {}
        self._default_config: Optional[MongoDBConfig] = None
        self._default_client: Optional[MongoDBClientWrapper] = None
        self._lock = asyncio.Lock()

    async def get_client(
        self, config: Optional[MongoDBConfig] = None, **connection_kwargs
    ) -> MongoDBClientWrapper:
        """
        Get MongoDB client

        Args:
            config: MongoDB configuration, use default if None
            **connection_kwargs: additional connection parameters

        Returns:
            MongoDBClientWrapper: MongoDB client wrapper
        """

        if config is None:
            config = await self._get_default_config()

        cache_key = config.get_cache_key()

        async with self._lock:
            # Check cache
            if cache_key in self._clients:
                return self._clients[cache_key]

            # Create new client
            logger.info("Creating new MongoDB client: %s", config)

            # Merge connection parameters
            conn_kwargs = {
                "serverSelectionTimeoutMS": 10000,  # PyMongo AsyncMongoClient requires longer timeout
                "connectTimeoutMS": 10000,  # connection timeout
                "socketTimeoutMS": 10000,  # socket timeout
                "maxPoolSize": 50,
                "minPoolSize": 5,
                "tz_aware": True,
                "tzinfo": timezone,
                **config.kwargs,
                **connection_kwargs,
            }

            try:
                client = AsyncMongoClient(config.get_connection_string(), **conn_kwargs)

                client_wrapper = MongoDBClientWrapper(client, config)

                # Test connection
                if not await client_wrapper.test_connection():
                    await client_wrapper.close()
                    raise RuntimeError(f"MongoDB connection test failed: {config}")

                # Cache client
                self._clients[cache_key] = client_wrapper
                logger.info("✅ MongoDB client created and cached: %s", config)

                return client_wrapper

            except Exception as e:
                logger.error(
                    "❌ Failed to create MongoDB client: %s, error: %s", config, e
                )
                raise

    async def get_default_client(self) -> MongoDBClientWrapper:
        """
        Get default MongoDB client

        Returns:
            MongoDBClientWrapper: default MongoDB client wrapper
        """
        if self._default_client is None:
            config = await self._get_default_config()
            self._default_client = await self.get_client(config)

        return self._default_client

    async def get_named_client(self, name: str) -> MongoDBClientWrapper:
        """
        Get MongoDB client by name.

        Convention: name as environment variable prefix, read configuration from "{name}_MONGODB_XXX".
        For example, when name="A", try to read "A_MONGODB_URI", "A_MONGODB_HOST", etc.

        Args:
            name: prefix name (i.e., environment variable prefix)

        Returns:
            MongoDBClientWrapper: MongoDB client wrapper
        """
        if name == DEFAULT_DATABASE:
            return await self.get_default_client()
        config = MongoDBConfig.from_env(prefix=name)
        logger.info("📋 Loading named MongoDB config [name=%s]: %s", name, config)
        return await self.get_client(config)

    async def _get_default_config(self) -> MongoDBConfig:
        """Get default config (internal method)"""
        if self._default_config is None:
            self._default_config = MongoDBConfig.from_env()
            logger.info("📋 Loading default MongoDB config: %s", self._default_config)

        return self._default_config

    async def create_client_with_config(
        self,
        host: str = "localhost",
        port: int = 27017,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "memsys",
        **kwargs,
    ) -> MongoDBClientWrapper:
        """
        Create client with specified configuration

        Args:
            host: MongoDB host
            port: MongoDB port
            username: username
            password: password
            database: database name
            **kwargs: other connection parameters

        Returns:
            MongoDBClientWrapper: MongoDB client wrapper
        """
        config = MongoDBConfig(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
            **kwargs,
        )

        return await self.get_client(config)

    async def close_client(self, config: Optional[MongoDBConfig] = None):
        """
        Close specified client

        Args:
            config: configuration, close default client if None
        """
        if config is None:
            if self._default_client:
                await self._default_client.close()
                self._default_client = None
                return

        cache_key = config.get_cache_key()

        async with self._lock:
            if cache_key in self._clients:
                await self._clients[cache_key].close()
                del self._clients[cache_key]

    async def close_all_clients(self):
        """Close all clients"""
        async with self._lock:
            for client_wrapper in self._clients.values():
                await client_wrapper.close()

            self._clients.clear()

            if self._default_client:
                self._default_client = None

            logger.info("🔌 All MongoDB clients closed")
