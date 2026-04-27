import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pymilvus import Collection, DataType, FieldSchema, utility, CollectionSchema
from pymilvus.client.types import ConsistencyLevel, LoadState

from pymilvus import connections
from core.oxm.milvus.async_collection import AsyncCollection
from common_utils.datetime_utils import get_now_with_timezone
from memory_layer.constants import VECTORIZE_DIMENSIONS

logger = logging.getLogger(__name__)


def generate_new_collection_name(alias: str) -> str:
    """Generate a new collection name with timestamp based on alias."""
    now = get_now_with_timezone()
    return f"{alias}_{now.strftime('%Y%m%d%H%M%S%f')}"


@dataclass
class IndexConfig:
    """
    Index configuration class

    Used to define indexes to be created (supports vector and scalar indexes)

    Attributes:
        field_name: Field name
        index_type: Index type (e.g., IVF_FLAT, HNSW, AUTOINDEX, etc.)
        metric_type: Metric type (required for vector indexes, e.g., L2, COSINE, IP)
        params: Index parameters (optional)
        index_name: Index name (optional, auto-generated if not specified)

    Examples:
        # Vector index
        IndexConfig(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 128}
        )

        # Scalar index
        IndexConfig(
            field_name="title",
            index_type="AUTOINDEX"
        )
    """

    field_name: str
    index_type: str
    metric_type: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    index_name: Optional[str] = None

    def to_index_params(self) -> Dict[str, Any]:
        """Convert to pymilvus index parameter format"""
        result = {"index_type": self.index_type}
        if self.metric_type:
            result["metric_type"] = self.metric_type
        if self.params:
            result["params"] = self.params
        return result


def get_collection_suffix(suffix: Optional[str] = None) -> str:
    """
    Get Collection name suffix, used in multi-tenant scenarios

    Args:
        suffix: Explicitly provided suffix; if given, return directly.
               If not provided, read from environment variable SELF_MILVUS_COLLECTION_NS

    Returns:
        Collection suffix string; return empty string if neither is set
    """
    if suffix is not None:
        return suffix
    return os.getenv("SELF_MILVUS_COLLECTION_NS", "")


class MilvusCollectionBase:
    """
    Milvus Collection base management class

    Responsibilities:
    1. Manage basic Collection information (name, Schema, index configuration)
    2. Provide lazily loaded Collection instance (internally cached)
    3. Provide utility methods (ensure_indexes, ensure_loaded)

    Applicable scenarios:
    - Simple Collection management
    - Read-only data sources (managed by other teams, only querying needed)
    - No need for suffix, timestamp, alias, or other complex logic

    Usage:
    1. Subclass defines:
       - _COLLECTION_NAME: Collection name (required)
       - _SCHEMA: Collection Schema (optional)
       - _INDEX_CONFIGS: List of index configurations (optional)
       - _DB_USING: Milvus connection alias (optional, default "default")

    2. Instantiation:
       mgr = MovieCollection()  # Use class-defined _DB_USING
       # or
       mgr = MovieCollection(using="custom_db")  # Override class definition

    3. Usage:
       mgr.ensure_loaded()  # Load into memory
       mgr.collection.search(...)  # Use Collection

    Example:
        # Read-only scenario (data source managed by other team)
        class ReadOnlyMovieCollection(MilvusCollectionBase):
            _COLLECTION_NAME = "external_movies"  # Fixed Collection name
            _DB_USING = "external_db"  # Use external database connection

        mgr = ReadOnlyMovieCollection()
        mgr.ensure_loaded()
        results = mgr.collection.search(...)
    """

    # Attributes that subclasses must define
    _COLLECTION_NAME: Optional[str] = None

    # Optional attributes that subclasses may define
    _SCHEMA: Optional[CollectionSchema] = None
    _INDEX_CONFIGS: Optional[List[IndexConfig]] = None
    _DB_USING: Optional[str] = "default"

    # Class-level instance cache
    _collection_instance: Optional[Collection] = None
    _async_collection_instance: Optional[AsyncCollection] = None

    def __init__(self):
        """Initialize configuration container"""
        if not self._COLLECTION_NAME:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define '_COLLECTION_NAME' class attribute"
            )

        # Use class attribute _DB_USING, default to "default" if not defined
        self._using = self._DB_USING if self._DB_USING is not None else "default"

    @classmethod
    def collection(cls) -> Collection:
        """Get Collection instance (class-level cache)"""
        if cls._collection_instance is None:
            raise ValueError(
                f"{cls.__name__} Collection instance not created, please call ensure_loaded() first"
            )
        return cls._collection_instance

    @classmethod
    def async_collection(cls) -> AsyncCollection:
        """Get asynchronous Collection instance (class-level cache)"""
        if cls._async_collection_instance is None:
            if cls._collection_instance is None:
                raise ValueError(
                    f"{cls.__name__} Collection instance not created, please call ensure_loaded() first"
                )
            cls._async_collection_instance = AsyncCollection(cls._collection_instance)
        return cls._async_collection_instance

    @property
    def name(self) -> str:
        """Get actual Collection name"""
        return self._COLLECTION_NAME

    @property
    def using(self) -> str:
        """Get connection alias"""
        return self._DB_USING if self._DB_USING is not None else "default"

    def ensure_connection_registered(self) -> None:
        """
        Ensure the pymilvus connection for self.using is established.

        Uses MilvusClientFactory to create a client, which internally registers
        the pymilvus connection alias needed by Collection().

        Subclasses (e.g., TenantAwareCollectionWithSuffix) may override this
        to handle tenant-specific connection registration.
        """
        from core.di.utils import get_bean

        using = self.using
        try:
            connections._fetch_handler(using)
            return
        except Exception:
            pass

        factory = get_bean("milvus_client_factory")
        factory.get_named_client(using)
        logger.info("Connection registered for using='%s'", using)

    def load_collection(self) -> Collection:
        """Load Collection (internal method)"""
        name = self.name
        if not utility.has_collection(name, using=self.using):
            raise ValueError(f"Collection '{name}' does not exist")

        coll = Collection(
            name=name,
            using=self.using,
            schema=self._SCHEMA,
            consistency_level=ConsistencyLevel.Bounded,
        )
        logger.info("Loaded Collection '%s'", name)
        return coll

    def ensure_loaded(self) -> None:
        """Ensure Collection is loaded into memory (class-level cache)"""
        # Lazy load Collection
        if self.__class__._collection_instance is None:
            self.__class__._collection_instance = self.load_collection()

        coll = self.__class__._collection_instance

        name = coll.name
        using = coll.using

        try:
            load_state = utility.load_state(name, using=using)

            if load_state == LoadState.NotLoad:
                logger.info("Collection '%s' not loaded, loading into memory...", name)
                coll.load()
                logger.info("Collection '%s' loaded successfully", name)
            elif load_state == LoadState.Loading:
                logger.info(
                    "Collection '%s' is loading, waiting for completion...", name
                )
                coll.load()
            else:
                logger.info("Collection '%s' already loaded", name)

        except Exception as e:
            logger.error("Error occurred while loading Collection: %s", e)
            raise

    def ensure_indexes(self) -> None:
        """Create all configured indexes (diff approach)"""
        if not self._INDEX_CONFIGS:
            logger.info(
                "Collection '%s' has no index configuration, skipping", self.name
            )
            return

        # Lazy load Collection
        if self._collection_instance is None:
            self._collection_instance = self.load_collection()

        coll = self._collection_instance
        self._create_indexes_for_collection(coll)

    @staticmethod
    def _get_existing_indexes(coll: Collection) -> Dict[str, Dict[str, Any]]:
        """Get existing index information in the specified Collection"""
        try:
            indexes_info = coll.indexes
            result = {}

            for index in indexes_info:
                field_name = index.field_name
                result[field_name] = {
                    "index_type": index.params.get("index_type"),
                    "metric_type": index.params.get("metric_type"),
                }

            return result

        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Error occurred while retrieving index information: %s", e)
            return {}

    def _create_indexes_for_collection(self, coll: Collection) -> None:
        """Create missing indexes for the specified Collection (reuse diff logic from ensure_indexes)"""
        try:
            existing_indexes = self._get_existing_indexes(coll)
            existing_field_names = set(existing_indexes.keys())

            logger.info(
                "Existing index fields in Collection '%s': %s",
                coll.name,
                existing_field_names,
            )

            index_configs = self._INDEX_CONFIGS or []
            for index_config in index_configs:
                field_name = index_config.field_name
                if field_name in existing_field_names:
                    logger.info("Field '%s' already has an index, skipping", field_name)
                    continue

                logger.info(
                    "Creating index for field '%s' (type: %s)...",
                    field_name,
                    index_config.index_type,
                )
                create_kwargs = {
                    "field_name": field_name,
                    "index_params": index_config.to_index_params(),
                    "timeout": 120,
                }
                if index_config.index_name:
                    create_kwargs["index_name"] = index_config.index_name
                coll.create_index(**create_kwargs)
                logger.info("Index creation for field '%s' succeeded", field_name)

            logger.info(
                "Index check and creation completed for Collection '%s'", coll.name
            )
        except Exception as e:
            logger.error(
                "Error occurred while creating indexes for Collection '%s': %s",
                coll.name,
                e,
            )
            raise

    @staticmethod
    def _get_collection_desc(collection_: Collection) -> Dict[str, Any]:
        conn = collection_._get_connection()
        return conn.describe_collection(collection_.name)

    def ensure_all(self) -> None:
        """
        Complete all initialization operations in one step

        Execution order:
        1. ensure_loaded(): Load into memory
        """
        self.ensure_loaded()
        logger.info("Collection '%s' initialization completed", self.name)


class MilvusCollectionWithSuffix(MilvusCollectionBase):
    """
    Milvus Collection management class with Suffix and Alias mechanism

    Inherits from MilvusCollectionBase, adds:
    1. Dynamic table name: Supports dynamically setting table name suffix via suffix or environment variable (multi-tenant scenario)
    2. Alias mechanism: Real table name includes timestamp, accessed via alias (convenient for future switching)
       - Alias: {base_name}_{suffix}
       - Real name: {base_name}_{suffix}-{timestamp}
    3. Creation management: Provides methods like ensure_create, ensure_all

    Applicable scenarios:
    - Multi-tenant scenarios requiring independent Collections for different customers
    - Need version management, retaining historical Collections
    - Need gray-scale switching, switching between different versions via alias

    Usage:
    1. Subclass defines:
       - _BASE_NAME: Base name of the Collection (required)
       - _SCHEMA: Schema definition of the Collection (required)
       - _INDEX_CONFIGS: List of index configurations (optional)
       - _DB_USING: Milvus connection alias (optional, default "default")

    2. Instantiation:
       mgr = MovieCollection(suffix="customer_a")
       # Alias: movies_customer_a
       # Real name: movies_customer_a-20231015123456789000

       # Or specify database connection
       mgr = MovieCollection(suffix="customer_a", using="custom_db")

    3. Initialization:
       mgr.ensure_all()  # One-step initialization

    4. Usage:
       mgr.collection.insert([...])
       mgr.collection.search(...)

    Example:
        class MovieCollection(MilvusCollectionWithSuffix):
            _BASE_NAME = "movies"
            _SCHEMA = CollectionSchema(fields=[...])
            _INDEX_CONFIGS = [
                IndexConfig(field_name="embedding", index_type="IVF_FLAT", ...),
                IndexConfig(field_name="year", index_type="AUTOINDEX")
            ]
            _DB_USING = "my_milvus"  # Optional: specify default database connection

        # Usage
        mgr = MovieCollection(suffix="customer_a")
        mgr.ensure_all()
        mgr.collection.insert([...])
    """

    def __init__(self, suffix: Optional[str] = None):
        """
        Initialize configuration container

        Args:
            suffix: Collection name suffix; if not provided, read from environment variable SELF_MILVUS_COLLECTION_NS
        """
        if not self._COLLECTION_NAME:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define '_COLLECTION_NAME' class attribute"
            )

        if not self._SCHEMA:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define '_SCHEMA' class attribute (required for creation scenarios)"
            )

        # Get suffix (supports parameter or environment variable)
        self._suffix = get_collection_suffix(suffix)

        # Construct alias name
        if self._suffix:
            self._alias_name = f"{self._COLLECTION_NAME}_{self._suffix}"
        else:
            self._alias_name = self._COLLECTION_NAME

        # Call parent class initialization
        super().__init__()

    @property
    def name(self) -> str:
        """Get Collection name"""
        return self._alias_name

    def load_collection(self) -> Collection:
        """
        Load or create Collection (internal method)

        Override parent method, add creation logic
        """
        # First check if alias exists
        name = self.name

        if not utility.has_collection(name, using=self._using):
            # Collection does not exist, create a new timestamped Collection
            _collection_name = generate_new_collection_name(name)

            logger.info(
                "Collection '%s' does not exist, creating new Collection: %s",
                name,
                _collection_name,
            )

            # Create Collection
            create_kwargs = {
                "name": _collection_name,
                "schema": self._SCHEMA,
                "using": self._using,
                "consistency_level": ConsistencyLevel.Bounded,
            }
            num_partitions = getattr(self, "_NUM_PARTITIONS", None)
            if num_partitions is not None:
                create_kwargs["num_partitions"] = num_partitions
            logger.info(
                "Creating Collection with kwargs: %s",
                {k: v for k, v in create_kwargs.items() if k != "schema"},
            )
            Collection(**create_kwargs)

            # Create alias pointing to new Collection
            # When deleting the actual Collection, the alias is not automatically deleted, so delete alias first
            utility.drop_alias(name, using=self._using)
            utility.create_alias(
                collection_name=_collection_name, alias=name, using=self._using
            )
            logger.info("Created Alias '%s' -> '%s'", name, _collection_name)

        # Uniformly load via alias (whether existing or newly created)
        coll = Collection(name=name, using=self._using)

        return coll

    def ensure_create(self) -> None:
        """
        Ensure Collection has been created

        This method triggers lazy loading of the Collection; if alias does not exist, create a new Collection
        """
        if self._collection_instance is None:
            self._collection_instance = self.load_collection()
        logger.info("Collection '%s' is ready", self.name)

    def ensure_all(self) -> None:
        """
        Complete all initialization operations in one step

        Execution order:
        1. ensure_create(): Create Collection and alias
        2. ensure_indexes(): Create all configured indexes
        3. ensure_loaded(): Load into memory
        """
        logger.info("Starting initialization of Collection '%s'", self.name)

        self.ensure_create()
        self.ensure_indexes()
        self.ensure_loaded()

        # Retrieve and print the actual collection name
        try:
            collection_desc = self._get_collection_desc(self._collection_instance)
            real_collection_name = collection_desc.get("collection_name", "unknown")
            logger.info(
                "Collection '%s' initialization completed, real name: %s",
                self.name,
                real_collection_name,
            )
        except Exception as e:
            logger.warning("Failed to retrieve real collection name: %s", e)
            logger.info("Collection '%s' initialization completed", self.name)

    def create_new_collection(self) -> Collection:
        """
        Create a new real Collection (without switching alias).
        - Create new collection using class-defined `_SCHEMA`
        - Create indexes and load for new collection according to `_INDEX_CONFIGS`

        Returns:
            New collection instance (indexes created and loaded)
        """
        if not self._SCHEMA:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define '_SCHEMA' to support collection creation"
            )

        alias_name = self._alias_name

        # Create new collection
        new_real_name = generate_new_collection_name(alias_name)
        create_kwargs = {
            "name": new_real_name,
            "schema": self._SCHEMA,
            "using": self._using,
            "consistency_level": ConsistencyLevel.Bounded,
        }
        num_partitions = getattr(self, "_NUM_PARTITIONS", None)
        if num_partitions is not None:
            create_kwargs["num_partitions"] = num_partitions
        Collection(**create_kwargs)

        # Create indexes for new collection
        try:
            new_coll = Collection(name=new_real_name, using=self._using)
            self._create_indexes_for_collection(new_coll)
            new_coll.load()
        except Exception as e:
            logger.warning(
                "Error occurred while creating indexes for new collection, can be ignored: %s",
                e,
            )

        return new_coll

    def switch_alias(self, new_collection: Collection, drop_old: bool = False) -> None:
        """
        Switch alias to the specified new collection, optionally delete old collection.
        - Prefer alter_alias; if fails, fall back to drop/create
        - Refresh class-level cache after switching
        """
        alias_name = self._alias_name
        new_real_name = new_collection.name

        # Get old collection real name (if exists)
        old_real_name: Optional[str] = None
        try:
            conn = connections._fetch_handler(self._using)
            desc = conn.describe_alias(alias_name)
            old_real_name = (
                desc.get("collection_name") if isinstance(desc, dict) else None
            )
        except Exception:
            old_real_name = None

        # Alias switching
        try:
            conn = connections._fetch_handler(self._using)
            conn.alter_alias(new_real_name, alias_name)
            logger.info("Alias '%s' switched to '%s'", alias_name, new_real_name)
        except Exception as e:
            logger.warning("alter_alias failed, attempting drop/create: %s", e)
            try:
                utility.drop_alias(alias_name, using=self._using)
            except Exception:
                pass
            utility.create_alias(
                collection_name=new_real_name, alias=alias_name, using=self._using
            )
            logger.info("Created alias '%s' -> '%s'", alias_name, new_real_name)

        # Optionally delete old collection (after switching completes)
        if drop_old and old_real_name:
            try:
                utility.drop_collection(old_real_name, using=self._using)
                logger.info("Deleted old collection: %s", old_real_name)
            except Exception as e:
                logger.warning(
                    "Failed to delete old collection (can be handled manually): %s", e
                )

        # Refresh class-level cache to alias collection
        try:
            self.__class__._collection_instance = Collection(
                name=alias_name, using=self._using
            )
        except Exception:
            pass

    def exists(self) -> bool:
        """Check if Collection exists (via alias)"""
        return utility.has_collection(self.name, using=self._using)

    def drop(self) -> None:
        """Delete current Collection (including alias and real Collection)"""
        try:
            if not self._collection_instance:
                self._collection_instance = Collection(
                    name=self.name, using=self._using
                )

            real_name = self._collection_instance.name
            logger.info(
                "Found real name corresponding to Collection '%s': %s",
                self.name,
                real_name,
            )

            utility.drop_collection(real_name, using=self._using)
            logger.info("Deleted Collection '%s'", real_name)

        except Exception as e:  # pylint: disable=broad-except
            logger.warning(
                "Collection '%s' does not exist or deletion failed: %s", self.name, e
            )


if __name__ == "__main__":
    connections.connect("default", host="localhost", port=19530)

    class TestCollection(MilvusCollectionWithSuffix):
        _COLLECTION_NAME = "test"
        _SCHEMA = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=VECTORIZE_DIMENSIONS,
                ),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="created_at", dtype=DataType.INT64),
                FieldSchema(name="updated_at", dtype=DataType.INT64),
                FieldSchema(name="deleted_at", dtype=DataType.INT64),
                FieldSchema(name="deleted_by", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(
                    name="deleted_reason", dtype=DataType.VARCHAR, max_length=255
                ),
            ]
        )
        _INDEX_CONFIGS = [
            IndexConfig(
                field_name="vector",
                index_type="HNSW",  # Efficient approximate nearest neighbor search
                metric_type="COSINE",  # Euclidean distance
                params={
                    "M": 16,  # Maximum number of edges per node
                    "efConstruction": 200,  # Search width during construction
                },
            )
        ]
        _DB_USING = "default"

    collection = TestCollection(suffix="zhanghui")
    collection.ensure_all()
    assert collection.name == "test_zhanghui"

    class TestCollection2(MilvusCollectionBase):
        _COLLECTION_NAME = "test_zhanghui"
        _SCHEMA = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=VECTORIZE_DIMENSIONS,
                ),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=255),
            ]
        )
        _INDEX_CONFIGS = [
            IndexConfig(
                field_name="vector",
                index_type="HNSW",  # Efficient approximate nearest neighbor search
                metric_type="COSINE",  # Euclidean distance
                params={
                    "M": 16,  # Maximum number of edges per node
                    "efConstruction": 200,  # Search width during construction
                },
            )
        ]
        _DB_USING = "default"

    collection2 = TestCollection2()
    collection2.ensure_all()
    assert collection2.name == "test_zhanghui"

    import asyncio

    asyncio.run(TestCollection.async_collection().insert([[1, 2, 3], [4, 5, 6]]))
