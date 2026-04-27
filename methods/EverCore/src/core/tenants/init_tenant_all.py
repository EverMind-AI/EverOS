"""
Tenant database initialization module

Initializes MongoDB, Milvus, and Elasticsearch databases for a tenant.

Tenant context is passed via TENANT_INIT_STORAGE_INFO environment variable,
which contains full storage_info JSON. Works for both shared mode and
exclusive mode.

Usage:
    # Shared mode (logical isolation, multiple tenants share the same storage):
    TENANT_INIT_STORAGE_INFO='{
        "tenant_id": "my_tenant",
        "isolation_mode": "shared",
        "storage_info": {
            "mongodb": {"database": "my_tenant_memsys"},
            "elasticsearch": {"index_prefix": "my_tenant"},
            "milvus": {"collection_prefix": "my_tenant", "num_partitions": 256}
        }
    }' python src/manage.py tenant-init

    # Exclusive mode (physical isolation, dedicated storage per tenant):
    TENANT_INIT_STORAGE_INFO='{
        "tenant_id": "my_tenant",
        "isolation_mode": "exclusive",
        "storage_info": {
            "mongodb": {"database": "my_tenant_memsys"},
            "elasticsearch": {"index_prefix": "my_tenant"},
            "milvus": {"collection_prefix": "my_tenant", "num_partitions": 1}
        }
    }' python src/manage.py tenant-init

Storage info fields:
    mongodb.database          — Target database name (e.g. "my_tenant_memsys")
    elasticsearch.index_prefix — ES index name prefix (e.g. "my_tenant")
    milvus.collection_prefix  — Milvus collection name prefix (e.g. "my_tenant")
    milvus.num_partitions     — Milvus partition count (optional, default from env MILVUS_NUM_PARTITIONS)
                                Recommended: 256 for shared mode, 1 for exclusive mode.
"""

import json
import os

from core.observation.logger import get_logger
from core.tenants.tenant_contextvar import set_current_tenant
from core.tenants.tenant_models import TenantInfo, TenantDetail
from core.lifespan.mongodb_lifespan import MongoDBLifespanProvider
from core.lifespan.milvus_lifespan import MilvusLifespanProvider
from core.lifespan.elasticsearch_lifespan import ElasticsearchLifespanProvider

logger = get_logger(__name__)

# Environment variable for passing full tenant context to init subprocess
TENANT_INIT_STORAGE_INFO_ENV = "TENANT_INIT_STORAGE_INFO"


def setup_tenant_context_from_env() -> str:
    """
    Set up tenant context from TENANT_INIT_STORAGE_INFO environment variable.

    Returns:
        tenant_id string for logging

    Raises:
        ValueError: If env var is not set or has invalid format
    """
    # Priority 1: Full storage_info from env (set by TenantInitService)
    storage_info_json = os.getenv(TENANT_INIT_STORAGE_INFO_ENV)
    if storage_info_json:
        try:
            data = json.loads(storage_info_json)
            tenant_id = data["tenant_id"]
            storage_info = data.get("storage_info", {})
            isolation_mode = data.get("isolation_mode", "shared")

            tenant_detail = TenantDetail(
                storage_info=storage_info, isolation_mode=isolation_mode
            )
            tenant_info = TenantInfo(
                tenant_id=tenant_id, tenant_detail=tenant_detail, origin_tenant_data={}
            )
            set_current_tenant(tenant_info)
            logger.info(
                "Tenant context set from %s: tenant_id=%s, mode=%s",
                TENANT_INIT_STORAGE_INFO_ENV,
                tenant_id,
                isolation_mode,
            )
            return tenant_id
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(
                f"Invalid {TENANT_INIT_STORAGE_INFO_ENV} format: {e}. "
                f"Expected JSON with tenant_id, storage_info, isolation_mode."
            ) from e

    raise ValueError(
        "Tenant context is not configured!\n"
        f"Set {TENANT_INIT_STORAGE_INFO_ENV} environment variable.\n"
        f"Example:\n"
        f'  {TENANT_INIT_STORAGE_INFO_ENV}=\'{{"tenant_id":"my_tenant","isolation_mode":"shared",'
        f'"storage_info":{{"mongodb":{{"database":"my_tenant_memsys"}},'
        f'"elasticsearch":{{"index_prefix":"my_tenant"}},'
        f'"milvus":{{"collection_prefix":"my_tenant","num_partitions":256}}}}}}\'\n'
        f"  python src/manage.py tenant-init"
    )


class _MockApp:
    """Mock FastAPI app for lifespan providers (only needs state attribute)."""

    class State:
        pass

    state = State()


async def init_mongodb() -> bool:
    """Initialize tenant's MongoDB database."""
    logger.info("=" * 60)
    logger.info("Starting initialization of tenant's MongoDB database...")
    logger.info("=" * 60)

    try:
        mongodb_provider = MongoDBLifespanProvider()
        mock_app = _MockApp()
        await mongodb_provider.startup(mock_app)
        logger.info("✅ Tenant's MongoDB database initialized successfully")
        await mongodb_provider.shutdown(mock_app)
        return True
    except Exception as e:
        logger.error("❌ Failed to initialize tenant's MongoDB database: %s", e)
        return False


async def init_milvus() -> bool:
    """Initialize tenant's Milvus collections."""
    logger.info("=" * 60)
    logger.info("Starting initialization of tenant's Milvus database...")
    logger.info("=" * 60)

    try:
        milvus_provider = MilvusLifespanProvider()
        mock_app = _MockApp()
        await milvus_provider.startup(mock_app)
        logger.info("✅ Tenant's Milvus database initialized successfully")
        await milvus_provider.shutdown(mock_app)
        return True
    except Exception as e:
        logger.error("❌ Failed to initialize tenant's Milvus database: %s", e)
        return False


async def init_elasticsearch() -> bool:
    """Initialize tenant's Elasticsearch indices."""
    logger.info("=" * 60)
    logger.info("Starting initialization of tenant's Elasticsearch database...")
    logger.info("=" * 60)

    try:
        es_provider = ElasticsearchLifespanProvider()
        mock_app = _MockApp()
        await es_provider.startup(mock_app)
        logger.info("✅ Tenant's Elasticsearch database initialized successfully")
        await es_provider.shutdown(mock_app)
        return True
    except Exception as e:
        logger.error("❌ Failed to initialize tenant's Elasticsearch database: %s", e)
        return False


async def run_tenant_init() -> bool:
    """
    Execute tenant database initialization.

    Reads tenant context from environment variables, then initializes
    MongoDB, Milvus, and Elasticsearch in sequence.

    Returns:
        Whether all initializations were successful
    """
    logger.info("*" * 60)
    logger.info("Tenant Database Initialization Tool")
    logger.info("*" * 60)

    # Set up tenant context
    tenant_id = setup_tenant_context_from_env()
    logger.info("Tenant ID: %s", tenant_id)
    logger.info("*" * 60)

    # Initialize all three storage engines
    mongodb_success = await init_mongodb()
    milvus_success = await init_milvus()
    es_success = await init_elasticsearch()

    # Summary
    logger.info("")
    logger.info("*" * 60)
    logger.info("Initialization Result Summary")
    logger.info("*" * 60)
    logger.info("Tenant ID: %s", tenant_id)
    logger.info("MongoDB: %s", "✅ Success" if mongodb_success else "❌ Failure")
    logger.info("Milvus: %s", "✅ Success" if milvus_success else "❌ Failure")
    logger.info("Elasticsearch: %s", "✅ Success" if es_success else "❌ Failure")
    logger.info("*" * 60)

    return mongodb_success and milvus_success and es_success
