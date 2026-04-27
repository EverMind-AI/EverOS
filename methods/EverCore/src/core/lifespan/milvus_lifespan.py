"""
Milvus lifespan provider implementation
"""

from collections import defaultdict
from fastapi import FastAPI
from typing import Any

from core.observation.logger import get_logger
from core.di.utils import get_bean, get_all_subclasses
from core.di.decorators import component
from core.lifespan.lifespan_interface import LifespanProvider
from core.oxm.milvus.milvus_collection_base import MilvusCollectionBase

logger = get_logger(__name__)


@component(name="milvus_lifespan_provider")
class MilvusLifespanProvider(LifespanProvider):
    """Milvus lifespan provider"""

    def __init__(self, name: str = "milvus", order: int = 18):
        """
        Initialize Milvus lifespan provider

        Args:
            name (str): Provider name
            order (int): Execution order, must start before business_lifespan(20)
        """
        super().__init__(name, order)
        self._milvus_factory = None
        self._milvus_clients = {}

    async def startup(self, app: FastAPI) -> Any:
        """
        Start Milvus connection and initialization

        Args:
            app (FastAPI): FastAPI application instance

        Returns:
            Any: Milvus client information
        """
        logger.info("Initializing Milvus connection...")

        try:
            # Get Milvus client factory
            self._milvus_factory = get_bean("milvus_client_factory")

            # Get all concrete Collection classes (by checking if _COLLECTION_NAME exists)
            all_collection_classes = [
                cls
                for cls in get_all_subclasses(MilvusCollectionBase)
                if cls._COLLECTION_NAME
                is not None  # Classes with _COLLECTION_NAME are concrete
            ]

            # Group by using
            using_collections = defaultdict(list)
            for collection_class in all_collection_classes:
                using = collection_class._DB_USING
                using_collections[using].append(collection_class)
                logger.info(
                    "Discovered Collection class: %s [using=%s]",
                    collection_class.__name__,
                    using,
                )

            # Get all required clients
            for using, collection_classes in using_collections.items():
                # Get client
                client = self._milvus_factory.get_named_client(using)
                self._milvus_clients[using] = client

                # Initialize each Collection
                for collection_class in collection_classes:
                    try:
                        collection = collection_class()
                        collection.ensure_all()
                        logger.info(
                            "✅ Collection '%s' initialized successfully [using=%s]",
                            collection.name,
                            using,
                        )
                    except Exception as e:
                        logger.error(
                            "❌ Failed to initialize Collection '%s' [using=%s]: %s",
                            collection_class._COLLECTION_NAME,
                            using,
                            e,
                        )
                        raise
            logger.info("✅ Milvus connection initialization completed")

        except Exception as e:
            logger.error("❌ Error during Milvus initialization: %s", str(e))
            raise

    async def shutdown(self, app: FastAPI) -> None:
        """
        Close Milvus connections

        Args:
            app (FastAPI): FastAPI application instance
        """
        logger.info("Closing Milvus connections...")

        if self._milvus_factory:
            try:
                self._milvus_factory.close_all_clients()
                logger.info("✅ Milvus connections closed successfully")
            except Exception as e:
                logger.error("❌ Error while closing Milvus connections: %s", str(e))

        # Clean up Milvus-related attributes in app.state
        for attr in ['milvus_clients', 'milvus_factory']:
            if hasattr(app.state, attr):
                delattr(app.state, attr)
