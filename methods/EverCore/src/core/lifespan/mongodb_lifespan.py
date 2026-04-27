"""
MongoDB Lifespan Provider Implementation
"""

from collections import defaultdict
from fastapi import FastAPI
from typing import Any

from core.observation.logger import get_logger
from core.di.utils import get_bean_by_type
from core.di.decorators import component
from core.lifespan.lifespan_interface import LifespanProvider
from core.oxm.mongo.document_base import DocumentBase
from core.di.utils import get_all_subclasses
from core.component.mongodb_client_factory import MongoDBClientFactory, MongoDBClientWrapper


logger = get_logger(__name__)


@component(name="mongodb_lifespan_provider")
class MongoDBLifespanProvider(LifespanProvider):
    """MongoDB Lifespan Provider"""

    def __init__(self, name: str = "mongodb", order: int = 15):
        """
        Initialize MongoDB Lifespan Provider

        Args:
            name (str): Provider name
            order (int): Execution order, MongoDB starts after database connections
        """
        super().__init__(name, order)
        self._mongodb_factory = None
        self._mongodb_client = None

    async def startup(self, app: FastAPI) -> Any:
        """
        Start MongoDB connection and initialization

        Args:
            app (FastAPI): FastAPI application instance

        Returns:
            Any: MongoDB client information
        """
        logger.info("Initializing MongoDB connection...")

        try:

            # Get MongoDB client factory
            self._mongodb_factory = get_bean_by_type(MongoDBClientFactory)

            # Get default client
            self._mongodb_client: MongoDBClientWrapper = (
                await self._mongodb_factory.get_default_client()
            )

            # Manually initialize Beanie ODM
            all_subclasses_of_document_base = get_all_subclasses(DocumentBase)
            db_document_models = defaultdict(list)
            for subclass in all_subclasses_of_document_base:
                db_document_models[subclass.get_bind_database()].append(subclass)

            # Get all DB names
            db_names = list(db_document_models.keys())
            db_clients = {
                db_name: await self._mongodb_factory.get_named_client(db_name)
                for db_name in db_names
            }

            # Initialize Beanie ODM
            for db_name, db_client in db_clients.items():
                await db_client.initialize_beanie(db_document_models[db_name])

            logger.info("✅ MongoDB connection initialization completed")

        except Exception as e:
            logger.error("❌ Error during MongoDB initialization: %s", str(e))
            raise

    async def shutdown(self, app: FastAPI) -> None:
        """
        Close MongoDB connection

        Args:
            app (FastAPI): FastAPI application instance
        """
        logger.info("Closing MongoDB connection...")

        if self._mongodb_factory:
            try:
                await self._mongodb_factory.close_all_clients()
                logger.info("✅ MongoDB connection closed successfully")
            except Exception as e:
                logger.error("❌ Error closing MongoDB connection: %s", str(e))
