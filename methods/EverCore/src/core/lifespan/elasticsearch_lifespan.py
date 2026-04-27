"""
Elasticsearch lifecycle provider implementation
"""

from fastapi import FastAPI
from typing import Any

from core.observation.logger import get_logger
from core.di.utils import get_all_subclasses, get_bean_by_type
from core.di.decorators import component
from core.lifespan.lifespan_interface import LifespanProvider
from core.oxm.es.doc_base import DocBase
from core.oxm.es.es_utils import EsIndexInitializer
from core.component.elasticsearch_client_factory import ElasticsearchClientFactory

logger = get_logger(__name__)


@component(name="elasticsearch_lifespan_provider")
class ElasticsearchLifespanProvider(LifespanProvider):
    """Elasticsearch lifecycle provider"""

    def __init__(self, name: str = "elasticsearch", order: int = 18):
        """
        Initialize the Elasticsearch lifecycle provider

        Args:
            name (str): Provider name
            order (int): Execution order, must start before business_lifespan(20)
        """
        super().__init__(name, order)
        self._es_factory = None

    async def startup(self, app: FastAPI) -> Any:
        """
        Start Elasticsearch connection and initialization

        Args:
            app (FastAPI): FastAPI application instance

        Returns:
            Any: Elasticsearch client information
        """
        logger.info("Initializing Elasticsearch connection...")

        try:
            # Get Elasticsearch client factory
            self._es_factory: ElasticsearchClientFactory = get_bean_by_type(
                ElasticsearchClientFactory
            )

            # Register a default client, mainly for backward compatibility with legacy code in single-tenant scenarios
            await self._es_factory.register_default_client()

            # Get all subclasses of DocBase - dynamically generated classes might not be found, reason unknown
            all_doc_classes = get_all_subclasses(DocBase)

            # Filter valid document classes
            document_classes = []
            for doc_class in all_doc_classes:
                index_name = doc_class.get_index_name()
                # Check if index name is valid
                document_classes.append(doc_class)
                logger.info(
                    "Discovered document class: %s -> %s",
                    doc_class.__name__,
                    index_name,
                )

            # Initialize indices (using utility class, tenant-aware support)
            if document_classes:
                initializer = EsIndexInitializer()
                await initializer.initialize_indices(document_classes)
            else:
                logger.info("No document classes found that require initialization")

            logger.info("✅ Elasticsearch connection initialization completed")

            return {
                "factory": self._es_factory,
                "document_classes": [cls.__name__ for cls in document_classes],
            }

        except Exception as e:
            logger.error("❌ Error during Elasticsearch initialization: %s", str(e))
            raise

    async def shutdown(self, app: FastAPI) -> None:
        """
        Close Elasticsearch connection

        Args:
            app (FastAPI): FastAPI application instance
        """
        logger.info("Closing Elasticsearch connection...")

        if self._es_factory:
            try:
                await self._es_factory.close_all_clients()
                logger.info("✅ Elasticsearch connection closed successfully")
            except Exception as e:
                logger.error("❌ Error closing Elasticsearch connection: %s", str(e))
