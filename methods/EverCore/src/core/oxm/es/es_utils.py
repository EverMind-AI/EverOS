from common_utils.datetime_utils import get_now_with_timezone
from typing import Type, List, Optional, TYPE_CHECKING
import os

from core.observation.logger import get_logger

if TYPE_CHECKING:
    from core.oxm.es.doc_base import DocBase

logger = get_logger(__name__)


def generate_index_name(cls: Type['DocBase']) -> str:
    """Generate index name with timestamp"""
    now = get_now_with_timezone()
    alias = cls.get_index_name()
    return f"{alias}-{now.strftime('%Y%m%d%H%M%S%f')}"


def get_index_ns() -> str:
    """Get index namespace"""
    return os.getenv("SELF_ES_INDEX_NS") or ""


def is_abstract_doc_class(doc_class: Type['DocBase']) -> bool:
    """
    Check if the document class is an abstract class

    Determine whether it is an abstract class by checking the Meta.abstract attribute.
    Abstract classes should not have their indices initialized.

    Args:
        doc_class: Document class

    Returns:
        bool: Whether it is an abstract class
    """
    # Check the abstract attribute of the Meta class
    pattern = getattr(doc_class, 'PATTERN', None) or None
    return (not pattern) or "Generated" in doc_class.__name__


class EsIndexInitializer:
    """
    Elasticsearch index initialization utility class

    Used to batch initialize indices and aliases corresponding to ES document classes.
    Uses doc_class._get_connection() to obtain the connection, supporting tenant awareness.
    """

    def __init__(self):
        self._initialized_classes: List[Type['DocBase']] = []

    async def initialize_indices(
        self, document_classes: Optional[List[Type['DocBase']]] = None
    ) -> None:
        """
        Initialize indices for multiple document classes

        Args:
            document_classes: List of document classes
        """
        if not document_classes:
            logger.info("No document classes need to be initialized")
            return

        try:
            logger.info(
                "Initializing Elasticsearch indices, total %d document classes",
                len(document_classes),
            )

            for doc_class in document_classes:
                await self.init_document_index(doc_class)

            self._initialized_classes.extend(document_classes)

            logger.info(
                "✅ Elasticsearch index initialization succeeded, processed %d document classes",
                len(document_classes),
            )

            for doc_class in document_classes:
                logger.info(
                    "📋 Initialized index: class=%s -> index=%s",
                    doc_class.__name__,
                    doc_class.get_index_name(),
                )

        except Exception as e:
            logger.error("❌ Elasticsearch index initialization failed: %s", e)
            raise

    async def init_document_index(self, doc_class: Type['DocBase']) -> None:
        """
        Initialize index for a single document class

        Args:
            doc_class: Document class
        """
        try:
            # Get alias name
            alias = doc_class.get_index_name()

            if not alias:
                logger.info(
                    "Document class has no index alias, skipping initialization %s",
                    doc_class.__name__,
                )
                return

            # Check if it is an abstract class
            if is_abstract_doc_class(doc_class):
                logger.debug(
                    "Document class is abstract, skipping initialization %s",
                    doc_class.__name__,
                )
                return

            # Get connection through document class (supports tenant awareness)
            client = doc_class._get_connection()

            # Check if alias exists
            logger.info(
                "Checking index alias: %s (document class: %s)",
                alias,
                doc_class.__name__,
            )
            alias_exists = await client.indices.exists(index=alias)

            if not alias_exists:
                # Generate target index name
                dst = doc_class.dest()

                # Create index
                await doc_class.init(index=dst, using=client)

                # Create alias
                await client.indices.update_aliases(
                    body={
                        "actions": [
                            {
                                "add": {
                                    "index": dst,
                                    "alias": alias,
                                    "is_write_index": True,
                                }
                            }
                        ]
                    }
                )
                logger.info("✅ Created index and alias: %s -> %s", dst, alias)
            else:
                logger.info("📋 Index alias already exists: %s", alias)

        except Exception as e:
            logger.error(
                "❌ Failed to initialize index for document class %s: %s",
                doc_class.__name__,
                e,
            )
            raise

    @property
    def initialized_classes(self) -> List[Type['DocBase']]:
        """Get list of initialized document classes"""
        return self._initialized_classes


async def initialize_document_indices(
    document_classes: Optional[List[Type['DocBase']]] = None,
) -> None:
    """
    Utility function: Initialize indices for multiple document classes

    Args:
        document_classes: List of document classes
    """
    initializer = EsIndexInitializer()
    await initializer.initialize_indices(document_classes)


async def init_single_document_index(doc_class: Type['DocBase']) -> None:
    """
    Utility function: Initialize index for a single document class

    Args:
        doc_class: Document class
    """
    initializer = EsIndexInitializer()
    await initializer.init_document_index(doc_class)
