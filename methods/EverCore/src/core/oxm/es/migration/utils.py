"""
Elasticsearch index migration tool

Provides generic Elasticsearch index rebuilding and migration functionality.
"""

import time
import traceback
from typing import Type, Any
from elasticsearch import NotFoundError, RequestError
from elasticsearch.dsl import AsyncDocument
from core.observation.logger import get_logger
from core.di.utils import get_all_subclasses
from core.oxm.es.doc_base import DocBase, get_index_ns
from core.oxm.es.es_utils import is_abstract_doc_class

logger = get_logger(__name__)


def find_document_class_by_index_name(index_name: str) -> Type[AsyncDocument]:
    """
    Find document class by index alias

    Uses `get_index_ns()` for namespace concatenation, ensuring consistency with `AliasDoc` alias rules.

    Args:
        index_name: Index alias (e.g., "episodic-memory")

    Returns:
        Matched ES document class

    Raises:
        ValueError: If no match or multiple matches are found
    """

    all_doc_classes = get_all_subclasses(DocBase)

    matched_classes: list[Type[AsyncDocument]] = []
    for cls in all_doc_classes:
        if not is_abstract_doc_class(cls) and index_name in cls.get_index_name():
            matched_classes.append(cls)

    if not matched_classes:
        available_indexes = [cls.get_index_name() for cls in all_doc_classes]
        logger.error(
            "Cannot find document class corresponding to index alias '%s'", index_name
        )
        logger.info("Available index aliases: %s", ", ".join(available_indexes))
        raise ValueError(
            f"Cannot find document class corresponding to index alias '{index_name}'"
        )

    if len(matched_classes) > 1:
        logger.error(
            "Found multiple document classes with index alias '%s': %s",
            index_name,
            [cls.__module__ + "." + cls.__name__ for cls in matched_classes],
        )
        raise ValueError(
            f"Found multiple document classes with index alias '{index_name}': {', '.join([f'{cls.__module__}.{cls.__name__}' for cls in matched_classes])}"
        )

    document_class = matched_classes[0]

    # Basic validation
    if not hasattr(document_class, "PATTERN"):
        raise ValueError(
            f"Document class {document_class.__name__} must have PATTERN attribute"
        )
    if not hasattr(document_class, "dest"):
        raise ValueError(
            f"Document class {document_class.__name__} must have dest() method"
        )

    return document_class


async def rebuild_index(
    document_class: Type[AsyncDocument],
    close_old: bool = False,
    delete_old: bool = False,
) -> None:
    """
    Rebuild Elasticsearch index

    Create a new index based on the existing one and update the alias to point to it. Supports closing or deleting the old index.

    Args:
        document_class: ES document class, must have PATTERN attribute and dest() method
        es_connect: Elasticsearch connection object
        close_old: Whether to close the old index
        delete_old: Whether to delete the old index

    Returns:
        None

    Raises:
        ValueError: If the document class is missing required attributes or methods
    """
    # Validate document class
    if not hasattr(document_class, 'PATTERN'):
        raise ValueError(
            "Document class %s must have PATTERN attribute" % document_class.__name__
        )
    if not hasattr(document_class, 'dest'):
        raise ValueError(
            "Document class %s must have dest() method" % document_class.__name__
        )

    # Get index information
    alias_name = document_class.get_index_name()
    es_connect = document_class._get_connection()
    pattern = document_class.PATTERN
    dest_index = document_class.dest()

    logger.info("Starting index rebuild: %s", alias_name)
    logger.info("Source index pattern: %s", pattern)
    logger.info("Destination index: %s", dest_index)

    # Check if destination index already exists
    if await es_connect.indices.exists(index=dest_index):
        logger.warning(
            "Destination index %s already exists, skipping rebuild", dest_index
        )
        return

    # Initialize new index
    await document_class.init(index=dest_index)
    logger.info("New index created: %s", dest_index)

    # Start reindexing task
    reindex_body = {"source": {"index": alias_name}, "dest": {"index": dest_index}}

    logger.info("Starting data migration...")
    result = await es_connect.reindex(body=reindex_body, wait_for_completion=False)
    task_id = result["task"]
    logger.info("Rebuild task ID: %s", task_id)

    # Wait for task completion
    await wait_for_task_completion(es_connect, task_id)

    # Update aliases
    await update_aliases(es_connect, alias_name, dest_index, close_old, delete_old)

    logger.info("Index rebuild completed: %s", alias_name)


async def wait_for_task_completion(es_connect: Any, task_id: str) -> None:
    """
    Wait for Elasticsearch task to complete

    Args:
        es_connect: Elasticsearch connection object
        task_id: Task ID
    """
    logger.info("Waiting for rebuild task to complete...")

    while True:
        try:
            task_result = await es_connect.tasks.get(task_id=task_id)

            if task_result.get("completed", False):
                logger.info("Rebuild task completed")
                break

            # Display progress information
            status = task_result.get("task", {}).get("status", {})
            if status:
                created = status.get("created", 0)
                total = status.get("total", 0)
                if total > 0:
                    progress = (created / total) * 100
                    logger.info(
                        "Rebuild progress: %d/%d (%.1f%%)", created, total, progress
                    )

            time.sleep(5)  # Check every 5 seconds

        except (NotFoundError, RequestError) as e:
            traceback.print_exc()
            logger.error("Failed to check task status: %s", e)
            time.sleep(10)  # Wait longer when error occurs


async def update_aliases(
    es_connect: Any,
    alias_name: str,
    dest_index: str,
    close_old: bool = False,
    delete_old: bool = False,
) -> None:
    """
    Update Elasticsearch index aliases

    Args:
        es_connect: Elasticsearch connection object
        alias_name: Alias name
        dest_index: Destination index
        close_old: Whether to close old index
        delete_old: Whether to delete old index
    """
    logger.info("Updating index aliases...")

    # Get indices currently pointed by the alias
    try:
        existing_indices = list(
            (await es_connect.indices.get_alias(name=alias_name)).keys()
        )
        logger.info(
            "Current indices pointed by alias %s: %s", alias_name, existing_indices
        )
    except NotFoundError:
        existing_indices = []
        logger.info("Alias %s does not exist, will create new alias", alias_name)

    # Refresh the new index
    await es_connect.indices.refresh(index=dest_index)

    # Build alias update operations
    actions = []

    # Remove old alias associations
    for old_index in existing_indices:
        actions.append({"remove": {"alias": alias_name, "index": old_index}})

    # Add new alias association
    actions.append(
        {"add": {"alias": alias_name, "index": dest_index, "is_write_index": True}}
    )

    # Execute alias update
    await es_connect.indices.update_aliases(body={"actions": actions})
    logger.info("Alias %s updated to point to %s", alias_name, dest_index)

    # Handle old indices
    for old_index in existing_indices:
        if close_old:
            logger.info("Closing old index: %s", old_index)
            await es_connect.indices.close(index=old_index)
        elif delete_old:
            logger.info("Deleting old index: %s", old_index)
            await es_connect.indices.delete(index=old_index)
