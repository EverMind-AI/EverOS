"""
Milvus collection rebuild and alias switching utility

Design goals:
- Consolidate infrastructure-related, reusable Milvus rebuild logic into the core layer
- Business or script layers only need to provide the client, alias, and options

Note:
- This tool only handles structure rebuilding and alias switching, not data migration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Type

from pymilvus import Collection, utility

from core.observation.logger import get_logger
from core.oxm.milvus.milvus_collection_base import (
    MilvusCollectionBase,
    MilvusCollectionWithSuffix,
)
from core.di.utils import get_all_subclasses


logger = get_logger(__name__)


@dataclass
class RebuildResult:
    """Result of Milvus collection rebuild."""

    alias: str
    source_collection: str
    dest_collection: str
    dropped_old: bool


def find_collection_manager_by_alias(alias: str) -> Type[MilvusCollectionBase]:
    """
    Find the corresponding Collection manager class by alias

    Args:
        alias: Collection alias

    Returns:
        Corresponding MilvusCollectionBase subclass

    Raises:
        ValueError: If no corresponding collection class is found
    """
    all_doc_classes = get_all_subclasses(MilvusCollectionBase)

    # Iterate through all subclasses to find the one matching the alias
    for doc_class in all_doc_classes:
        # Skip abstract classes
        # pylint: disable=protected-access  # Internal framework usage, accessing subclass configuration attributes
        if (
            not hasattr(doc_class, '_COLLECTION_NAME')
            or doc_class._COLLECTION_NAME is None
        ):
            continue

        # Check if it's a MilvusCollectionWithSuffix type
        if issubclass(doc_class, MilvusCollectionWithSuffix):
            # Temporarily instantiate to get alias (requires parsing suffix)
            try:
                # Try to parse suffix from alias
                base_name = (
                    doc_class._COLLECTION_NAME
                )  # pylint: disable=protected-access
                if alias.startswith(base_name):
                    return doc_class
            except (
                Exception
            ):  # pylint: disable=broad-except  # Ignore instantiation failure, continue to next class
                continue
        else:
            # For MilvusCollectionBase, directly compare _COLLECTION_NAME
            if doc_class._COLLECTION_NAME == alias:  # pylint: disable=protected-access
                return doc_class

    raise ValueError(f"Cannot find collection class corresponding to alias '{alias}'")


def rebuild_collection(
    alias: str,
    drop_old: bool = False,
    populate_fn: Optional[Callable[[Collection, Collection], None]] = None,
) -> RebuildResult:
    """
    Rebuild Milvus collection based on alias:
    1) Find the corresponding Collection manager class by alias
    2) Call create_new_collection() to create a new collection (automatically create index and load)
    3) Call optional data population callback (implemented by caller)
    4) Call switch_alias() to switch alias, optionally delete old collection

    Args:
        alias: Collection alias
        drop_old: Whether to delete the old collection
        populate_fn: Optional callback to populate data after index creation and before alias switching.
            Function signature: (old_collection: Collection, new_collection: Collection) -> None

    Returns:
        RebuildResult: Information about the rebuild result

    Raises:
        ValueError: If no corresponding collection class is found
        MilvusException: If Milvus operation fails
    """
    logger.info(
        "Starting to rebuild Collection: alias=%s, drop_old=%s", alias, drop_old
    )

    # 1. Find the corresponding Collection manager class by alias
    collection_class = find_collection_manager_by_alias(alias)
    logger.info("Found collection class: %s", collection_class.__name__)

    # 2. Instantiate manager (parse suffix from alias)
    if issubclass(collection_class, MilvusCollectionWithSuffix):
        # Parse suffix from alias
        base_name = (
            collection_class._COLLECTION_NAME
        )  # pylint: disable=protected-access
        suffix = None
        if alias != base_name and alias.startswith(base_name + "_"):
            suffix = alias[len(base_name) + 1 :]
        manager = collection_class(suffix=suffix)
    else:
        raise NotImplementedError(
            "Unsupported collection type: %s", collection_class.__name__
        )

    # Load the old collection WITHOUT passing schema to avoid SchemaNotReadyException
    # when code-defined schema differs from the existing collection's schema.
    # This is the whole point of rebuild: the schema has changed.
    manager.ensure_connection_registered()

    alias_name = manager.name
    using = manager.using

    if not utility.has_collection(alias_name, using=using):
        raise ValueError(
            f"Collection '{alias_name}' does not exist in Milvus, nothing to rebuild"
        )

    old_collection = Collection(name=alias_name, using=using)
    old_collection.load()
    old_real_name = old_collection.name
    logger.info("Original collection real name: %s", old_real_name)

    # 3. Create new collection (automatically create index and load)
    logger.info("Starting to create new collection...")
    new_collection = manager.create_new_collection()
    new_real_name = new_collection.name
    logger.info("New collection created: %s", new_real_name)

    # 4. Call data population callback if provided
    if populate_fn:
        logger.info("Starting data population callback...")
        try:
            populate_fn(old_collection, new_collection)
            logger.info("Data population completed")
        except Exception as e:
            logger.error("Data population failed: %s", e)
            raise

    # 5. Verify data consistency before switching
    #    Use query-based count instead of num_entities (which is approximate
    #    and includes soft-deleted records not yet compacted).
    if populate_fn:
        new_collection.flush()
        old_count = old_collection.query(expr='id != ""', output_fields=["count(*)"])[
            0
        ]["count(*)"]
        new_count = new_collection.query(expr='id != ""', output_fields=["count(*)"])[
            0
        ]["count(*)"]
        logger.info("Data verification: old=%d, new=%d", old_count, new_count)
        if old_count != new_count:
            raise RuntimeError(
                f"Data count mismatch after migration: "
                f"old={old_count}, new={new_count}. "
                f"Aborting alias switch. New collection '{new_real_name}' "
                f"is left intact for inspection."
            )
        logger.info("Data verification passed")

    # 6. Switch alias to new collection and optionally delete old collection
    logger.info("Switching alias '%s' to new collection '%s'...", alias, new_real_name)
    manager.switch_alias(new_collection, drop_old=drop_old)

    logger.info(
        "Rebuild completed: alias=%s, src=%s -> dest=%s, dropped_old=%s",
        alias,
        old_real_name,
        new_real_name,
        drop_old,
    )

    return RebuildResult(
        alias=alias,
        source_collection=old_real_name,
        dest_collection=new_real_name,
        dropped_old=drop_old,
    )
