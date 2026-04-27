"""
Milvus rebuild script (calling core common tools)

Implemented based on methods provided by MilvusCollectionBase:
- Find the corresponding Collection management class by alias
- Call create_new_collection() to create a new collection (automatically create index and load)
- Perform data migration (supports batch processing to avoid memory overflow)
- Call switch_alias() to switch the alias to the new collection
- Optionally delete the old Collection

Supports:
- Single collection rebuild: -a <alias>
- All collections rebuild:   --all

Usage (via bootstrap with SKIP_LIFESPAN to avoid schema validation on startup):
  SKIP_LIFESPAN=true TENANT_INIT_STORAGE_INFO='...' python src/bootstrap.py src/devops_scripts/data_fix/milvus_rebuild_collection.py --all

Note: This script migrates data by default (in batches of 3000).
To disable data migration, use the --no-migrate-data option.
"""

import argparse
import sys
import traceback
from typing import Optional, List

from pymilvus import Collection

from core.observation.logger import get_logger
from core.di.utils import get_all_subclasses
from core.oxm.milvus.migration.utils import rebuild_collection
from core.oxm.milvus.milvus_collection_base import (
    MilvusCollectionBase,
    MilvusCollectionWithSuffix,
)
from devops_scripts.progress import ProgressReporter, StdoutProgressReporter


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data migration callback
# ---------------------------------------------------------------------------


def migrate_data_callback(
    old_collection: Collection,
    new_collection: Collection,
    batch_size: int = 3000,
    progress: Optional[ProgressReporter] = None,
    alias: str = "",
) -> None:
    """
    Data migration callback function (using offset pagination + sorting to ensure data integrity)

    Args:
        old_collection: Old collection instance
        new_collection: New collection instance
        batch_size: Number of records processed per batch, default is 3000
        progress: Optional progress reporter for real-time status updates
        alias: Collection alias (for progress reporting)

    Note:
        Use offset + limit + order_by for paginated queries to avoid:
        1. Data loss (unordered queries may return in unpredictable order)
        2. Data duplication (pagination position may drift)

        Although queries with large offsets are less efficient, they are acceptable for one-time data migration,
        and ensure data completeness and accuracy.

        When schema has changed (fields added/removed), only fields present in the
        new collection schema are migrated. Removed fields are dropped automatically.
    """
    # Collect new schema field names for filtering removed fields
    new_field_names = {f.name for f in new_collection.schema.fields}

    logger.info(
        "Start migrating data: %s -> %s (batch size: %d)",
        old_collection.name,
        new_collection.name,
        batch_size,
    )

    # Query total record count before migration starts
    total_records = old_collection.query(expr='id != ""', output_fields=["count(*)"])[
        0
    ]["count(*)"]
    logger.info("Total records to migrate: %d", total_records)

    if progress:
        progress.emit(
            {"event": "migrate_start", "alias": alias, "total_records": total_records}
        )

    total_migrated = 0  # Total number of records migrated
    offset = 0  # Current query offset
    batch_num = 0  # Current batch number

    try:
        while True:
            batch_num += 1
            logger.info(
                "Querying batch %d, offset: %d, limit: %d",
                batch_num,
                offset,
                batch_size,
            )

            # Use offset+limit for pagination, and sort by id
            # Note: STL_SORT index on the id field is required to use order_by
            # Without an index, it may raise an error or have poor performance
            try:
                # Try using the order_by parameter (pymilvus 2.4+)
                query_result = old_collection.query(
                    expr="",  # Query all data
                    output_fields=["*"],
                    limit=batch_size,
                    offset=offset,
                    order_by=[("id", "asc")],  # Sort by id in ascending order
                )
            except TypeError:
                # If order_by is not supported (older version), fall back to unordered query
                # In this case, there's still a risk of data loss or duplication
                logger.warning(
                    "Current pymilvus version does not support order_by parameter, using unordered query"
                )
                logger.warning(
                    "It is recommended to upgrade pymilvus to version 2.4+, or create an STL_SORT index on the id field"
                )
                query_result = old_collection.query(
                    expr="", output_fields=["*"], limit=batch_size, offset=offset
                )
            except Exception as e:
                # If the error is due to missing index, prompt user to create one
                if "index" in str(e).lower() or "sort" in str(e).lower():
                    logger.error(
                        "Query failed, possibly because there is no STL_SORT index on the id field: %s",
                        e,
                    )
                    logger.error(
                        "Please create an STL_SORT index on the id field of the old collection, or use unordered query"
                    )
                raise

            # If query result is empty, no more data to migrate
            if not query_result:
                logger.info("No more data, migration completed")
                break

            # Filter each row to only include fields in new schema (handles field removal)
            filtered_result = [
                {k: v for k, v in row.items() if k in new_field_names}
                for row in query_result
            ]

            batch_count = len(filtered_result)
            logger.info(
                "Retrieved %d records, starting to insert into new collection...",
                batch_count,
            )

            # Insert into new collection
            new_collection.insert(filtered_result)
            new_collection.flush()

            # Update statistics
            total_migrated += batch_count
            offset += batch_count  # Update offset
            logger.info("Migrated %d records", total_migrated)

            # Report migration progress
            if progress:
                progress.emit(
                    {
                        "event": "migrate_progress",
                        "alias": alias,
                        "batch": batch_num,
                        "batch_count": batch_count,
                        "total_migrated": total_migrated,
                        "total_records": total_records,
                    }
                )

            # If the number of records retrieved is less than batch_size, it's the last batch
            if batch_count < batch_size:
                logger.info("Last batch, migration completed")
                break

    except Exception as e:
        logger.error("Error occurred during data migration: %s", e)
        raise

    logger.info("Data migration completed: total %d records", total_migrated)


# ---------------------------------------------------------------------------
# Discovery & execution
# ---------------------------------------------------------------------------


def discover_all_aliases() -> List[str]:
    """
    Discover all concrete collection aliases by scanning MilvusCollectionBase subclasses.

    Returns:
        List of collection base names (e.g., ["v1_episodic_memory", "v1_user_profile"])  #skip-sensitive-check
    """
    aliases = []
    for cls in get_all_subclasses(MilvusCollectionBase):
        if cls._COLLECTION_NAME is None:
            continue
        if not issubclass(cls, MilvusCollectionWithSuffix):
            continue
        aliases.append(cls._COLLECTION_NAME)
    return aliases


def run(
    alias: str,
    drop_old: bool,
    migrate_data: bool,
    batch_size: int,
    progress: ProgressReporter,
) -> None:
    """
    Execute rebuild logic for a single collection (delegated to core tools)

    Args:
        alias: Collection alias
        drop_old: Whether to delete the old collection
        migrate_data: Whether to migrate data
        batch_size: Number of records processed per batch
        progress: Progress reporter
    """
    progress.emit({"event": "collection_start", "alias": alias})

    try:
        # Determine whether to pass the callback function based on whether data migration is needed
        if migrate_data:
            populate_fn = lambda old_col, new_col: migrate_data_callback(
                old_col, new_col, batch_size, progress=progress, alias=alias
            )
        else:
            populate_fn = None

        result = rebuild_collection(
            alias=alias, drop_old=drop_old, populate_fn=populate_fn
        )

        progress.emit(
            {
                "event": "collection_done",
                "alias": alias,
                "status": "ok",
                "source": result.source_collection,
                "dest": result.dest_collection,
                "dropped_old": result.dropped_old,
            }
        )
    except Exception as exc:
        progress.emit(
            {
                "event": "collection_done",
                "alias": alias,
                "status": "fail",
                "error": str(exc)[:500],
            }
        )
        logger.error("Milvus rebuild failed: %s", exc)
        traceback.print_exc()
        raise


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main function: parse command-line arguments and execute rebuild

    Args:
        argv: List of command-line arguments

    Returns:
        Exit code (0 indicates success)
    """
    parser = argparse.ArgumentParser(
        description="Rebuild and switch Milvus Collection alias",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Rebuild a single collection
  ... -a v1_episodic_memory

  # Rebuild multiple specific collections
  ... -a v1_episodic_memory v1_user_profile

  # Rebuild ALL discovered collections
  ... --all

  # With tenant context (recommended)
  SKIP_LIFESPAN=true TENANT_INIT_STORAGE_INFO='{"tenant_id":"s0001","isolation_mode":"shared","storage_info":{"milvus":{"collection_prefix":"s0001"}}}' python src/bootstrap.py ... --all  #skip-sensitive-check

  # Rebuild without migrating data
  ... -a v1_episodic_memory --no-migrate-data

  # Rebuild with custom batch size and drop old collection
  ... --all --batch-size 5000 --drop-old
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--alias",
        "-a",
        nargs="+",
        help="Collection alias(es), e.g.: -a v1_episodic_memory v1_user_profile",
    )
    group.add_argument(
        "--all",
        action="store_true",
        dest="rebuild_all",
        help="Rebuild all discovered collections",
    )
    parser.add_argument(
        "--drop-old",
        "-x",
        action="store_true",
        help="Whether to delete old collection (default: keep)",
    )
    parser.add_argument(
        "--no-migrate-data",
        action="store_true",
        help="Do not migrate data (default: migrate data)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=3000,
        help="Number of records per migration batch (default: 3000)",
    )

    args = parser.parse_args(argv)
    migrate_data = not args.no_migrate_data
    progress = StdoutProgressReporter()

    # Set up tenant context from TENANT_INIT_STORAGE_INFO if available
    import os

    if os.getenv("TENANT_INIT_STORAGE_INFO"):
        from core.tenants.init_tenant_all import setup_tenant_context_from_env

        tenant_id = setup_tenant_context_from_env()
        logger.info("Rebuild running with tenant context: %s", tenant_id)

    # Determine which aliases to rebuild
    if args.rebuild_all:
        aliases = discover_all_aliases()
    else:
        aliases = args.alias  # list from nargs="+"

    if not aliases:
        logger.warning("No collections to rebuild")
        return 0

    progress.emit({"event": "start", "total": len(aliases), "aliases": aliases})

    # Single alias: run directly (fail fast)
    if len(aliases) == 1:
        run(
            alias=aliases[0],
            drop_old=args.drop_old,
            migrate_data=migrate_data,
            batch_size=args.batch_size,
            progress=progress,
        )
        progress.emit({"event": "summary", "total": 1, "success": 1, "failed": 0})
        return 0

    # Multiple aliases: run with summary
    failed = 0
    for i, alias in enumerate(aliases, 1):
        logger.info("=" * 60)
        logger.info("[%d/%d] Rebuilding collection: %s", i, len(aliases), alias)
        logger.info("=" * 60)
        try:
            run(
                alias=alias,
                drop_old=args.drop_old,
                migrate_data=migrate_data,
                batch_size=args.batch_size,
                progress=progress,
            )
            logger.info("[OK] %s", alias)
        except Exception:
            logger.error("[FAIL] %s", alias)
            failed += 1

    progress.emit(
        {
            "event": "summary",
            "total": len(aliases),
            "success": len(aliases) - failed,
            "failed": failed,
        }
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info(
        "Rebuild summary: total=%d, success=%d, failed=%d",
        len(aliases),
        len(aliases) - failed,
        failed,
    )
    logger.info("=" * 60)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
