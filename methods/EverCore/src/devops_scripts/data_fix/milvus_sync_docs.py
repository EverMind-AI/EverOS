"""
Sync MongoDB data to Milvus

Main entry script that calls the corresponding sync implementation based on Collection name.
Supports command-line arguments for batch size, processing limits, and time range.

Usage (recommended to run via bootstrap, which automatically loads application context and dependencies):
  python src/bootstrap.py src/devops_scripts/data_fix/milvus_sync_docs.py --collection-name episodic_memory --batch-size 500

Arguments:
  --collection-name, -c  Milvus Collection name (required), e.g.: episodic_memory
  --batch-size, -b       Batch size (default 500)
  --limit, -l            Limit the number of documents to process (default: all)
  --days, -d             Only process documents created in the past N days (default: all)
"""

import argparse
import asyncio
import traceback

from core.observation.logger import get_logger


logger = get_logger(__name__)


async def run(
    collection_name: str, batch_size: int, limit_: int | None, days: int | None
) -> None:
    """
    Sync MongoDB data to the specified Milvus Collection.

    Routes to the specific sync implementation based on the Collection name.

    Args:
        collection_name: Milvus Collection name, e.g.: episodic_memory
        batch_size: Batch size, default 500
        limit_: Limit the number of documents to process, None means process all
        days: Only process documents created in the past N days, None means process all

    Raises:
        ValueError: If the Collection name is not supported
        Exception: If an error occurs during synchronization
    """
    try:
        logger.info("Starting sync to Milvus Collection: %s", collection_name)

        # Route to specific implementation based on Collection name
        if collection_name == "episodic_memory":
            from devops_scripts.data_fix.milvus_sync_episodic_memory_docs import (
                sync_episodic_memory_docs,
            )

            await sync_episodic_memory_docs(
                batch_size=batch_size, limit=limit_, days=days
            )
        else:
            raise ValueError(f"Unsupported Collection type: {collection_name}")

    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to sync documents: %s", exc)
        traceback.print_exc()
        raise


def main(argv: list[str] | None = None) -> int:
    """
    Command-line entry function.

    Parses command-line arguments and calls the sync function.

    Args:
        argv: List of command-line arguments, None means use sys.argv

    Returns:
        int: Exit code, 0 indicates success

    Examples:
        # Sync all episodic_memory documents
        python milvus_sync_docs.py --collection-name episodic_memory

        # Sync only documents from the last 7 days, with batch size 1000
        python milvus_sync_docs.py --collection-name episodic_memory --batch-size 1000 --days 7

        # Limit processing to 10,000 documents
        python milvus_sync_docs.py --collection-name episodic_memory --limit 10000
    """
    parser = argparse.ArgumentParser(
        description="Sync MongoDB data to Milvus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --collection-name episodic_memory
  %(prog)s --collection-name episodic_memory --batch-size 1000 --days 7
  %(prog)s --collection-name episodic_memory --limit 10000
        """,
    )

    parser.add_argument(
        "--collection-name",
        "-c",
        required=True,
        help="Milvus Collection name, e.g.: episodic_memory",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=500, help="Batch size, default 500"
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit the number of documents to process, default: all",
    )
    parser.add_argument(
        "--days",
        "-d",
        type=int,
        default=None,
        help="Only process documents created in the past N days, default: all",
    )

    args = parser.parse_args(argv)

    # Run async sync task
    asyncio.run(run(args.collection_name, args.batch_size, args.limit, args.days))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
