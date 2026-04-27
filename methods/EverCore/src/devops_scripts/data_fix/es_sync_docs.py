import argparse
import asyncio
import traceback
from elasticsearch.dsl import AsyncDocument

from core.observation.logger import get_logger
from core.oxm.es.migration.utils import find_document_class_by_index_name


logger = get_logger(__name__)


async def run(
    index_name: str, batch_size: int, limit_: int | None, days: int | None
) -> None:
    """Synchronize MongoDB data to the specified Elasticsearch index."""
    try:
        document_class: type[AsyncDocument] = find_document_class_by_index_name(
            index_name
        )
        logger.info(
            "Found document class: %s.%s",
            document_class.__module__,
            document_class.__name__,
        )

        doc_alias = document_class.get_index_name()
        logger.info("Index alias: %s", doc_alias)

        if "episodic-memory" in str(doc_alias):
            from devops_scripts.data_fix.es_sync_episodic_memory_docs import (
                sync_episodic_memory_docs,
            )

            await sync_episodic_memory_docs(
                batch_size=batch_size, limit=limit_, days=days
            )
        else:
            raise ValueError(f"Unsupported index type: {doc_alias}")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to synchronize documents: %s", exc)
        traceback.print_exc()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synchronize MongoDB data to Elasticsearch"
    )
    parser.add_argument(
        "--index-name", "-i", required=True, help="Index alias, e.g.: episodic-memory"
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=500, help="Batch size, default 500"
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit the number of documents to process, default all",
    )
    parser.add_argument(
        "--days",
        "-d",
        type=int,
        default=None,
        help="Process only documents created in the last N days, default all",
    )
    args = parser.parse_args(argv)

    asyncio.run(run(args.index_name, args.batch_size, args.limit, args.days))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
