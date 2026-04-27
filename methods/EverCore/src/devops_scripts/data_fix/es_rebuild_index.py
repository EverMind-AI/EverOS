import argparse
import asyncio
import traceback

from elasticsearch.dsl import AsyncDocument

from core.observation.logger import get_logger
from core.oxm.es.migration.utils import find_document_class_by_index_name, rebuild_index


logger = get_logger(__name__)


async def run(index_name: str, close_old: bool, delete_old: bool) -> None:
    try:
        document_class: type[AsyncDocument] = find_document_class_by_index_name(
            index_name
        )
        logger.info(
            "Found document class: %s.%s",
            document_class.__module__,
            document_class.__name__,
        )
        logger.info("Index alias: %s", document_class.get_index_name())

        await rebuild_index(document_class, close_old=close_old, delete_old=delete_old)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to rebuild index: %s", exc)
        traceback.print_exc()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild and switch Elasticsearch index alias"
    )
    parser.add_argument(
        "--index-name", "-i", required=True, help="Index alias, e.g.: episodic-memory"
    )
    parser.add_argument(
        "--close-old", "-c", action="store_true", help="Whether to close the old index"
    )
    parser.add_argument(
        "--delete-old",
        "-x",
        action="store_true",
        help="Whether to delete the old index",
    )
    args = parser.parse_args(argv)

    asyncio.run(run(args.index_name, args.close_old, args.delete_old))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
