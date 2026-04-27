"""
Milvus Collection Browser & Cleanup Tool

Browse all Milvus collections with detailed info (row count, aliases, fields),
and interactively delete selected collections.

Usage:
    python src/bootstrap.py src/devops_scripts/milvus_admin/browse_collections.py
    python src/bootstrap.py src/devops_scripts/milvus_admin/browse_collections.py --db default
    python src/bootstrap.py src/devops_scripts/milvus_admin/browse_collections.py --filter v1_episodic
    python src/bootstrap.py src/devops_scripts/milvus_admin/browse_collections.py --delete
    python src/bootstrap.py src/devops_scripts/milvus_admin/browse_collections.py --prefix v1_episodic --drop
"""

import argparse
import sys
from typing import Optional, List

from pymilvus import MilvusClient, utility, connections

from core.component.milvus_client_factory import get_milvus_config
from core.observation.logger import get_logger

logger = get_logger(__name__)


def _connect(db_name: str = "") -> MilvusClient:
    """Create a MilvusClient from environment config."""
    config = get_milvus_config()
    client = MilvusClient(
        uri=config["uri"],
        user=config["user"],
        password=config["password"],
        db_name=db_name or config["db_name"],
    )
    # Also create a pymilvus connection for utility calls
    connections.connect(
        alias="_admin",
        uri=config["uri"],
        user=config["user"],
        password=config["password"],
        db_name=db_name or config["db_name"],
    )
    return client


def _format_row_count(count: int) -> str:
    """Format row count for display."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


def list_collections(
    client: MilvusClient,
    name_filter: Optional[str] = None,
    prefix: Optional[str] = None,
) -> List[str]:
    """List all collections, optionally filtered by name substring or prefix."""
    all_names = client.list_collections()
    all_names.sort()

    if prefix:
        all_names = [n for n in all_names if n.startswith(prefix)]
    elif name_filter:
        all_names = [n for n in all_names if name_filter.lower() in n.lower()]

    return all_names


def show_collection_details(client: MilvusClient, names: List[str]) -> None:
    """Print detailed info for each collection."""
    if not names:
        print("\nNo collections found.")
        return

    print(f"\n{'='*80}")
    print(f"  Found {len(names)} collection(s)")
    print(f"{'='*80}\n")

    header = f"{'#':<4} {'Collection Name':<50} {'Rows':>10} {'Aliases'}"
    print(header)
    print("-" * len(header) + "-" * 20)

    for idx, name in enumerate(names, 1):
        try:
            stats = client.get_collection_stats(name)
            row_count = int(stats.get("row_count", 0))
        except Exception:
            row_count = -1

        try:
            aliases = utility.list_aliases(collection_name=name, using="_admin")
            alias_str = ", ".join(aliases) if aliases else "-"
        except Exception:
            alias_str = "?"

        row_str = _format_row_count(row_count) if row_count >= 0 else "?"
        print(f"{idx:<4} {name:<50} {row_str:>10} {alias_str}")

    print()


def interactive_delete(client: MilvusClient, names: List[str]) -> None:
    """Interactively select and delete collections."""
    if not names:
        print("No collections available for deletion.")
        return

    print("Enter collection numbers to delete (comma-separated), or 'q' to quit.")
    print("Example: 1,3,5  or  2-6  or  all\n")

    user_input = input("Delete> ").strip()
    if not user_input or user_input.lower() == "q":
        print("Cancelled.")
        return

    # Parse selection
    selected_indices = set()
    if user_input.lower() == "all":
        selected_indices = set(range(len(names)))
    else:
        for part in user_input.split(","):
            part = part.strip()
            if "-" in part:
                try:
                    start, end = part.split("-", 1)
                    for i in range(int(start), int(end) + 1):
                        selected_indices.add(i - 1)
                except ValueError:
                    print(f"Invalid range: {part}")
                    return
            else:
                try:
                    selected_indices.add(int(part) - 1)
                except ValueError:
                    print(f"Invalid number: {part}")
                    return

    # Validate indices
    selected_names = []
    for i in sorted(selected_indices):
        if 0 <= i < len(names):
            selected_names.append(names[i])
        else:
            print(f"Index out of range: {i + 1}")
            return

    if not selected_names:
        print("No valid collections selected.")
        return

    # Confirm
    print(f"\nAbout to DELETE {len(selected_names)} collection(s):")
    for name in selected_names:
        print(f"  - {name}")

    confirm = input("\nType 'yes' to confirm deletion: ").strip()
    if confirm.lower() != "yes":
        print("Cancelled.")
        return

    # Execute deletion
    for name in selected_names:
        drop_collection(client, name)

    print("\nDone.")


def drop_collection(client: MilvusClient, name: str) -> bool:
    """Drop a single collection and its aliases. Returns True on success."""
    try:
        try:
            aliases = utility.list_aliases(collection_name=name, using="_admin")
            for alias in aliases:
                utility.drop_alias(alias, using="_admin")
                print(f"  Dropped alias: {alias}")
        except Exception:
            pass

        client.drop_collection(name)
        print(f"  Deleted: {name}")
        return True
    except Exception as e:
        print(f"  Failed to delete {name}: {e}")
        return False


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Browse and manage Milvus collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # List all collections
  python src/bootstrap.py src/devops_scripts/milvus_admin/browse_collections.py

  # Filter by name
  python src/bootstrap.py src/devops_scripts/milvus_admin/browse_collections.py --filter episodic

  # Enter interactive delete mode
  python src/bootstrap.py src/devops_scripts/milvus_admin/browse_collections.py --delete

  # Delete all collections with a specific prefix
  python src/bootstrap.py src/devops_scripts/milvus_admin/browse_collections.py --prefix v1_episodic --drop

  # Specify database
  python src/bootstrap.py src/devops_scripts/milvus_admin/browse_collections.py --db my_database
        """,
    )

    parser.add_argument(
        "--filter",
        "-f",
        type=str,
        default=None,
        help="Filter collections by name substring (case-insensitive)",
    )
    parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        default=None,
        help="Match collections by name prefix (exact, case-sensitive)",
    )
    parser.add_argument(
        "--delete",
        "-d",
        action="store_true",
        help="Enter interactive delete mode after listing",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Delete matched collections (use with --prefix or --filter)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="",
        help="Milvus database name (default: from env or 'default')",
    )

    args = parser.parse_args(argv)

    client = _connect(db_name=args.db)

    try:
        # --prefix --drop: batch delete by prefix
        if args.drop and (args.prefix or args.filter):
            names = list_collections(
                client, name_filter=args.filter, prefix=args.prefix
            )
            show_collection_details(client, names)
            if names:
                label = (
                    f"prefix '{args.prefix}'"
                    if args.prefix
                    else f"filter '{args.filter}'"
                )
                print(
                    f"About to DELETE all {len(names)} collection(s) matching {label}."
                )
                confirm = input("Type 'yes' to confirm deletion: ").strip()
                if confirm.lower() == "yes":
                    success = sum(1 for n in names if drop_collection(client, n))
                    print(f"\nDone. Deleted: {success}, Failed: {len(names) - success}")
                else:
                    print("Cancelled.")
        elif args.drop:
            print("Error: --drop requires --prefix or --filter to select collections.")
            return 1
        else:
            names = list_collections(
                client, name_filter=args.filter, prefix=args.prefix
            )
            show_collection_details(client, names)
            if args.delete:
                interactive_delete(client, names)
    finally:
        try:
            client.close()
            connections.disconnect("_admin")
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
