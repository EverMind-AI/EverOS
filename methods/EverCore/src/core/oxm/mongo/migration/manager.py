"""
MongoDB migration manager module.

This module provides a high-level interface for managing MongoDB database migrations
using Beanie as the underlying migration engine.
"""

import os
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from common_utils.datetime_utils import get_now_with_timezone
from common_utils.project_path import CURRENT_DIR
from pymongo import MongoClient

# Module-level logger for this file
logger = logging.getLogger(__name__)


class MigrationManager:
    """Migration manager for MongoDB using Beanie"""

    MIGRATIONS_DIR = CURRENT_DIR / "migrations" / "mongodb"

    # Default migration template
    MIGRATION_TEMPLATE = '''"""
{description}

Created at: {created_at}
"""

from beanie import Document
from beanie import iterative_migration, free_fall_migration
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT


class Forward:
    """Forward migration"""
    
    # Example: Iterative migration (recommended)
    # @iterative_migration()
    # async def update_field(self, input_document: OldModel, output_document: NewModel):
    #     output_document.new_field = input_document.old_field
    
    # Example: Free fall migration (flexible)
    # @free_fall_migration(document_models=[YourModel])
    # async def create_indexes(self, session):
    #     # Get collection
    #     collection = YourModel.get_pymongo_collection()
    #     
    #     # Create indexes
    #     indexes = [
    #         IndexModel([("field_name", ASCENDING)], name="idx_field_name")
    #     ]
    #     await collection.create_indexes(indexes)
    
    pass


class Backward:
    """Backward migration"""
    
    # @iterative_migration()
    # async def revert_field(self, input_document: NewModel, output_document: OldModel):
    #     output_document.old_field = input_document.new_field
    
    # @free_fall_migration(document_models=[YourModel])
    # async def drop_indexes(self, session):
    #     collection = YourModel.get_pymongo_collection()
    #     await collection.drop_index("idx_field_name")
    
    pass
'''

    def __init__(
        self,
        uri: Optional[str] = None,
        database: Optional[str] = None,
        migrations_path: Optional[Path] = None,
        use_transaction: bool = True,
        distance: Optional[int] = None,
        backward: bool = False,
        stream_output: bool = True,
    ):
        """
        Initialize migration manager

        Args:
            uri: MongoDB connection URI. If not provided, load from env.
            database: MongoDB database name. If not provided, load from env.
            migrations_path: Directory of migration files. Defaults to MIGRATIONS_DIR.
            use_transaction: Whether to use transactions (requires replica set).
            distance: Number of migrations to apply (positive integer).
            backward: Whether to perform rollback.
        """
        self.uri = uri or self._get_mongodb_uri()
        self.database = database or self._get_mongodb_database()
        self.migrations_path = migrations_path or self.MIGRATIONS_DIR
        self.use_transaction = use_transaction
        self.distance = distance
        self.backward = backward
        self.stream_output = stream_output

        if not self.uri:
            raise ValueError("MongoDB URI cannot be empty")
        if not self.database:
            raise ValueError("MongoDB database name cannot be empty")
        if not self.migrations_path:
            raise ValueError("Migrations path cannot be empty")

        self._ensure_migrations_dir()

    @classmethod
    def _get_mongodb_uri(cls) -> str:
        """Get MongoDB URI from environment variables"""
        base_uri = None
        if uri := os.getenv("MONGODB_URI"):
            base_uri = uri
        else:
            # Build URI from separate environment variables
            host = os.getenv("MONGODB_HOST", "localhost")
            port = os.getenv("MONGODB_PORT", "27017")
            username = os.getenv("MONGODB_USERNAME", "")
            password = os.getenv("MONGODB_PASSWORD", "")
            database = cls._get_mongodb_database()

            if username and password:
                base_uri = f"mongodb://{username}:{password}@{host}:{port}/{database}"
            else:
                base_uri = f"mongodb://{host}:{port}/{database}"

        # Append URI parameters (if any)
        uri_params = os.getenv("MONGODB_URI_PARAMS", "").strip()
        if uri_params:
            separator = '&' if ('?' in base_uri) else '?'
            return f"{base_uri}{separator}{uri_params}"
        return base_uri

    @staticmethod
    def _get_mongodb_database() -> str:
        """Get MongoDB database name from environment"""
        return os.getenv("MONGODB_DATABASE", "memsys")

    def _ensure_migrations_dir(self):
        """Ensure migrations directory exists"""
        self.migrations_path.mkdir(parents=True, exist_ok=True)

    def create_migration(self, migration_name: str) -> Path:
        """
        Create a new migration file

        Args:
            migration_name: Name of the migration

        Returns:
            Path to the created migration file

        Raises:
            FileExistsError: If migration file already exists
        """
        # Generate timestamp
        timestamp = get_now_with_timezone().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{migration_name}.py"
        filepath = self.migrations_path / filename

        # Check if file already exists
        if filepath.exists():
            raise FileExistsError(f"Migration file already exists: {filepath}")

        # Generate migration content
        content = self.MIGRATION_TEMPLATE.format(
            description=migration_name.replace("_", " ").title(),
            created_at=get_now_with_timezone().isoformat(),
        )

        # Write file
        filepath.write_text(content, encoding='utf-8')
        logger.info(f"✅ Created migration file: {filepath}")

        return filepath

    def run_migration(self) -> int:
        """
        Run migration using Beanie

        Returns:
            Exit code from Beanie command
        """
        # Build beanie args
        beanie_args = ["migrate"]
        if self.distance is not None:
            if self.distance <= 0:
                raise ValueError("Migration distance must be positive")
            beanie_args.extend(["--distance", str(self.distance)])
        if self.backward:
            beanie_args.append("--backward")
        if not self.use_transaction:
            beanie_args.append("--no-use-transaction")

        # Build complete command
        cmd = [
            "beanie",
            *beanie_args,
            "-uri",
            self.uri,
            "-db",
            self.database,
            "-p",
            str(self.migrations_path),
        ]

        logger.info(f"🚀 Executing command: {' '.join(cmd[3:])}")  # Hide python path
        logger.info(f"📍 Database: {self.database}")
        logger.info(f"📁 Migration directory: {self.migrations_path}")

        # Check if there are migration files in the directory
        migration_files = list(self.migrations_path.glob("*.py"))
        migration_files = [f for f in migration_files if not f.name.startswith("_")]
        if not migration_files:
            logger.info("🧭 No migration files found in directory, skipping migration")
            return 0
        logger.info(f"📄 Found {len(migration_files)} migration files")

        # Snapshot migration logs before running
        before_names, before_current = self._snapshot_migration_log()
        if before_names is not None:
            logger.info(f"🧭 Number of records before migration: {len(before_names)}")
            logger.info(
                f"⭐ Current pointer before migration: {before_current or '<none>'}"
            )
        else:
            logger.info(
                "🧭 migrations_log collection not initialized (first migration)"
            )
        try:
            # Execute command
            if self.stream_output:
                # Redirect subprocess output to current process stdout/stderr for real-time printing
                result = subprocess.run(
                    cmd,
                    check=True,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    text=True,
                    env=os.environ.copy(),
                )
                # In streaming mode, output is printed directly, no need to log result.stdout/stderr again
            else:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    env=os.environ.copy(),
                )

                # Log buffered output at the end
                if result.stdout:
                    logger.info(result.stdout)
                if result.stderr:
                    logger.warning(result.stderr)

            # Snapshot and log diff after success
            self._log_migration_diff(before_names, before_current)
            return result.returncode

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Command execution failed: {e}")
            if e.stdout:
                logger.info(f"Standard output: {e.stdout}")
            if e.stderr:
                logger.error(f"Error output: {e.stderr}")
            # Snapshot and log diff even on failure (migration may have partially executed)
            self._log_migration_diff(before_names, before_current)
            return e.returncode

        except FileNotFoundError:
            logger.error(
                "❌ beanie command not found, please ensure beanie is installed"
            )
            logger.error("Installation command: pip install beanie")
            # Snapshot and log diff even if command not found (should be no changes)
            self._log_migration_diff(before_names, before_current)
            return 1

    # ---------- Helper methods for migration log inspection ----------
    def _get_sync_mongo_client(self) -> MongoClient:
        """Create a short-lived sync MongoDB client for inspections."""
        return MongoClient(self.uri)

    def _read_migration_logs(self):
        """Read migrations_log documents sorted by ts ascending.

        Returns:
            Tuple[List[str], Optional[str]] | (None, None) if any error occurs.
        """
        try:
            with self._get_sync_mongo_client() as client:
                db = client[self.database]
                coll = db["migrations_log"]
                docs = list(
                    coll.find({}, {"_id": 0, "name": 1, "is_current": 1, "ts": 1}).sort(
                        "ts", 1
                    )
                )
                names = [d.get("name") for d in docs if d.get("name")]
                current = None
                for d in reversed(docs):
                    if d.get("is_current"):
                        current = d.get("name")
                        break
                return names, current
        except Exception as e:
            logger.warning("Failed to read migration logs: %s", str(e))
            return None, None

    def _snapshot_migration_log(self):
        """Wrapper to snapshot current migration log state."""
        names, current = self._read_migration_logs()
        if names is None:
            return None, None
        return set(names), current

    def _log_migration_diff(self, before_names, before_current) -> None:
        """Compare before/after migration log snapshots and print diffs."""
        after_names, after_current = self._snapshot_migration_log()
        if after_names is None:
            logger.info("🧭 Unable to read post-migration log snapshot")
            return

        logger.info("🧭 Number of records after migration: %d", len(after_names))
        if after_current:
            logger.info("⭐ Current pointer after migration: %s", after_current)
        else:
            logger.info("⭐ Current pointer after migration: <none>")

        if before_names is None:
            return

        added = sorted(list(after_names - before_names))
        removed = sorted(list(before_names - after_names))

        if added:
            logger.info("✅ Newly executed scripts: %s", ", ".join(added))
        else:
            logger.info("✅ Newly executed scripts: <none>")

        if removed:
            logger.info("↩️ Scripts removed due to rollback: %s", ", ".join(removed))
        else:
            logger.info("↩️ Scripts removed due to rollback: <none>")

        if before_current != after_current:
            logger.info(
                "📍 Current pointer changed: %s -> %s",
                before_current or "<none>",
                after_current or "<none>",
            )

    # ---------- Public utility for manual query ----------
    def get_migration_history(self):
        """Return full migration history from migrations_log (sorted by ts asc)."""
        try:
            with self._get_sync_mongo_client() as client:
                db = client[self.database]
                coll = db["migrations_log"]
                docs = list(
                    coll.find({}, {"_id": 0, "name": 1, "is_current": 1, "ts": 1}).sort(
                        "ts", 1
                    )
                )
                return docs
        except Exception as e:
            logger.warning("Failed to get migration history: %s", str(e))
            return []

    def log_migration_history(self) -> None:
        """Log migration history and current pointer."""
        names, current = self._snapshot_migration_log()
        if names is None:
            logger.info("Unable to read migration history")
            return
        logger.info(
            "📜 Recorded migration scripts (%d): %s",
            len(names),
            ", ".join(sorted(names)),
        )
        logger.info("⭐ Current pointer: %s", current or "<none>")

    @classmethod
    def run_migrations_on_startup(cls, enabled: bool = True) -> int:
        """
        Run MongoDB database migrations on application startup

        Execute all pending migration scripts using default configuration (connection info from environment variables)

        Args:
            enabled: Whether to enable migration, False to skip migration step

        Returns:
            int: Exit code from migration execution, 0 means success, -1 means skipped
        """
        if not enabled:
            logger.info(
                "MongoDB startup migration is disabled, skipping migration step"
            )
            return -1

        logger.info("Running MongoDB database migrations...")

        try:
            # Create migration manager instance with default configuration
            migration_manager = cls(
                use_transaction=False,  # Default not to use transaction
                distance=None,  # Execute all pending migrations
                backward=False,  # Do not rollback
                stream_output=True,  # Stream output in real time
            )

            # Execute migration
            logger.info("Starting MongoDB migration operation...")
            exit_code = migration_manager.run_migration()

            if exit_code != 0:
                logger.warning(
                    "⚠️ MongoDB migration process returned non-zero exit code: %s",
                    exit_code,
                )
            else:
                logger.info("✅ MongoDB database migration completed")

            return exit_code

        except Exception as e:
            logger.error("❌ Error during MongoDB migration: %s", str(e))
            return 1
