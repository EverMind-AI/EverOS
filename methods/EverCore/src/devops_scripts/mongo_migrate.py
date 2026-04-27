#!/usr/bin/env python3
"""
MongoDB migration management script.

This is a wrapper around the core.oxm.mongo.migration module for backward compatibility.
Please use 'python -m core.oxm.mongo.migration.cli' for new migrations.

Usage:
    python migrate.py new-migration -n migration_name    # Create new migration
    python migrate.py migrate                           # Run all migrations
    python migrate.py migrate --distance 1              # Run 1 migration
    python migrate.py migrate --backward                # Roll back all migrations
    python migrate.py migrate --backward --distance 1   # Roll back 1 migration
    python migrate.py migrate --no-use-transaction      # Run without transactions
    python migrate.py --uri mongodb://...              # Specify MongoDB URI
    python migrate.py --help                           # Show help
"""

from core.oxm.mongo.migration.cli import main

if __name__ == "__main__":
    main()
