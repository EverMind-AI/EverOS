#!/usr/bin/env python3
"""
Add Timestamp Shard

Add timestamp-based sharding configuration to the MemCell collection
Created: 2025-09-11T23:37:54.703305
"""

import asyncio
import logging
from common_utils.datetime_utils import get_now_with_timezone

from pymongo.errors import OperationFailure

from infra_layer.adapters.out.persistence.document.memory.memcell import MemCell

logger = logging.getLogger(__name__)


async def enable_timestamp_sharding(session=None):
    """
    Enable timestamp sharding for the MemCell collection
    """
    try:
        # Get MongoDB collection and client
        collection = MemCell.get_pymongo_collection()
        db = collection.database
        client = db.client
        admin_db = client.admin

        logger.info("🔧 Starting timestamp sharding configuration...")

        # 1. Check if it's a sharded cluster
        try:
            shard_status = await admin_db.command('listShards')
            if not shard_status.get('shards'):
                logger.warning(
                    "⚠️  Current environment is not a sharded cluster, skipping sharding configuration"
                )
                return
            logger.info(
                f"✅ Sharded cluster detected, total {len(shard_status['shards'])} shards"
            )
        except OperationFailure as e:
            logger.warning(
                f"⚠️  Unable to check sharding status: {e}, may not be a sharded environment"
            )
            return

        # 2. Enable database sharding
        try:
            await admin_db.command('enableSharding', db.name)
            logger.info(f"✅ Sharding enabled for database '{db.name}'")
        except OperationFailure as e:
            if "already enabled" in str(e).lower():
                logger.info(f"📝 Sharding already exists for database '{db.name}'")
            else:
                logger.error(f"❌ Failed to enable database sharding: {e}")
                raise

        # 3. Set collection shard key - timestamp
        collection_name = f"{db.name}.memcells"
        try:
            await admin_db.command(
                'shardCollection', collection_name, key={"timestamp": 1}
            )
            logger.info("✅ Shard key configuration for MemCell collection completed")
        except OperationFailure as e:
            if "already sharded" in str(e).lower():
                logger.info("📝 Sharding already exists for MemCell collection")
            else:
                logger.error(f"❌ Failed to set collection sharding: {e}")
                raise

        # 4. Create pre-split chunks (optional, improves initial performance)
        try:
            from datetime import timedelta

            # Create pre-split points for the next 12 months
            base_date = get_now_with_timezone().replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )
            split_points = []

            for i in range(1, 13):  # Next 12 months
                split_date = base_date + timedelta(days=30 * i)
                split_points.append({"timestamp": split_date})

            # Execute pre-splitting
            for point in split_points:
                try:
                    await admin_db.command('split', collection_name, middle=point)
                    logger.debug(f"📅 Created split point: {point['timestamp']}")
                except OperationFailure as e:
                    if "already exists" not in str(e).lower():
                        logger.debug(f"Failed to create pre-split point: {e}")

            logger.info(f"✅ Created {len(split_points)} pre-split points")

        except Exception as e:
            logger.warning(f"⚠️  Pre-splitting creation failed: {e}")

        # 5. Verify sharding configuration
        try:
            shard_info = await db.command('collStats', 'memcells')

            if shard_info.get('sharded'):
                logger.info(
                    "✅ MemCell collection sharding configuration verified successfully"
                )
                logger.info(f"📊 Shard key: {shard_info.get('shardKey', {})}")
            else:
                logger.warning("⚠️  Sharding configuration verification failed")

        except Exception as e:
            logger.warning(f"⚠️  Sharding verification failed: {e}")

        logger.info("🎉 Timestamp sharding configuration completed")

    except Exception as e:
        logger.error(f"❌ Error occurred during sharding configuration: {e}")
        raise


async def disable_timestamp_sharding(session=None):
    """
    Warning: Disabling sharding is a dangerous operation, generally not recommended in production environments
    """
    logger.warning(
        "⚠️  Disabling sharding is a dangerous operation, requires manual handling by administrator"
    )
    logger.info(
        "📝 Please manually execute the following MongoDB commands to disable sharding:"
    )
    logger.info("   1. Stop balancer: sh.stopBalancer()")
    logger.info("   2. Wait for balancer to complete: sh.waitForBalancer()")
    logger.info(
        "   3. Removing sharding configuration requires recreating the collection"
    )


async def main():
    """Main function"""
    # Execute sharding configuration
    await enable_timestamp_sharding()


if __name__ == "__main__":
    # Run main function
    asyncio.run(main())
