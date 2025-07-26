#!/usr/bin/env python3
"""
Initialize local Cosmos DB emulator with a test database and containers.
"""

import logging
from cosmos_mode_manager import CosmosModeManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Initialize the local Cosmos DB with test data."""

    # Create manager in local mode
    manager = CosmosModeManager("local")

    logger.info("🚀 Initializing local Cosmos DB emulator...")
    logger.info(f"📍 Endpoint: {manager.config['endpoint']}")
    logger.info(f"📊 Database: {manager.config['database']}")

    try:
        # Get client
        client = manager.get_cosmos_client()
        database_name = manager.config["database"]

        # Create database
        logger.info(f"📂 Creating database: {database_name}")
        database = client.create_database_if_not_exists(database_name)
        logger.info(f"✅ Database ready: {database_name}")

        # Create some test containers
        test_containers = ["knowledge", "memory", "repositories", "diagrams"]

        for container_name in test_containers:
            logger.info(f"📦 Creating container: {container_name}")
            container = database.create_container_if_not_exists(
                id=container_name, partition_key={"paths": ["/id"], "kind": "Hash"}
            )

            # Add a test document
            test_doc = {
                "id": f"test-{container_name}-001",
                "type": "test_document",
                "container": container_name,
                "created_by": "initialization_script",
                "data": f"Sample data for {container_name}",
            }
            container.create_item(test_doc)
            logger.info(f"   • Added test document to {container_name}")

        logger.info("🎉 Local Cosmos DB initialization completed!")
        logger.info("💡 You can now test the cleanup script with real data")

        return True

    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
