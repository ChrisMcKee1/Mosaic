#!/usr/bin/env python3
"""
Simple Cosmos DB Cleanup Script
Supports both local Docker emulator and Azure cloud modes.
Clears all data from Cosmos DB containers for fresh ingestion testing.

Usage:
    python simple_cosmos_cleanup.py --mode local --check    # Check local emulator
    python simple_cosmos_cleanup.py --mode azure --check    # Check Azure cloud
    python simple_cosmos_cleanup.py --mode local --force    # Clean local emulator
    python simple_cosmos_cleanup.py --mode azure --force    # Clean Azure cloud
"""

import logging
from typing import List, Optional

# Import our dual-mode manager
from cosmos_mode_manager import CosmosModeManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Check if we can import Azure SDK
try:
    from azure.cosmos import CosmosClient, exceptions as cosmos_exceptions
    from azure.identity import DefaultAzureCredential

    AZURE_SDK_AVAILABLE = True
    logger.info("✅ Azure Cosmos SDK available")
except ImportError as e:
    logger.warning(f"⚠️ Azure Cosmos SDK not available: {e}")
    AZURE_SDK_AVAILABLE = False


class SimpleCosmosCleaner:
    """Simple Cosmos DB cleaner with dual-mode support."""

    def __init__(self, mode: Optional[str] = None):
        self.stats = {"containers_found": 0, "documents_deleted": 0, "errors": 0}

        # Initialize the dual-mode manager
        self.cosmos_manager = CosmosModeManager(mode)

        # Common container names used in Mosaic
        self.expected_containers = [
            "knowledge",
            "memory",
            "golden_nodes",
            "diagrams",
            "code_entities",
            "code_relationships",
            "repositories",
            "mosaic",
            "context",
        ]

    def test_connection(self) -> bool:
        """Test connection to Cosmos DB."""
        if not AZURE_SDK_AVAILABLE:
            logger.error("❌ Azure SDK not available")
            return False

        client = self.cosmos_manager.get_cosmos_client()
        if not client:
            logger.error("❌ Failed to create Cosmos client")
            return False

        try:
            database_name = self.cosmos_manager.config["database"]
            database = client.get_database_client(database_name)
            # Try to read database properties
            database.read()
            logger.info(f"✅ Connected to database: {database_name}")
            logger.info(f"🔧 Mode: {self.cosmos_manager.get_current_mode()}")
            return True
        except cosmos_exceptions.CosmosResourceNotFoundError:
            logger.warning(f"⚠️ Database '{database_name}' not found")
            return False
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False

    def list_containers(self) -> List[str]:
        """List all containers in the database."""
        client = self.cosmos_manager.get_cosmos_client()
        if not client:
            return []

        try:
            database_name = self.cosmos_manager.config["database"]
            database = client.get_database_client(database_name)
            containers = list(database.list_containers())
            container_names = [c["id"] for c in containers]

            self.stats["containers_found"] = len(container_names)
            logger.info(f"📋 Found {len(container_names)} containers:")
            for name in container_names:
                logger.info(f"   • {name}")

            return container_names

        except Exception as e:
            logger.error(f"❌ Failed to list containers: {e}")
            self.stats["errors"] += 1
            return []

    def count_documents(self, container_name: str) -> int:
        """Count documents in a container."""
        client = self.cosmos_manager.get_cosmos_client()
        if not client:
            return 0

        try:
            database_name = self.cosmos_manager.config["database"]
            database = client.get_database_client(database_name)
            container = database.get_container_client(container_name)

            # Query to count all documents
            query = "SELECT VALUE COUNT(1) FROM c"
            results = list(
                container.query_items(query, enable_cross_partition_query=True)
            )
            count = results[0] if results else 0

            logger.info(f"📊 Container '{container_name}': {count} documents")
            return count

        except Exception as e:
            logger.error(f"❌ Failed to count documents in '{container_name}': {e}")
            self.stats["errors"] += 1
            return 0

    def delete_all_documents(self, container_name: str) -> int:
        """Delete all documents from a container."""
        client = self.cosmos_manager.get_cosmos_client()
        if not client:
            return 0

        try:
            database_name = self.cosmos_manager.config["database"]
            database = client.get_database_client(database_name)
            container = database.get_container_client(container_name)

            # Get all document IDs and partition keys
            query = "SELECT c.id, c._partitionKey FROM c"
            documents = list(
                container.query_items(query, enable_cross_partition_query=True)
            )

            deleted_count = 0
            for doc in documents:
                try:
                    container.delete_item(
                        item=doc["id"],
                        partition_key=doc.get("_partitionKey", doc["id"]),
                    )
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"⚠️ Failed to delete document {doc['id']}: {e}")
                    self.stats["errors"] += 1

            logger.info(f"🗑️ Deleted {deleted_count} documents from '{container_name}'")
            self.stats["documents_deleted"] += deleted_count
            return deleted_count

        except Exception as e:
            logger.error(f"❌ Failed to delete documents from '{container_name}': {e}")
            self.stats["errors"] += 1
            return 0

    def cleanup_database(self, force: bool = False) -> bool:
        """Clean up all containers in the database."""
        logger.info("🧹 Starting Cosmos DB cleanup...")
        logger.info(f"🔧 Mode: {self.cosmos_manager.get_current_mode()}")

        # Test connection
        if not self.test_connection():
            return False

        # List containers
        containers = self.list_containers()
        if not containers:
            logger.info("✅ No containers found - nothing to clean")
            return True

        # Count total documents
        total_docs = 0
        for container_name in containers:
            count = self.count_documents(container_name)
            total_docs += count

        if total_docs == 0:
            logger.info("✅ No documents found - database is already clean")
            return True

        # Confirm deletion
        if not force:
            logger.warning(
                f"⚠️ This will delete {total_docs} documents from {len(containers)} containers!"
            )
            response = input("Do you want to continue? (yes/no): ").lower().strip()
            if response not in ["yes", "y"]:
                logger.info("❌ Cleanup cancelled by user")
                return False

        # Delete all documents
        logger.info(f"🗑️ Deleting {total_docs} documents...")
        for container_name in containers:
            self.delete_all_documents(container_name)

        # Report results
        logger.info("✅ Cleanup completed!")
        logger.info("📊 Final stats:")
        logger.info(f"   • Containers processed: {self.stats['containers_found']}")
        logger.info(f"   • Documents deleted: {self.stats['documents_deleted']}")
        logger.info(f"   • Errors: {self.stats['errors']}")

        return self.stats["errors"] == 0


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Clean up Cosmos DB for fresh testing")
    parser.add_argument(
        "--mode",
        choices=["local", "azure"],
        help="Cosmos DB mode (local emulator or Azure cloud)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompts"
    )
    parser.add_argument("--check", action="store_true", help="Only check current state")

    args = parser.parse_args()

    cleaner = SimpleCosmosCleaner(mode=args.mode)

    if args.check:
        # Just check the current state
        if cleaner.test_connection():
            containers = cleaner.list_containers()
            total_docs = 0
            for container_name in containers:
                count = cleaner.count_documents(container_name)
                total_docs += count
            logger.info(f"📊 Total documents in database: {total_docs}")
        return 0

    # Run cleanup
    success = cleaner.cleanup_database(force=args.force)
    return 0 if success else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
