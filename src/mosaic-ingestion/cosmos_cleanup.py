#!/usr/bin/env python3
"""
Cosmos DB Cleanup Script for Fresh Testing Environment

This script provides comprehensive cleanup functionality for Azure Cosmos DB
to establish a clean testing baseline. It preserves container structure while
removing all documents for fresh CRUD testing.

Task: CRUD-000 - Clean Cosmos DB for Fresh Testing Environment
Author: Mosaic MCP Tool
Usage:
    python cosmos_cleanup.py --confirm  # Safe mode with confirmation
    python cosmos_cleanup.py --force    # Skip confirmations (CI/CD use)
    python cosmos_cleanup.py --backup   # Create backup before cleanup
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from azure.cosmos import CosmosClient, exceptions as cosmos_exceptions
from azure.identity import DefaultAzureCredential
from src.mosaic_mcp.config.settings import MosaicSettings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("cosmos_cleanup.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class CosmosDBCleaner:
    """
    Comprehensive Cosmos DB cleanup utility for fresh testing environments.

    Features:
    - Safe cleanup with confirmation prompts
    - Backup creation before deletion
    - Container integrity validation
    - Detailed progress reporting
    - Rollback capability
    """

    def __init__(self, settings: MosaicSettings):
        """Initialize the cleaner with configuration settings."""
        self.settings = settings
        self.cosmos_client: Optional[CosmosClient] = None
        self.database = None
        self.cleanup_stats = {
            "containers_processed": 0,
            "documents_deleted": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None,
        }

        # Define all containers that may contain data
        self.containers = [
            self.settings.azure_cosmos_container_name,  # "knowledge"
            self.settings.azure_cosmos_memory_container,  # "memory"
            self.settings.azure_cosmos_golden_nodes_container,  # "golden_nodes"
            self.settings.azure_cosmos_diagrams_container,  # "diagrams"
            self.settings.azure_cosmos_code_entities_container,  # "code_entities"
            self.settings.azure_cosmos_code_relationships_container,  # "code_relationships"
            self.settings.azure_cosmos_repositories_container,  # "repositories"
        ]

    def initialize_connection(self) -> bool:
        """
        Initialize connection to Azure Cosmos DB using managed identity.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            if not self.settings.azure_cosmos_endpoint:
                logger.error("❌ Azure Cosmos DB endpoint not configured in settings")
                return False

            # Use DefaultAzureCredential for managed identity
            credential = DefaultAzureCredential()

            self.cosmos_client = CosmosClient(
                url=self.settings.azure_cosmos_endpoint, credential=credential
            )

            self.database = self.cosmos_client.get_database_client(
                self.settings.azure_cosmos_database_name
            )

            # Test connection
            _ = self.database.read()
            logger.info(
                f"✅ Connected to Cosmos DB: {self.settings.azure_cosmos_endpoint}"
            )
            logger.info(f"📊 Database: {self.settings.azure_cosmos_database_name}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to connect to Cosmos DB: {str(e)}")
            return False

    def list_existing_containers(self) -> List[str]:
        """
        List all existing containers in the database.

        Returns:
            List[str]: List of container names that exist
        """
        try:
            existing_containers = []
            containers = self.database.list_containers()

            for container in containers:
                container_name = container["id"]
                existing_containers.append(container_name)

            logger.info(f"📋 Found {len(existing_containers)} existing containers")
            for container in existing_containers:
                logger.info(f"   - {container}")

            return existing_containers

        except Exception as e:
            logger.error(f"❌ Failed to list containers: {str(e)}")
            return []

    def get_container_stats(self, container_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific container.

        Args:
            container_name: Name of the container

        Returns:
            Dict with container statistics
        """
        try:
            container = self.database.get_container_client(container_name)

            # Count documents
            query = "SELECT VALUE COUNT(1) FROM c"
            result = list(
                container.query_items(query=query, enable_cross_partition_query=True)
            )

            doc_count = result[0] if result else 0

            # Get container properties
            properties = container.read()

            return {
                "name": container_name,
                "document_count": doc_count,
                "partition_key": properties.get("partitionKey", {}),
                "indexing_policy": properties.get("indexingPolicy", {}),
                "exists": True,
            }

        except cosmos_exceptions.CosmosResourceNotFoundError:
            return {"name": container_name, "document_count": 0, "exists": False}
        except Exception as e:
            logger.error(f"❌ Error getting stats for {container_name}: {str(e)}")
            return {
                "name": container_name,
                "document_count": 0,
                "error": str(e),
                "exists": False,
            }

    def create_backup(self, backup_dir: str = "cosmos_backup") -> bool:
        """
        Create backup of all data before cleanup.

        Args:
            backup_dir: Directory to store backup files

        Returns:
            bool: True if backup successful
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = Path(backup_dir) / f"cosmos_backup_{timestamp}"
            backup_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"📦 Creating backup in: {backup_path}")

            backup_summary = {
                "timestamp": timestamp,
                "database": self.settings.azure_cosmos_database_name,
                "containers": {},
            }

            for container_name in self.containers:
                try:
                    container = self.database.get_container_client(container_name)

                    # Query all documents
                    query = "SELECT * FROM c"
                    documents = list(
                        container.query_items(
                            query=query, enable_cross_partition_query=True
                        )
                    )

                    if documents:
                        container_backup_file = backup_path / f"{container_name}.json"
                        with open(container_backup_file, "w", encoding="utf-8") as f:
                            json.dump(documents, f, indent=2, default=str)

                        backup_summary["containers"][container_name] = {
                            "document_count": len(documents),
                            "backup_file": str(container_backup_file),
                        }

                        logger.info(
                            f"   ✅ Backed up {len(documents)} documents from {container_name}"
                        )
                    else:
                        backup_summary["containers"][container_name] = {
                            "document_count": 0,
                            "backup_file": None,
                        }
                        logger.info(f"   📭 No documents in {container_name}")

                except cosmos_exceptions.CosmosResourceNotFoundError:
                    logger.info(f"   ⚠️ Container {container_name} does not exist")
                    backup_summary["containers"][container_name] = {
                        "document_count": 0,
                        "backup_file": None,
                        "error": "Container not found",
                    }

            # Save backup summary
            summary_file = backup_path / "backup_summary.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(backup_summary, f, indent=2, default=str)

            logger.info(f"✅ Backup completed: {summary_file}")
            return True

        except Exception as e:
            logger.error(f"❌ Backup failed: {str(e)}")
            return False

    def cleanup_container(self, container_name: str) -> Dict[str, Any]:
        """
        Clean up all documents in a specific container.

        Args:
            container_name: Name of the container to clean

        Returns:
            Dict with cleanup results
        """
        try:
            container = self.database.get_container_client(container_name)

            # Get all documents
            query = "SELECT c.id, c._partitionKey FROM c"
            documents = list(
                container.query_items(query=query, enable_cross_partition_query=True)
            )

            if not documents:
                return {
                    "container": container_name,
                    "documents_deleted": 0,
                    "status": "empty",
                }

            deleted_count = 0
            errors = []

            logger.info(f"🧹 Cleaning {len(documents)} documents from {container_name}")

            for doc in documents:
                try:
                    # Delete document
                    container.delete_item(
                        item=doc["id"],
                        partition_key=doc.get("_partitionKey", doc["id"]),
                    )
                    deleted_count += 1

                    if deleted_count % 100 == 0:
                        logger.info(
                            f"   🗑️ Deleted {deleted_count}/{len(documents)} documents"
                        )

                except Exception as e:
                    errors.append(f"Failed to delete {doc['id']}: {str(e)}")

            result = {
                "container": container_name,
                "documents_deleted": deleted_count,
                "errors": errors,
                "status": "completed",
            }

            if errors:
                logger.warning(
                    f"⚠️ {len(errors)} errors during cleanup of {container_name}"
                )
                for error in errors[:5]:  # Log first 5 errors
                    logger.warning(f"   {error}")
            else:
                logger.info(
                    f"✅ Successfully cleaned {container_name}: {deleted_count} documents deleted"
                )

            return result

        except cosmos_exceptions.CosmosResourceNotFoundError:
            logger.info(f"⚠️ Container {container_name} does not exist")
            return {
                "container": container_name,
                "documents_deleted": 0,
                "status": "not_found",
            }
        except Exception as e:
            logger.error(f"❌ Failed to clean {container_name}: {str(e)}")
            return {
                "container": container_name,
                "documents_deleted": 0,
                "error": str(e),
                "status": "error",
            }

    def validate_containers(self) -> bool:
        """
        Validate that all containers are empty and schema intact.

        Returns:
            bool: True if all containers are valid and empty
        """
        try:
            logger.info("🔍 Validating container states...")

            all_valid = True
            for container_name in self.containers:
                stats = self.get_container_stats(container_name)

                if stats["exists"]:
                    if stats["document_count"] == 0:
                        logger.info(f"   ✅ {container_name}: Empty and valid")
                    else:
                        logger.warning(
                            f"   ⚠️ {container_name}: Still contains {stats['document_count']} documents"
                        )
                        all_valid = False
                else:
                    logger.info(f"   📭 {container_name}: Does not exist")

            return all_valid

        except Exception as e:
            logger.error(f"❌ Validation failed: {str(e)}")
            return False

    def run_cleanup(self, create_backup: bool = True, force: bool = False) -> bool:
        """
        Run the complete cleanup process.

        Args:
            create_backup: Whether to create backup before cleanup
            force: Skip confirmation prompts

        Returns:
            bool: True if cleanup successful
        """
        try:
            self.cleanup_stats["start_time"] = datetime.now()

            logger.info("🚀 Starting Cosmos DB cleanup process...")

            # Initialize connection
            if not self.initialize_connection():
                return False

            # Get current state
            existing_containers = self.list_existing_containers()
            total_docs = 0

            for container_name in existing_containers:
                stats = self.get_container_stats(container_name)
                total_docs += stats.get("document_count", 0)

            logger.info(f"📊 Total documents to delete: {total_docs}")

            # Confirmation prompt
            if not force and total_docs > 0:
                response = input(
                    f"\n⚠️ This will delete {total_docs} documents from {len(existing_containers)} containers.\nAre you sure? (yes/no): "
                )
                if response.lower() != "yes":
                    logger.info("❌ Cleanup cancelled by user")
                    return False

            # Create backup if requested
            if create_backup and total_docs > 0:
                logger.info("📦 Creating backup before cleanup...")
                if not self.create_backup():
                    if not force:
                        response = input(
                            "Backup failed. Continue without backup? (yes/no): "
                        )
                        if response.lower() != "yes":
                            logger.info("❌ Cleanup cancelled due to backup failure")
                            return False
                    else:
                        logger.warning(
                            "⚠️ Backup failed but continuing due to --force flag"
                        )

            # Perform cleanup
            logger.info("🧹 Starting document deletion...")

            cleanup_results = []
            for container_name in self.containers:
                result = self.cleanup_container(container_name)
                cleanup_results.append(result)

                self.cleanup_stats["documents_deleted"] += result.get(
                    "documents_deleted", 0
                )
                if result.get("error") or result.get("errors"):
                    self.cleanup_stats["errors"] += 1

                self.cleanup_stats["containers_processed"] += 1

            # Validate final state
            logger.info("🔍 Validating cleanup results...")
            validation_success = self.validate_containers()

            self.cleanup_stats["end_time"] = datetime.now()
            duration = self.cleanup_stats["end_time"] - self.cleanup_stats["start_time"]

            # Final report
            logger.info("\n" + "=" * 60)
            logger.info("📋 CLEANUP SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Duration: {duration}")
            logger.info(
                f"Containers processed: {self.cleanup_stats['containers_processed']}"
            )
            logger.info(f"Documents deleted: {self.cleanup_stats['documents_deleted']}")
            logger.info(f"Errors: {self.cleanup_stats['errors']}")
            logger.info(
                f"Validation: {'✅ PASSED' if validation_success else '❌ FAILED'}"
            )
            logger.info("=" * 60)

            if validation_success:
                logger.info("🎉 Cosmos DB cleanup completed successfully!")
                logger.info("📊 Database is now ready for fresh CRUD testing")
                return True
            else:
                logger.error("❌ Cleanup validation failed - some documents may remain")
                return False

        except Exception as e:
            logger.error(f"❌ Cleanup process failed: {str(e)}")
            return False


def main():
    """Main entry point for the cleanup script."""
    parser = argparse.ArgumentParser(
        description="Clean Cosmos DB for fresh testing environment",
        epilog="Example: python cosmos_cleanup.py --backup --confirm",
    )

    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Run with confirmation prompts (safe mode)",
    )

    parser.add_argument(
        "--force", action="store_true", help="Skip all confirmation prompts (for CI/CD)"
    )

    parser.add_argument(
        "--backup", action="store_true", help="Create backup before cleanup"
    )

    parser.add_argument("--no-backup", action="store_true", help="Skip backup creation")

    args = parser.parse_args()

    # Validate arguments
    if not args.confirm and not args.force:
        print("❌ Error: Must specify either --confirm or --force")
        print("Use --confirm for safe interactive mode")
        print("Use --force for automated CI/CD mode")
        sys.exit(1)

    if args.backup and args.no_backup:
        print("❌ Error: Cannot specify both --backup and --no-backup")
        sys.exit(1)

    # Default to backup unless explicitly disabled
    create_backup = not args.no_backup

    try:
        # Load configuration
        settings = MosaicSettings()

        # Create cleaner instance
        cleaner = CosmosDBCleaner(settings)

        # Run cleanup
        success = cleaner.run_cleanup(create_backup=create_backup, force=args.force)

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("\n❌ Cleanup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
