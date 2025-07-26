"""
Test script for Cosmos DB cleanup functionality.

This script tests the cosmos_cleanup.py script in a safe manner
without actually deleting production data.

Task: CRUD-000 - Clean Cosmos DB for Fresh Testing Environment
Test Coverage: Cleanup script functionality and validation
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, Mock

# Add parent directories to path for imports
test_dir = os.path.dirname(__file__)
ingestion_dir = os.path.dirname(test_dir)
src_dir = os.path.dirname(ingestion_dir)
root_dir = os.path.dirname(src_dir)

sys.path.insert(0, root_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, ingestion_dir)

# Import from the parent src directory structure
sys.path.insert(0, os.path.join(src_dir, "mosaic-mcp"))

try:
    from config.settings import MosaicSettings
except ImportError:
    # Fallback to direct path
    sys.path.insert(0, os.path.join(root_dir, "src", "mosaic-mcp"))
    from config.settings import MosaicSettings

from cosmos_cleanup import CosmosDBCleaner


class TestCosmosDBCleaner(unittest.TestCase):
    """Test cases for CosmosDBCleaner class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_settings = MosaicSettings()
        self.mock_settings.azure_cosmos_endpoint = (
            "https://test-cosmos.documents.azure.com:443/"
        )
        self.mock_settings.azure_cosmos_database_name = "test-db"

        self.cleaner = CosmosDBCleaner(self.mock_settings)

    def test_initialization(self):
        """Test cleaner initialization."""
        self.assertIsNotNone(self.cleaner.settings)
        self.assertIsNone(self.cleaner.cosmos_client)
        self.assertEqual(self.cleaner.cleanup_stats["documents_deleted"], 0)
        self.assertEqual(len(self.cleaner.containers), 7)  # All configured containers

    @patch("cosmos_cleanup.DefaultAzureCredential")
    @patch("cosmos_cleanup.CosmosClient")
    def test_initialize_connection_success(self, mock_cosmos_client, mock_credential):
        """Test successful connection initialization."""
        # Setup mocks
        mock_client_instance = MagicMock()
        mock_database = MagicMock()

        mock_cosmos_client.return_value = mock_client_instance
        mock_client_instance.get_database_client.return_value = mock_database
        mock_database.read.return_value = {"id": "test-db"}

        # Test
        result = self.cleaner.initialize_connection()

        # Assertions
        self.assertTrue(result)
        self.assertIsNotNone(self.cleaner.cosmos_client)
        self.assertIsNotNone(self.cleaner.database)
        mock_cosmos_client.assert_called_once()
        mock_database.read.assert_called_once()

    @patch("cosmos_cleanup.DefaultAzureCredential")
    @patch("cosmos_cleanup.CosmosClient")
    def test_initialize_connection_failure(self, mock_cosmos_client, mock_credential):
        """Test connection initialization failure."""
        # Setup mock to raise exception
        mock_cosmos_client.side_effect = Exception("Connection failed")

        # Test
        result = self.cleaner.initialize_connection()

        # Assertions
        self.assertFalse(result)
        self.assertIsNone(self.cleaner.cosmos_client)

    def test_initialize_connection_no_endpoint(self):
        """Test connection failure when no endpoint configured."""
        self.cleaner.settings.azure_cosmos_endpoint = None

        result = self.cleaner.initialize_connection()

        self.assertFalse(result)

    def test_get_container_stats_existing(self):
        """Test getting stats for existing container."""
        # Setup mocks
        mock_container = MagicMock()
        mock_container.query_items.return_value = [5]  # Document count
        mock_container.read.return_value = {
            "partitionKey": {"paths": ["/id"]},
            "indexingPolicy": {"automatic": True},
        }

        self.cleaner.database = MagicMock()
        self.cleaner.database.get_container_client.return_value = mock_container

        # Test
        stats = self.cleaner.get_container_stats("test-container")

        # Assertions
        self.assertEqual(stats["name"], "test-container")
        self.assertEqual(stats["document_count"], 5)
        self.assertTrue(stats["exists"])
        self.assertIn("partition_key", stats)

    def test_get_container_stats_not_found(self):
        """Test getting stats for non-existent container."""
        from azure.cosmos import exceptions as cosmos_exceptions

        # Setup mocks
        self.cleaner.database = MagicMock()
        self.cleaner.database.get_container_client.side_effect = (
            cosmos_exceptions.CosmosResourceNotFoundError()
        )

        # Test
        stats = self.cleaner.get_container_stats("non-existent")

        # Assertions
        self.assertEqual(stats["name"], "non-existent")
        self.assertEqual(stats["document_count"], 0)
        self.assertFalse(stats["exists"])

    def test_cleanup_container_empty(self):
        """Test cleanup of empty container."""
        # Setup mocks
        mock_container = MagicMock()
        mock_container.query_items.return_value = []  # No documents

        self.cleaner.database = MagicMock()
        self.cleaner.database.get_container_client.return_value = mock_container

        # Test
        result = self.cleaner.cleanup_container("empty-container")

        # Assertions
        self.assertEqual(result["container"], "empty-container")
        self.assertEqual(result["documents_deleted"], 0)
        self.assertEqual(result["status"], "empty")

    def test_cleanup_container_with_documents(self):
        """Test cleanup of container with documents."""
        # Setup mocks
        mock_documents = [
            {"id": "doc1", "_partitionKey": "partition1"},
            {"id": "doc2", "_partitionKey": "partition2"},
        ]

        mock_container = MagicMock()
        mock_container.query_items.return_value = mock_documents
        mock_container.delete_item.return_value = None

        self.cleaner.database = MagicMock()
        self.cleaner.database.get_container_client.return_value = mock_container

        # Test
        result = self.cleaner.cleanup_container("test-container")

        # Assertions
        self.assertEqual(result["container"], "test-container")
        self.assertEqual(result["documents_deleted"], 2)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["errors"]), 0)

        # Verify delete_item called for each document
        self.assertEqual(mock_container.delete_item.call_count, 2)

    def test_list_existing_containers(self):
        """Test listing existing containers."""
        # Setup mocks
        mock_containers = [
            {"id": "container1"},
            {"id": "container2"},
            {"id": "container3"},
        ]

        self.cleaner.database = MagicMock()
        self.cleaner.database.list_containers.return_value = mock_containers

        # Test
        containers = self.cleaner.list_existing_containers()

        # Assertions
        self.assertEqual(len(containers), 3)
        self.assertIn("container1", containers)
        self.assertIn("container2", containers)
        self.assertIn("container3", containers)

    @patch("cosmos_cleanup.Path")
    @patch("builtins.open")
    @patch("json.dump")
    def test_create_backup_success(self, mock_json_dump, mock_open, mock_path):
        """Test successful backup creation."""
        # Setup mocks
        mock_documents = [{"id": "doc1", "data": "test"}]

        mock_container = MagicMock()
        mock_container.query_items.return_value = mock_documents

        self.cleaner.database = MagicMock()
        self.cleaner.database.get_container_client.return_value = mock_container

        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.mkdir.return_value = None

        # Test
        result = self.cleaner.create_backup()

        # Assertions
        self.assertTrue(result)
        mock_path_instance.mkdir.assert_called()
        mock_json_dump.assert_called()

    def test_validate_containers_all_empty(self):
        """Test container validation when all are empty."""

        # Mock get_container_stats to return empty containers
        def mock_get_stats(container_name):
            return {"name": container_name, "document_count": 0, "exists": True}

        self.cleaner.get_container_stats = Mock(side_effect=mock_get_stats)

        # Test
        result = self.cleaner.validate_containers()

        # Assertions
        self.assertTrue(result)

    def test_validate_containers_some_not_empty(self):
        """Test container validation when some containers have documents."""

        # Mock get_container_stats to return mixed results
        def mock_get_stats(container_name):
            if container_name == "knowledge":
                return {
                    "name": container_name,
                    "document_count": 5,  # Not empty
                    "exists": True,
                }
            else:
                return {"name": container_name, "document_count": 0, "exists": True}

        self.cleaner.get_container_stats = Mock(side_effect=mock_get_stats)

        # Test
        result = self.cleaner.validate_containers()

        # Assertions
        self.assertFalse(result)


class TestCosmosCleanupIntegration(unittest.TestCase):
    """Integration tests for the cleanup process."""

    @patch("cosmos_cleanup.input")
    @patch.object(CosmosDBCleaner, "initialize_connection")
    @patch.object(CosmosDBCleaner, "list_existing_containers")
    @patch.object(CosmosDBCleaner, "get_container_stats")
    @patch.object(CosmosDBCleaner, "cleanup_container")
    @patch.object(CosmosDBCleaner, "validate_containers")
    def test_run_cleanup_with_confirmation(
        self,
        mock_validate,
        mock_cleanup_container,
        mock_get_stats,
        mock_list_containers,
        mock_init_connection,
        mock_input,
    ):
        """Test complete cleanup process with user confirmation."""
        # Setup mocks
        mock_init_connection.return_value = True
        mock_list_containers.return_value = ["knowledge", "memory"]
        mock_get_stats.return_value = {"document_count": 10}
        mock_input.return_value = "yes"  # User confirms
        mock_cleanup_container.return_value = {
            "container": "test",
            "documents_deleted": 10,
            "status": "completed",
        }
        mock_validate.return_value = True

        # Create cleaner
        settings = MosaicSettings()
        settings.azure_cosmos_endpoint = "https://test.cosmos.com"
        cleaner = CosmosDBCleaner(settings)

        # Test
        result = cleaner.run_cleanup(create_backup=False, force=False)

        # Assertions
        self.assertTrue(result)
        mock_input.assert_called_once()
        mock_validate.assert_called_once()

    @patch.object(CosmosDBCleaner, "initialize_connection")
    def test_run_cleanup_connection_failure(self, mock_init_connection):
        """Test cleanup failure when connection fails."""
        mock_init_connection.return_value = False

        settings = MosaicSettings()
        cleaner = CosmosDBCleaner(settings)

        result = cleaner.run_cleanup(create_backup=False, force=True)

        self.assertFalse(result)


if __name__ == "__main__":
    # Run specific test categories
    import argparse

    parser = argparse.ArgumentParser(description="Run Cosmos cleanup tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests only"
    )
    parser.add_argument("--coverage", action="store_true", help="Run with coverage")

    args = parser.parse_args()

    if args.coverage:
        try:
            import coverage

            cov = coverage.Coverage()
            cov.start()

            unittest.main(exit=False, argv=[""])

            cov.stop()
            cov.save()
            print("\nCoverage Report:")
            cov.report()
        except ImportError:
            print("Coverage package not installed. Running without coverage.")
            unittest.main()
    elif args.unit:
        suite = unittest.TestLoader().loadTestsFromTestCase(TestCosmosDBCleaner)
        unittest.TextTestRunner(verbosity=2).run(suite)
    elif args.integration:
        suite = unittest.TestLoader().loadTestsFromTestCase(
            TestCosmosCleanupIntegration
        )
        unittest.TextTestRunner(verbosity=2).run(suite)
    else:
        unittest.main(verbosity=2)
