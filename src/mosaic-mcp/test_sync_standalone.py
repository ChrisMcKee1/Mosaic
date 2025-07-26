#!/usr/bin/env python3
"""
Standalone test for GraphSynchronizer to avoid module import dependencies
"""

import unittest
from unittest.mock import MagicMock, patch
import asyncio
from dataclasses import asdict
import logging
import sys
from pathlib import Path

# Import directly without going through plugins module
sys.path.append(str(Path(__file__).parent / "plugins"))
from graph_synchronizer import GraphSynchronizer, SyncMetrics, SyncState


class TestSyncMetrics(unittest.TestCase):
    """Test SyncMetrics dataclass"""

    def test_initialization(self):
        """Test SyncMetrics initialization with defaults"""
        metrics = SyncMetrics()
        self.assertEqual(metrics.total_documents, 0)
        self.assertEqual(metrics.processed_documents, 0)
        self.assertEqual(metrics.total_triples, 0)
        self.assertEqual(metrics.added_triples, 0)
        self.assertEqual(metrics.errors, 0)
        self.assertEqual(metrics.start_time, 0.0)
        self.assertEqual(metrics.end_time, 0.0)
        self.assertEqual(metrics.memory_usage_mb, 0.0)

    def test_to_dict(self):
        """Test conversion to dictionary"""
        metrics = SyncMetrics(
            total_documents=100,
            processed_documents=100,
            added_triples=500,
            start_time=1000.0,
            end_time=1005.0,
        )
        result = asdict(metrics)
        self.assertEqual(result["total_documents"], 100)
        self.assertEqual(result["processed_documents"], 100)
        self.assertEqual(result["added_triples"], 500)
        self.assertEqual(result["start_time"], 1000.0)
        self.assertEqual(result["end_time"], 1005.0)


class TestSyncState(unittest.TestCase):
    """Test SyncState dataclass"""

    def test_initialization(self):
        """Test SyncState initialization with defaults"""
        state = SyncState()
        self.assertIsNone(state.last_sync_timestamp)
        self.assertIsNone(state.continuation_token)
        self.assertIsNone(state.last_processed_id)
        self.assertEqual(state.total_triples_synced, 0)
        self.assertEqual(state.sync_errors, [])


class TestGraphSynchronizer(unittest.TestCase):
    """Test GraphSynchronizer class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock cosmos client
        self.mock_cosmos_client = MagicMock()

        # Create synchronizer instance
        self.synchronizer = GraphSynchronizer(self.mock_cosmos_client)

    def test_initialization(self):
        """Test GraphSynchronizer initialization"""
        self.assertEqual(self.synchronizer.database_name, "mosaic-knowledge")
        self.assertEqual(self.synchronizer.container_name, "code-entities")
        self.assertEqual(self.synchronizer.batch_size, 1000)
        self.assertEqual(self.synchronizer.max_memory_mb, 2048)
        self.assertFalse(self.synchronizer.is_syncing)

    def test_graph_namespace_bindings(self):
        """Test that RDF graph has required namespace bindings"""
        namespaces = dict(self.synchronizer.graph.namespaces())
        self.assertIn("mosaic", str(namespaces))
        self.assertIn("rdf", str(namespaces))
        self.assertIn("rdfs", str(namespaces))

    def test_parse_rdf_triple_document_valid(self):
        """Test parsing a valid RDF triple document"""
        document = {
            "id": "test-1",
            "type": "rdf_triple",
            "subject": "http://mosaic.local/entity/test-entity",
            "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "object": "http://mosaic.local/class/TestClass",
        }

        triple = self.synchronizer._parse_rdf_triple_document(document)
        self.assertIsNotNone(triple)
        self.assertEqual(len(triple), 3)

    def test_parse_rdf_triple_document_invalid(self):
        """Test parsing an invalid RDF triple document"""
        document = {
            "id": "test-invalid",
            "type": "other",  # Wrong type
            # Missing required fields
        }

        triple = self.synchronizer._parse_rdf_triple_document(document)
        self.assertIsNone(triple)

    def test_memory_check_under_limit(self):
        """Test memory check when under limit"""
        # Mock psutil to return low memory usage
        with patch("graph_synchronizer.psutil") as mock_psutil:
            mock_process = MagicMock()
            mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
            mock_psutil.Process.return_value = mock_process

            memory_mb = self.synchronizer._get_memory_usage()
            self.assertLess(memory_mb, 2048)  # Under max_memory_mb

    def test_memory_check_over_limit(self):
        """Test memory check when over limit"""
        # Mock psutil to return high memory usage
        with patch("graph_synchronizer.psutil") as mock_psutil:
            mock_process = MagicMock()
            mock_process.memory_info.return_value.rss = 3000 * 1024 * 1024  # 3GB
            mock_psutil.Process.return_value = mock_process

            memory_mb = self.synchronizer._get_memory_usage()
            self.assertGreater(memory_mb, 2048)  # Over max_memory_mb


class TestGraphSynchronizerAsync(unittest.TestCase):
    """Test async methods of GraphSynchronizer"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_cosmos_client = MagicMock()
        self.synchronizer = GraphSynchronizer(self.mock_cosmos_client)

    def test_basic_async_initialization(self):
        """Test async initialization works"""

        async def run_test():
            # Mock the database and container clients
            mock_database = MagicMock()
            mock_container = MagicMock()
            mock_container.query_items.return_value = []

            self.mock_cosmos_client.get_database_client.return_value = mock_database
            mock_database.get_container_client.return_value = mock_container

            # Test basic properties are accessible
            self.assertEqual(self.synchronizer.database_name, "mosaic-knowledge")
            self.assertEqual(self.synchronizer.container_name, "code-entities")

        asyncio.run(run_test())


def run_tests():
    """Run all tests"""
    # Set up logging to see test output
    logging.basicConfig(level=logging.INFO)

    # Create test suite using modern approach
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTest(loader.loadTestsFromTestCase(TestSyncMetrics))
    suite.addTest(loader.loadTestsFromTestCase(TestSyncState))
    suite.addTest(loader.loadTestsFromTestCase(TestGraphSynchronizer))
    suite.addTest(loader.loadTestsFromTestCase(TestGraphSynchronizerAsync))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
