"""
Comprehensive tests for GraphSynchronizer.

Tests bulk synchronization, incremental sync, consistency validation,
and performance optimization for RDF graph synchronization from Cosmos DB.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from rdflib import URIRef, Literal
from rdflib.namespace import RDF

# Mock the Azure imports for testing
import sys

sys.modules["azure.cosmos.aio"] = MagicMock()
sys.modules["azure.cosmos"] = MagicMock()
sys.modules["azure.identity.aio"] = MagicMock()

# Add parent directory to path to import plugins
import sys

sys.path.append(str(Path(__file__).parent.parent))

from plugins.graph_synchronizer import (
    GraphSynchronizer,
    SyncMetrics,
    SyncState,
    GraphSynchronizerError,
    create_graph_synchronizer,
)


class TestSyncMetrics:
    """Test SyncMetrics dataclass functionality."""

    def test_sync_metrics_duration_calculation(self):
        """Test duration calculation in SyncMetrics."""
        metrics = SyncMetrics()
        metrics.start_time = 100.0
        metrics.end_time = 105.0

        assert metrics.duration_seconds == 5.0

    def test_sync_metrics_triples_per_second(self):
        """Test triples per second calculation."""
        metrics = SyncMetrics()
        metrics.start_time = 100.0
        metrics.end_time = 110.0
        metrics.added_triples = 50

        assert metrics.triples_per_second == 5.0

    def test_sync_metrics_zero_duration(self):
        """Test handling of zero duration."""
        metrics = SyncMetrics()
        metrics.start_time = 100.0
        metrics.end_time = 100.0
        metrics.added_triples = 50

        assert metrics.triples_per_second == 0.0


class TestSyncState:
    """Test SyncState dataclass functionality."""

    def test_sync_state_initialization(self):
        """Test SyncState initialization with default values."""
        state = SyncState()

        assert state.last_sync_timestamp is None
        assert state.continuation_token is None
        assert state.last_processed_id is None
        assert state.total_triples_synced == 0
        assert state.sync_errors == []

    def test_sync_state_with_errors(self):
        """Test SyncState with custom error list."""
        errors = ["Error 1", "Error 2"]
        state = SyncState(sync_errors=errors)

        assert state.sync_errors == errors


class TestGraphSynchronizer:
    """Test GraphSynchronizer class functionality."""

    @pytest.fixture
    def mock_cosmos_client(self):
        """Create a mock Cosmos client."""
        client = AsyncMock()
        client.get_database_client.return_value = AsyncMock()
        client.get_database_client.return_value.get_container_client.return_value = (
            AsyncMock()
        )
        return client

    @pytest.fixture
    def synchronizer(self, mock_cosmos_client):
        """Create a GraphSynchronizer instance for testing."""
        with patch("plugins.graph_synchronizer.get_settings") as mock_settings:
            mock_settings.return_value.AZURE_COSMOS_DB_ACCOUNT_NAME = "test-account"
            return GraphSynchronizer(mock_cosmos_client)

    def test_initialization(self, synchronizer):
        """Test GraphSynchronizer initialization."""
        assert synchronizer.database_name == "mosaic-knowledge"
        assert synchronizer.container_name == "code-entities"
        assert synchronizer.batch_size == 1000
        assert synchronizer.max_memory_mb == 2048
        assert not synchronizer.is_syncing

    def test_graph_namespace_bindings(self, synchronizer):
        """Test that graph has proper namespace bindings."""
        namespaces = dict(synchronizer.graph.namespaces())

        assert "mosaic" in namespaces
        assert "code" in namespaces
        assert "rdf" in namespaces
        assert "rdfs" in namespaces

    def test_parse_rdf_triple_document_valid(self, synchronizer):
        """Test parsing valid RDF triple document."""
        document = {
            "type": "rdf_triple",
            "subject": "http://mosaic.ai/graph#function/test.py/main",
            "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "object": "http://mosaic.ai/ontology/code_base#Function",
            "id": "test-id",
        }

        triple = synchronizer._parse_rdf_triple_document(document)

        assert triple is not None
        subject, predicate, obj = triple
        assert isinstance(subject, URIRef)
        assert isinstance(predicate, URIRef)
        assert isinstance(obj, URIRef)

    def test_parse_rdf_triple_document_literal(self, synchronizer):
        """Test parsing RDF triple with literal object."""
        document = {
            "type": "rdf_triple",
            "subject": "http://mosaic.ai/graph#function/test.py/main",
            "predicate": "http://mosaic.ai/ontology/code_base#name",
            "object": '"main"',
            "id": "test-id",
        }

        triple = synchronizer._parse_rdf_triple_document(document)

        assert triple is not None
        subject, predicate, obj = triple
        assert isinstance(obj, Literal)
        assert str(obj) == "main"

    def test_parse_rdf_triple_document_invalid_type(self, synchronizer):
        """Test parsing document with wrong type."""
        document = {
            "type": "code_entity",
            "subject": "http://mosaic.ai/graph#function/test.py/main",
            "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "object": "http://mosaic.ai/ontology/code_base#Function",
            "id": "test-id",
        }

        triple = synchronizer._parse_rdf_triple_document(document)
        assert triple is None

    def test_parse_rdf_triple_document_missing_fields(self, synchronizer):
        """Test parsing document with missing required fields."""
        document = {
            "type": "rdf_triple",
            "subject": "http://mosaic.ai/graph#function/test.py/main",
            # Missing predicate and object
            "id": "test-id",
        }

        triple = synchronizer._parse_rdf_triple_document(document)
        assert triple is None

    @pytest.mark.asyncio
    async def test_process_document_batch(self, synchronizer):
        """Test processing a batch of documents."""
        documents = [
            {
                "type": "rdf_triple",
                "subject": "http://mosaic.ai/graph#function/test.py/main",
                "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "object": "http://mosaic.ai/ontology/code_base#Function",
                "id": "test-id-1",
            },
            {
                "type": "rdf_triple",
                "subject": "http://mosaic.ai/graph#function/test.py/main",
                "predicate": "http://mosaic.ai/ontology/code_base#name",
                "object": '"main"',
                "id": "test-id-2",
            },
            {
                "type": "code_entity",  # Should be ignored
                "subject": "ignored",
                "predicate": "ignored",
                "object": "ignored",
                "id": "test-id-3",
            },
        ]

        triples_added = await synchronizer._process_document_batch(documents)

        assert triples_added == 2
        assert synchronizer.metrics.processed_documents == 3
        assert len(synchronizer.graph) == 2

    @pytest.mark.asyncio
    async def test_validate_graph_consistency_empty_graph(self, synchronizer):
        """Test consistency validation with empty graph."""
        result = await synchronizer._validate_graph_consistency()
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_graph_consistency_with_triples(self, synchronizer):
        """Test consistency validation with actual triples."""
        # Add some test triples
        synchronizer.graph.add(
            (
                URIRef("http://mosaic.ai/graph#function/test.py/main"),
                RDF.type,
                URIRef("http://mosaic.ai/ontology/code_base#Function"),
            )
        )

        result = await synchronizer._validate_graph_consistency()
        assert result is True

    def test_get_sync_status(self, synchronizer):
        """Test getting synchronization status."""
        status = synchronizer.get_sync_status()

        assert "is_syncing" in status
        assert "total_triples" in status
        assert "last_sync_timestamp" in status
        assert "total_triples_synced" in status
        assert "recent_errors" in status
        assert "memory_usage_mb" in status
        assert "metrics" in status

        assert status["is_syncing"] is False
        assert status["total_triples"] == 0

    def test_clear_graph(self, synchronizer):
        """Test clearing graph and resetting state."""
        # Add some test data
        synchronizer.graph.add(
            (
                URIRef("http://test.com/subject"),
                URIRef("http://test.com/predicate"),
                URIRef("http://test.com/object"),
            )
        )
        synchronizer.sync_state.total_triples_synced = 100

        assert len(synchronizer.graph) == 1

        synchronizer.clear_graph()

        assert len(synchronizer.graph) == 0
        assert synchronizer.sync_state.total_triples_synced == 0

    @pytest.mark.asyncio
    async def test_repair_graph_consistency(self, synchronizer):
        """Test graph consistency repair functionality."""
        # Add some valid triples
        synchronizer.graph.add(
            (
                URIRef("http://mosaic.ai/graph#function/test.py/main"),
                RDF.type,
                URIRef("http://mosaic.ai/ontology/code_base#Function"),
            )
        )

        repairs_made = await synchronizer.repair_graph_consistency()

        # Should be no repairs needed for valid graph
        assert repairs_made == 0

    def test_export_graph(self, synchronizer):
        """Test graph export functionality."""
        # Add test triple
        synchronizer.graph.add(
            (
                URIRef("http://test.com/subject"),
                URIRef("http://test.com/predicate"),
                Literal("test value"),
            )
        )

        # Test string export
        turtle_output = synchronizer.export_graph(format="turtle")
        assert isinstance(turtle_output, str)
        assert "test.com/subject" in turtle_output

    @pytest.mark.asyncio
    async def test_get_performance_metrics(self, synchronizer):
        """Test performance metrics collection."""
        # Add test triples
        for i in range(10):
            synchronizer.graph.add(
                (
                    URIRef(f"http://test.com/subject{i}"),
                    URIRef("http://test.com/predicate"),
                    Literal(f"value{i}"),
                )
            )

        metrics = await synchronizer.get_performance_metrics()

        assert "graph_size" in metrics
        assert "query_performance" in metrics
        assert "sync_metrics" in metrics

        assert metrics["graph_size"]["total_triples"] == 10
        assert "memory_usage_mb" in metrics["graph_size"]

    @pytest.mark.asyncio
    async def test_sync_from_cosmos_not_syncing(self, synchronizer):
        """Test that sync_from_cosmos prevents concurrent syncing."""
        synchronizer.is_syncing = True

        with pytest.raises(GraphSynchronizerError, match="already in progress"):
            await synchronizer.sync_from_cosmos()

    @pytest.mark.asyncio
    async def test_incremental_sync_not_syncing(self, synchronizer):
        """Test that incremental_sync prevents concurrent syncing."""
        synchronizer.is_syncing = True

        with pytest.raises(GraphSynchronizerError, match="already in progress"):
            await synchronizer.incremental_sync()


class TestGraphSynchronizerIntegration:
    """Integration tests for GraphSynchronizer."""

    @pytest.mark.asyncio
    async def test_create_graph_synchronizer_factory(self):
        """Test factory function for creating GraphSynchronizer."""
        with patch("plugins.graph_synchronizer.get_settings") as mock_settings:
            mock_settings.return_value.AZURE_COSMOS_DB_ACCOUNT_NAME = "test-account"

            with patch("azure.identity.aio.DefaultAzureCredential"):
                with patch("azure.cosmos.aio.CosmosClient") as mock_client:
                    mock_client.return_value = AsyncMock()

                    synchronizer = await create_graph_synchronizer()

                    assert isinstance(synchronizer, GraphSynchronizer)

    def test_sync_state_persistence(self, tmp_path):
        """Test sync state persistence to file."""
        with patch("plugins.graph_synchronizer.get_settings") as mock_settings:
            mock_settings.return_value.AZURE_COSMOS_DB_ACCOUNT_NAME = "test-account"

            synchronizer = GraphSynchronizer()
            synchronizer.state_file_path = tmp_path / "test_sync_state.json"

            # Set some state
            synchronizer.sync_state.total_triples_synced = 500
            synchronizer.sync_state.last_sync_timestamp = "2025-07-25T10:00:00Z"

            # Save state
            synchronizer._save_sync_state()

            # Create new synchronizer and load state
            synchronizer2 = GraphSynchronizer()
            synchronizer2.state_file_path = tmp_path / "test_sync_state.json"
            synchronizer2._load_sync_state()

            assert synchronizer2.sync_state.total_triples_synced == 500
            assert (
                synchronizer2.sync_state.last_sync_timestamp == "2025-07-25T10:00:00Z"
            )


if __name__ == "__main__":
    # Run basic tests without pytest

    print("Running GraphSynchronizer Tests...")

    # Test SyncMetrics
    print("\n1. Testing SyncMetrics...")
    metrics = SyncMetrics()
    metrics.start_time = 100.0
    metrics.end_time = 105.0
    metrics.added_triples = 25

    assert metrics.duration_seconds == 5.0
    assert metrics.triples_per_second == 5.0
    print("✓ SyncMetrics tests passed")

    # Test SyncState
    print("\n2. Testing SyncState...")
    state = SyncState()
    assert state.sync_errors == []
    print("✓ SyncState tests passed")

    # Test GraphSynchronizer basic functionality
    print("\n3. Testing GraphSynchronizer basic functionality...")
    with patch("plugins.graph_synchronizer.get_settings") as mock_settings:
        mock_settings.return_value.AZURE_COSMOS_DB_ACCOUNT_NAME = "test-account"

        synchronizer = GraphSynchronizer()

        # Test namespace bindings
        namespaces = dict(synchronizer.graph.namespaces())
        assert "mosaic" in namespaces
        assert "code" in namespaces
        print("✓ Namespace bindings correct")

        # Test document parsing
        document = {
            "type": "rdf_triple",
            "subject": "http://mosaic.ai/graph#function/test.py/main",
            "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "object": "http://mosaic.ai/ontology/code_base#Function",
            "id": "test-id",
        }

        triple = synchronizer._parse_rdf_triple_document(document)
        assert triple is not None
        print("✓ RDF triple parsing works")

        # Test sync status
        status = synchronizer.get_sync_status()
        assert "is_syncing" in status
        assert status["is_syncing"] is False
        print("✓ Sync status retrieval works")

    print(
        "\n✅ All basic tests passed! GraphSynchronizer implementation is working correctly."
    )
    print("\nNext steps:")
    print("- Install Azure SDK dependencies (pip install azure-cosmos azure-identity)")
    print("- Add integration with existing MCP plugins")
    print("- Test with actual Cosmos DB connection")
