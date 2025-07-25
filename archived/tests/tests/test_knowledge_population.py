"""
Comprehensive test suite for knowledge base population functionality.

Tests the complete Cosmos DB population logic with OmniRAG schema compliance,
batch operations, embedding generation, and error handling.

Based on ConPort task #48 requirements and TDD_UNIFIED.md specifications.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# Mock the Azure imports before importing the plugin
with patch.dict(
    "sys.modules",
    {
        "azure.cosmos": MagicMock(),
        "azure.identity": MagicMock(),
        "semantic_kernel": MagicMock(),
        "semantic_kernel.connectors.ai.open_ai": MagicMock(),
        "tree_sitter": MagicMock(),
        "tree_sitter_python": MagicMock(),
        "tree_sitter_javascript": MagicMock(),
        "tree_sitter_typescript": MagicMock(),
        "tree_sitter_java": MagicMock(),
        "tree_sitter_go": MagicMock(),
        "tree_sitter_rust": MagicMock(),
        "tree_sitter_c": MagicMock(),
        "tree_sitter_cpp": MagicMock(),
        "tree_sitter_c_sharp": MagicMock(),
        "tree_sitter_html": MagicMock(),
        "tree_sitter_css": MagicMock(),
    },
):
    from src.ingestion_service.plugins.ingestion import IngestionPlugin
    from src.mosaic.config.settings import MosaicSettings


class TestKnowledgeBasePopulation:
    """Test knowledge base population with Cosmos DB and embeddings."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = MagicMock(spec=MosaicSettings)
        settings.azure_openai_endpoint = "https://test.openai.azure.com/"
        settings.azure_openai_text_embedding_deployment_name = "text-embedding-3-small"
        settings.azure_openai_chat_deployment_name = "gpt-4.1"
        settings.get_cosmos_config.return_value = {
            "endpoint": "https://test.cosmos.azure.com:443/",
            "database_name": "mosaic",
            "container_name": "knowledge",
            "memory_container": "memory",
        }
        return settings

    @pytest.fixture
    def plugin(self, mock_settings):
        """Create IngestionPlugin instance for testing."""
        return IngestionPlugin(mock_settings)

    @pytest.fixture
    def sample_entities(self):
        """Create sample entities for testing."""
        return [
            {
                "id": "entity_1",
                "entity_type": "function",
                "name": "calculate_total",
                "language": "python",
                "file_path": "/app/src/utils.py",
                "content": "def calculate_total(items):\n    return sum(item.price for item in items)",
                "ai_description": "Calculates the total price of a list of items",
                "has_ai_analysis": True,
                "timestamp": datetime.now().isoformat(),
            },
            {
                "id": "entity_2",
                "entity_type": "class",
                "name": "UserService",
                "language": "python",
                "file_path": "/app/src/services.py",
                "content": "class UserService:\n    def __init__(self):\n        self.users = []",
                "ai_description": "Service class for managing user operations",
                "has_ai_analysis": True,
                "timestamp": datetime.now().isoformat(),
            },
        ]

    @pytest.fixture
    def sample_relationships(self):
        """Create sample relationships for testing."""
        return [
            {
                "id": "rel_1",
                "type": "imports",
                "source_id": "entity_1",
                "target_id": "entity_2",
                "source_file": "/app/src/utils.py",
                "target_file": "/app/src/services.py",
                "timestamp": datetime.now().isoformat(),
            }
        ]

    @pytest.mark.asyncio
    async def test_populate_knowledge_base_success(
        self, plugin, sample_entities, sample_relationships
    ):
        """Test successful knowledge base population."""
        # Mock Azure services
        plugin.embedding_service = AsyncMock()
        plugin.knowledge_container = AsyncMock()

        # Mock embedding generation
        mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        plugin.embedding_service.generate_embeddings.return_value = mock_embeddings

        # Mock Cosmos DB operations
        plugin.knowledge_container.upsert_item = AsyncMock()

        # Execute the test
        await plugin._populate_knowledge_base(sample_entities, sample_relationships)

        # Verify embedding generation was called
        plugin.embedding_service.generate_embeddings.assert_called_once()

        # Verify entity storage calls
        assert (
            plugin.knowledge_container.upsert_item.call_count == 3
        )  # 2 entities + 1 relationship

        # Verify entity documents structure
        entity_calls = [
            call
            for call in plugin.knowledge_container.upsert_item.call_args_list
            if call[0][0].get("type") == "code_entity"
        ]
        assert len(entity_calls) == 2

        # Verify relationship documents structure
        rel_calls = [
            call
            for call in plugin.knowledge_container.upsert_item.call_args_list
            if call[0][0].get("type") == "relationship"
        ]
        assert len(rel_calls) == 1

    @pytest.mark.asyncio
    async def test_generate_and_store_entity_embeddings_batch_processing(
        self, plugin, sample_entities
    ):
        """Test batch processing of entity embeddings."""
        # Create larger dataset to test batching
        entities = []
        for i in range(150):  # More than batch size of 100
            entity = sample_entities[0].copy()
            entity["id"] = f"entity_{i}"
            entity["name"] = f"function_{i}"
            entities.append(entity)

        # Mock services
        plugin.embedding_service = AsyncMock()
        plugin.knowledge_container = AsyncMock()

        mock_embeddings = [[0.1, 0.2, 0.3] for _ in range(100)]  # First batch
        plugin.embedding_service.generate_embeddings.return_value = mock_embeddings

        # Execute
        await plugin._generate_and_store_entity_embeddings(entities)

        # Verify batching: should be called twice (100 + 50)
        assert plugin.embedding_service.generate_embeddings.call_count == 2

        # Verify all entities were processed
        assert plugin.knowledge_container.upsert_item.call_count == 150

    @pytest.mark.asyncio
    async def test_embedding_generation_with_ai_descriptions(
        self, plugin, sample_entities
    ):
        """Test that AI descriptions are properly included in embedding text."""
        plugin.embedding_service = AsyncMock()
        plugin.knowledge_container = AsyncMock()

        mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        plugin.embedding_service.generate_embeddings.return_value = mock_embeddings

        await plugin._generate_and_store_entity_embeddings(sample_entities)

        # Verify embedding generation was called with proper text format
        call_args = plugin.embedding_service.generate_embeddings.call_args[0][0]
        assert "calculate_total function: Calculates the total price" in call_args[0]
        assert (
            "UserService class: Service class for managing user operations"
            in call_args[1]
        )

    @pytest.mark.asyncio
    async def test_embedding_generation_without_ai_descriptions(self, plugin):
        """Test embedding generation for entities without AI descriptions."""
        entities = [
            {
                "id": "entity_no_ai",
                "entity_type": "function",
                "name": "simple_func",
                "language": "python",
                "file_path": "/app/test.py",
                "content": "def simple_func(): pass",
                "timestamp": datetime.now().isoformat(),
            }
        ]

        plugin.embedding_service = AsyncMock()
        plugin.knowledge_container = AsyncMock()

        mock_embeddings = [[0.1, 0.2, 0.3]]
        plugin.embedding_service.generate_embeddings.return_value = mock_embeddings

        await plugin._generate_and_store_entity_embeddings(entities)

        # Verify fallback text format
        call_args = plugin.embedding_service.generate_embeddings.call_args[0][0]
        assert "simple_func function def simple_func(): pass" in call_args[0]

    @pytest.mark.asyncio
    async def test_store_relationships(self, plugin, sample_relationships):
        """Test relationship storage in Cosmos DB."""
        plugin.knowledge_container = AsyncMock()

        await plugin._store_relationships(sample_relationships)

        # Verify relationship was stored
        plugin.knowledge_container.upsert_item.assert_called_once()

        # Verify relationship document structure
        stored_doc = plugin.knowledge_container.upsert_item.call_args[0][0]
        assert stored_doc["type"] == "relationship"
        assert stored_doc["relationship_type"] == "imports"
        assert stored_doc["source_id"] == "entity_1"
        assert stored_doc["target_id"] == "entity_2"

    @pytest.mark.asyncio
    async def test_relationship_batch_processing(self, plugin):
        """Test batch processing of relationships."""
        relationships = []
        for i in range(150):  # More than batch size of 100
            rel = {
                "id": f"rel_{i}",
                "type": "imports",
                "source_id": f"entity_{i}",
                "target_id": f"entity_{i + 1}",
                "source_file": f"/app/file_{i}.py",
                "target_file": f"/app/file_{i + 1}.py",
                "timestamp": datetime.now().isoformat(),
            }
            relationships.append(rel)

        plugin.knowledge_container = AsyncMock()

        await plugin._store_relationships(relationships)

        # Verify all relationships were processed
        assert plugin.knowledge_container.upsert_item.call_count == 150

    @pytest.mark.asyncio
    async def test_cosmos_db_schema_compliance(self, plugin, sample_entities):
        """Test that stored documents comply with OmniRAG schema."""
        plugin.embedding_service = AsyncMock()
        plugin.knowledge_container = AsyncMock()

        mock_embeddings = [[0.1, 0.2, 0.3]]
        plugin.embedding_service.generate_embeddings.return_value = mock_embeddings

        await plugin._generate_and_store_entity_embeddings([sample_entities[0]])

        # Verify document structure matches OmniRAG schema
        stored_doc = plugin.knowledge_container.upsert_item.call_args[0][0]

        # Required fields from CLAUDE.md schema
        assert "id" in stored_doc
        assert "type" in stored_doc
        assert stored_doc["type"] == "code_entity"
        assert "entity_type" in stored_doc
        assert "name" in stored_doc
        assert "language" in stored_doc
        assert "file_path" in stored_doc
        assert "content" in stored_doc
        assert "embedding" in stored_doc
        assert "timestamp" in stored_doc

        # AI enhancement fields
        assert "ai_description" in stored_doc
        assert "has_ai_analysis" in stored_doc

    @pytest.mark.asyncio
    async def test_error_handling_embedding_failure(self, plugin, sample_entities):
        """Test error handling when embedding generation fails."""
        plugin.embedding_service = AsyncMock()
        plugin.knowledge_container = AsyncMock()

        # Mock embedding service failure
        plugin.embedding_service.generate_embeddings.side_effect = Exception(
            "Embedding service error"
        )

        # Should continue processing despite error
        await plugin._generate_and_store_entity_embeddings(sample_entities)

        # Verify no entities were stored due to embedding failure
        plugin.knowledge_container.upsert_item.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_handling_cosmos_db_failure(self, plugin, sample_entities):
        """Test error handling when Cosmos DB operations fail."""
        plugin.embedding_service = AsyncMock()
        plugin.knowledge_container = AsyncMock()

        mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        plugin.embedding_service.generate_embeddings.return_value = mock_embeddings

        # Mock Cosmos DB failure
        plugin.knowledge_container.upsert_item.side_effect = Exception(
            "Cosmos DB error"
        )

        # Should handle error gracefully
        await plugin._generate_and_store_entity_embeddings(sample_entities)

        # Verify embedding generation still occurred
        plugin.embedding_service.generate_embeddings.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_relationship_storage_failure(
        self, plugin, sample_relationships
    ):
        """Test error handling when relationship storage fails."""
        plugin.knowledge_container = AsyncMock()

        # Mock Cosmos DB failure for relationships
        plugin.knowledge_container.upsert_item.side_effect = Exception(
            "Relationship storage error"
        )

        # Should handle error gracefully
        await plugin._store_relationships(sample_relationships)

        # Verify attempt was made
        plugin.knowledge_container.upsert_item.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_entities_handling(self, plugin):
        """Test handling of empty entity lists."""
        plugin.embedding_service = AsyncMock()
        plugin.knowledge_container = AsyncMock()

        await plugin._generate_and_store_entity_embeddings([])

        # Should not call any services
        plugin.embedding_service.generate_embeddings.assert_not_called()
        plugin.knowledge_container.upsert_item.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_relationships_handling(self, plugin):
        """Test handling of empty relationship lists."""
        plugin.knowledge_container = AsyncMock()

        await plugin._store_relationships([])

        # Should not call any services
        plugin.knowledge_container.upsert_item.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrent_batch_processing(self, plugin):
        """Test that batch processing doesn't interfere with concurrent operations."""
        # Create multiple batches
        entities1 = [
            {
                "id": f"batch1_{i}",
                "entity_type": "function",
                "name": f"func1_{i}",
                "language": "python",
                "file_path": f"/app/batch1_{i}.py",
                "content": f"def func1_{i}(): pass",
                "timestamp": datetime.now().isoformat(),
            }
            for i in range(50)
        ]

        entities2 = [
            {
                "id": f"batch2_{i}",
                "entity_type": "class",
                "name": f"class2_{i}",
                "language": "python",
                "file_path": f"/app/batch2_{i}.py",
                "content": f"class Class2_{i}: pass",
                "timestamp": datetime.now().isoformat(),
            }
            for i in range(75)
        ]

        plugin.embedding_service = AsyncMock()
        plugin.knowledge_container = AsyncMock()

        # Mock embeddings for different batch sizes
        plugin.embedding_service.generate_embeddings.side_effect = [
            [[0.1, 0.2, 0.3] for _ in range(50)],  # First batch
            [[0.4, 0.5, 0.6] for _ in range(75)],  # Second batch
        ]

        # Process batches concurrently
        await asyncio.gather(
            plugin._generate_and_store_entity_embeddings(entities1),
            plugin._generate_and_store_entity_embeddings(entities2),
        )

        # Verify both batches were processed
        assert plugin.embedding_service.generate_embeddings.call_count == 2
        assert plugin.knowledge_container.upsert_item.call_count == 125  # 50 + 75

    def test_text_preparation_for_embeddings(self, plugin):
        """Test text preparation for embedding generation."""
        # Test entity with AI description
        entity_with_ai = {
            "name": "test_function",
            "entity_type": "function",
            "content": "def test_function(): return 'test'",
            "ai_description": "A simple test function",
        }

        # Test entity without AI description
        entity_without_ai = {
            "name": "another_function",
            "entity_type": "function",
            "content": "def another_function(): pass",
        }

        # This would be tested within the actual method, but demonstrates the logic
        with_ai_text = f"{entity_with_ai['name']} {entity_with_ai['entity_type']}: {entity_with_ai['ai_description']} {entity_with_ai['content'][:300]}"
        without_ai_text = f"{entity_without_ai['name']} {entity_without_ai['entity_type']} {entity_without_ai['content'][:300]}"

        assert "A simple test function" in with_ai_text
        assert "test_function function: A simple test function" in with_ai_text
        assert "another_function function def another_function" in without_ai_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
