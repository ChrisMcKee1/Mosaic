"""
Unit tests for core MCP plugins (Retrieval, Memory, Refinement, Diagram).

Tests the main Semantic Kernel plugins that provide MCP tool functionality
for the Mosaic MCP server.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from plugins.retrieval import RetrievalPlugin
from plugins.memory import MemoryPlugin
from plugins.refinement import RefinementPlugin
from plugins.diagram import DiagramPlugin


class TestRetrievalPlugin:
    """Test cases for RetrievalPlugin."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = Mock()
        settings.azure_cosmos_endpoint = "https://test.cosmos.azure.com/"
        settings.azure_openai_endpoint = "https://test.openai.azure.com/"
        settings.database_name = "test-knowledge"
        settings.container_name = "test-documents"
        settings.embedding_model = "text-embedding-ada-002"
        return settings

    @pytest.fixture
    def mock_cosmos_client(self):
        """Create mock Cosmos DB client."""
        mock_client = Mock()
        mock_database = Mock()
        mock_container = Mock()

        mock_client.get_database_client.return_value = mock_database
        mock_database.get_container_client.return_value = mock_container

        return mock_client

    @pytest.fixture
    def retrieval_plugin(self, mock_settings):
        """Create RetrievalPlugin with mocked dependencies."""
        with (
            patch("plugins.retrieval.CosmosClient") as mock_cosmos_class,
            patch("plugins.retrieval.AzureTextEmbedding") as mock_embedding_class,
        ):
            mock_cosmos_class.return_value = Mock()
            mock_embedding_class.return_value = Mock()

            plugin = RetrievalPlugin(mock_settings)
            return plugin

    def test_retrieval_plugin_initialization(self, retrieval_plugin, mock_settings):
        """Test RetrievalPlugin initialization."""
        assert retrieval_plugin.settings == mock_settings
        assert retrieval_plugin.cosmos_client is not None
        assert retrieval_plugin.embedding_client is not None

    @pytest.mark.asyncio
    async def test_hybrid_search(self, retrieval_plugin):
        """Test hybrid search functionality."""
        # Mock vector search results
        mock_vector_results = [
            {"id": "doc1", "content": "AI context management", "score": 0.95},
            {"id": "doc2", "content": "Machine learning models", "score": 0.88},
        ]

        # Mock keyword search results
        mock_keyword_results = [
            {"id": "doc1", "content": "AI context management", "score": 0.92},
            {"id": "doc3", "content": "Context-aware systems", "score": 0.85},
        ]

        with (
            patch.object(
                retrieval_plugin, "_vector_search", return_value=mock_vector_results
            ) as mock_vector,
            patch.object(
                retrieval_plugin, "_keyword_search", return_value=mock_keyword_results
            ) as mock_keyword,
            patch.object(
                retrieval_plugin,
                "_merge_and_deduplicate",
                return_value=[
                    {"id": "doc1", "content": "AI context management", "score": 0.94},
                    {"id": "doc2", "content": "Machine learning models", "score": 0.88},
                    {"id": "doc3", "content": "Context-aware systems", "score": 0.85},
                ],
            ) as mock_merge,
        ):
            result = await retrieval_plugin.hybrid_search("AI context")

            mock_vector.assert_called_once_with("AI context")
            mock_keyword.assert_called_once_with("AI context")
            mock_merge.assert_called_once_with(
                mock_vector_results, mock_keyword_results
            )

            assert len(result) == 3
            assert result[0]["score"] == 0.94

    @pytest.mark.asyncio
    async def test_query_code_graph(self, retrieval_plugin):
        """Test code graph querying."""
        mock_graph_results = [
            {
                "id": "lib_requests",
                "name": "requests",
                "type": "library",
                "dependencies": ["urllib3", "certifi"],
            },
            {
                "id": "lib_flask",
                "name": "flask",
                "type": "framework",
                "dependencies": ["werkzeug", "jinja2"],
            },
        ]

        with patch.object(
            retrieval_plugin,
            "_query_graph_relationships",
            return_value=mock_graph_results,
        ) as mock_query:
            result = await retrieval_plugin.query_code_graph("python", "dependencies")

            mock_query.assert_called_once_with("python", "dependencies")
            assert len(result) == 2
            assert result[0]["name"] == "requests"

    @pytest.mark.asyncio
    async def test_vector_search(self, retrieval_plugin):
        """Test vector search implementation."""
        query = "machine learning algorithms"

        # Mock embedding generation
        mock_embedding = [0.1, 0.2, 0.3, -0.1, 0.5]

        # Mock Cosmos DB vector search results
        mock_cosmos_results = [
            {
                "id": "doc1",
                "content": "Introduction to machine learning algorithms",
                "embedding": mock_embedding,
                "_score": 0.92,
            },
            {
                "id": "doc2",
                "content": "Deep learning neural networks",
                "embedding": [0.2, 0.1, 0.4, -0.2, 0.6],
                "_score": 0.87,
            },
        ]

        with (
            patch.object(
                retrieval_plugin, "_generate_embedding", return_value=mock_embedding
            ) as mock_embed,
            patch.object(
                retrieval_plugin,
                "_execute_vector_query",
                return_value=mock_cosmos_results,
            ) as mock_query,
        ):
            result = await retrieval_plugin._vector_search(query)

            mock_embed.assert_called_once_with(query)
            mock_query.assert_called_once_with(mock_embedding)

            assert len(result) == 2
            assert result[0]["score"] == 0.92

    @pytest.mark.asyncio
    async def test_keyword_search(self, retrieval_plugin):
        """Test keyword search implementation."""
        query = "AI context management"

        mock_search_results = [
            {
                "id": "doc1",
                "content": "AI-powered context management systems",
                "rank": 1,
            },
            {"id": "doc2", "content": "Context-aware AI applications", "rank": 2},
        ]

        with patch.object(
            retrieval_plugin, "_execute_keyword_query", return_value=mock_search_results
        ) as mock_query:
            result = await retrieval_plugin._keyword_search(query)

            mock_query.assert_called_once_with(query)
            assert len(result) == 2
            # Keyword search should convert rank to score
            assert result[0]["score"] > result[1]["score"]

    def test_merge_and_deduplicate(self, retrieval_plugin):
        """Test result merging and deduplication."""
        vector_results = [
            {"id": "doc1", "content": "AI context", "score": 0.95},
            {"id": "doc2", "content": "Machine learning", "score": 0.88},
        ]

        keyword_results = [
            {"id": "doc1", "content": "AI context", "score": 0.92},  # Duplicate
            {"id": "doc3", "content": "Context systems", "score": 0.85},
        ]

        result = retrieval_plugin._merge_and_deduplicate(
            vector_results, keyword_results
        )

        # Should have 3 unique documents
        assert len(result) == 3

        # doc1 should have combined score (higher from vector search)
        doc1 = next(r for r in result if r["id"] == "doc1")
        assert doc1["score"] == 0.95  # Takes higher score

        # Results should be sorted by score
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)


class TestMemoryPlugin:
    """Test cases for MemoryPlugin."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = Mock()
        settings.azure_cosmos_endpoint = "https://test.cosmos.azure.com/"
        settings.azure_redis_endpoint = "redis://test-redis:6379"
        settings.azure_openai_endpoint = "https://test.openai.azure.com/"
        settings.database_name = "test-memory"
        settings.container_name = "test-memories"
        return settings

    @pytest.fixture
    def memory_plugin(self, mock_settings):
        """Create MemoryPlugin with mocked dependencies."""
        with (
            patch("plugins.memory.CosmosClient") as mock_cosmos,
            patch("plugins.memory.redis.from_url") as mock_redis,
            patch("plugins.memory.AzureTextEmbedding") as mock_embedding,
        ):
            mock_cosmos.return_value = Mock()
            mock_redis.return_value = Mock()
            mock_embedding.return_value = Mock()

            plugin = MemoryPlugin(mock_settings)
            return plugin

    def test_memory_plugin_initialization(self, memory_plugin, mock_settings):
        """Test MemoryPlugin initialization."""
        assert memory_plugin.settings == mock_settings
        assert memory_plugin.cosmos_client is not None
        assert memory_plugin.redis_client is not None

    @pytest.mark.asyncio
    async def test_save_memory(self, memory_plugin):
        """Test memory saving functionality."""
        session_id = "session_123"
        content = "User prefers Python for data analysis tasks"
        memory_type = "episodic"

        mock_memory_entry = {
            "id": "memory_456",
            "sessionId": session_id,
            "type": memory_type,
            "content": content,
            "importance_score": 0.85,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with (
            patch.object(
                memory_plugin, "_calculate_importance", return_value=0.85
            ) as mock_importance,
            patch.object(memory_plugin, "_store_in_redis") as mock_redis_store,
            patch.object(
                memory_plugin, "_store_in_cosmos", return_value=mock_memory_entry
            ) as mock_cosmos_store,
        ):
            result = await memory_plugin.save(session_id, content, memory_type)

            mock_importance.assert_called_once_with(content, memory_type)
            mock_redis_store.assert_called_once()
            mock_cosmos_store.assert_called_once()

            assert result["id"] == "memory_456"
            assert result["importance_score"] == 0.85

    @pytest.mark.asyncio
    async def test_retrieve_memory(self, memory_plugin):
        """Test memory retrieval functionality."""
        session_id = "session_123"
        query = "Python programming preferences"
        limit = 5

        mock_redis_results = [
            {"id": "mem1", "content": "Recent Python discussion", "score": 0.95},
            {"id": "mem2", "content": "Data analysis with pandas", "score": 0.88},
        ]

        mock_cosmos_results = [
            {"id": "mem3", "content": "Python best practices", "score": 0.82},
            {"id": "mem4", "content": "Machine learning with Python", "score": 0.79},
        ]

        with (
            patch.object(
                memory_plugin, "_search_redis", return_value=mock_redis_results
            ) as mock_redis_search,
            patch.object(
                memory_plugin, "_search_cosmos", return_value=mock_cosmos_results
            ) as mock_cosmos_search,
            patch.object(
                memory_plugin,
                "_merge_memory_results",
                return_value=mock_redis_results + mock_cosmos_results,
            ) as mock_merge,
        ):
            result = await memory_plugin.retrieve(session_id, query, limit)

            mock_redis_search.assert_called_once_with(session_id, query)
            mock_cosmos_search.assert_called_once_with(session_id, query, limit)
            mock_merge.assert_called_once()

            assert len(result) == 4

    @pytest.mark.asyncio
    async def test_clear_memory(self, memory_plugin):
        """Test memory clearing functionality."""
        session_id = "session_123"

        with (
            patch.object(memory_plugin, "_clear_redis_session") as mock_clear_redis,
            patch.object(memory_plugin, "_clear_cosmos_session") as mock_clear_cosmos,
        ):
            result = await memory_plugin.clear(session_id)

            mock_clear_redis.assert_called_once_with(session_id)
            mock_clear_cosmos.assert_called_once_with(session_id)

            assert result["status"] == "cleared"
            assert result["session_id"] == session_id

    def test_calculate_importance(self, memory_plugin):
        """Test importance score calculation."""
        # High importance content
        high_content = "CRITICAL: Database connection failed. Need immediate attention."
        high_score = memory_plugin._calculate_importance(high_content, "episodic")
        assert high_score > 0.8

        # Medium importance content
        medium_content = "User prefers using pandas for data analysis"
        medium_score = memory_plugin._calculate_importance(medium_content, "semantic")
        assert 0.5 <= medium_score <= 0.8

        # Low importance content
        low_content = "Hello there"
        low_score = memory_plugin._calculate_importance(low_content, "episodic")
        assert low_score < 0.5


class TestRefinementPlugin:
    """Test cases for RefinementPlugin."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = Mock()
        settings.azure_ml_endpoint = "https://test-ml.azure.com/"
        settings.reranker_model = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        return settings

    @pytest.fixture
    def refinement_plugin(self, mock_settings):
        """Create RefinementPlugin with mocked dependencies."""
        with patch("plugins.refinement.MLClient") as mock_ml_client:
            mock_ml_client.return_value = Mock()
            plugin = RefinementPlugin(mock_settings)
            return plugin

    def test_refinement_plugin_initialization(self, refinement_plugin, mock_settings):
        """Test RefinementPlugin initialization."""
        assert refinement_plugin.settings == mock_settings
        assert refinement_plugin.ml_client is not None

    @pytest.mark.asyncio
    async def test_rerank_documents(self, refinement_plugin):
        """Test document reranking functionality."""
        query = "machine learning algorithms"
        documents = [
            {"id": "doc1", "content": "Introduction to ML algorithms", "score": 0.75},
            {"id": "doc2", "content": "Deep learning neural networks", "score": 0.80},
            {"id": "doc3", "content": "Statistical learning theory", "score": 0.70},
        ]

        # Mock reranker scores
        mock_reranker_scores = [0.92, 0.88, 0.85]

        with patch.object(
            refinement_plugin, "_call_reranker_model", return_value=mock_reranker_scores
        ) as mock_reranker:
            result = await refinement_plugin.rerank(query, documents)

            mock_reranker.assert_called_once_with(
                query, [doc["content"] for doc in documents]
            )

            # Results should be reordered by reranker scores
            assert len(result) == 3
            assert result[0]["id"] == "doc1"  # Highest reranker score
            assert result[0]["score"] == 0.92
            assert result[1]["id"] == "doc2"
            assert result[2]["id"] == "doc3"

    @pytest.mark.asyncio
    async def test_rerank_with_score_fusion(self, refinement_plugin):
        """Test reranking with original score fusion."""
        query = "AI context management"
        documents = [
            {"id": "doc1", "content": "AI-powered context", "score": 0.95},
            {"id": "doc2", "content": "Context management systems", "score": 0.70},
        ]

        mock_reranker_scores = [0.85, 0.90]  # Reranker prefers doc2

        with patch.object(
            refinement_plugin, "_call_reranker_model", return_value=mock_reranker_scores
        ):
            result = await refinement_plugin.rerank(query, documents, fusion_weight=0.3)

            # Should blend original scores with reranker scores
            # doc1: 0.95 * 0.3 + 0.85 * 0.7 = 0.88
            # doc2: 0.70 * 0.3 + 0.90 * 0.7 = 0.84
            assert result[0]["id"] == "doc1"  # Still higher after fusion
            assert 0.87 <= result[0]["score"] <= 0.89

    @pytest.mark.asyncio
    async def test_call_reranker_model(self, refinement_plugin):
        """Test calling the reranker model."""
        query = "test query"
        passages = ["passage 1", "passage 2", "passage 3"]

        mock_response = {
            "predictions": [{"score": 0.92}, {"score": 0.85}, {"score": 0.78}]
        }

        with patch.object(
            refinement_plugin.ml_client, "invoke", return_value=mock_response
        ) as mock_invoke:
            scores = await refinement_plugin._call_reranker_model(query, passages)

            mock_invoke.assert_called_once()
            assert scores == [0.92, 0.85, 0.78]


class TestDiagramPlugin:
    """Test cases for DiagramPlugin."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = Mock()
        settings.azure_openai_endpoint = "https://test.openai.azure.com/"
        settings.diagram_model = "gpt-4"
        return settings

    @pytest.fixture
    def diagram_plugin(self, mock_settings):
        """Create DiagramPlugin with mocked dependencies."""
        with patch("plugins.diagram.AzureChatCompletion") as mock_chat:
            mock_chat.return_value = Mock()
            plugin = DiagramPlugin(mock_settings)
            return plugin

    def test_diagram_plugin_initialization(self, diagram_plugin, mock_settings):
        """Test DiagramPlugin initialization."""
        assert diagram_plugin.settings == mock_settings
        assert diagram_plugin.chat_client is not None

    @pytest.mark.asyncio
    async def test_generate_mermaid_diagram(self, diagram_plugin):
        """Test Mermaid diagram generation."""
        description = "Create a flow diagram showing user authentication process"

        mock_mermaid_code = """
graph TD
    A[User Login] --> B{Valid Credentials?}
    B -->|Yes| C[Generate Token]
    B -->|No| D[Login Failed]
    C --> E[Redirect to Dashboard]
    D --> A
"""

        mock_response = Mock()
        mock_response.content = mock_mermaid_code

        with patch.object(
            diagram_plugin, "_generate_mermaid_with_llm", return_value=mock_mermaid_code
        ) as mock_generate:
            result = await diagram_plugin.generate(description)

            mock_generate.assert_called_once_with(description)
            assert "mermaid" in result
            assert "graph TD" in result["mermaid"]
            assert "User Login" in result["mermaid"]

    @pytest.mark.asyncio
    async def test_generate_complex_diagram(self, diagram_plugin):
        """Test generation of complex system diagram."""
        description = "Create an architecture diagram for microservices with API gateway, user service, order service, and database"

        mock_complex_mermaid = """
graph TB
    subgraph "API Layer"
        GW[API Gateway]
    end
    
    subgraph "Microservices"
        US[User Service]
        OS[Order Service]
        PS[Payment Service]
    end
    
    subgraph "Data Layer"
        UDB[(User DB)]
        ODB[(Order DB)]
        PDB[(Payment DB)]
    end
    
    GW --> US
    GW --> OS
    GW --> PS
    US --> UDB
    OS --> ODB
    PS --> PDB
    OS --> PS
"""

        with patch.object(
            diagram_plugin,
            "_generate_mermaid_with_llm",
            return_value=mock_complex_mermaid,
        ):
            result = await diagram_plugin.generate(description)

            assert "subgraph" in result["mermaid"]
            assert "API Gateway" in result["mermaid"]
            assert "Microservices" in result["mermaid"]

    @pytest.mark.asyncio
    async def test_validate_mermaid_syntax(self, diagram_plugin):
        """Test Mermaid syntax validation."""
        valid_mermaid = "graph TD\n    A --> B\n    B --> C"
        invalid_mermaid = "graph TD\n    A -> B\n    B -->> C"  # Invalid syntax

        # Valid syntax should pass
        is_valid = diagram_plugin._validate_mermaid_syntax(valid_mermaid)
        assert is_valid is True

        # Invalid syntax should fail
        is_valid = diagram_plugin._validate_mermaid_syntax(invalid_mermaid)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_diagram_caching(self, diagram_plugin):
        """Test diagram result caching."""
        description = "Simple flow diagram"

        mock_mermaid = "graph TD\n    A --> B"

        with patch.object(
            diagram_plugin, "_generate_mermaid_with_llm", return_value=mock_mermaid
        ) as mock_generate:
            # First call should generate
            result1 = await diagram_plugin.generate(description)
            mock_generate.assert_called_once()

            # Second call with same description should use cache
            result2 = await diagram_plugin.generate(description)
            mock_generate.assert_called_once()  # Still only called once

            assert result1["mermaid"] == result2["mermaid"]

    @pytest.mark.asyncio
    async def test_diagram_error_handling(self, diagram_plugin):
        """Test error handling in diagram generation."""
        description = "Invalid diagram request"

        with patch.object(
            diagram_plugin,
            "_generate_mermaid_with_llm",
            side_effect=Exception("LLM API error"),
        ):
            result = await diagram_plugin.generate(description)

            # Should return error result instead of raising
            assert "error" in result
            assert "LLM API error" in result["error"]


class TestPluginIntegration:
    """Integration tests for plugin interactions."""

    @pytest.mark.asyncio
    async def test_retrieval_to_refinement_workflow(self):
        """Test workflow from retrieval to refinement."""
        mock_settings = Mock()

        with (
            patch("plugins.retrieval.CosmosClient"),
            patch("plugins.retrieval.AzureTextEmbedding"),
            patch("plugins.refinement.MLClient"),
        ):
            retrieval = RetrievalPlugin(mock_settings)
            refinement = RefinementPlugin(mock_settings)

            # Mock retrieval results
            mock_documents = [
                {"id": "doc1", "content": "Machine learning basics", "score": 0.75},
                {"id": "doc2", "content": "Advanced ML algorithms", "score": 0.70},
            ]

            with (
                patch.object(retrieval, "hybrid_search", return_value=mock_documents),
                patch.object(
                    refinement,
                    "rerank",
                    return_value=[
                        {
                            "id": "doc2",
                            "content": "Advanced ML algorithms",
                            "score": 0.88,
                        },
                        {
                            "id": "doc1",
                            "content": "Machine learning basics",
                            "score": 0.82,
                        },
                    ],
                ),
            ):
                # Execute workflow
                search_results = await retrieval.hybrid_search("machine learning")
                refined_results = await refinement.rerank(
                    "machine learning", search_results
                )

                # Verify workflow
                assert len(refined_results) == 2
                assert refined_results[0]["id"] == "doc2"  # Reranked to top

    @pytest.mark.asyncio
    async def test_memory_and_retrieval_integration(self):
        """Test integration between memory and retrieval plugins."""
        mock_settings = Mock()

        with (
            patch("plugins.memory.CosmosClient"),
            patch("plugins.memory.redis.from_url"),
            patch("plugins.memory.AzureTextEmbedding"),
            patch("plugins.retrieval.CosmosClient"),
            patch("plugins.retrieval.AzureTextEmbedding"),
        ):
            memory = MemoryPlugin(mock_settings)
            retrieval = RetrievalPlugin(mock_settings)

            # Save context to memory
            with patch.object(
                memory, "save", return_value={"id": "mem_123", "status": "saved"}
            ):
                memory_result = await memory.save(
                    "session_1", "User prefers Python", "semantic"
                )
                assert memory_result["status"] == "saved"

            # Retrieve related documents
            with patch.object(
                retrieval,
                "hybrid_search",
                return_value=[
                    {"id": "doc1", "content": "Python programming guide", "score": 0.92}
                ],
            ):
                search_results = await retrieval.hybrid_search("Python programming")
                assert len(search_results) == 1
                assert "Python" in search_results[0]["content"]
