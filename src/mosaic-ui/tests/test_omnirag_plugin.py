"""
Comprehensive tests for OmniRAG Plugin functionality.

Tests cover:
- OmniRAG strategy determination and routing
- Database RAG implementation
- Vector RAG semantic search
- Graph RAG relationship traversal
- Query processing and response formatting
- Azure service integration
- Error handling and fallbacks
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import json
from typing import Dict, List, Any


# Mock the OmniRAG plugin since it has Azure dependencies
class MockOmniRAGPlugin:
    """Mock OmniRAG plugin for testing purposes."""

    def __init__(self, settings=None):
        self.settings = settings or MagicMock()
        self.cosmos_client = None
        self.database = None
        self.embedding_service = None
        self._initialize_services()

    def _initialize_services(self):
        """Mock service initialization."""
        # Simulate Azure service availability based on settings
        if (
            hasattr(self.settings, "azure_cosmos_endpoint")
            and self.settings.azure_cosmos_endpoint
        ):
            self.cosmos_client = MagicMock()
            self.database = MagicMock()

        if (
            hasattr(self.settings, "azure_openai_endpoint")
            and self.settings.azure_openai_endpoint
        ):
            self.embedding_service = AsyncMock()

    def _determine_query_strategy(self, query: str) -> str:
        """Mock query strategy determination."""
        query_lower = query.lower()

        # Graph RAG indicators
        graph_keywords = [
            "dependencies",
            "depends on",
            "uses",
            "used by",
            "relationships",
            "connects to",
            "imports",
            "imported by",
            "related to",
            "hierarchy",
        ]

        # Database RAG indicators
        db_keywords = [
            "what is",
            "show me",
            "describe",
            "definition",
            "details about",
            "information about",
            "tell me about",
            "find entity",
            "get",
        ]

        # Vector RAG indicators
        vector_keywords = [
            "similar",
            "like",
            "resembles",
            "related concepts",
            "semantic search",
            "find similar",
            "similar to",
            "comparable",
            "equivalent",
        ]

        if any(keyword in query_lower for keyword in graph_keywords):
            return "graph_rag"
        elif any(keyword in query_lower for keyword in vector_keywords):
            return "vector_rag"
        else:
            return "database_rag"

    async def omnirag_query(self, query: str) -> str:
        """Mock main OmniRAG query method."""
        try:
            strategy = self._determine_query_strategy(query)

            if strategy == "database_rag":
                results = await self._database_rag(query)
            elif strategy == "vector_rag":
                results = await self._vector_rag(query)
            elif strategy == "graph_rag":
                results = await self._graph_rag(query)
            else:
                results = {"error": f"Unknown strategy: {strategy}"}

            return json.dumps(
                {
                    "query": query,
                    "strategy": strategy,
                    "results": results,
                    "omnirag_version": "1.0",
                },
                indent=2,
            )

        except Exception as e:
            return json.dumps({"error": str(e), "query": query, "strategy": "error"})

    async def _database_rag(self, query: str) -> Dict[str, Any]:
        """Mock database RAG implementation."""
        if not self.cosmos_client:
            return self._simulated_database_response(query)

        # Mock entity search
        entity_keywords = self._extract_entity_keywords(query)

        # Simulate database query results
        mock_entities = [
            {
                "id": "entity_1",
                "name": "Flask",
                "entity_type": "framework",
                "content": "Flask is a micro web framework",
                "file_path": "src/app.py",
            },
            {
                "id": "entity_2",
                "name": "FastAPI",
                "entity_type": "framework",
                "content": "FastAPI is a modern web framework",
                "file_path": "src/api.py",
            },
        ]

        return {
            "method": "database_rag",
            "entities_found": len(mock_entities),
            "entities": mock_entities,
            "keywords_searched": entity_keywords,
        }

    async def _vector_rag(self, query: str) -> Dict[str, Any]:
        """Mock vector RAG implementation."""
        if not self.cosmos_client or not self.embedding_service:
            return self._simulated_vector_response(query)

        # Mock embedding generation
        mock_embedding = [0.1] * 1536  # Typical OpenAI embedding size

        # Mock vector search results
        mock_matches = [
            {
                "name": "similar_component_1",
                "entity_type": "function",
                "content": "Similar functionality",
                "similarity_score": 0.85,
            },
            {
                "name": "related_class_2",
                "entity_type": "class",
                "content": "Related implementation",
                "similarity_score": 0.78,
            },
        ]

        return {
            "method": "vector_rag",
            "semantic_matches": len(mock_matches),
            "top_matches": mock_matches,
            "query_embedding_dims": len(mock_embedding),
        }

    async def _graph_rag(self, query: str) -> Dict[str, Any]:
        """Mock graph RAG implementation."""
        if not self.cosmos_client:
            return self._simulated_graph_response(query)

        entity_keywords = self._extract_entity_keywords(query)

        # Mock relationship data
        mock_relationships = [
            {
                "id": "rel_1",
                "source_name": "FastAPI",
                "target_name": "Pydantic",
                "relationship_type": "depends_on",
                "relationship_strength": 0.9,
            },
            {
                "id": "rel_2",
                "source_name": "FastAPI",
                "target_name": "Uvicorn",
                "relationship_type": "uses",
                "relationship_strength": 0.8,
            },
        ]

        related_entities = {"FastAPI", "Pydantic", "Uvicorn"}

        return {
            "method": "graph_rag",
            "relationships_found": len(mock_relationships),
            "related_entities": list(related_entities),
            "graph_data": mock_relationships,
            "keywords_searched": entity_keywords,
        }

    def _extract_entity_keywords(self, query: str) -> List[str]:
        """Mock entity keyword extraction."""
        import re

        stop_words = {
            "what",
            "is",
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "show",
            "me",
            "find",
            "get",
            "tell",
            "about",
        }

        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]*\b", query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        return keywords[:3]

    def _simulated_database_response(self, query: str) -> Dict[str, Any]:
        """Mock simulated database response."""
        return {
            "method": "database_rag",
            "mode": "simulated",
            "message": "Database RAG would search code entities for direct matches",
            "example_entities": [
                {"name": "semantic_kernel", "type": "module", "language": "python"},
                {"name": "FastMCP", "type": "class", "language": "python"},
            ],
            "query": query,
        }

    def _simulated_vector_response(self, query: str) -> Dict[str, Any]:
        """Mock simulated vector response."""
        return {
            "method": "vector_rag",
            "mode": "simulated",
            "message": "Vector RAG would perform semantic similarity search",
            "example_matches": [
                {"name": "similar_concept_1", "similarity": 0.85},
                {"name": "related_function_2", "similarity": 0.78},
            ],
            "query": query,
        }

    def _simulated_graph_response(self, query: str) -> Dict[str, Any]:
        """Mock simulated graph response."""
        return {
            "method": "graph_rag",
            "mode": "simulated",
            "message": "Graph RAG would traverse relationships and dependencies",
            "example_relationships": [
                {
                    "source": "FastMCP",
                    "target": "semantic_kernel",
                    "type": "depends_on",
                },
                {"source": "OmniRAG", "target": "Azure_Cosmos_DB", "type": "uses"},
            ],
            "query": query,
        }

    async def get_filtered_graph(self, query: str) -> str:
        """Mock filtered graph data retrieval."""
        try:
            # Get OmniRAG results first
            omnirag_result = await self.omnirag_query(query)
            result_data = json.loads(omnirag_result)

            # Extract relevant entities
            relevant_entities = set()

            if result_data.get("strategy") == "database_rag":
                for entity in result_data.get("results", {}).get("entities", []):
                    relevant_entities.add(entity.get("name", ""))

            elif result_data.get("strategy") == "graph_rag":
                relevant_entities.update(
                    result_data.get("results", {}).get("related_entities", [])
                )

            elif result_data.get("strategy") == "vector_rag":
                for match in result_data.get("results", {}).get("top_matches", []):
                    relevant_entities.add(match.get("name", ""))

            # Generate filtered graph
            filtered_graph = self._generate_filtered_graph_data(relevant_entities)

            return json.dumps(filtered_graph, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e)})

    def _generate_filtered_graph_data(self, relevant_entities: set) -> Dict[str, Any]:
        """Mock filtered graph data generation."""
        if not relevant_entities:
            return {
                "nodes": [],
                "links": [],
                "message": "No specific entities found",
                "filter_applied": False,
            }

        # Generate mock filtered data
        filtered_nodes = []
        filtered_links = []

        for entity in list(relevant_entities)[:10]:
            filtered_nodes.append(
                {"id": entity, "name": entity, "type": "filtered_entity", "group": 1}
            )

        # Add connections between filtered nodes
        entities_list = list(relevant_entities)
        for i in range(min(len(entities_list) - 1, 5)):
            filtered_links.append(
                {
                    "source": entities_list[i],
                    "target": entities_list[i + 1],
                    "type": "filtered_relationship",
                }
            )

        return {
            "nodes": filtered_nodes,
            "links": filtered_links,
            "total_nodes": len(filtered_nodes),
            "total_links": len(filtered_links),
            "filter_applied": True,
            "filtered_entities": list(relevant_entities),
        }


@pytest.fixture
def mock_omnirag_plugin():
    """Create mock OmniRAG plugin for testing."""
    settings = MagicMock()
    settings.azure_cosmos_endpoint = "https://test.cosmos.azure.com"
    settings.azure_openai_endpoint = "https://test.openai.azure.com"
    return MockOmniRAGPlugin(settings)


@pytest.fixture
def mock_omnirag_plugin_no_services():
    """Create mock OmniRAG plugin without Azure services."""
    settings = MagicMock()
    settings.azure_cosmos_endpoint = None
    settings.azure_openai_endpoint = None
    return MockOmniRAGPlugin(settings)


class TestOmniRAGStrategyDetermination:
    """Test OmniRAG strategy determination logic."""

    def test_database_rag_strategy(self, mock_omnirag_plugin):
        """Test database RAG strategy identification."""
        database_queries = [
            "what is Flask",
            "show me React components",
            "describe the authentication system",
            "tell me about the API endpoints",
            "get information about the database model",
        ]

        for query in database_queries:
            strategy = mock_omnirag_plugin._determine_query_strategy(query)
            assert strategy == "database_rag", (
                f"Query '{query}' should use database_rag"
            )

    def test_vector_rag_strategy(self, mock_omnirag_plugin):
        """Test vector RAG strategy identification."""
        vector_queries = [
            "find similar components to Flask",
            "components like React",
            "find related concepts to authentication",
            "semantic search for API patterns",
            "similar to authentication middleware",
        ]

        for query in vector_queries:
            strategy = mock_omnirag_plugin._determine_query_strategy(query)
            assert strategy == "vector_rag", f"Query '{query}' should use vector_rag"

    def test_graph_rag_strategy(self, mock_omnirag_plugin):
        """Test graph RAG strategy identification."""
        graph_queries = [
            "dependencies of Flask",
            "what uses React components",
            "relationships between services",
            "imports of the authentication module",
            "components that connect to the database",
            "hierarchy of the API structure",
        ]

        for query in graph_queries:
            strategy = mock_omnirag_plugin._determine_query_strategy(query)
            assert strategy == "graph_rag", f"Query '{query}' should use graph_rag"

    def test_strategy_edge_cases(self, mock_omnirag_plugin):
        """Test strategy determination edge cases."""
        edge_cases = [
            ("", "database_rag"),  # Empty query defaults to database
            ("simple query", "database_rag"),  # Simple query defaults to database
            ("FIND SIMILAR COMPONENTS", "vector_rag"),  # Case insensitive
            ("Dependencies and relationships", "graph_rag"),  # Multiple keywords
        ]

        for query, expected_strategy in edge_cases:
            strategy = mock_omnirag_plugin._determine_query_strategy(query)
            assert strategy == expected_strategy, (
                f"Query '{query}' should use {expected_strategy}"
            )


class TestDatabaseRAG:
    """Test Database RAG functionality."""

    @pytest.mark.asyncio
    async def test_database_rag_with_services(self, mock_omnirag_plugin):
        """Test database RAG with Azure services available."""
        query = "what is Flask"
        result = await mock_omnirag_plugin._database_rag(query)

        assert result["method"] == "database_rag"
        assert "entities_found" in result
        assert "entities" in result
        assert "keywords_searched" in result

        # Should have found mock entities
        assert result["entities_found"] == 2
        assert len(result["entities"]) == 2

        # Verify entity structure
        entity = result["entities"][0]
        assert "id" in entity
        assert "name" in entity
        assert "entity_type" in entity
        assert "content" in entity

    @pytest.mark.asyncio
    async def test_database_rag_without_services(self, mock_omnirag_plugin_no_services):
        """Test database RAG without Azure services (simulated mode)."""
        query = "what is Flask"
        result = await mock_omnirag_plugin_no_services._database_rag(query)

        assert result["method"] == "database_rag"
        assert result["mode"] == "simulated"
        assert "message" in result
        assert "example_entities" in result
        assert result["query"] == query

    @pytest.mark.asyncio
    async def test_database_rag_keyword_extraction(self, mock_omnirag_plugin):
        """Test keyword extraction for database queries."""
        query = "show me the Flask application components"
        result = await mock_omnirag_plugin._database_rag(query)

        keywords = result["keywords_searched"]

        # Should extract meaningful keywords
        assert "flask" in keywords or "Flask" in [k.title() for k in keywords]
        assert "application" in keywords
        assert "components" in keywords

        # Should filter out stop words
        assert "show" not in keywords
        assert "me" not in keywords
        assert "the" not in keywords

    @pytest.mark.asyncio
    async def test_database_rag_empty_query(self, mock_omnirag_plugin):
        """Test database RAG with empty query."""
        result = await mock_omnirag_plugin._database_rag("")

        assert result["method"] == "database_rag"
        assert "keywords_searched" in result
        assert len(result["keywords_searched"]) == 0


class TestVectorRAG:
    """Test Vector RAG functionality."""

    @pytest.mark.asyncio
    async def test_vector_rag_with_services(self, mock_omnirag_plugin):
        """Test vector RAG with Azure services available."""
        query = "find similar components"
        result = await mock_omnirag_plugin._vector_rag(query)

        assert result["method"] == "vector_rag"
        assert "semantic_matches" in result
        assert "top_matches" in result
        assert "query_embedding_dims" in result

        # Should have found mock matches
        assert result["semantic_matches"] == 2
        assert len(result["top_matches"]) == 2
        assert result["query_embedding_dims"] == 1536  # OpenAI embedding size

        # Verify match structure
        match = result["top_matches"][0]
        assert "name" in match
        assert "entity_type" in match
        assert "content" in match
        assert "similarity_score" in match
        assert isinstance(match["similarity_score"], float)

    @pytest.mark.asyncio
    async def test_vector_rag_without_services(self, mock_omnirag_plugin_no_services):
        """Test vector RAG without Azure services (simulated mode)."""
        query = "find similar components"
        result = await mock_omnirag_plugin_no_services._vector_rag(query)

        assert result["method"] == "vector_rag"
        assert result["mode"] == "simulated"
        assert "message" in result
        assert "example_matches" in result
        assert result["query"] == query

    @pytest.mark.asyncio
    async def test_vector_rag_similarity_scores(self, mock_omnirag_plugin):
        """Test vector RAG similarity score validation."""
        query = "semantic search test"
        result = await mock_omnirag_plugin._vector_rag(query)

        matches = result["top_matches"]

        for match in matches:
            similarity = match["similarity_score"]
            assert 0.0 <= similarity <= 1.0, (
                f"Similarity score {similarity} should be between 0 and 1"
            )

        # Scores should be in descending order (most similar first)
        if len(matches) > 1:
            for i in range(len(matches) - 1):
                assert (
                    matches[i]["similarity_score"] >= matches[i + 1]["similarity_score"]
                )


class TestGraphRAG:
    """Test Graph RAG functionality."""

    @pytest.mark.asyncio
    async def test_graph_rag_with_services(self, mock_omnirag_plugin):
        """Test graph RAG with Azure services available."""
        query = "dependencies of FastAPI"
        result = await mock_omnirag_plugin._graph_rag(query)

        assert result["method"] == "graph_rag"
        assert "relationships_found" in result
        assert "related_entities" in result
        assert "graph_data" in result
        assert "keywords_searched" in result

        # Should have found mock relationships
        assert result["relationships_found"] == 2
        assert len(result["graph_data"]) == 2

        # Verify relationship structure
        relationship = result["graph_data"][0]
        assert "id" in relationship
        assert "source_name" in relationship
        assert "target_name" in relationship
        assert "relationship_type" in relationship
        assert "relationship_strength" in relationship

    @pytest.mark.asyncio
    async def test_graph_rag_without_services(self, mock_omnirag_plugin_no_services):
        """Test graph RAG without Azure services (simulated mode)."""
        query = "dependencies of FastAPI"
        result = await mock_omnirag_plugin_no_services._graph_rag(query)

        assert result["method"] == "graph_rag"
        assert result["mode"] == "simulated"
        assert "message" in result
        assert "example_relationships" in result
        assert result["query"] == query

    @pytest.mark.asyncio
    async def test_graph_rag_relationship_types(self, mock_omnirag_plugin):
        """Test graph RAG relationship type validation."""
        query = "what uses FastAPI"
        result = await mock_omnirag_plugin._graph_rag(query)

        relationships = result["graph_data"]

        valid_relationship_types = {
            "depends_on",
            "uses",
            "imports",
            "inherits",
            "implements",
            "calls",
            "creates",
            "configures",
            "extends",
        }

        for rel in relationships:
            rel_type = rel["relationship_type"]
            assert (
                rel_type in valid_relationship_types or rel_type == "depends_on"
            )  # Mock data

    @pytest.mark.asyncio
    async def test_graph_rag_entity_extraction(self, mock_omnirag_plugin):
        """Test entity extraction for graph traversal."""
        query = "relationships between React and Redux components"
        result = await mock_omnirag_plugin._graph_rag(query)

        related_entities = result["related_entities"]

        # Should have extracted relevant entities
        assert len(related_entities) > 0
        assert all(isinstance(entity, str) for entity in related_entities)


class TestOmniRAGQueryProcessing:
    """Test main OmniRAG query processing."""

    @pytest.mark.asyncio
    async def test_omnirag_query_database_strategy(self, mock_omnirag_plugin):
        """Test complete OmniRAG query with database strategy."""
        query = "what is Flask"
        response = await mock_omnirag_plugin.omnirag_query(query)

        result = json.loads(response)

        assert result["query"] == query
        assert result["strategy"] == "database_rag"
        assert result["omnirag_version"] == "1.0"
        assert "results" in result

        # Verify database RAG results
        results = result["results"]
        assert results["method"] == "database_rag"
        assert "entities_found" in results

    @pytest.mark.asyncio
    async def test_omnirag_query_vector_strategy(self, mock_omnirag_plugin):
        """Test complete OmniRAG query with vector strategy."""
        query = "find similar components to Flask"
        response = await mock_omnirag_plugin.omnirag_query(query)

        result = json.loads(response)

        assert result["query"] == query
        assert result["strategy"] == "vector_rag"
        assert "results" in result

        # Verify vector RAG results
        results = result["results"]
        assert results["method"] == "vector_rag"
        assert "semantic_matches" in results

    @pytest.mark.asyncio
    async def test_omnirag_query_graph_strategy(self, mock_omnirag_plugin):
        """Test complete OmniRAG query with graph strategy."""
        query = "dependencies of Flask"
        response = await mock_omnirag_plugin.omnirag_query(query)

        result = json.loads(response)

        assert result["query"] == query
        assert result["strategy"] == "graph_rag"
        assert "results" in result

        # Verify graph RAG results
        results = result["results"]
        assert results["method"] == "graph_rag"
        assert "relationships_found" in results

    @pytest.mark.asyncio
    async def test_omnirag_query_error_handling(self, mock_omnirag_plugin):
        """Test OmniRAG query error handling."""
        # Patch _database_rag to raise an exception
        with patch.object(
            mock_omnirag_plugin, "_database_rag", side_effect=Exception("Test error")
        ):
            query = "what is Flask"
            response = await mock_omnirag_plugin.omnirag_query(query)

            result = json.loads(response)

            assert "error" in result
            assert result["query"] == query
            assert result["strategy"] == "error"
            assert "Test error" in result["error"]

    @pytest.mark.asyncio
    async def test_omnirag_query_json_format(self, mock_omnirag_plugin):
        """Test OmniRAG query JSON response format."""
        query = "test query"
        response = await mock_omnirag_plugin.omnirag_query(query)

        # Should be valid JSON
        result = json.loads(response)

        # Required fields
        required_fields = ["query", "strategy", "results", "omnirag_version"]
        for field in required_fields:
            assert field in result

        # Proper data types
        assert isinstance(result["query"], str)
        assert isinstance(result["strategy"], str)
        assert isinstance(result["results"], dict)
        assert isinstance(result["omnirag_version"], str)


class TestFilteredGraphGeneration:
    """Test filtered graph generation functionality."""

    @pytest.mark.asyncio
    async def test_get_filtered_graph_database_strategy(self, mock_omnirag_plugin):
        """Test filtered graph generation from database RAG results."""
        query = "what is Flask"
        response = await mock_omnirag_plugin.get_filtered_graph(query)

        result = json.loads(response)

        assert "nodes" in result
        assert "links" in result
        assert "total_nodes" in result
        assert "total_links" in result
        assert "filter_applied" in result

        # Should have extracted entities from database results
        if result["filter_applied"]:
            assert len(result["nodes"]) > 0
            assert "filtered_entities" in result

    @pytest.mark.asyncio
    async def test_get_filtered_graph_vector_strategy(self, mock_omnirag_plugin):
        """Test filtered graph generation from vector RAG results."""
        query = "find similar components"
        response = await mock_omnirag_plugin.get_filtered_graph(query)

        result = json.loads(response)

        assert "nodes" in result
        assert "links" in result

        # Should have extracted entities from vector search results
        if result["filter_applied"]:
            assert len(result["nodes"]) > 0

    @pytest.mark.asyncio
    async def test_get_filtered_graph_graph_strategy(self, mock_omnirag_plugin):
        """Test filtered graph generation from graph RAG results."""
        query = "dependencies of FastAPI"
        response = await mock_omnirag_plugin.get_filtered_graph(query)

        result = json.loads(response)

        assert "nodes" in result
        assert "links" in result

        # Graph RAG should provide related entities directly
        if result["filter_applied"]:
            assert len(result["nodes"]) > 0
            assert "filtered_entities" in result

    @pytest.mark.asyncio
    async def test_filtered_graph_node_structure(self, mock_omnirag_plugin):
        """Test filtered graph node structure."""
        query = "what is Flask"
        response = await mock_omnirag_plugin.get_filtered_graph(query)

        result = json.loads(response)

        if result["filter_applied"] and result["nodes"]:
            node = result["nodes"][0]

            required_fields = ["id", "name", "type", "group"]
            for field in required_fields:
                assert field in node

    @pytest.mark.asyncio
    async def test_filtered_graph_link_structure(self, mock_omnirag_plugin):
        """Test filtered graph link structure."""
        query = "dependencies of FastAPI"
        response = await mock_omnirag_plugin.get_filtered_graph(query)

        result = json.loads(response)

        if result["filter_applied"] and result["links"]:
            link = result["links"][0]

            required_fields = ["source", "target", "type"]
            for field in required_fields:
                assert field in link

    @pytest.mark.asyncio
    async def test_filtered_graph_empty_results(self, mock_omnirag_plugin):
        """Test filtered graph with empty results."""
        # Mock empty results by patching the filtered graph data generation
        with patch.object(
            mock_omnirag_plugin, "_generate_filtered_graph_data"
        ) as mock_gen:
            mock_gen.return_value = {
                "nodes": [],
                "links": [],
                "message": "No specific entities found",
                "filter_applied": False,
            }

            query = "nonexistent entity"
            response = await mock_omnirag_plugin.get_filtered_graph(query)

            result = json.loads(response)

            assert len(result["nodes"]) == 0
            assert len(result["links"]) == 0
            assert result["filter_applied"] is False


class TestKeywordExtraction:
    """Test keyword extraction functionality."""

    def test_basic_keyword_extraction(self, mock_omnirag_plugin):
        """Test basic keyword extraction."""
        query = "show me the Flask application components"
        keywords = mock_omnirag_plugin._extract_entity_keywords(query)

        # Should extract meaningful keywords
        expected_keywords = ["flask", "application", "components"]
        for expected in expected_keywords:
            assert any(expected.lower() in k.lower() for k in keywords)

        # Should filter out stop words
        stop_words = ["show", "me", "the"]
        for stop_word in stop_words:
            assert stop_word not in keywords

    def test_keyword_extraction_limits(self, mock_omnirag_plugin):
        """Test keyword extraction limits."""
        long_query = "find all the React components that use Redux state management with authentication middleware patterns"
        keywords = mock_omnirag_plugin._extract_entity_keywords(long_query)

        # Should limit to 3 keywords
        assert len(keywords) <= 3

    def test_keyword_extraction_special_characters(self, mock_omnirag_plugin):
        """Test keyword extraction with special characters."""
        query = "find Flask_API components with @decorators"
        keywords = mock_omnirag_plugin._extract_entity_keywords(query)

        # Should handle underscores in identifiers
        assert any("flask_api" in k.lower() for k in keywords) or any(
            "flask" in k.lower() for k in keywords
        )

    def test_keyword_extraction_empty_query(self, mock_omnirag_plugin):
        """Test keyword extraction with empty query."""
        keywords = mock_omnirag_plugin._extract_entity_keywords("")

        assert isinstance(keywords, list)
        assert len(keywords) == 0


class TestServiceIntegration:
    """Test Azure service integration."""

    def test_initialization_with_services(self, mock_omnirag_plugin):
        """Test initialization with Azure services available."""
        assert mock_omnirag_plugin.cosmos_client is not None
        assert mock_omnirag_plugin.database is not None
        assert mock_omnirag_plugin.embedding_service is not None

    def test_initialization_without_services(self, mock_omnirag_plugin_no_services):
        """Test initialization without Azure services."""
        assert mock_omnirag_plugin_no_services.cosmos_client is None
        assert mock_omnirag_plugin_no_services.database is None
        assert mock_omnirag_plugin_no_services.embedding_service is None

    @pytest.mark.asyncio
    async def test_fallback_to_simulated_mode(self, mock_omnirag_plugin_no_services):
        """Test fallback to simulated mode when services unavailable."""
        # All RAG methods should fall back to simulated mode
        db_result = await mock_omnirag_plugin_no_services._database_rag("test")
        vector_result = await mock_omnirag_plugin_no_services._vector_rag("test")
        graph_result = await mock_omnirag_plugin_no_services._graph_rag("test")

        assert db_result["mode"] == "simulated"
        assert vector_result["mode"] == "simulated"
        assert graph_result["mode"] == "simulated"


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_query_handling(self, mock_omnirag_plugin):
        """Test handling of empty queries."""
        response = await mock_omnirag_plugin.omnirag_query("")
        result = json.loads(response)

        assert result["query"] == ""
        assert "strategy" in result
        assert "results" in result

    @pytest.mark.asyncio
    async def test_very_long_query_handling(self, mock_omnirag_plugin):
        """Test handling of very long queries."""
        long_query = "find " * 1000 + "components"  # Very long query
        response = await mock_omnirag_plugin.omnirag_query(long_query)
        result = json.loads(response)

        assert result["query"] == long_query
        assert "strategy" in result

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self, mock_omnirag_plugin):
        """Test handling of special characters in queries."""
        special_query = "find @decorators & $variables with #tags"
        response = await mock_omnirag_plugin.omnirag_query(special_query)
        result = json.loads(response)

        assert result["query"] == special_query
        assert "strategy" in result

    @pytest.mark.asyncio
    async def test_non_english_query_handling(self, mock_omnirag_plugin):
        """Test handling of non-English queries."""
        non_english_query = "trouvez des composants similaires"  # French
        response = await mock_omnirag_plugin.omnirag_query(non_english_query)
        result = json.loads(response)

        assert result["query"] == non_english_query
        assert "strategy" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
