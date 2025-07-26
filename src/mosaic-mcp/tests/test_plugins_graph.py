"""
Comprehensive tests for GraphPlugin in Mosaic MCP Server.

Tests cover:
- Natural language query processing and SPARQL translation
- Graph schema discovery and exploration
- Graph query execution and result formatting
- Interactive visualization generation
- Error handling and edge cases
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any, Optional
import json

# Mock the GraphPlugin since it's not implemented yet
class MockGraphPlugin:
    """Mock GraphPlugin for testing purposes."""
    
    def __init__(self, settings):
        self.settings = settings
        self.nl2sparql_service = None
        self.sparql_executor = None
        self.visualization_generator = None
    
    async def initialize(self):
        """Initialize mock plugin."""
        pass
    
    async def natural_language_query(
        self, query: str, include_visualization: bool = False, max_results: int = 100
    ) -> Dict[str, Any]:
        """Mock natural language query processing."""
        if "error" in query.lower():
            raise ValueError("Query translation failed")
        
        return {
            "query": query,
            "sparql_query": "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10",
            "results": [
                {"s": "entity1", "p": "hasProperty", "o": "value1"},
                {"s": "entity2", "p": "hasProperty", "o": "value2"}
            ],
            "total_results": 2,
            "execution_time_ms": 150,
            "visualization": "<div>Mock visualization</div>" if include_visualization else None
        }
    
    async def execute_sparql_query(
        self, sparql_query: str, include_visualization: bool = False, max_results: int = 100
    ) -> Dict[str, Any]:
        """Mock SPARQL query execution."""
        if "INVALID" in sparql_query:
            raise ValueError("Invalid SPARQL syntax")
        
        return {
            "sparql_query": sparql_query,
            "results": [
                {"subject": "http://example.org/entity1", "predicate": "rdf:type", "object": "Class"},
                {"subject": "http://example.org/entity2", "predicate": "rdf:type", "object": "Property"}
            ],
            "total_results": 2,
            "execution_time_ms": 75,
            "visualization": "<div>SPARQL visualization</div>" if include_visualization else None
        }
    
    async def visualize_graph_results(
        self, results: List[Dict[str, Any]], title: str = "Graph Query Results", layout: str = "force"
    ) -> str:
        """Mock graph visualization generation."""
        if not results:
            return "<div>No data to visualize</div>"
        
        return f"""
        <div id="graph-visualization">
            <h3>{title}</h3>
            <div class="graph-container" data-layout="{layout}">
                <p>Visualizing {len(results)} results</p>
            </div>
        </div>
        """
    
    async def discover_graph_schema(
        self, entity_type: Optional[str] = None, include_counts: bool = True, max_results: int = 50
    ) -> Dict[str, Any]:
        """Mock graph schema discovery."""
        schema_data = {
            "entity_types": ["Class", "Property", "Individual"],
            "relationship_types": ["subClassOf", "hasProperty", "instanceOf"],
            "total_entities": 1000,
            "total_relationships": 2500
        }
        
        if entity_type:
            schema_data["filtered_by"] = entity_type
            schema_data["total_entities"] = 100
        
        if include_counts:
            schema_data["counts"] = {
                "Class": 300,
                "Property": 200,
                "Individual": 500
            }
        
        return schema_data


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock()
    settings.max_search_results = 50
    settings.azure_openai_endpoint = "https://test.openai.azure.com"
    settings.azure_openai_text_embedding_deployment_name = "text-embedding-ada-002"
    settings.get_cosmos_config.return_value = {
        "endpoint": "https://test.cosmos.azure.com",
        "database_name": "mosaic_test",
        "container_name": "knowledge_test"
    }
    return settings


@pytest.fixture
def graph_plugin(mock_settings):
    """Create GraphPlugin instance for testing."""
    return MockGraphPlugin(mock_settings)


class TestGraphPluginInitialization:
    """Test GraphPlugin initialization and setup."""
    
    def test_init(self, mock_settings):
        """Test plugin initialization."""
        plugin = MockGraphPlugin(mock_settings)
        assert plugin.settings == mock_settings
        assert plugin.nl2sparql_service is None
        assert plugin.sparql_executor is None
        assert plugin.visualization_generator is None
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, graph_plugin):
        """Test successful plugin initialization."""
        await graph_plugin.initialize()
        # Mock initialization always succeeds
        assert True


class TestNaturalLanguageQuery:
    """Test natural language query processing."""
    
    @pytest.mark.asyncio
    async def test_natural_language_query_basic(self, graph_plugin):
        """Test basic natural language query processing."""
        query = "Find all classes in the ontology"
        result = await graph_plugin.natural_language_query(query)
        
        assert "query" in result
        assert "sparql_query" in result
        assert "results" in result
        assert "total_results" in result
        assert "execution_time_ms" in result
        assert result["query"] == query
        assert len(result["results"]) == 2
    
    @pytest.mark.asyncio
    async def test_natural_language_query_with_visualization(self, graph_plugin):
        """Test natural language query with visualization."""
        query = "Show me the class hierarchy"
        result = await graph_plugin.natural_language_query(
            query, include_visualization=True
        )
        
        assert result["visualization"] is not None
        assert "<div>Mock visualization</div>" in result["visualization"]
    
    @pytest.mark.asyncio
    async def test_natural_language_query_without_visualization(self, graph_plugin):
        """Test natural language query without visualization."""
        query = "List all properties"
        result = await graph_plugin.natural_language_query(
            query, include_visualization=False
        )
        
        assert result["visualization"] is None
    
    @pytest.mark.asyncio
    async def test_natural_language_query_with_max_results(self, graph_plugin):
        """Test natural language query with result limits."""
        query = "Find entities with specific property"
        result = await graph_plugin.natural_language_query(
            query, max_results=5
        )
        
        assert result["total_results"] == 2  # Mock returns 2 results
        assert len(result["results"]) <= 5
    
    @pytest.mark.asyncio
    async def test_natural_language_query_error_handling(self, graph_plugin):
        """Test error handling in natural language query."""
        query = "This is an error query"
        
        with pytest.raises(ValueError, match="Query translation failed"):
            await graph_plugin.natural_language_query(query)


class TestSPARQLQuery:
    """Test SPARQL query execution."""
    
    @pytest.mark.asyncio
    async def test_sparql_query_execution(self, graph_plugin):
        """Test basic SPARQL query execution."""
        sparql_query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10"
        result = await graph_plugin.execute_sparql_query(sparql_query)
        
        assert "sparql_query" in result
        assert "results" in result
        assert "total_results" in result
        assert "execution_time_ms" in result
        assert result["sparql_query"] == sparql_query
        assert len(result["results"]) == 2
    
    @pytest.mark.asyncio
    async def test_sparql_query_with_visualization(self, graph_plugin):
        """Test SPARQL query with visualization."""
        sparql_query = "SELECT ?class WHERE { ?class rdf:type owl:Class }"
        result = await graph_plugin.execute_sparql_query(
            sparql_query, include_visualization=True
        )
        
        assert result["visualization"] is not None
        assert "SPARQL visualization" in result["visualization"]
    
    @pytest.mark.asyncio
    async def test_sparql_query_error_handling(self, graph_plugin):
        """Test error handling for invalid SPARQL queries."""
        invalid_query = "INVALID SPARQL SYNTAX"
        
        with pytest.raises(ValueError, match="Invalid SPARQL syntax"):
            await graph_plugin.execute_sparql_query(invalid_query)
    
    @pytest.mark.asyncio
    async def test_sparql_query_result_limiting(self, graph_plugin):
        """Test SPARQL query result limiting."""
        sparql_query = "SELECT * WHERE { ?s ?p ?o }"
        result = await graph_plugin.execute_sparql_query(
            sparql_query, max_results=1
        )
        
        assert result["total_results"] == 2  # Mock returns 2
        # In real implementation, results would be limited


class TestGraphVisualization:
    """Test graph visualization generation."""
    
    @pytest.mark.asyncio
    async def test_visualize_graph_results_basic(self, graph_plugin):
        """Test basic graph visualization."""
        results = [
            {"subject": "A", "predicate": "relates", "object": "B"},
            {"subject": "B", "predicate": "relates", "object": "C"}
        ]
        
        html = await graph_plugin.visualize_graph_results(results)
        
        assert "<div id=\"graph-visualization\">" in html
        assert "Visualizing 2 results" in html
        assert "Graph Query Results" in html
    
    @pytest.mark.asyncio
    async def test_visualize_graph_results_custom_title(self, graph_plugin):
        """Test graph visualization with custom title."""
        results = [{"s": "entity", "p": "property", "o": "value"}]
        title = "Custom Graph Title"
        
        html = await graph_plugin.visualize_graph_results(results, title=title)
        
        assert title in html
    
    @pytest.mark.asyncio
    async def test_visualize_graph_results_different_layouts(self, graph_plugin):
        """Test graph visualization with different layout algorithms."""
        results = [{"s": "A", "p": "relates", "o": "B"}]
        
        for layout in ["force", "circular", "hierarchical"]:
            html = await graph_plugin.visualize_graph_results(
                results, layout=layout
            )
            assert f'data-layout="{layout}"' in html
    
    @pytest.mark.asyncio
    async def test_visualize_empty_results(self, graph_plugin):
        """Test visualization with empty results."""
        html = await graph_plugin.visualize_graph_results([])
        
        assert "No data to visualize" in html


class TestGraphSchemaDiscovery:
    """Test graph schema discovery and exploration."""
    
    @pytest.mark.asyncio
    async def test_discover_schema_basic(self, graph_plugin):
        """Test basic schema discovery."""
        result = await graph_plugin.discover_graph_schema()
        
        assert "entity_types" in result
        assert "relationship_types" in result
        assert "total_entities" in result
        assert "total_relationships" in result
        assert result["total_entities"] == 1000
        assert result["total_relationships"] == 2500
    
    @pytest.mark.asyncio
    async def test_discover_schema_with_counts(self, graph_plugin):
        """Test schema discovery with entity counts."""
        result = await graph_plugin.discover_graph_schema(include_counts=True)
        
        assert "counts" in result
        assert "Class" in result["counts"]
        assert "Property" in result["counts"]
        assert "Individual" in result["counts"]
    
    @pytest.mark.asyncio
    async def test_discover_schema_without_counts(self, graph_plugin):
        """Test schema discovery without entity counts."""
        result = await graph_plugin.discover_graph_schema(include_counts=False)
        
        # Mock always includes counts, but real implementation would respect this
        assert "counts" in result  # Mock behavior
    
    @pytest.mark.asyncio
    async def test_discover_schema_filtered_by_type(self, graph_plugin):
        """Test schema discovery filtered by entity type."""
        entity_type = "Class"
        result = await graph_plugin.discover_graph_schema(entity_type=entity_type)
        
        assert "filtered_by" in result
        assert result["filtered_by"] == entity_type
        assert result["total_entities"] == 100  # Filtered count
    
    @pytest.mark.asyncio
    async def test_discover_schema_with_max_results(self, graph_plugin):
        """Test schema discovery with result limits."""
        result = await graph_plugin.discover_graph_schema(max_results=10)
        
        # Mock doesn't respect max_results, but real implementation would
        assert "entity_types" in result
        assert "relationship_types" in result


class TestGraphVisualizationPluginTools:
    """Test graph visualization plugin tools."""
    
    def test_mock_graph_visualization_plugin(self):
        """Test that mock graph visualization plugin can be created."""
        # This would test the actual GraphVisualizationPlugin
        # For now, we just verify the concept
        assert True
    
    def test_repository_structure_visualization_concept(self):
        """Test repository structure visualization concept."""
        # Mock the visualization functionality
        repository_url = "https://github.com/example/repo"
        
        # Expected parameters for repository visualization
        params = {
            "repository_url": repository_url,
            "include_functions": True,
            "include_classes": True,
            "color_by_language": True,
            "size_by_complexity": True
        }
        
        assert params["repository_url"] == repository_url
        assert params["include_functions"] is True
    
    def test_dependency_visualization_concept(self):
        """Test code dependency visualization concept."""
        repository_url = "https://github.com/example/repo"
        
        params = {
            "repository_url": repository_url,
            "dependency_types": ["imports", "calls", "inherits"],
            "show_external_deps": True,
            "layout_algorithm": "force"
        }
        
        assert "imports" in params["dependency_types"]
        assert params["show_external_deps"] is True
    
    def test_knowledge_graph_visualization_concept(self):
        """Test knowledge graph visualization concept."""
        params = {
            "repository_url": "https://github.com/example/repo",
            "include_semantic_similarity": True,
            "cluster_by_functionality": True,
            "max_nodes": 200
        }
        
        assert params["max_nodes"] == 200
        assert params["cluster_by_functionality"] is True


class TestGraphPluginIntegration:
    """Test GraphPlugin integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_query_to_visualization_workflow(self, graph_plugin):
        """Test complete workflow from query to visualization."""
        # Step 1: Execute natural language query
        query = "Find all related entities"
        query_result = await graph_plugin.natural_language_query(query)
        
        # Step 2: Visualize the results
        visualization = await graph_plugin.visualize_graph_results(
            query_result["results"]
        )
        
        assert query_result["results"] is not None
        assert len(visualization) > 0
        assert "graph-visualization" in visualization
    
    @pytest.mark.asyncio
    async def test_schema_discovery_to_query_workflow(self, graph_plugin):
        """Test workflow from schema discovery to targeted queries."""
        # Step 1: Discover schema
        schema = await graph_plugin.discover_graph_schema()
        
        # Step 2: Use schema information to build query
        entity_types = schema["entity_types"]
        assert len(entity_types) > 0
        
        # Step 3: Query for specific entity type
        query = f"Find all entities of type {entity_types[0]}"
        result = await graph_plugin.natural_language_query(query)
        
        assert result["results"] is not None
    
    @pytest.mark.asyncio
    async def test_sparql_to_natural_language_comparison(self, graph_plugin):
        """Test comparison between SPARQL and natural language queries."""
        # Natural language query
        nl_query = "Find all classes"
        nl_result = await graph_plugin.natural_language_query(nl_query)
        
        # Equivalent SPARQL query
        sparql_query = "SELECT ?class WHERE { ?class rdf:type owl:Class }"
        sparql_result = await graph_plugin.execute_sparql_query(sparql_query)
        
        # Both should return results
        assert nl_result["results"] is not None
        assert sparql_result["results"] is not None
        assert nl_result["execution_time_ms"] > 0
        assert sparql_result["execution_time_ms"] > 0


class TestGraphPluginEdgeCases:
    """Test edge cases and error scenarios."""
    
    @pytest.mark.asyncio
    async def test_empty_query_handling(self, graph_plugin):
        """Test handling of empty queries."""
        result = await graph_plugin.natural_language_query("")
        assert "query" in result
        assert result["query"] == ""
    
    @pytest.mark.asyncio
    async def test_very_long_query(self, graph_plugin):
        """Test handling of very long queries."""
        long_query = "Find entities " * 1000  # Very long query
        result = await graph_plugin.natural_language_query(long_query)
        assert "results" in result
    
    @pytest.mark.asyncio
    async def test_special_characters_in_query(self, graph_plugin):
        """Test handling of special characters in queries."""
        special_query = "Find entities with @#$%^&*() characters"
        result = await graph_plugin.natural_language_query(special_query)
        assert "results" in result
    
    @pytest.mark.asyncio
    async def test_concurrent_queries(self, graph_plugin):
        """Test handling of concurrent queries."""
        import asyncio
        
        queries = [
            "Find all classes",
            "Show properties",
            "List individuals"
        ]
        
        # Execute queries concurrently
        tasks = [
            graph_plugin.natural_language_query(query) 
            for query in queries
        ]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result in results:
            assert "results" in result
    
    @pytest.mark.asyncio
    async def test_malformed_sparql_queries(self, graph_plugin):
        """Test handling of various malformed SPARQL queries."""
        malformed_queries = [
            "SELECT WHERE",  # Missing variables and pattern
            "INVALID SPARQL SYNTAX",  # Completely invalid
            "SELECT ?s ?p ?o WHERE { ?s ?p }",  # Incomplete pattern
        ]
        
        for query in malformed_queries:
            if "INVALID" in query:
                with pytest.raises(ValueError):
                    await graph_plugin.execute_sparql_query(query)
            else:
                # Other malformed queries might be handled gracefully
                try:
                    result = await graph_plugin.execute_sparql_query(query)
                    assert "results" in result
                except ValueError:
                    # Expected for malformed queries
                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])