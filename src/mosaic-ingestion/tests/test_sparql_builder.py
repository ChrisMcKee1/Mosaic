"""
Unit tests for SPARQL Builder functionality.

Tests the SparqlBuilder class for loading RDF triples from Cosmos DB
and executing SPARQL queries for code relationship analysis.

Task: OMR-P1-007 - Implement Basic SPARQL Query Capability
"""

import pytest
from unittest.mock import Mock
from rdflib import Graph, URIRef, Literal, RDF

from rdf.sparql_builder import SparqlBuilder


class TestSparqlBuilder:
    """Test cases for SparqlBuilder class."""

    @pytest.fixture
    def mock_cosmos_client(self):
        """Create a mock Cosmos DB client."""
        return Mock()

    @pytest.fixture
    def sparql_builder(self, mock_cosmos_client):
        """Create a SparqlBuilder instance with mocked dependencies."""
        return SparqlBuilder(mock_cosmos_client)

    @pytest.fixture
    def sample_rdf_documents(self):
        """Sample Cosmos DB documents containing RDF triples."""
        return [
            {
                "id": "doc1",
                "rdf_triples": [
                    {
                        "subject": "http://mosaic.local/code/function1",
                        "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                        "object": "http://mosaic.local/code/Function",
                        "object_type": "uri",
                    },
                    {
                        "subject": "http://mosaic.local/code/function1",
                        "predicate": "http://mosaic.local/code/name",
                        "object": "calculate_metrics",
                        "object_type": "literal",
                    },
                    {
                        "subject": "http://mosaic.local/code/function1",
                        "predicate": "http://mosaic.local/code/inModule",
                        "object": "http://mosaic.local/code/module1",
                        "object_type": "uri",
                    },
                ],
            },
            {
                "id": "doc2",
                "rdf_triples": [
                    {
                        "subject": "http://mosaic.local/code/function2",
                        "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                        "object": "http://mosaic.local/code/Function",
                        "object_type": "uri",
                    },
                    {
                        "subject": "http://mosaic.local/code/function2",
                        "predicate": "http://mosaic.local/code/name",
                        "object": "process_data",
                        "object_type": "literal",
                    },
                    {
                        "subject": "http://mosaic.local/code/function1",
                        "predicate": "http://mosaic.local/code/calls",
                        "object": "http://mosaic.local/code/function2",
                        "object_type": "uri",
                    },
                ],
            },
        ]

    def test_sparql_builder_initialization(self, sparql_builder):
        """Test SparqlBuilder initialization."""
        assert sparql_builder.cosmos_client is not None
        assert isinstance(sparql_builder.graph, Graph)
        assert "code" in sparql_builder.namespaces
        assert "rdf" in sparql_builder.namespaces
        assert "rdfs" in sparql_builder.namespaces

    def test_load_triples_from_documents(self, sparql_builder, sample_rdf_documents):
        """Test loading RDF triples from Cosmos DB documents."""
        # Mock Cosmos DB client behavior
        mock_database = Mock()
        mock_container = Mock()
        mock_container.query_items.return_value = sample_rdf_documents
        mock_database.get_container_client.return_value = mock_container
        sparql_builder.cosmos_client.get_database_client.return_value = mock_database

        # Load triples
        triple_count = sparql_builder.load_triples_from_documents("test-container")

        # Verify results
        assert triple_count == 6  # Total triples from sample documents
        assert len(sparql_builder.graph) == 6

        # Verify Cosmos DB calls
        sparql_builder.cosmos_client.get_database_client.assert_called_once_with(
            "mosaic-knowledge"
        )
        mock_database.get_container_client.assert_called_once_with("test-container")
        mock_container.query_items.assert_called_once()

    def test_query_functions_in_module(self, sparql_builder):
        """Test querying functions in a specific module."""
        # Add test triples to graph
        code_ns = sparql_builder.namespaces["code"]
        sparql_builder.graph.add(
            (URIRef(f"{code_ns}function1"), RDF.type, URIRef(f"{code_ns}Function"))
        )
        sparql_builder.graph.add(
            (
                URIRef(f"{code_ns}function1"),
                URIRef(f"{code_ns}name"),
                Literal("test_function"),
            )
        )
        sparql_builder.graph.add(
            (
                URIRef(f"{code_ns}function1"),
                URIRef(f"{code_ns}inModule"),
                URIRef(f"{code_ns}test_module"),
            )
        )

        # Execute query
        results = sparql_builder.query_functions_in_module("test_module")

        # Verify results
        assert len(results) == 1
        assert results[0]["functionName"] == "test_function"

    def test_query_function_calls(self, sparql_builder):
        """Test querying function call relationships."""
        # Add test triples to graph
        code_ns = sparql_builder.namespaces["code"]
        sparql_builder.graph.add(
            (
                URIRef(f"{code_ns}function1"),
                URIRef(f"{code_ns}calls"),
                URIRef(f"{code_ns}function2"),
            )
        )
        sparql_builder.graph.add(
            (URIRef(f"{code_ns}function1"), URIRef(f"{code_ns}name"), Literal("caller"))
        )
        sparql_builder.graph.add(
            (URIRef(f"{code_ns}function2"), URIRef(f"{code_ns}name"), Literal("callee"))
        )

        # Execute query without specific function
        results = sparql_builder.query_function_calls()

        # Verify results
        assert len(results) == 1
        assert results[0]["callerName"] == "caller"
        assert results[0]["calleeName"] == "callee"

        # Execute query with specific function
        specific_results = sparql_builder.query_function_calls("callee")
        assert len(specific_results) == 1

    def test_query_inheritance_hierarchy(self, sparql_builder):
        """Test querying class inheritance relationships."""
        # Add test triples to graph
        code_ns = sparql_builder.namespaces["code"]
        sparql_builder.graph.add(
            (
                URIRef(f"{code_ns}childClass"),
                URIRef(f"{code_ns}inherits"),
                URIRef(f"{code_ns}parentClass"),
            )
        )
        sparql_builder.graph.add(
            (
                URIRef(f"{code_ns}childClass"),
                URIRef(f"{code_ns}name"),
                Literal("ChildClass"),
            )
        )
        sparql_builder.graph.add(
            (
                URIRef(f"{code_ns}parentClass"),
                URIRef(f"{code_ns}name"),
                Literal("ParentClass"),
            )
        )

        # Execute query
        results = sparql_builder.query_inheritance_hierarchy()

        # Verify results
        assert len(results) == 1
        assert results[0]["childName"] == "ChildClass"
        assert results[0]["parentName"] == "ParentClass"

    def test_query_transitive_dependencies(self, sparql_builder):
        """Test querying transitive module dependencies."""
        # Add test triples to graph
        code_ns = sparql_builder.namespaces["code"]
        sparql_builder.graph.add(
            (
                URIRef(f"{code_ns}moduleA"),
                URIRef(f"{code_ns}imports"),
                URIRef(f"{code_ns}moduleB"),
            )
        )
        sparql_builder.graph.add(
            (
                URIRef(f"{code_ns}moduleB"),
                URIRef(f"{code_ns}imports"),
                URIRef(f"{code_ns}moduleC"),
            )
        )
        sparql_builder.graph.add(
            (URIRef(f"{code_ns}moduleA"), URIRef(f"{code_ns}name"), Literal("ModuleA"))
        )
        sparql_builder.graph.add(
            (URIRef(f"{code_ns}moduleB"), URIRef(f"{code_ns}name"), Literal("ModuleB"))
        )
        sparql_builder.graph.add(
            (URIRef(f"{code_ns}moduleC"), URIRef(f"{code_ns}name"), Literal("ModuleC"))
        )

        # Execute query
        results = sparql_builder.query_transitive_dependencies("ModuleA")

        # Verify results (should include both direct and transitive dependencies)
        assert len(results) >= 1

    def test_query_code_complexity_metrics(self, sparql_builder):
        """Test querying code complexity metrics."""
        # Add test triples to graph
        code_ns = sparql_builder.namespaces["code"]
        sparql_builder.graph.add(
            (URIRef(f"{code_ns}function1"), RDF.type, URIRef(f"{code_ns}Function"))
        )
        sparql_builder.graph.add(
            (
                URIRef(f"{code_ns}function1"),
                URIRef(f"{code_ns}name"),
                Literal("complex_function"),
            )
        )
        sparql_builder.graph.add(
            (
                URIRef(f"{code_ns}function1"),
                URIRef(f"{code_ns}calls"),
                URIRef(f"{code_ns}function2"),
            )
        )
        sparql_builder.graph.add(
            (
                URIRef(f"{code_ns}function1"),
                URIRef(f"{code_ns}calls"),
                URIRef(f"{code_ns}function3"),
            )
        )

        # Execute query
        results = sparql_builder.query_code_complexity_metrics()

        # Verify results
        assert len(results) == 1
        assert results[0]["functionName"] == "complex_function"
        assert int(results[0]["callCount"]) == 2

    def test_query_unused_functions(self, sparql_builder):
        """Test querying unused functions."""
        # Add test triples to graph
        code_ns = sparql_builder.namespaces["code"]
        sparql_builder.graph.add(
            (URIRef(f"{code_ns}function1"), RDF.type, URIRef(f"{code_ns}Function"))
        )
        sparql_builder.graph.add(
            (
                URIRef(f"{code_ns}function1"),
                URIRef(f"{code_ns}name"),
                Literal("unused_function"),
            )
        )
        sparql_builder.graph.add(
            (URIRef(f"{code_ns}function2"), RDF.type, URIRef(f"{code_ns}Function"))
        )
        sparql_builder.graph.add(
            (
                URIRef(f"{code_ns}function2"),
                URIRef(f"{code_ns}name"),
                Literal("used_function"),
            )
        )
        sparql_builder.graph.add(
            (
                URIRef(f"{code_ns}function1"),
                URIRef(f"{code_ns}calls"),
                URIRef(f"{code_ns}function2"),
            )
        )

        # Execute query
        results = sparql_builder.query_unused_functions()

        # Verify results - function1 should be unused (not called by anyone)
        unused_names = [result["functionName"] for result in results]
        assert "unused_function" in unused_names
        assert "used_function" not in unused_names

    def test_get_graph_statistics(self, sparql_builder):
        """Test getting graph statistics."""
        # Add test triples to graph
        code_ns = sparql_builder.namespaces["code"]
        sparql_builder.graph.add(
            (
                URIRef(f"{code_ns}subject1"),
                URIRef(f"{code_ns}predicate1"),
                URIRef(f"{code_ns}object1"),
            )
        )
        sparql_builder.graph.add(
            (
                URIRef(f"{code_ns}subject2"),
                URIRef(f"{code_ns}predicate2"),
                Literal("literal1"),
            )
        )

        # Get statistics
        stats = sparql_builder.get_graph_statistics()

        # Verify statistics
        assert stats["total_triples"] == 2
        assert stats["unique_subjects"] == 2
        assert stats["unique_predicates"] == 2
        assert stats["unique_objects"] == 2

    def test_clear_graph(self, sparql_builder):
        """Test clearing the RDF graph."""
        # Add test triples
        code_ns = sparql_builder.namespaces["code"]
        sparql_builder.graph.add(
            (
                URIRef(f"{code_ns}subject"),
                URIRef(f"{code_ns}predicate"),
                URIRef(f"{code_ns}object"),
            )
        )

        # Verify graph has triples
        assert len(sparql_builder.graph) == 1

        # Clear graph
        sparql_builder.clear_graph()

        # Verify graph is empty
        assert len(sparql_builder.graph) == 0

    def test_execute_query_error_handling(self, sparql_builder):
        """Test error handling in SPARQL query execution."""
        # Test with invalid SPARQL syntax
        with pytest.raises(Exception):
            sparql_builder._execute_query("INVALID SPARQL QUERY")

    def test_load_triples_error_handling(self, sparql_builder):
        """Test error handling in loading triples from Cosmos DB."""
        # Mock Cosmos DB to raise exception
        sparql_builder.cosmos_client.get_database_client.side_effect = Exception(
            "Connection failed"
        )

        # Test error handling
        with pytest.raises(Exception):
            sparql_builder.load_triples_from_documents("test-container")


class TestSparqlBuilderIntegration:
    """Integration tests for SparqlBuilder with realistic data."""

    @pytest.fixture
    def sparql_builder_with_data(self):
        """Create SparqlBuilder with realistic test data."""
        mock_cosmos_client = Mock()
        builder = SparqlBuilder(mock_cosmos_client)

        # Add realistic code relationship data
        code_ns = builder.namespaces["code"]

        # Add modules
        builder.graph.add(
            (URIRef(f"{code_ns}analytics_module"), RDF.type, URIRef(f"{code_ns}Module"))
        )
        builder.graph.add(
            (
                URIRef(f"{code_ns}analytics_module"),
                URIRef(f"{code_ns}name"),
                Literal("analytics"),
            )
        )

        builder.graph.add(
            (URIRef(f"{code_ns}utils_module"), RDF.type, URIRef(f"{code_ns}Module"))
        )
        builder.graph.add(
            (
                URIRef(f"{code_ns}utils_module"),
                URIRef(f"{code_ns}name"),
                Literal("utils"),
            )
        )

        # Add functions
        builder.graph.add(
            (
                URIRef(f"{code_ns}calculate_metrics"),
                RDF.type,
                URIRef(f"{code_ns}Function"),
            )
        )
        builder.graph.add(
            (
                URIRef(f"{code_ns}calculate_metrics"),
                URIRef(f"{code_ns}name"),
                Literal("calculate_metrics"),
            )
        )
        builder.graph.add(
            (
                URIRef(f"{code_ns}calculate_metrics"),
                URIRef(f"{code_ns}inModule"),
                URIRef(f"{code_ns}analytics_module"),
            )
        )

        builder.graph.add(
            (
                URIRef(f"{code_ns}helper_function"),
                RDF.type,
                URIRef(f"{code_ns}Function"),
            )
        )
        builder.graph.add(
            (
                URIRef(f"{code_ns}helper_function"),
                URIRef(f"{code_ns}name"),
                Literal("helper_function"),
            )
        )
        builder.graph.add(
            (
                URIRef(f"{code_ns}helper_function"),
                URIRef(f"{code_ns}inModule"),
                URIRef(f"{code_ns}utils_module"),
            )
        )

        # Add function calls
        builder.graph.add(
            (
                URIRef(f"{code_ns}calculate_metrics"),
                URIRef(f"{code_ns}calls"),
                URIRef(f"{code_ns}helper_function"),
            )
        )

        # Add module imports
        builder.graph.add(
            (
                URIRef(f"{code_ns}analytics_module"),
                URIRef(f"{code_ns}imports"),
                URIRef(f"{code_ns}utils_module"),
            )
        )

        return builder

    def test_complete_code_analysis_workflow(self, sparql_builder_with_data):
        """Test complete workflow of code analysis using SPARQL queries."""
        builder = sparql_builder_with_data

        # Test finding functions in analytics module
        analytics_functions = builder.query_functions_in_module("analytics")
        assert len(analytics_functions) == 1
        assert analytics_functions[0]["functionName"] == "calculate_metrics"

        # Test finding function calls
        function_calls = builder.query_function_calls()
        assert len(function_calls) == 1
        assert function_calls[0]["callerName"] == "calculate_metrics"
        assert function_calls[0]["calleeName"] == "helper_function"

        # Test finding transitive dependencies
        dependencies = builder.query_transitive_dependencies("analytics")
        assert len(dependencies) >= 1

        # Verify graph statistics
        stats = builder.get_graph_statistics()
        assert stats["total_triples"] > 0
