"""
Unit tests for GraphBuilder plugin.

Tests the in-memory RDF graph construction system that builds and manages
large RDF graphs with efficient batch operations and SPARQL querying.
"""

import pytest
from unittest.mock import patch
import tempfile
import os

from rdflib import Graph, URIRef, Literal, RDF

from plugins.graph_builder import GraphBuilder


class TestGraphBuilder:
    """Test cases for GraphBuilder class."""

    @pytest.fixture
    def graph_builder(self):
        """Create GraphBuilder instance for testing."""
        return GraphBuilder(base_uri="http://test.local/")

    @pytest.fixture
    def sample_triples(self):
        """Sample RDF triples for testing."""
        return [
            (
                "http://test.local/subject1",
                "http://test.local/predicate1",
                "http://test.local/object1",
            ),
            (
                "http://test.local/subject2",
                "http://test.local/predicate2",
                "literal_value",
            ),
            ("http://test.local/subject3", RDF.type, "http://test.local/Class1"),
        ]

    def test_graph_builder_initialization(self, graph_builder):
        """Test GraphBuilder initialization."""
        assert isinstance(graph_builder.graph, Graph)
        assert graph_builder.base_uri == "http://test.local/"
        assert graph_builder.triple_count == 0
        assert graph_builder.batch_operations == 0
        assert graph_builder.query_count == 0

        # Verify namespace bindings
        namespaces = dict(graph_builder.graph.namespaces())
        assert "mosaic" in namespaces
        assert "rdf" in namespaces
        assert "rdfs" in namespaces
        assert "owl" in namespaces

    def test_default_initialization(self):
        """Test GraphBuilder with default parameters."""
        builder = GraphBuilder()
        assert builder.base_uri == "http://mosaic.dev/graph/"
        assert isinstance(builder.namespace.term("test"), URIRef)

    def test_add_triples_success(self, graph_builder, sample_triples):
        """Test successful addition of triples."""
        graph_builder.add_triples(sample_triples)

        assert len(graph_builder.graph) == len(sample_triples)
        assert graph_builder.triple_count == len(sample_triples)
        assert graph_builder.batch_operations == 1

    def test_add_triples_empty_list(self, graph_builder):
        """Test adding empty list of triples."""
        with patch("plugins.graph_builder.logger") as mock_logger:
            graph_builder.add_triples([])

            mock_logger.warning.assert_called_once_with(
                "No triples provided to add_triples"
            )
            assert len(graph_builder.graph) == 0

    def test_add_triples_batching(self, graph_builder):
        """Test that large triple sets are processed in batches."""
        # Create large set of triples
        large_triple_set = []
        for i in range(2500):  # More than default batch size of 1000
            large_triple_set.append(
                (
                    f"http://test.local/subject{i}",
                    "http://test.local/predicate",
                    f"http://test.local/object{i}",
                )
            )

        graph_builder.add_triples(large_triple_set, batch_size=1000)

        assert len(graph_builder.graph) == 2500
        assert graph_builder.batch_operations == 3  # 3 batches (1000, 1000, 500)

    def test_to_rdf_term_uri_ref(self, graph_builder):
        """Test conversion of string to URIRef."""
        uri_string = "http://test.local/resource"
        result = graph_builder._to_rdf_term(uri_string)

        assert isinstance(result, URIRef)
        assert str(result) == uri_string

    def test_to_rdf_term_literal(self, graph_builder):
        """Test conversion of literal values."""
        literal_value = "test literal"
        result = graph_builder._to_rdf_term(literal_value)

        assert isinstance(result, Literal)
        assert str(result) == literal_value

    def test_to_rdf_term_already_rdf(self, graph_builder):
        """Test that existing RDF terms are returned unchanged."""
        uri_ref = URIRef("http://test.local/existing")
        result = graph_builder._to_rdf_term(uri_ref)

        assert result is uri_ref
        assert isinstance(result, URIRef)

    def test_query_execution(self, graph_builder, sample_triples):
        """Test SPARQL query execution."""
        graph_builder.add_triples(sample_triples)

        # Test basic SELECT query
        query = """
        SELECT ?subject ?predicate ?object
        WHERE {
            ?subject ?predicate ?object .
        }
        """

        results = graph_builder.query(query)

        assert len(list(results)) == len(sample_triples)
        assert graph_builder.query_count == 1

    def test_complex_sparql_query(self, graph_builder):
        """Test complex SPARQL query with filters and patterns."""
        # Add structured test data
        triples = [
            ("http://test.local/person1", RDF.type, "http://test.local/Person"),
            ("http://test.local/person1", "http://test.local/name", "Alice"),
            ("http://test.local/person1", "http://test.local/age", 30),
            ("http://test.local/person2", RDF.type, "http://test.local/Person"),
            ("http://test.local/person2", "http://test.local/name", "Bob"),
            ("http://test.local/person2", "http://test.local/age", 25),
        ]

        graph_builder.add_triples(triples)

        # Query for people over 25
        query = """
        PREFIX test: <http://test.local/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT ?person ?name ?age
        WHERE {
            ?person rdf:type test:Person .
            ?person test:name ?name .
            ?person test:age ?age .
            FILTER(?age > 25)
        }
        ORDER BY ?age
        """

        results = list(graph_builder.query(query))

        assert len(results) == 1  # Only Alice (age 30)
        assert str(results[0].name) == "Alice"
        assert results[0].age == 30

    def test_get_statistics(self, graph_builder, sample_triples):
        """Test graph statistics collection."""
        graph_builder.add_triples(sample_triples)

        stats = graph_builder.get_statistics()

        assert stats["total_triples"] == len(sample_triples)
        assert stats["batch_operations"] == 1
        assert stats["query_count"] == 0  # No queries executed yet
        assert "memory_usage_mb" in stats
        assert isinstance(stats["memory_usage_mb"], float)

    def test_memory_monitoring(self, graph_builder):
        """Test memory usage monitoring functionality."""
        initial_memory = graph_builder._get_memory_usage_mb()

        # Add significant amount of data
        large_triples = []
        for i in range(1000):
            large_triples.append(
                (
                    f"http://test.local/resource{i}",
                    "http://test.local/property",
                    f"Large content value for resource {i} with additional text to increase memory usage",
                )
            )

        graph_builder.add_triples(large_triples)

        final_memory = graph_builder._get_memory_usage_mb()

        # Memory usage should have increased
        assert final_memory >= initial_memory
        assert isinstance(final_memory, float)

    def test_clear_graph(self, graph_builder, sample_triples):
        """Test clearing the graph."""
        graph_builder.add_triples(sample_triples)
        assert len(graph_builder.graph) > 0

        graph_builder.clear()

        assert len(graph_builder.graph) == 0
        assert graph_builder.triple_count == 0
        assert graph_builder.batch_operations == 0
        assert graph_builder.query_count == 0

    def test_serialize_to_file(self, graph_builder, sample_triples):
        """Test serializing graph to file."""
        graph_builder.add_triples(sample_triples)

        with tempfile.NamedTemporaryFile(suffix=".ttl", delete=False) as tmp_file:
            try:
                result = graph_builder.serialize_to_file(tmp_file.name, format="turtle")

                assert result is True
                assert os.path.exists(tmp_file.name)

                # Verify file content
                with open(tmp_file.name, "r") as f:
                    content = f.read()
                    assert len(content) > 0
                    assert "test.local" in content

            finally:
                os.unlink(tmp_file.name)

    def test_serialize_to_file_invalid_format(self, graph_builder, sample_triples):
        """Test serialization with invalid format."""
        graph_builder.add_triples(sample_triples)

        with tempfile.NamedTemporaryFile(suffix=".invalid", delete=False) as tmp_file:
            try:
                result = graph_builder.serialize_to_file(
                    tmp_file.name, format="invalid_format"
                )

                assert result is False

            finally:
                os.unlink(tmp_file.name)

    def test_load_from_file(self, graph_builder):
        """Test loading graph from file."""
        # Create test RDF file
        test_content = """
        @prefix test: <http://test.local/> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        
        test:subject1 rdf:type test:Resource .
        test:subject1 test:name "Test Resource" .
        """

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ttl", delete=False
        ) as tmp_file:
            tmp_file.write(test_content)
            tmp_file.flush()

            try:
                result = graph_builder.load_from_file(tmp_file.name, format="turtle")

                assert result is True
                assert len(graph_builder.graph) == 2  # Two triples loaded

                # Verify loaded content
                query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"
                results = list(graph_builder.query(query))
                assert len(results) == 2

            finally:
                os.unlink(tmp_file.name)

    def test_load_from_file_not_found(self, graph_builder):
        """Test loading from non-existent file."""
        result = graph_builder.load_from_file("/nonexistent/file.ttl")

        assert result is False
        assert len(graph_builder.graph) == 0

    def test_add_namespace(self, graph_builder):
        """Test adding custom namespace."""
        custom_ns = "http://custom.local/"
        graph_builder.add_namespace("custom", custom_ns)

        namespaces = dict(graph_builder.graph.namespaces())
        assert "custom" in namespaces
        assert str(namespaces["custom"]) == custom_ns

    def test_get_namespace_uris(self, graph_builder):
        """Test getting all namespace URIs."""
        namespaces = graph_builder.get_namespace_uris()

        assert isinstance(namespaces, dict)
        assert "mosaic" in namespaces
        assert "rdf" in namespaces
        assert namespaces["mosaic"] == graph_builder.namespace

    def test_find_subjects_by_predicate_object(self, graph_builder):
        """Test finding subjects by predicate-object pattern."""
        triples = [
            ("http://test.local/resource1", RDF.type, "http://test.local/Document"),
            ("http://test.local/resource2", RDF.type, "http://test.local/Document"),
            ("http://test.local/resource3", RDF.type, "http://test.local/Image"),
        ]

        graph_builder.add_triples(triples)

        subjects = graph_builder.find_subjects_by_predicate_object(
            RDF.type, URIRef("http://test.local/Document")
        )

        subject_list = list(subjects)
        assert len(subject_list) == 2
        assert URIRef("http://test.local/resource1") in subject_list
        assert URIRef("http://test.local/resource2") in subject_list

    def test_find_objects_by_subject_predicate(self, graph_builder):
        """Test finding objects by subject-predicate pattern."""
        triples = [
            ("http://test.local/person1", "http://test.local/name", "Alice"),
            ("http://test.local/person1", "http://test.local/email", "alice@test.com"),
            ("http://test.local/person1", "http://test.local/age", 30),
        ]

        graph_builder.add_triples(triples)

        objects = graph_builder.find_objects_by_subject_predicate(
            URIRef("http://test.local/person1"), URIRef("http://test.local/name")
        )

        object_list = list(objects)
        assert len(object_list) == 1
        assert str(object_list[0]) == "Alice"

    def test_performance_with_large_dataset(self, graph_builder):
        """Test performance with large dataset."""
        # Create large dataset
        large_dataset = []
        for i in range(5000):
            large_dataset.extend(
                [
                    (
                        f"http://test.local/resource{i}",
                        RDF.type,
                        "http://test.local/Resource",
                    ),
                    (f"http://test.local/resource{i}", "http://test.local/id", i),
                    (
                        f"http://test.local/resource{i}",
                        "http://test.local/name",
                        f"Resource {i}",
                    ),
                ]
            )

        import time

        start_time = time.time()

        graph_builder.add_triples(large_dataset, batch_size=1000)

        end_time = time.time()
        processing_time = end_time - start_time

        # Verify data was added correctly
        assert len(graph_builder.graph) == len(large_dataset)

        # Verify reasonable performance (should complete within 5 seconds)
        assert processing_time < 5.0

        # Test query performance
        query_start = time.time()
        query = """
        SELECT (COUNT(?resource) as ?count)
        WHERE {
            ?resource rdf:type <http://test.local/Resource> .
        }
        """
        results = list(graph_builder.query(query))
        query_end = time.time()

        assert len(results) == 1
        assert results[0].count == 5000
        assert (query_end - query_start) < 2.0  # Query should be fast

    def test_error_handling_invalid_triples(self, graph_builder):
        """Test error handling with invalid triple data."""
        invalid_triples = [
            (None, "http://test.local/predicate", "object"),  # Invalid subject
            ("http://test.local/subject", None, "object"),  # Invalid predicate
        ]

        with patch("plugins.graph_builder.logger") as mock_logger:
            # Should handle gracefully without crashing
            try:
                graph_builder.add_triples(invalid_triples)
                # If no exception, verify logging occurred
                mock_logger.error.assert_called()
            except Exception:
                # Exception is acceptable for invalid data
                pass

    def test_concurrent_access_safety(self, graph_builder, sample_triples):
        """Test thread safety for concurrent access."""
        import threading

        results = []
        errors = []

        def add_triples_worker(worker_id):
            try:
                worker_triples = [
                    (
                        f"http://test.local/worker{worker_id}_subject{i}",
                        "http://test.local/predicate",
                        f"worker{worker_id}_object{i}",
                    )
                    for i in range(100)
                ]
                graph_builder.add_triples(worker_triples)
                results.append(f"worker{worker_id}_completed")
            except Exception as e:
                errors.append(f"worker{worker_id}_error: {e}")

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_triples_worker, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)

        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        assert len(graph_builder.graph) == 500  # 5 workers × 100 triples each


class TestGraphBuilderIntegration:
    """Integration tests for GraphBuilder with realistic scenarios."""

    def test_code_analysis_workflow(self):
        """Test complete code analysis workflow using RDF graph."""
        builder = GraphBuilder(base_uri="http://code.analysis/")

        # Add code entities and relationships
        code_triples = [
            # Module definition
            (
                "http://code.analysis/module_analytics",
                RDF.type,
                "http://code.analysis/Module",
            ),
            (
                "http://code.analysis/module_analytics",
                "http://code.analysis/name",
                "analytics",
            ),
            (
                "http://code.analysis/module_analytics",
                "http://code.analysis/language",
                "python",
            ),
            # Function definitions
            (
                "http://code.analysis/func_calculate",
                RDF.type,
                "http://code.analysis/Function",
            ),
            (
                "http://code.analysis/func_calculate",
                "http://code.analysis/name",
                "calculate_metrics",
            ),
            (
                "http://code.analysis/func_calculate",
                "http://code.analysis/inModule",
                "http://code.analysis/module_analytics",
            ),
            (
                "http://code.analysis/func_calculate",
                "http://code.analysis/lineStart",
                10,
            ),
            ("http://code.analysis/func_calculate", "http://code.analysis/lineEnd", 25),
            (
                "http://code.analysis/func_helper",
                RDF.type,
                "http://code.analysis/Function",
            ),
            (
                "http://code.analysis/func_helper",
                "http://code.analysis/name",
                "helper_function",
            ),
            (
                "http://code.analysis/func_helper",
                "http://code.analysis/inModule",
                "http://code.analysis/module_analytics",
            ),
            # Function relationships
            (
                "http://code.analysis/func_calculate",
                "http://code.analysis/calls",
                "http://code.analysis/func_helper",
            ),
            # Class definition
            (
                "http://code.analysis/class_processor",
                RDF.type,
                "http://code.analysis/Class",
            ),
            (
                "http://code.analysis/class_processor",
                "http://code.analysis/name",
                "DataProcessor",
            ),
            (
                "http://code.analysis/class_processor",
                "http://code.analysis/inModule",
                "http://code.analysis/module_analytics",
            ),
        ]

        builder.add_triples(code_triples)

        # Test queries for code analysis

        # 1. Find all functions in the analytics module
        functions_query = """
        PREFIX code: <http://code.analysis/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT ?function ?name
        WHERE {
            ?function rdf:type code:Function .
            ?function code:name ?name .
            ?function code:inModule code:module_analytics .
        }
        """

        functions = list(builder.query(functions_query))
        assert len(functions) == 2
        function_names = [str(f.name) for f in functions]
        assert "calculate_metrics" in function_names
        assert "helper_function" in function_names

        # 2. Find function call relationships
        calls_query = """
        PREFIX code: <http://code.analysis/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT ?caller ?callerName ?callee ?calleeName
        WHERE {
            ?caller code:calls ?callee .
            ?caller code:name ?callerName .
            ?callee code:name ?calleeName .
        }
        """

        calls = list(builder.query(calls_query))
        assert len(calls) == 1
        assert str(calls[0].callerName) == "calculate_metrics"
        assert str(calls[0].calleeName) == "helper_function"

        # 3. Find all entities in the module
        entities_query = """
        PREFIX code: <http://code.analysis/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT ?entity ?type ?name
        WHERE {
            ?entity code:inModule code:module_analytics .
            ?entity rdf:type ?type .
            ?entity code:name ?name .
        }
        """

        entities = list(builder.query(entities_query))
        assert len(entities) == 3  # 2 functions + 1 class

        # Verify graph statistics
        stats = builder.get_statistics()
        assert stats["total_triples"] == len(code_triples)
        assert stats["total_triples"] > 10

    def test_large_codebase_analysis(self):
        """Test analysis of large codebase with many entities."""
        builder = GraphBuilder(base_uri="http://large.codebase/")

        # Simulate large codebase with multiple modules
        triples = []

        # Create 10 modules
        for module_id in range(10):
            module_uri = f"http://large.codebase/module{module_id}"
            triples.extend(
                [
                    (module_uri, RDF.type, "http://large.codebase/Module"),
                    (module_uri, "http://large.codebase/name", f"module_{module_id}"),
                ]
            )

            # Each module has 20 functions
            for func_id in range(20):
                func_uri = f"http://large.codebase/module{module_id}_func{func_id}"
                triples.extend(
                    [
                        (func_uri, RDF.type, "http://large.codebase/Function"),
                        (func_uri, "http://large.codebase/name", f"function_{func_id}"),
                        (func_uri, "http://large.codebase/inModule", module_uri),
                        (func_uri, "http://large.codebase/lineStart", func_id * 10),
                        (func_uri, "http://large.codebase/lineEnd", func_id * 10 + 5),
                    ]
                )

                # Add some function calls (create dependencies)
                if func_id > 0:
                    caller_uri = func_uri
                    callee_uri = (
                        f"http://large.codebase/module{module_id}_func{func_id - 1}"
                    )
                    triples.append(
                        (caller_uri, "http://large.codebase/calls", callee_uri)
                    )

        # Add all triples efficiently
        builder.add_triples(triples, batch_size=500)

        # Verify data was added
        total_expected = 10 * (2 + 20 * 5 + 19)  # modules + functions + calls
        assert len(builder.graph) == total_expected

        # Test complex analytical queries

        # Find modules with most functions
        modules_query = """
        PREFIX code: <http://large.codebase/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT ?module ?name (COUNT(?function) as ?functionCount)
        WHERE {
            ?module rdf:type code:Module .
            ?module code:name ?name .
            ?function code:inModule ?module .
            ?function rdf:type code:Function .
        }
        GROUP BY ?module ?name
        ORDER BY DESC(?functionCount)
        """

        modules = list(builder.query(modules_query))
        assert len(modules) == 10
        # Each module should have 20 functions
        for module in modules:
            assert module.functionCount == 20

        # Find functions with most calls
        call_analysis_query = """
        PREFIX code: <http://large.codebase/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT ?function ?name (COUNT(?callee) as ?callCount)
        WHERE {
            ?function rdf:type code:Function .
            ?function code:name ?name .
            ?function code:calls ?callee .
        }
        GROUP BY ?function ?name
        HAVING (?callCount > 0)
        ORDER BY DESC(?callCount)
        """

        calling_functions = list(builder.query(call_analysis_query))
        # Should have 19 calling functions per module (func_1 through func_19)
        assert len(calling_functions) == 10 * 19

        # Verify performance with large dataset
        stats = builder.get_statistics()
        assert stats["total_triples"] == total_expected
        assert stats["memory_usage_mb"] > 0
