"""
Test suite for GraphBuilder - In-Memory RDF Graph Construction System
Tests OMR-P1-005 implementation
"""

import pytest
import tempfile
from pathlib import Path
from typing import List, Dict, Any


# Import the GraphBuilder class
import sys

sys.path.append(str(Path(__file__).parent.parent))
from plugins.graph_builder import GraphBuilder


class TestGraphBuilder:
    """Test suite for GraphBuilder class."""

    @pytest.fixture
    def graph_builder(self) -> GraphBuilder:
        """Create a fresh GraphBuilder instance for each test."""
        return GraphBuilder(base_uri="http://test.mosaic.dev/")

    @pytest.fixture
    def sample_triples(self) -> List[tuple]:
        """Sample triples for testing."""
        return [
            (
                "http://test.mosaic.dev/person1",
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "http://test.mosaic.dev/Person",
            ),
            ("http://test.mosaic.dev/person1", "http://test.mosaic.dev/name", "Alice"),
            ("http://test.mosaic.dev/person1", "http://test.mosaic.dev/age", "30"),
            (
                "http://test.mosaic.dev/person2",
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "http://test.mosaic.dev/Person",
            ),
            ("http://test.mosaic.dev/person2", "http://test.mosaic.dev/name", "Bob"),
            ("http://test.mosaic.dev/person2", "http://test.mosaic.dev/age", "25"),
        ]

    @pytest.fixture
    def triple_generator_output(self) -> List[Dict[str, Any]]:
        """Sample TripleGenerator output format."""
        return [
            {
                "subject": "http://test.mosaic.dev/class1",
                "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "object": "http://test.mosaic.dev/CodeClass",
            },
            {
                "subject": "http://test.mosaic.dev/class1",
                "predicate": "http://test.mosaic.dev/className",
                "object": "DatabaseManager",
            },
            {
                "subject": "http://test.mosaic.dev/class1",
                "predicate": "http://test.mosaic.dev/hasMethod",
                "object": "http://test.mosaic.dev/method1",
            },
        ]

    def test_initialization(self, graph_builder):
        """Test GraphBuilder initialization."""
        assert graph_builder.base_uri == "http://test.mosaic.dev/"
        assert graph_builder.triple_count == 0
        assert graph_builder.batch_operations == 0
        assert graph_builder.query_count == 0
        assert len(graph_builder.graph) == 0

        # Check namespace bindings
        namespaces = dict(graph_builder.graph.namespaces())
        assert "mosaic" in namespaces
        assert "rdf" in namespaces
        assert "rdfs" in namespaces

    def test_add_triples_basic(self, graph_builder, sample_triples):
        """Test basic triple addition."""
        graph_builder.add_triples(sample_triples)

        assert len(graph_builder.graph) == len(sample_triples)
        assert graph_builder.triple_count == len(sample_triples)
        assert graph_builder.batch_operations == 1

    def test_add_triples_empty(self, graph_builder):
        """Test adding empty triple list."""
        graph_builder.add_triples([])

        assert len(graph_builder.graph) == 0
        assert graph_builder.triple_count == 0
        assert graph_builder.batch_operations == 0

    def test_add_triples_batch_processing(self, graph_builder):
        """Test batch processing with large triple sets."""
        # Create a large set of triples
        large_triple_set = []
        for i in range(2500):  # More than default batch size
            large_triple_set.append(
                (f"http://test.mosaic.dev/item{i}", "http://test.mosaic.dev/id", str(i))
            )

        graph_builder.add_triples(large_triple_set, batch_size=1000)

        assert len(graph_builder.graph) == 2500
        assert graph_builder.triple_count == 2500
        assert graph_builder.batch_operations == 1

    def test_rdf_term_conversion(self, graph_builder):
        """Test RDF term conversion for different data types."""
        triples = [
            (
                "http://test.mosaic.dev/item1",
                "http://test.mosaic.dev/prop",
                "string_value",
            ),
            ("http://test.mosaic.dev/item2", "http://test.mosaic.dev/prop", "42"),
            ("http://test.mosaic.dev/item3", "http://test.mosaic.dev/prop", "true"),
        ]

        graph_builder.add_triples(triples)

        assert len(graph_builder.graph) == 3

        # Verify terms are properly converted
        query = """
        SELECT ?s ?p ?o WHERE {
            ?s ?p ?o .
        }
        """
        results = graph_builder.query(query)
        assert len(results) == 3

    def test_sparql_query_basic(self, graph_builder, sample_triples):
        """Test basic SPARQL query execution."""
        graph_builder.add_triples(sample_triples)

        query = """
        PREFIX mosaic: <http://test.mosaic.dev/>
        SELECT ?person ?name WHERE {
            ?person mosaic:name ?name .
        }
        """

        results = graph_builder.query(query)
        assert len(results) == 2  # Alice and Bob
        assert graph_builder.query_count == 1

        # Extract names
        names = [str(row[1]) for row in results]
        assert "Alice" in names
        assert "Bob" in names

    def test_sparql_query_complex(self, graph_builder, sample_triples):
        """Test complex SPARQL query with filters."""
        graph_builder.add_triples(sample_triples)

        query = """
        PREFIX mosaic: <http://test.mosaic.dev/>
        SELECT ?person ?name ?age WHERE {
            ?person mosaic:name ?name .
            ?person mosaic:age ?age .
            FILTER(?age > "25")
        }
        """

        results = graph_builder.query(query)
        assert len(results) == 1  # Only Alice (age 30)

        person, name, age = results[0]
        assert str(name) == "Alice"
        assert str(age) == "30"

    def test_sparql_query_error(self, graph_builder):
        """Test SPARQL query error handling."""
        invalid_query = "INVALID SPARQL SYNTAX"

        with pytest.raises(Exception):
            graph_builder.query(invalid_query)

    def test_serialization_turtle(self, graph_builder, sample_triples):
        """Test Turtle format serialization."""
        graph_builder.add_triples(sample_triples)

        turtle_output = graph_builder.serialize(format="turtle")

        assert isinstance(turtle_output, str)
        assert "@prefix" in turtle_output
        assert "mosaic:" in turtle_output
        assert len(turtle_output) > 0

    def test_serialization_to_file(self, graph_builder, sample_triples):
        """Test serialization to file."""
        graph_builder.add_triples(sample_triples)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "test_graph.ttl"

            turtle_output = graph_builder.serialize(
                format="turtle", destination=output_file
            )

            assert output_file.exists()
            assert output_file.stat().st_size > 0

            # Verify content
            file_content = output_file.read_text(encoding="utf-8")
            assert file_content == turtle_output

    def test_serialization_formats(self, graph_builder, sample_triples):
        """Test multiple serialization formats."""
        graph_builder.add_triples(sample_triples)

        formats = ["turtle", "nt", "xml"]

        for fmt in formats:
            output = graph_builder.serialize(format=fmt)
            assert isinstance(output, str)
            assert len(output) > 0

    def test_memory_usage(self, graph_builder, sample_triples):
        """Test memory usage monitoring."""
        # Get initial memory usage
        initial_stats = graph_builder.get_memory_usage()
        assert "memory_mb" in initial_stats
        assert "memory_percent" in initial_stats
        assert "triple_count" in initial_stats
        assert "triples_per_mb" in initial_stats

        # Add triples and check memory usage changes
        graph_builder.add_triples(sample_triples)

        after_stats = graph_builder.get_memory_usage()
        assert after_stats["triple_count"] == len(sample_triples)
        assert after_stats["memory_mb"] > 0

    def test_clear_graph(self, graph_builder, sample_triples):
        """Test graph clearing functionality."""
        graph_builder.add_triples(sample_triples)
        assert len(graph_builder.graph) == len(sample_triples)

        graph_builder.clear()

        assert len(graph_builder.graph) == 0
        assert graph_builder.triple_count == 0

        # Check namespaces are still bound after clear
        namespaces = dict(graph_builder.graph.namespaces())
        assert "mosaic" in namespaces

    def test_statistics(self, graph_builder, sample_triples):
        """Test comprehensive statistics."""
        graph_builder.add_triples(sample_triples)

        # Execute a query to increment query count
        graph_builder.query("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")

        stats = graph_builder.get_statistics()

        expected_keys = [
            "triple_count",
            "batch_operations",
            "query_count",
            "memory_usage_mb",
            "memory_percent",
            "triples_per_mb",
            "base_uri",
            "bound_namespaces",
        ]

        for key in expected_keys:
            assert key in stats

        assert stats["triple_count"] == len(sample_triples)
        assert stats["batch_operations"] == 1
        assert stats["query_count"] == 1
        assert stats["base_uri"] == "http://test.mosaic.dev/"

    def test_add_from_triple_generator(self, graph_builder, triple_generator_output):
        """Test integration with TripleGenerator output."""
        graph_builder.add_from_triple_generator(triple_generator_output)

        assert len(graph_builder.graph) == len(triple_generator_output)
        assert graph_builder.triple_count == len(triple_generator_output)

        # Verify specific triples were added
        query = """
        PREFIX mosaic: <http://test.mosaic.dev/>
        SELECT ?class ?name WHERE {
            ?class mosaic:className ?name .
        }
        """

        results = graph_builder.query(query)
        assert len(results) == 1
        assert str(results[0][1]) == "DatabaseManager"

    def test_add_from_triple_generator_invalid(self, graph_builder):
        """Test TripleGenerator integration with invalid data."""
        invalid_output = [
            {
                "subject": "http://test.mosaic.dev/item1",
                "predicate": None,
                "object": "value",
            },
            {
                "subject": None,
                "predicate": "http://test.mosaic.dev/prop",
                "object": "value",
            },
            {"incomplete": "data"},
        ]

        graph_builder.add_from_triple_generator(invalid_output)

        # Should handle invalid triples gracefully
        assert len(graph_builder.graph) == 0
        assert graph_builder.triple_count == 0

    def test_add_from_triple_generator_empty(self, graph_builder):
        """Test TripleGenerator integration with empty data."""
        graph_builder.add_from_triple_generator([])

        assert len(graph_builder.graph) == 0
        assert graph_builder.triple_count == 0

    def test_performance_large_dataset(self, graph_builder):
        """Test performance with larger dataset."""
        # Generate a larger test dataset
        large_dataset = []
        for i in range(10000):
            large_dataset.extend(
                [
                    (
                        f"http://test.mosaic.dev/entity{i}",
                        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                        "http://test.mosaic.dev/Entity",
                    ),
                    (
                        f"http://test.mosaic.dev/entity{i}",
                        "http://test.mosaic.dev/id",
                        str(i),
                    ),
                    (
                        f"http://test.mosaic.dev/entity{i}",
                        "http://test.mosaic.dev/category",
                        f"category{i % 10}",
                    ),
                ]
            )

        # Should handle large datasets efficiently
        graph_builder.add_triples(large_dataset, batch_size=5000)

        assert len(graph_builder.graph) == 30000
        assert graph_builder.triple_count == 30000

        # Test query performance on large dataset
        query = """
        PREFIX mosaic: <http://test.mosaic.dev/>
        SELECT (COUNT(?entity) as ?count) WHERE {
            ?entity mosaic:category "category5" .
        }
        """

        results = graph_builder.query(query)
        assert len(results) == 1
        # Should find 1000 entities (every 10th from 10000)
        count = int(results[0][0])
        assert count == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
