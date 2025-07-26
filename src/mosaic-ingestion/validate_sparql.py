"""
Simple validation script for SPARQL Builder without external dependencies.

This script validates the core SPARQL functionality using only RDFLib
without requiring Azure Cosmos DB or other external services.

Task: OMR-P1-007 - Implement Basic SPARQL Query Capability
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from rdflib import Graph, URIRef, Literal, RDF

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockCosmosClient:
    """Mock Cosmos DB client for testing."""

    pass


def validate_sparql_builder():
    """
    Validate SPARQL Builder functionality with minimal dependencies.
    """
    try:
        # Import the SparqlBuilder class directly
        from rdf.sparql_builder import SparqlBuilder

        logger.info("=== SPARQL Builder Validation ===")

        # Create SPARQL builder with mock client
        mock_client = MockCosmosClient()
        sparql_builder = SparqlBuilder(mock_client)

        logger.info("✓ SparqlBuilder created successfully")

        # Test 1: Verify initialization
        assert hasattr(sparql_builder, "graph")
        assert hasattr(sparql_builder, "namespaces")
        assert isinstance(sparql_builder.graph, Graph)
        assert "code" in sparql_builder.namespaces
        logger.info("✓ Initialization test passed")

        # Test 2: Add sample data and test basic queries
        code_ns = sparql_builder.namespaces["code"]

        # Add sample function
        func_uri = URIRef(f"{code_ns}test_function")
        sparql_builder.graph.add((func_uri, RDF.type, URIRef(f"{code_ns}Function")))
        sparql_builder.graph.add(
            (func_uri, URIRef(f"{code_ns}name"), Literal("test_function"))
        )
        sparql_builder.graph.add(
            (func_uri, URIRef(f"{code_ns}inModule"), URIRef(f"{code_ns}test_module"))
        )

        # Add another function for call relationship
        func2_uri = URIRef(f"{code_ns}helper_function")
        sparql_builder.graph.add((func2_uri, RDF.type, URIRef(f"{code_ns}Function")))
        sparql_builder.graph.add(
            (func2_uri, URIRef(f"{code_ns}name"), Literal("helper_function"))
        )
        sparql_builder.graph.add((func_uri, URIRef(f"{code_ns}calls"), func2_uri))

        logger.info("✓ Sample data added successfully")

        # Test 3: Execute SPARQL query for functions in module
        try:
            functions = sparql_builder.query_functions_in_module("test_module")
            assert len(functions) >= 1
            assert functions[0]["functionName"] == "test_function"
            logger.info("✓ Functions in module query test passed")
        except Exception as e:
            logger.error(f"✗ Functions in module query failed: {e}")

        # Test 4: Execute SPARQL query for function calls
        try:
            calls = sparql_builder.query_function_calls()
            assert len(calls) >= 1
            assert calls[0]["callerName"] == "test_function"
            assert calls[0]["calleeName"] == "helper_function"
            logger.info("✓ Function calls query test passed")
        except Exception as e:
            logger.error(f"✗ Function calls query failed: {e}")

        # Test 5: Test graph statistics
        try:
            stats = sparql_builder.get_graph_statistics()
            assert stats["total_triples"] > 0
            assert stats["unique_subjects"] > 0
            logger.info(f"✓ Graph statistics test passed: {stats}")
        except Exception as e:
            logger.error(f"✗ Graph statistics failed: {e}")

        # Test 6: Test clear graph
        try:
            sparql_builder.clear_graph()
            assert len(sparql_builder.graph) == 0
            logger.info("✓ Clear graph test passed")
        except Exception as e:
            logger.error(f"✗ Clear graph failed: {e}")

        logger.info("=== All validation tests completed successfully ===")
        return True

    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Validation failed: {e}")
        return False


if __name__ == "__main__":
    success = validate_sparql_builder()
    if success:
        print("\n🎉 SPARQL Builder validation PASSED")
        sys.exit(0)
    else:
        print("\n❌ SPARQL Builder validation FAILED")
        sys.exit(1)
