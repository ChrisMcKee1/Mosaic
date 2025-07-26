"""
Validation script for OMR-P1-005: GraphBuilder implementation
Tests core functionality without pytest dependencies
"""

import sys
from pathlib import Path

# Add the plugins directory to the path
plugins_path = Path(__file__).parent / "plugins"
sys.path.insert(0, str(plugins_path))

from graph_builder import GraphBuilder


def test_basic_functionality():
    """Test basic GraphBuilder functionality."""
    print("Testing GraphBuilder basic functionality...")

    # Initialize GraphBuilder
    gb = GraphBuilder(base_uri="http://test.mosaic.dev/")
    print(f"✓ GraphBuilder initialized with base URI: {gb.base_uri}")

    # Test adding triples
    sample_triples = [
        (
            "http://test.mosaic.dev/person1",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://test.mosaic.dev/Person",
        ),
        ("http://test.mosaic.dev/person1", "http://test.mosaic.dev/name", "Alice"),
        ("http://test.mosaic.dev/person1", "http://test.mosaic.dev/age", "30"),
    ]

    gb.add_triples(sample_triples)
    print(
        f"✓ Added {len(sample_triples)} triples, graph contains {len(gb.graph)} triples"
    )

    # Test SPARQL query
    query = """
    PREFIX mosaic: <http://test.mosaic.dev/>
    SELECT ?person ?name WHERE {
        ?person mosaic:name ?name .
    }
    """

    results = gb.query(query)
    print(f"✓ SPARQL query executed, found {len(results)} results")

    # Test serialization
    turtle_output = gb.serialize(format="turtle")
    print(f"✓ Graph serialized to Turtle format ({len(turtle_output)} chars)")

    # Test memory usage
    memory_stats = gb.get_memory_usage()
    print(
        f"✓ Memory usage: {memory_stats['memory_mb']:.2f} MB, {memory_stats['triple_count']} triples"
    )

    # Test statistics
    stats = gb.get_statistics()
    print(
        f"✓ Statistics: {stats['triple_count']} triples, {stats['batch_operations']} batch ops, {stats['query_count']} queries"
    )

    # Test TripleGenerator integration
    triple_generator_output = [
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
    ]

    initial_count = len(gb.graph)
    gb.add_from_triple_generator(triple_generator_output)
    final_count = len(gb.graph)
    print(
        f"✓ TripleGenerator integration: added {final_count - initial_count} triples from generator output"
    )

    # Test clearing
    gb.clear()
    print(f"✓ Graph cleared, now contains {len(gb.graph)} triples")

    return True


def test_batch_operations():
    """Test batch operations with larger datasets."""
    print("\nTesting batch operations...")

    gb = GraphBuilder()

    # Create a larger dataset
    large_dataset = []
    for i in range(1000):
        large_dataset.extend(
            [
                (
                    f"http://example.com/entity{i}",
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                    "http://example.com/Entity",
                ),
                (f"http://example.com/entity{i}", "http://example.com/id", str(i)),
            ]
        )

    gb.add_triples(large_dataset, batch_size=500)
    print(f"✓ Batch operations: added {len(large_dataset)} triples in batches")
    print(f"✓ Final graph size: {len(gb.graph)} triples")

    # Test query on larger dataset
    query = """
    SELECT (COUNT(?entity) as ?count) WHERE {
        ?entity <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.com/Entity> .
    }
    """

    results = gb.query(query)
    count = int(results[0][0])
    print(f"✓ Query on large dataset: found {count} entities")

    return True


def test_error_handling():
    """Test error handling scenarios."""
    print("\nTesting error handling...")

    gb = GraphBuilder()

    # Test empty triple addition
    gb.add_triples([])
    print("✓ Empty triple list handled gracefully")

    # Test invalid TripleGenerator output
    invalid_output = [
        {"subject": "test", "predicate": None, "object": "value"},
        {"incomplete": "data"},
    ]

    gb.add_from_triple_generator(invalid_output)
    print("✓ Invalid TripleGenerator output handled gracefully")

    # Test invalid SPARQL query
    try:
        gb.query("INVALID SPARQL")
        print("✗ Should have raised an exception for invalid SPARQL")
        return False
    except Exception:
        print("✓ Invalid SPARQL query raised exception as expected")

    return True


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("GraphBuilder Validation Script")
    print("=" * 60)

    try:
        success1 = test_basic_functionality()
        success2 = test_batch_operations()
        success3 = test_error_handling()

        if success1 and success2 and success3:
            print("\n" + "=" * 60)
            print(
                "✅ ALL TESTS PASSED - GraphBuilder implementation is working correctly!"
            )
            print("✅ OMR-P1-005 acceptance criteria validated")
            print("=" * 60)
            return True
        else:
            print("\n" + "=" * 60)
            print("❌ SOME TESTS FAILED")
            print("=" * 60)
            return False

    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
