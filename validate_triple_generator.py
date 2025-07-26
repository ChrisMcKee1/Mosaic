"""Simple validation script for TripleGenerator.

Test basic imports and functionality without complex test framework.
"""

import os
import sys

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def test_basic_imports():
    """Test that we can import all the required components."""
    print("Testing basic imports...")

    try:
        from src.mosaic_ingestion.rdf.triple_generator import (
            TripleGenerator,
            TripleGeneratorError,
            generate_triples_for_entities,
        )

        print("✓ TripleGenerator imports successful")
    except ImportError as e:
        print(f"✗ TripleGenerator import failed: {e}")
        return False

    try:
        from src.mosaic_ingestion.models.golden_node import (
            CodeEntity,
            EntityType,
            LanguageType,
        )

        print("✓ GoldenNode model imports successful")
    except ImportError as e:
        print(f"✗ GoldenNode model import failed: {e}")
        return False

    try:
        from rdflib import Graph, Literal, URIRef

        print("✓ RDFLib imports successful")
    except ImportError as e:
        print(f"✗ RDFLib import failed: {e}")
        return False

    return True


def test_basic_functionality():
    """Test basic TripleGenerator functionality."""
    print("\nTesting basic functionality...")

    try:
        from src.mosaic_ingestion.models.golden_node import (
            CodeEntity,
            EntityType,
            LanguageType,
        )
        from src.mosaic_ingestion.rdf.triple_generator import TripleGenerator

        # Create a simple test entity
        entity = CodeEntity(
            name="test_function",
            entity_type=EntityType.FUNCTION,
            language=LanguageType.PYTHON,
            content="def test_function():\n    pass",
            signature="def test_function():",
        )

        # Create TripleGenerator
        generator = TripleGenerator(base_namespace="http://test.example/")

        # Generate triples
        graph = generator.generate_triples_for_entities(
            entities=[entity],
            file_path="test.py",
            validate=False,  # Skip validation for basic test
        )

        print(f"✓ Generated {len(graph)} triples for test entity")
        print(f"✓ Statistics: {generator.get_statistics()}")

        # Test serialization
        turtle_output = graph.serialize(format="turtle")
        if len(turtle_output) > 0:
            print("✓ Graph serialization successful")
            return True
        print("✗ Graph serialization produced empty output")
        return False

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def test_performance_basic():
    """Test basic performance with multiple entities."""
    print("\nTesting basic performance...")

    try:
        import time

        from src.mosaic_ingestion.models.golden_node import (
            CodeEntity,
            EntityType,
            LanguageType,
        )
        from src.mosaic_ingestion.rdf.triple_generator import TripleGenerator

        # Create 100 test entities
        entities = []
        for i in range(100):
            entity = CodeEntity(
                name=f"function_{i}",
                entity_type=EntityType.FUNCTION,
                language=LanguageType.PYTHON,
                content=f"def function_{i}():\n    pass",
                signature=f"def function_{i}():",
            )
            entities.append(entity)

        # Time the generation
        generator = TripleGenerator()
        start_time = time.time()

        graph = generator.generate_triples_for_entities(
            entities=entities, file_path="test_large.py", validate=False
        )

        end_time = time.time()
        generation_time = end_time - start_time

        print(
            f"✓ Generated {len(graph)} triples for {len(entities)} entities in {generation_time:.3f}s"
        )
        print(f"✓ Performance: {len(entities) / generation_time:.1f} entities/second")

        return True

    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=== TripleGenerator Validation Script ===\n")

    results = []

    # Test imports
    results.append(test_basic_imports())

    # Test basic functionality
    results.append(test_basic_functionality())

    # Test performance
    results.append(test_performance_basic())

    # Summary
    print("\n=== Validation Summary ===")
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✓ All {total} tests passed!")
        return True
    print(f"✗ {passed}/{total} tests passed")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
