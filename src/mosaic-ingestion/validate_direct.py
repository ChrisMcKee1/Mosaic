"""
Direct validation script for TripleGenerator

Test basic imports and functionality from the src directory.
"""

import sys
import os


def test_basic_imports():
    """Test that we can import all the required components."""
    print("Testing basic imports...")

    try:
        # Add the parent directory to the path to import mosaic-ingestion
        sys.path.insert(0, os.path.dirname(__file__))

        from rdflib import Graph, URIRef, Literal, Namespace
        from rdflib.namespace import RDF, RDFS, XSD

        print("✓ RDFLib imports successful")
    except ImportError as e:
        print(f"✗ RDFLib import failed: {e}")
        return False

    # Test importing from the rdf module directly
    try:
        from rdf.ontology_manager import ontology_manager

        print("✓ Ontology manager import successful")
    except ImportError as e:
        print(f"✗ Ontology manager import failed: {e}")
        return False

    return True


def test_triple_generation():
    """Test basic triple generation with RDFLib."""
    print("\nTesting basic triple generation...")

    try:
        from rdflib import Graph, Literal, Namespace
        from rdflib.namespace import RDF, XSD

        # Create a simple graph
        graph = Graph()

        # Define namespaces
        ns = Namespace("http://test.example/")
        code_ns = Namespace("http://mosaic.ai/ontology/code_base#")

        # Create a simple entity
        entity_uri = ns.test_function

        # Add triples
        graph.add((entity_uri, RDF.type, code_ns.Function))
        graph.add(
            (entity_uri, code_ns.name, Literal("test_function", datatype=XSD.string))
        )
        graph.add(
            (entity_uri, code_ns.file_path, Literal("test.py", datatype=XSD.string))
        )

        print(f"✓ Generated {len(graph)} basic triples")

        # Test serialization
        turtle_output = graph.serialize(format="turtle")
        if len(turtle_output) > 0:
            print("✓ Graph serialization successful")
            print("Sample output:")
            print(
                turtle_output[:200] + "..."
                if len(turtle_output) > 200
                else turtle_output
            )
            return True
        else:
            print("✗ Graph serialization produced empty output")
            return False

    except Exception as e:
        print(f"✗ Basic triple generation test failed: {e}")
        return False


def test_ontology_loading():
    """Test ontology loading."""
    print("\nTesting ontology loading...")

    try:
        from rdf.ontology_manager import ontology_manager

        # Try to load ontologies
        print("Attempting to load ontologies...")

        # Get ontology info
        info = ontology_manager.get_ontology_info()
        print("✓ Ontology manager accessible")
        print(f"✓ Available ontologies: {list(info.keys())}")

        return True

    except Exception as e:
        print(f"✗ Ontology loading test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=== Direct TripleGenerator Validation ===\n")

    results = []

    # Test imports
    results.append(test_basic_imports())

    # Test basic triple generation
    results.append(test_triple_generation())

    # Test ontology loading
    results.append(test_ontology_loading())

    # Summary
    print("\n=== Validation Summary ===")
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✓ All {total} tests passed!")
        return True
    else:
        print(f"✗ {passed}/{total} tests passed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
