"""
Unit tests for AST to RDF Triple Generator

Tests the TripleGenerator class functionality including:
- Entity to RDF triple conversion
- URI generation consistency
- Relationship mapping
- Ontology validation
- Performance benchmarks
"""

import pytest
from typing import List
from unittest.mock import Mock, patch
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, XSD

# Import the components we're testing
from src.mosaic_ingestion.rdf.triple_generator import (
    TripleGenerator,
    generate_triples_for_entities,
)
from src.mosaic_ingestion.models.golden_node import CodeEntity, EntityType, LanguageType


class TestTripleGenerator:
    """Test suite for TripleGenerator class."""

    @pytest.fixture
    def sample_entities(self) -> List[CodeEntity]:
        """Create sample CodeEntity objects for testing."""
        return [
            CodeEntity(
                name="TestClass",
                entity_type=EntityType.CLASS,
                language=LanguageType.PYTHON,
                content="class TestClass:\n    pass",
                signature="class TestClass:",
                file_path="test_module.py",
                line_start=1,
                line_end=2,
                scope="public",
                is_exported=True,
            ),
            CodeEntity(
                name="test_function",
                entity_type=EntityType.FUNCTION,
                language=LanguageType.PYTHON,
                content="def test_function():\n    return 'hello'",
                signature="def test_function():",
                file_path="test_module.py",
                line_start=4,
                line_end=5,
                parent_entity="TestClass",
                scope="public",
                is_exported=False,
                calls=["print", "len"],
                imports=["os", "sys"],
            ),
            CodeEntity(
                name="helper_method",
                entity_type=EntityType.METHOD,
                language=LanguageType.PYTHON,
                content="def helper_method(self):\n    return self.value",
                signature="def helper_method(self):",
                file_path="test_module.py",
                line_start=7,
                line_end=8,
                parent_entity="TestClass",
                scope="private",
                calls=["test_function"],
            ),
        ]

    @pytest.fixture
    def triple_generator(self) -> TripleGenerator:
        """Create TripleGenerator instance for testing."""
        return TripleGenerator(base_namespace="http://test.example/graph#")

    def test_initialization(self, triple_generator):
        """Test TripleGenerator initialization."""
        assert triple_generator.base_namespace == "http://test.example/graph#"
        assert triple_generator.ns == Namespace("http://test.example/graph#")
        assert triple_generator.stats["triples_generated"] == 0
        assert triple_generator.stats["entities_processed"] == 0

    def test_uri_generation_for_class(self, triple_generator, sample_entities):
        """Test URI generation for class entities."""
        class_entity = sample_entities[0]  # TestClass
        uri = triple_generator._generate_entity_uri(class_entity, "test_module.py")

        expected = URIRef("http://test.example/graph#class/test_module.py/TestClass")
        assert uri == expected

    def test_uri_generation_for_nested_entity(self, triple_generator, sample_entities):
        """Test URI generation for nested entities."""
        function_entity = sample_entities[1]  # test_function (nested in TestClass)
        uri = triple_generator._generate_entity_uri(function_entity, "test_module.py")

        expected = URIRef(
            "http://test.example/graph#function/test_module.py/TestClass/test_function"
        )
        assert uri == expected

    def test_uri_generation_special_characters(self, triple_generator):
        """Test URI generation handles special characters."""
        entity = CodeEntity(
            name="test function with spaces",
            entity_type=EntityType.FUNCTION,
            language=LanguageType.PYTHON,
            content="def test():\n    pass",
        )

        uri = triple_generator._generate_entity_uri(
            entity, "path with spaces/test file.py"
        )

        # Should properly encode spaces and special characters
        assert "test%20function%20with%20spaces" in str(uri)
        assert "path%20with%20spaces" in str(uri) or "path with spaces" in str(uri)

    def test_entity_type_mapping_python(self, triple_generator):
        """Test entity type mapping for Python-specific types."""
        # Test Python function mapping
        python_func_class = triple_generator._map_entity_type_to_ontology_class(
            EntityType.FUNCTION, LanguageType.PYTHON
        )
        assert python_func_class == triple_generator.python_ns.PythonFunction

        # Test Python class mapping
        python_class_class = triple_generator._map_entity_type_to_ontology_class(
            EntityType.CLASS, LanguageType.PYTHON
        )
        assert python_class_class == triple_generator.python_ns.PythonClass

    def test_entity_type_mapping_generic(self, triple_generator):
        """Test entity type mapping for generic languages."""
        # Test generic function mapping
        generic_func_class = triple_generator._map_entity_type_to_ontology_class(
            EntityType.FUNCTION, LanguageType.JAVASCRIPT
        )
        assert generic_func_class == triple_generator.code_ns.Function

        # Test fallback to CodeEntity
        unknown_class = triple_generator._map_entity_type_to_ontology_class(
            EntityType.OTHER, LanguageType.PYTHON
        )
        assert unknown_class == triple_generator.code_ns.CodeEntity

    def test_generate_entity_triples(self, triple_generator, sample_entities):
        """Test generation of triples for a single entity."""
        graph = Graph()
        entity = sample_entities[0]  # TestClass

        entity_uri = triple_generator._generate_entity_triples(
            graph, entity, "test_module.py"
        )

        # Check that entity URI is returned
        assert isinstance(entity_uri, URIRef)

        # Check that basic triples are generated
        triples = list(graph)
        assert len(triples) > 0

        # Check for type triple
        type_triples = list(graph.triples((entity_uri, RDF.type, None)))
        assert len(type_triples) == 1
        assert type_triples[0][2] == triple_generator.python_ns.PythonClass

        # Check for name triple
        name_triples = list(
            graph.triples((entity_uri, triple_generator.code_ns.name, None))
        )
        assert len(name_triples) == 1
        assert name_triples[0][2] == Literal("TestClass", datatype=XSD.string)

    def test_generate_relationship_triples(self, triple_generator, sample_entities):
        """Test generation of relationship triples."""
        graph = Graph()

        # Generate entity triples first to get URIs
        entity_uris = {}
        for entity in sample_entities:
            uri = triple_generator._generate_entity_triples(
                graph, entity, "test_module.py"
            )
            entity_uris[entity.name] = uri

        # Generate relationships for function entity
        function_entity = sample_entities[1]  # test_function
        triple_generator._generate_relationship_triples(
            graph, function_entity, entity_uris, "test_module.py"
        )

        # Check parent relationship exists
        function_uri = entity_uris["test_function"]
        class_uri = entity_uris["TestClass"]

        parent_triples = list(
            graph.triples((function_uri, triple_generator.rel_ns.definedIn, class_uri))
        )
        assert len(parent_triples) == 1

        child_triples = list(
            graph.triples((class_uri, triple_generator.rel_ns.defines, function_uri))
        )
        assert len(child_triples) == 1

    def test_external_entity_references(self, triple_generator, sample_entities):
        """Test handling of external entity references."""
        graph = Graph()
        entity_uris = {
            "test_function": URIRef(
                "http://test.example/graph#function/test_module.py/TestClass/test_function"
            )
        }

        # Function calls external entities (print, len)
        function_entity = sample_entities[1]  # test_function

        triple_generator._generate_relationship_triples(
            graph, function_entity, entity_uris, "test_module.py"
        )

        # Check that external function references are created
        function_uri = entity_uris["test_function"]
        call_triples = list(
            graph.triples((function_uri, triple_generator.rel_ns.calls, None))
        )

        # Should have calls to print and len (external functions)
        assert len(call_triples) >= 2

    @patch("src.mosaic_ingestion.rdf.triple_generator.ontology_manager")
    def test_generate_triples_for_entities_integration(
        self, mock_ontology_manager, triple_generator, sample_entities
    ):
        """Test complete integration of triple generation."""
        # Mock ontology manager methods
        mock_ontology_manager.get_class.return_value = Mock()
        mock_ontology_manager.get_property.return_value = Mock()

        graph = triple_generator.generate_triples_for_entities(
            entities=sample_entities,
            file_path="test_module.py",
            validate=False,  # Skip validation for this test
        )

        # Check that graph has content
        assert len(graph) > 0

        # Check statistics
        stats = triple_generator.get_statistics()
        assert stats["entities_processed"] == 3
        assert stats["triples_generated"] > 0
        assert stats["relationships_created"] > 0

    def test_validation_warnings(self, triple_generator, sample_entities):
        """Test validation warnings for unknown classes/properties."""
        with patch(
            "src.mosaic_ingestion.rdf.triple_generator.ontology_manager"
        ) as mock_ontology_manager:
            # Make ontology manager raise exceptions for unknown entities
            mock_ontology_manager.get_class.side_effect = Exception("Class not found")
            mock_ontology_manager.get_property.side_effect = Exception(
                "Property not found"
            )

            # Should not raise exception but generate warnings
            graph = triple_generator.generate_triples_for_entities(
                entities=sample_entities, file_path="test_module.py", validate=True
            )

            # Should still generate triples despite validation warnings
            assert len(graph) > 0

            # Should have validation errors recorded
            stats = triple_generator.get_statistics()
            assert stats["validation_errors"] > 0

    def test_performance_benchmark(self, triple_generator):
        """Test performance with large number of entities."""
        import time

        # Create 1000 test entities
        entities = []
        for i in range(1000):
            entity = CodeEntity(
                name=f"function_{i}",
                entity_type=EntityType.FUNCTION,
                language=LanguageType.PYTHON,
                content=f"def function_{i}():\n    pass",
                file_path="large_module.py",
                line_start=i * 2 + 1,
                line_end=i * 2 + 2,
            )
            entities.append(entity)

        # Measure generation time
        start_time = time.time()
        graph = triple_generator.generate_triples_for_entities(
            entities=entities,
            file_path="large_module.py",
            validate=False,  # Skip validation for performance test
        )
        end_time = time.time()

        generation_time = end_time - start_time

        # Should process 1000 functions in under 10 seconds (acceptance criteria)
        assert generation_time < 10.0, (
            f"Generation took {generation_time:.2f}s, should be under 10s"
        )

        # Should generate appropriate number of triples
        assert (
            len(graph) >= 3000
        )  # At least 3 triples per entity (type, name, file_path)

        # Check statistics
        stats = triple_generator.get_statistics()
        assert stats["entities_processed"] == 1000
        assert stats["triples_generated"] == len(graph)

    def test_convenience_function(self, sample_entities):
        """Test module-level convenience function."""
        graph = generate_triples_for_entities(
            entities=sample_entities,
            file_path="test_module.py",
            base_namespace="http://test.convenience/graph#",
            validate=False,
        )

        assert isinstance(graph, Graph)
        assert len(graph) > 0

        # Check that custom namespace is used
        found_custom_namespace = False
        for s, p, o in graph:
            if "test.convenience" in str(s):
                found_custom_namespace = True
                break
        assert found_custom_namespace

    def test_error_handling(self, triple_generator):
        """Test error handling in triple generation."""
        # Test with invalid entity
        invalid_entity = CodeEntity(
            name="",  # Empty name should cause issues
            entity_type=EntityType.FUNCTION,
            language=LanguageType.PYTHON,
            content="invalid",
        )

        # Should handle gracefully
        graph = triple_generator.generate_triples_for_entities(
            entities=[invalid_entity], file_path="invalid.py", validate=False
        )

        # Should return empty or minimal graph, not crash
        assert isinstance(graph, Graph)

    def test_statistics_reset(self, triple_generator):
        """Test statistics reset functionality."""
        # Generate some triples first
        triple_generator.stats["triples_generated"] = 100
        triple_generator.stats["entities_processed"] = 10

        # Reset statistics
        triple_generator.reset_statistics()

        # Should be back to zero
        assert triple_generator.stats["triples_generated"] == 0
        assert triple_generator.stats["entities_processed"] == 0
        assert triple_generator.stats["relationships_created"] == 0
        assert triple_generator.stats["validation_errors"] == 0

    def test_namespace_binding(self, triple_generator, sample_entities):
        """Test that generated graphs have proper namespace bindings."""
        graph = triple_generator.generate_triples_for_entities(
            entities=sample_entities, file_path="test_module.py", validate=False
        )

        # Check namespace bindings
        namespaces = dict(graph.namespaces())

        assert "mosaic" in namespaces
        assert "code" in namespaces
        assert "python" in namespaces
        assert "rel" in namespaces
        assert "rdf" in namespaces


class TestTripleGeneratorIntegration:
    """Integration tests for TripleGenerator with AI code parser."""

    @pytest.fixture
    def mock_entities(self) -> List[CodeEntity]:
        """Create realistic CodeEntity objects as would come from AI parser."""
        return [
            CodeEntity(
                name="UserService",
                entity_type=EntityType.CLASS,
                language=LanguageType.PYTHON,
                content="class UserService:\n    def __init__(self):\n        self.db = Database()",
                signature="class UserService:",
                file_path="services/user_service.py",
                line_start=1,
                line_end=15,
                scope="public",
                is_exported=True,
                imports=["database", "logging"],
                calls=[],
            ),
            CodeEntity(
                name="get_user",
                entity_type=EntityType.METHOD,
                language=LanguageType.PYTHON,
                content="def get_user(self, user_id):\n    return self.db.find_user(user_id)",
                signature="def get_user(self, user_id):",
                file_path="services/user_service.py",
                line_start=5,
                line_end=6,
                parent_entity="UserService",
                scope="public",
                is_exported=False,
                calls=["find_user", "validate_id"],
                imports=[],
            ),
            CodeEntity(
                name="validate_user_data",
                entity_type=EntityType.FUNCTION,
                language=LanguageType.PYTHON,
                content="def validate_user_data(data):\n    if not data:\n        raise ValueError('Invalid data')",
                signature="def validate_user_data(data):",
                file_path="services/user_service.py",
                line_start=10,
                line_end=12,
                scope="private",
                is_exported=False,
                calls=["ValueError"],
                imports=[],
            ),
        ]

    def test_realistic_code_structure(self, mock_entities):
        """Test triple generation with realistic code structure."""
        generator = TripleGenerator(base_namespace="http://mosaic.ai/graph#")

        graph = generator.generate_triples_for_entities(
            entities=mock_entities, file_path="services/user_service.py", validate=False
        )

        # Should generate comprehensive triples
        assert len(graph) >= 15  # Multiple triples per entity

        # Check that all entities are represented
        entity_types = set()
        for s, p, o in graph:
            if p == RDF.type:
                entity_types.add(str(o))

        # Should have both class and function types
        assert any("PythonClass" in t for t in entity_types)
        assert any("PythonMethod" in t or "PythonFunction" in t for t in entity_types)

        # Check relationships exist
        relationship_count = 0
        for s, p, o in graph:
            if "calls" in str(p) or "definedIn" in str(p) or "imports" in str(p):
                relationship_count += 1

        assert relationship_count > 0

    def test_serialization_formats(self, mock_entities):
        """Test that generated graphs can be serialized to different formats."""
        generator = TripleGenerator()

        graph = generator.generate_triples_for_entities(
            entities=mock_entities, file_path="services/user_service.py", validate=False
        )

        # Test different serialization formats
        turtle_output = graph.serialize(format="turtle")
        assert len(turtle_output) > 0
        assert "UserService" in turtle_output

        json_ld_output = graph.serialize(format="json-ld")
        assert len(json_ld_output) > 0

        xml_output = graph.serialize(format="xml")
        assert len(xml_output) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
