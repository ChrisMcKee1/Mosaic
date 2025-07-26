"""
AST to RDF Triple Generator

Converts CodeEntity objects from AI-powered parsing into RDF triples using
the defined ontologies. Provides consistent URI generation, relationship
mapping, and ontology validation for semantic code representation.

Key Features:
- Convert CodeEntity objects to RDF triples
- Consistent URI generation scheme
- Relationship extraction and representation
- Ontology validation against schemas
- Performance optimized for large codebases
- Integration with existing AI parser pipeline
"""

import logging
from typing import List, Dict
from urllib.parse import quote
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, XSD

from .ontology_manager import ontology_manager
from models.golden_node import CodeEntity, EntityType, LanguageType


class TripleGeneratorError(Exception):
    """Base exception for triple generation errors."""

    pass


class URIGenerationError(TripleGeneratorError):
    """Exception raised when URI generation fails."""

    pass


class ValidationError(TripleGeneratorError):
    """Exception raised when triple validation fails."""

    pass


class TripleGenerator:
    """
    Converts CodeEntity objects to RDF triples using defined ontologies.

    The TripleGenerator creates a semantic representation of code structure
    by mapping CodeEntity objects to RDF triples using the core ontologies:
    - code_base.owl: Core code entities and properties
    - python.owl: Python-specific extensions
    - relationships.owl: Code relationships and dependencies
    """

    def __init__(self, base_namespace: str = "http://mosaic.ai/graph#"):
        """
        Initialize the triple generator with base namespace.

        Args:
            base_namespace: Base namespace for generated URIs
        """
        self.logger = logging.getLogger("mosaic.triple_generator")
        self.base_namespace = base_namespace

        # Create namespace objects for URI generation
        self.ns = Namespace(base_namespace)
        self.code_ns = Namespace("http://mosaic.ai/ontology/code_base#")
        self.python_ns = Namespace("http://mosaic.ai/ontology/python#")
        self.rel_ns = Namespace("http://mosaic.ai/ontology/relationships#")

        # Initialize statistics
        self.stats = {
            "triples_generated": 0,
            "entities_processed": 0,
            "relationships_created": 0,
            "validation_errors": 0,
        }

        self.logger.info(
            f"TripleGenerator initialized with namespace: {base_namespace}"
        )

    def generate_triples_for_entities(
        self, entities: List[CodeEntity], file_path: str, validate: bool = True
    ) -> Graph:
        """
        Generate RDF triples for a list of CodeEntity objects.

        Args:
            entities: List of CodeEntity objects to convert
            file_path: Source file path for context
            validate: Whether to validate triples against ontologies

        Returns:
            RDF Graph containing all generated triples

        Raises:
            TripleGeneratorError: If triple generation fails
        """
        try:
            self.logger.info(
                f"Generating RDF triples for {len(entities)} entities from {file_path}"
            )

            # Reset statistics
            self.stats = {
                "triples_generated": 0,
                "entities_processed": 0,
                "relationships_created": 0,
                "validation_errors": 0,
            }

            # Create RDF graph
            graph = Graph()

            # Bind namespaces for readable output
            graph.bind("mosaic", self.ns)
            graph.bind("code", self.code_ns)
            graph.bind("python", self.python_ns)
            graph.bind("rel", self.rel_ns)
            graph.bind("rdf", RDF)
            graph.bind("rdfs", RDFS)
            graph.bind("xsd", XSD)

            # Generate entity triples
            entity_uris = {}  # Map entity names to URIs for relationship generation

            for entity in entities:
                try:
                    entity_uri = self._generate_entity_triples(graph, entity, file_path)
                    entity_uris[entity.name] = entity_uri
                    self.stats["entities_processed"] += 1
                except Exception as e:
                    self.logger.error(
                        f"Failed to generate triples for entity {entity.name}: {e}"
                    )
                    if validate:
                        raise TripleGeneratorError(
                            f"Entity triple generation failed: {e}"
                        )

            # Generate relationship triples
            for entity in entities:
                try:
                    self._generate_relationship_triples(
                        graph, entity, entity_uris, file_path
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to generate relationships for entity {entity.name}: {e}"
                    )
                    if validate:
                        raise TripleGeneratorError(
                            f"Relationship triple generation failed: {e}"
                        )

            # Validate against ontologies if requested
            if validate:
                self._validate_graph_against_ontologies(graph)

            self.stats["triples_generated"] = len(graph)

            self.logger.info(
                f"Successfully generated {self.stats['triples_generated']} triples "
                f"for {self.stats['entities_processed']} entities "
                f"with {self.stats['relationships_created']} relationships"
            )

            return graph

        except Exception as e:
            self.logger.error(f"Triple generation failed for {file_path}: {e}")
            raise TripleGeneratorError(f"Failed to generate triples: {e}")

    def _generate_entity_triples(
        self, graph: Graph, entity: CodeEntity, file_path: str
    ) -> URIRef:
        """
        Generate RDF triples for a single CodeEntity.

        Args:
            graph: RDF graph to add triples to
            entity: CodeEntity to convert
            file_path: Source file path for URI generation

        Returns:
            URI of the generated entity
        """
        # Generate unique URI for this entity
        entity_uri = self._generate_entity_uri(entity, file_path)

        # Determine ontology classes based on entity type
        rdf_type = self._map_entity_type_to_ontology_class(
            entity.entity_type, entity.language
        )

        # Core entity triples
        graph.add((entity_uri, RDF.type, rdf_type))

        # Basic properties
        if entity.name:
            graph.add(
                (
                    entity_uri,
                    self.code_ns.name,
                    Literal(entity.name, datatype=XSD.string),
                )
            )

        if entity.signature:
            graph.add(
                (
                    entity_uri,
                    self.code_ns.signature,
                    Literal(entity.signature, datatype=XSD.string),
                )
            )

        if hasattr(entity, "file_path") and entity.file_path:
            graph.add(
                (
                    entity_uri,
                    self.code_ns.file_path,
                    Literal(entity.file_path, datatype=XSD.string),
                )
            )

        # Line numbers
        if hasattr(entity, "line_start") and entity.line_start:
            graph.add(
                (
                    entity_uri,
                    self.code_ns.start_line,
                    Literal(entity.line_start, datatype=XSD.integer),
                )
            )

        if hasattr(entity, "line_end") and entity.line_end:
            graph.add(
                (
                    entity_uri,
                    self.code_ns.end_line,
                    Literal(entity.line_end, datatype=XSD.integer),
                )
            )

        # Python-specific properties
        if entity.language == LanguageType.PYTHON:
            self._add_python_specific_triples(graph, entity_uri, entity)

        # Scope and visibility
        if hasattr(entity, "scope") and entity.scope:
            # Create a literal for scope (public, private, protected, etc.)
            graph.add(
                (
                    entity_uri,
                    self.code_ns.scope,
                    Literal(entity.scope, datatype=XSD.string),
                )
            )

        if hasattr(entity, "is_exported") and entity.is_exported is not None:
            graph.add(
                (
                    entity_uri,
                    self.code_ns.is_exported,
                    Literal(entity.is_exported, datatype=XSD.boolean),
                )
            )

        return entity_uri

    def _generate_relationship_triples(
        self,
        graph: Graph,
        entity: CodeEntity,
        entity_uris: Dict[str, URIRef],
        file_path: str,
    ) -> None:
        """
        Generate relationship triples for a CodeEntity.

        Args:
            graph: RDF graph to add triples to
            entity: Source entity for relationships
            entity_uris: Mapping of entity names to URIs
            file_path: Source file path for context
        """
        entity_uri = entity_uris.get(entity.name)
        if not entity_uri:
            self.logger.warning(f"No URI found for entity {entity.name}")
            return

        # Parent-child relationships
        if entity.parent_entity and entity.parent_entity in entity_uris:
            parent_uri = entity_uris[entity.parent_entity]
            graph.add((entity_uri, self.rel_ns.definedIn, parent_uri))
            graph.add((parent_uri, self.rel_ns.defines, entity_uri))
            self.stats["relationships_created"] += 1

        # Function calls
        if hasattr(entity, "calls") and entity.calls:
            for called_function in entity.calls:
                if called_function in entity_uris:
                    called_uri = entity_uris[called_function]
                    graph.add((entity_uri, self.rel_ns.calls, called_uri))
                    graph.add((called_uri, self.rel_ns.calledBy, entity_uri))
                    self.stats["relationships_created"] += 1
                else:
                    # Create external function reference
                    external_uri = self._generate_external_entity_uri(
                        called_function, file_path
                    )
                    graph.add((entity_uri, self.rel_ns.calls, external_uri))
                    graph.add((external_uri, RDF.type, self.code_ns.Function))
                    graph.add(
                        (
                            external_uri,
                            self.code_ns.name,
                            Literal(called_function, datatype=XSD.string),
                        )
                    )
                    self.stats["relationships_created"] += 1

        # Import dependencies
        if hasattr(entity, "imports") and entity.imports:
            for imported_module in entity.imports:
                if imported_module in entity_uris:
                    imported_uri = entity_uris[imported_module]
                    graph.add((entity_uri, self.rel_ns.imports, imported_uri))
                    graph.add((imported_uri, self.rel_ns.importedBy, entity_uri))
                    self.stats["relationships_created"] += 1
                else:
                    # Create external module reference
                    external_uri = self._generate_external_entity_uri(
                        imported_module, file_path, "Module"
                    )
                    graph.add((entity_uri, self.rel_ns.imports, external_uri))
                    graph.add((external_uri, RDF.type, self.code_ns.Module))
                    graph.add(
                        (
                            external_uri,
                            self.code_ns.name,
                            Literal(imported_module, datatype=XSD.string),
                        )
                    )
                    self.stats["relationships_created"] += 1

    def _generate_entity_uri(self, entity: CodeEntity, file_path: str) -> URIRef:
        """
        Generate a consistent URI for a code entity.

        Args:
            entity: CodeEntity to generate URI for
            file_path: Source file path for context

        Returns:
            Unique URI for the entity
        """
        try:
            # Normalize file path for URI
            normalized_path = file_path.replace("\\", "/").replace(" ", "_")
            if normalized_path.startswith("/"):
                normalized_path = normalized_path[1:]

            # Create URI components
            entity_type = (
                entity.entity_type.value.lower()
                if hasattr(entity.entity_type, "value")
                else str(entity.entity_type).lower()
            )
            entity_name = quote(entity.name.replace(" ", "_"), safe="")

            # Generate hierarchical URI
            if entity.parent_entity:
                parent_part = quote(entity.parent_entity.replace(" ", "_"), safe="")
                uri_path = (
                    f"{entity_type}/{normalized_path}/{parent_part}/{entity_name}"
                )
            else:
                uri_path = f"{entity_type}/{normalized_path}/{entity_name}"

            return URIRef(f"{self.base_namespace}{uri_path}")

        except Exception as e:
            raise URIGenerationError(
                f"Failed to generate URI for entity {entity.name}: {e}"
            )

    def _generate_external_entity_uri(
        self, entity_name: str, file_path: str, entity_type: str = "Function"
    ) -> URIRef:
        """Generate URI for external entity reference."""
        try:
            normalized_path = file_path.replace("\\", "/").replace(" ", "_")
            entity_name_safe = quote(entity_name.replace(" ", "_"), safe="")
            uri_path = (
                f"external/{entity_type.lower()}/{normalized_path}/{entity_name_safe}"
            )
            return URIRef(f"{self.base_namespace}{uri_path}")
        except Exception as e:
            raise URIGenerationError(
                f"Failed to generate external URI for {entity_name}: {e}"
            )

    def _map_entity_type_to_ontology_class(
        self, entity_type: EntityType, language: LanguageType
    ) -> URIRef:
        """
        Map EntityType to appropriate ontology class.

        Args:
            entity_type: Type of the code entity
            language: Programming language

        Returns:
            URI of the appropriate ontology class
        """
        # Python-specific mappings
        if language == LanguageType.PYTHON:
            python_mappings = {
                EntityType.FUNCTION: self.python_ns.PythonFunction,
                EntityType.CLASS: self.python_ns.PythonClass,
                EntityType.MODULE: self.python_ns.PythonModule,
                EntityType.METHOD: self.python_ns.PythonMethod,
                EntityType.PROPERTY: self.python_ns.PythonProperty,
            }
            if entity_type in python_mappings:
                return python_mappings[entity_type]

        # Generic mappings for all languages
        generic_mappings = {
            EntityType.FUNCTION: self.code_ns.Function,
            EntityType.CLASS: self.code_ns.Class,
            EntityType.MODULE: self.code_ns.Module,
            EntityType.LIBRARY: self.code_ns.Library,
            EntityType.VARIABLE: self.code_ns.Variable,
            EntityType.INTERFACE: self.code_ns.Class,  # Map to Class for now
            EntityType.ENUM: self.code_ns.Class,  # Map to Class for now
            EntityType.STRUCT: self.code_ns.Class,  # Map to Class for now
            EntityType.METHOD: self.code_ns.Function,  # Map to Function for generic
            EntityType.PROPERTY: self.code_ns.Variable,  # Map to Variable for generic
        }

        return generic_mappings.get(entity_type, self.code_ns.CodeEntity)

    def _add_python_specific_triples(
        self, graph: Graph, entity_uri: URIRef, entity: CodeEntity
    ) -> None:
        """Add Python-specific triples for an entity."""
        # Add Python-specific properties if available
        if hasattr(entity, "is_async") and entity.is_async is not None:
            graph.add(
                (
                    entity_uri,
                    self.python_ns.is_async,
                    Literal(entity.is_async, datatype=XSD.boolean),
                )
            )

        if hasattr(entity, "is_generator") and entity.is_generator is not None:
            graph.add(
                (
                    entity_uri,
                    self.python_ns.is_generator,
                    Literal(entity.is_generator, datatype=XSD.boolean),
                )
            )

        if hasattr(entity, "has_docstring") and entity.has_docstring is not None:
            graph.add(
                (
                    entity_uri,
                    self.python_ns.has_docstring,
                    Literal(entity.has_docstring, datatype=XSD.boolean),
                )
            )

    def _validate_graph_against_ontologies(self, graph: Graph) -> None:
        """
        Validate generated RDF graph against loaded ontologies.

        Args:
            graph: RDF graph to validate

        Raises:
            ValidationError: If validation fails
        """
        try:
            self.logger.debug("Validating generated triples against ontologies")

            # Check that all used classes exist in ontologies
            used_classes = set()
            used_properties = set()

            for s, p, o in graph:
                if p == RDF.type and isinstance(o, URIRef):
                    used_classes.add(o)
                if isinstance(p, URIRef):
                    used_properties.add(p)

            # Validate classes exist in ontologies
            for cls in used_classes:
                if cls.startswith(self.code_ns) or cls.startswith(self.python_ns):
                    try:
                        # Try to get class from ontology manager
                        ontology_manager.get_class(str(cls))
                    except Exception:
                        self.logger.warning(f"Class {cls} not found in ontologies")
                        self.stats["validation_errors"] += 1

            # Validate properties exist in ontologies
            for prop in used_properties:
                if (
                    prop.startswith(self.rel_ns)
                    or prop.startswith(self.code_ns)
                    or prop.startswith(self.python_ns)
                ):
                    try:
                        # Try to get property from ontology manager
                        ontology_manager.get_property(str(prop))
                    except Exception:
                        self.logger.warning(f"Property {prop} not found in ontologies")
                        self.stats["validation_errors"] += 1

            if self.stats["validation_errors"] > 0:
                self.logger.warning(
                    f"Validation completed with {self.stats['validation_errors']} warnings"
                )
            else:
                self.logger.debug("Graph validation completed successfully")

        except Exception as e:
            raise ValidationError(f"Graph validation failed: {e}")

    def get_statistics(self) -> Dict[str, int]:
        """Get generation statistics."""
        return self.stats.copy()

    def reset_statistics(self) -> None:
        """Reset generation statistics."""
        self.stats = {
            "triples_generated": 0,
            "entities_processed": 0,
            "relationships_created": 0,
            "validation_errors": 0,
        }


# Module-level convenience function
def generate_triples_for_entities(
    entities: List[CodeEntity],
    file_path: str,
    base_namespace: str = "http://mosaic.ai/graph#",
    validate: bool = True,
) -> Graph:
    """
    Convenience function to generate RDF triples for CodeEntity objects.

    Args:
        entities: List of CodeEntity objects
        file_path: Source file path
        base_namespace: Base namespace for URIs
        validate: Whether to validate against ontologies

    Returns:
        RDF Graph containing generated triples
    """
    generator = TripleGenerator(base_namespace=base_namespace)
    return generator.generate_triples_for_entities(entities, file_path, validate)
