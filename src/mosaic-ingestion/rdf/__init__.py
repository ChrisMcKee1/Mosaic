"""
RDF processing module for OmniRAG architecture.

This module provides RDF graph management, triple generation, and SPARQL query
capabilities as part of the Mosaic MCP Tool's OmniRAG implementation.

Key Components:
- RDF graph creation and management
- Ontology loading and validation
- SPARQL query execution
- Triple generation from code analysis

Dependencies:
- rdflib: Core RDF library
- owlready2: OWL ontology management
- SPARQLWrapper: SPARQL endpoint interface
- networkx: Graph analysis

Author: Mosaic MCP Tool
Date: 2025-01-25
Phase: OMR-P1-001 (RDF Infrastructure Setup)
"""

# Version information
__version__ = "0.1.1"
__phase__ = "OMR-P1-004"

# Import and export ontology manager
from .ontology_manager import (
    OntologyManager,
    OntologyError,
    OntologyLoadError,
    OntologyValidationError,
    OntologyNotFoundError,
    ontology_manager,
)

# Import and export triple generator (NEW)
from .triple_generator import (
    TripleGenerator,
    TripleGeneratorError,
    URIGenerationError,
    ValidationError,
    generate_triples_for_entities,
)

# Export key components
__all__ = [
    # Ontology management (implemented)
    "OntologyManager",
    "OntologyError",
    "OntologyLoadError",
    "OntologyValidationError",
    "OntologyNotFoundError",
    "ontology_manager",
    # Triple generation (implemented - OMR-P1-004)
    "TripleGenerator",
    "TripleGeneratorError",
    "URIGenerationError",
    "ValidationError",
    "generate_triples_for_entities",
    # RDF management (to be implemented)
    "RDFGraphManager",
    # SPARQL operations (to be implemented)
    "SPARQLQueryEngine",
    "QueryBuilder",
    # Utilities (to be implemented)
    "namespace_manager",
    "rdf_utils",
]
