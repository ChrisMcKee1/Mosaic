"""
Schema definitions for OmniRAG data structures.

This module contains JSON Schema, SHACL shapes, and validation rules
for ensuring data quality and consistency in the Mosaic knowledge graph.

Structure:
- rdf_schemas.py: RDF/SPARQL validation schemas
- json_schemas.py: JSON Schema definitions
- shacl_shapes.ttl: SHACL constraint definitions
- validation_rules.py: Custom validation logic

Author: Mosaic MCP Tool
Date: 2025-01-25
Phase: OMR-P1-001 (RDF Infrastructure Setup)
"""

# Version information
__version__ = "0.1.0"
__phase__ = "OMR-P1-001"

# Export key components (to be implemented in subsequent tasks)
__all__ = [
    "RDFSchemaValidator",
    "JSONSchemaValidator",
    "SHACLValidator",
    "ValidationEngine",
    "schema_registry",
    "validation_utils",
]
