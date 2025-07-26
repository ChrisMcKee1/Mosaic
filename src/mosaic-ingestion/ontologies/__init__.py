"""
Ontology definitions and management for OmniRAG architecture.

This module contains OWL ontology files and Python classes for managing
domain-specific ontologies used in the Mosaic MCP Tool's knowledge graph.

Structure:
- core.owl: Core Mosaic ontology definitions
- code.owl: Programming language and code structure ontology
- project.owl: Software project and repository ontology
- developer.owl: Developer and team collaboration ontology

Author: Mosaic MCP Tool
Date: 2025-01-25
Phase: OMR-P1-001 (RDF Infrastructure Setup)
"""

# Version information
__version__ = "0.1.0"
__phase__ = "OMR-P1-001"

# Export key components (to be implemented in subsequent tasks)
__all__ = [
    "MosaicCoreOntology",
    "CodeOntology",
    "ProjectOntology",
    "DeveloperOntology",
    "ontology_loader",
    "ontology_validator",
]
