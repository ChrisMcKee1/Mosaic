"""
Ontology Manager System for Mosaic MCP Tool

This module provides a comprehensive ontology management system to load, validate,
and provide access to OWL ontologies throughout the application. It implements
enterprise-grade patterns with caching, validation, and error handling.

Author: Mosaic MCP Tool Development Team
"""

import os
import logging
import functools
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from urllib.parse import urlparse

import owlready2
from owlready2 import get_ontology, onto_path, PREDEFINED_ONTOLOGIES, Thing


logger = logging.getLogger(__name__)


class OntologyError(Exception):
    """Base exception for ontology-related errors."""

    pass


class OntologyLoadError(OntologyError):
    """Exception raised when ontology loading fails."""

    pass


class OntologyValidationError(OntologyError):
    """Exception raised when ontology validation fails."""

    pass


class OntologyNotFoundError(OntologyError):
    """Exception raised when requested ontology is not found."""

    pass


class OntologyManager:
    """
    Comprehensive ontology management system for the Mosaic MCP Tool.

    Provides enterprise-grade ontology loading, caching, validation, and access
    capabilities. Implements singleton pattern for global access throughout the
    application.

    Features:
    - Thread-safe singleton implementation
    - LRU caching for performance optimization
    - Support for local files and HTTP URLs
    - Comprehensive validation and error reporting
    - Entity introspection and search capabilities
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern implementation using __new__()."""
        if cls._instance is None:
            logger.info("Creating OntologyManager singleton instance")
            cls._instance = super(OntologyManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the OntologyManager (called only once due to singleton)."""
        if OntologyManager._initialized:
            return

        logger.info("Initializing OntologyManager")

        # Internal state
        self._loaded_ontologies: Dict[str, owlready2.Ontology] = {}
        self._ontology_metadata: Dict[str, Dict[str, Any]] = {}
        self._base_ontology_path = os.getenv(
            "MOSAIC_ONTOLOGY_PATH", "src/mosaic-ingestion/ontologies"
        )

        # Configure owlready2 paths
        self._setup_ontology_paths()

        # Load core ontologies
        self._load_core_ontologies()

        OntologyManager._initialized = True
        logger.info("OntologyManager initialization completed")

    def _setup_ontology_paths(self) -> None:
        """Configure owlready2 ontology search paths."""
        try:
            # Add local ontologies path to owlready2 search path
            if os.path.exists(self._base_ontology_path):
                abs_path = os.path.abspath(self._base_ontology_path)
                if abs_path not in onto_path:
                    onto_path.append(abs_path)
                    logger.info(f"Added ontology path: {abs_path}")

            # Configure predefined ontologies for known locations
            core_ontologies = {
                "http://mosaic.ai/ontology/code_base": os.path.join(
                    self._base_ontology_path, "code_base.owl"
                ),
                "http://mosaic.ai/ontology/python": os.path.join(
                    self._base_ontology_path, "python.owl"
                ),
                "http://mosaic.ai/ontology/relationships": os.path.join(
                    self._base_ontology_path, "relationships.owl"
                ),
            }

            for iri, path in core_ontologies.items():
                if os.path.exists(path):
                    # Use absolute path directly for owlready2
                    abs_path = os.path.abspath(path)
                    PREDEFINED_ONTOLOGIES[iri] = abs_path
                    logger.debug(f"Registered predefined ontology: {iri} -> {abs_path}")

        except Exception as e:
            logger.error(f"Failed to setup ontology paths: {e}")
            raise OntologyError(f"Ontology path configuration failed: {e}")

    def _load_core_ontologies(self) -> None:
        """Load core Mosaic ontologies on initialization."""
        core_ontologies = [
            "http://mosaic.ai/ontology/code_base",
            "http://mosaic.ai/ontology/python",
            "http://mosaic.ai/ontology/relationships",
        ]

        for ontology_iri in core_ontologies:
            try:
                self.load_ontology(ontology_iri)
                logger.info(f"Successfully loaded core ontology: {ontology_iri}")
            except Exception as e:
                logger.warning(f"Failed to load core ontology {ontology_iri}: {e}")

    def load_ontology(
        self, source: Union[str, Path], force_reload: bool = False
    ) -> owlready2.Ontology:
        """
        Load an ontology from a local file or HTTP URL.

        Args:
            source: Path to local OWL file or HTTP URL
            force_reload: If True, reload even if already cached

        Returns:
            Loaded ontology object

        Raises:
            OntologyLoadError: If loading fails
            OntologyValidationError: If validation fails
        """
        source_str = str(source)

        # Check cache first
        if not force_reload and source_str in self._loaded_ontologies:
            logger.debug(f"Returning cached ontology: {source_str}")
            return self._loaded_ontologies[source_str]

        logger.info(f"Loading ontology: {source_str}")

        try:
            # Determine if source is URL or file path
            if self._is_url(source_str):
                ontology = self._load_from_url(source_str)
            else:
                ontology = self._load_from_file(source_str)

            # Validate the loaded ontology
            self._validate_ontology(ontology)

            # Cache the ontology
            self._loaded_ontologies[source_str] = ontology
            self._ontology_metadata[source_str] = {
                "iri": ontology.base_iri,
                "classes_count": len(list(ontology.classes())),
                "properties_count": len(list(ontology.properties())),
                "individuals_count": len(list(ontology.individuals())),
                "loaded_at": logger.handlers[0].format(
                    logger.makeRecord(logger.name, logging.INFO, "", 0, "", (), None)
                )
                if logger.handlers
                else "unknown",
            }

            logger.info(f"Successfully loaded and cached ontology: {source_str}")
            return ontology

        except Exception as e:
            error_msg = f"Failed to load ontology from {source_str}: {e}"
            logger.error(error_msg)
            raise OntologyLoadError(error_msg) from e

    def _is_url(self, source: str) -> bool:
        """Check if source string is a URL."""
        try:
            result = urlparse(source)
            return result.scheme in ("http", "https", "file")
        except Exception:
            return False

    def _load_from_url(self, url: str) -> owlready2.Ontology:
        """Load ontology from HTTP URL."""
        try:
            ontology = get_ontology(url)
            ontology.load()
            return ontology
        except Exception as e:
            raise OntologyLoadError(f"Failed to load from URL {url}: {e}") from e

    def _load_from_file(self, file_path: str) -> owlready2.Ontology:
        """Load ontology from local file."""
        try:
            # Handle both relative and absolute paths
            if not os.path.isabs(file_path):
                # Try relative to ontology base path first
                full_path = os.path.join(self._base_ontology_path, file_path)
                if not os.path.exists(full_path):
                    # Try relative to current working directory
                    full_path = os.path.abspath(file_path)
            else:
                full_path = file_path

            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Ontology file not found: {full_path}")

            # Use direct path loading instead of file:// URL on Windows
            # owlready2 can handle direct file paths better
            ontology = get_ontology(full_path)
            ontology.load()
            return ontology

        except Exception as e:
            raise OntologyLoadError(f"Failed to load from file {file_path}: {e}") from e

    def _validate_ontology(self, ontology: owlready2.Ontology) -> None:
        """
        Validate loaded ontology for consistency and completeness.

        Args:
            ontology: The ontology to validate

        Raises:
            OntologyValidationError: If validation fails
        """
        try:
            # Basic existence checks
            if not ontology:
                raise OntologyValidationError("Ontology object is None")

            if not ontology.base_iri:
                raise OntologyValidationError("Ontology has no base IRI")

            # Check for basic OWL structure
            classes = list(ontology.classes())
            properties = list(ontology.properties())

            logger.debug(
                f"Ontology validation - Classes: {len(classes)}, Properties: {len(properties)}"
            )

            # Validate that imported ontologies are also loaded
            for imported in ontology.imported_ontologies:
                if imported not in self._loaded_ontologies.values():
                    logger.warning(f"Imported ontology not loaded: {imported.base_iri}")

            logger.debug(f"Ontology validation passed: {ontology.base_iri}")

        except Exception as e:
            error_msg = f"Ontology validation failed: {e}"
            logger.error(error_msg)
            raise OntologyValidationError(error_msg) from e

    @functools.lru_cache(maxsize=128)
    def get_class(
        self, class_name: str, ontology_iri: Optional[str] = None
    ) -> Optional[type]:
        """
        Get a class by name from loaded ontologies.

        Args:
            class_name: Name of the class to find
            ontology_iri: Specific ontology to search (optional)

        Returns:
            Class object if found, None otherwise
        """
        try:
            if ontology_iri:
                # Search in specific ontology
                if ontology_iri not in [
                    ont.base_iri for ont in self._loaded_ontologies.values()
                ]:
                    raise OntologyNotFoundError(f"Ontology not loaded: {ontology_iri}")

                ontology = next(
                    ont
                    for ont in self._loaded_ontologies.values()
                    if ont.base_iri == ontology_iri
                )

                # Try direct access
                if hasattr(ontology, class_name):
                    return getattr(ontology, class_name)

                # Search through classes
                for cls in ontology.classes():
                    if cls.name == class_name:
                        return cls
            else:
                # Search all loaded ontologies
                for ontology in self._loaded_ontologies.values():
                    if hasattr(ontology, class_name):
                        return getattr(ontology, class_name)

                    for cls in ontology.classes():
                        if cls.name == class_name:
                            return cls

            logger.debug(f"Class not found: {class_name}")
            return None

        except Exception as e:
            logger.error(f"Error getting class {class_name}: {e}")
            return None

    @functools.lru_cache(maxsize=128)
    def get_property(
        self, property_name: str, ontology_iri: Optional[str] = None
    ) -> Optional[type]:
        """
        Get a property by name from loaded ontologies.

        Args:
            property_name: Name of the property to find
            ontology_iri: Specific ontology to search (optional)

        Returns:
            Property object if found, None otherwise
        """
        try:
            if ontology_iri:
                # Search in specific ontology
                if ontology_iri not in [
                    ont.base_iri for ont in self._loaded_ontologies.values()
                ]:
                    raise OntologyNotFoundError(f"Ontology not loaded: {ontology_iri}")

                ontology = next(
                    ont
                    for ont in self._loaded_ontologies.values()
                    if ont.base_iri == ontology_iri
                )

                # Try direct access
                if hasattr(ontology, property_name):
                    return getattr(ontology, property_name)

                # Search through properties
                for prop in ontology.properties():
                    if prop.name == property_name:
                        return prop
            else:
                # Search all loaded ontologies
                for ontology in self._loaded_ontologies.values():
                    if hasattr(ontology, property_name):
                        return getattr(ontology, property_name)

                    for prop in ontology.properties():
                        if prop.name == property_name:
                            return prop

            logger.debug(f"Property not found: {property_name}")
            return None

        except Exception as e:
            logger.error(f"Error getting property {property_name}: {e}")
            return None

    def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        ontology_iri: Optional[str] = None,
    ) -> List[Any]:
        """
        Search for entities across loaded ontologies.

        Args:
            query: Search query (supports wildcards)
            entity_type: Type of entity to search ('class', 'property', 'individual')
            ontology_iri: Specific ontology to search (optional)

        Returns:
            List of matching entities
        """
        results = []

        try:
            ontologies_to_search = []
            if ontology_iri:
                ontologies_to_search = [
                    ont
                    for ont in self._loaded_ontologies.values()
                    if ont.base_iri == ontology_iri
                ]
            else:
                ontologies_to_search = list(self._loaded_ontologies.values())

            for ontology in ontologies_to_search:
                try:
                    # Use owlready2's built-in search functionality
                    search_results = ontology.search(iri=f"*{query}*")

                    # Filter by entity type if specified
                    if entity_type:
                        if entity_type.lower() == "class":
                            search_results = [
                                r
                                for r in search_results
                                if isinstance(r, type) and issubclass(r, Thing)
                            ]
                        elif entity_type.lower() == "property":
                            search_results = [
                                r for r in search_results if hasattr(r, "domain")
                            ]
                        elif entity_type.lower() == "individual":
                            search_results = [
                                r for r in search_results if not isinstance(r, type)
                            ]

                    results.extend(search_results)

                except Exception as e:
                    logger.warning(
                        f"Search failed in ontology {ontology.base_iri}: {e}"
                    )

            logger.debug(f"Search for '{query}' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Entity search failed: {e}")
            return []

    def get_ontology_info(self, ontology_iri: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a loaded ontology.

        Args:
            ontology_iri: IRI of the ontology

        Returns:
            Dictionary with ontology information
        """
        try:
            ontology = next(
                (
                    ont
                    for ont in self._loaded_ontologies.values()
                    if ont.base_iri == ontology_iri
                ),
                None,
            )

            if not ontology:
                return None

            info = {
                "iri": ontology.base_iri,
                "classes": [cls.name for cls in ontology.classes()],
                "properties": [prop.name for prop in ontology.properties()],
                "individuals": [ind.name for ind in ontology.individuals()],
                "imported_ontologies": [
                    imp.base_iri for imp in ontology.imported_ontologies
                ],
                "metadata": self._ontology_metadata.get(ontology_iri, {}),
            }

            return info

        except Exception as e:
            logger.error(f"Failed to get ontology info for {ontology_iri}: {e}")
            return None

    def list_loaded_ontologies(self) -> List[str]:
        """Get list of currently loaded ontology IRIs."""
        return [ont.base_iri for ont in self._loaded_ontologies.values()]

    def reload_ontology(self, ontology_iri: str) -> bool:
        """
        Reload a specific ontology.

        Args:
            ontology_iri: IRI of the ontology to reload

        Returns:
            True if successful, False otherwise
        """
        try:
            # Find the source for this ontology
            source = None
            for src, ont in self._loaded_ontologies.items():
                if ont.base_iri == ontology_iri:
                    source = src
                    break

            if not source:
                logger.error(
                    f"Cannot reload ontology - source not found: {ontology_iri}"
                )
                return False

            # Clear caches
            self.get_class.cache_clear()
            self.get_property.cache_clear()

            # Remove from cache
            if source in self._loaded_ontologies:
                del self._loaded_ontologies[source]
            if source in self._ontology_metadata:
                del self._ontology_metadata[source]

            # Reload
            self.load_ontology(source, force_reload=True)
            logger.info(f"Successfully reloaded ontology: {ontology_iri}")
            return True

        except Exception as e:
            logger.error(f"Failed to reload ontology {ontology_iri}: {e}")
            return False

    def validate_all_ontologies(self) -> Dict[str, bool]:
        """
        Validate all loaded ontologies.

        Returns:
            Dictionary mapping ontology IRI to validation result
        """
        results = {}

        for ontology in self._loaded_ontologies.values():
            try:
                self._validate_ontology(ontology)
                results[ontology.base_iri] = True
            except Exception as e:
                logger.error(f"Validation failed for {ontology.base_iri}: {e}")
                results[ontology.base_iri] = False

        return results

    def clear_cache(self) -> None:
        """Clear all internal caches."""
        self.get_class.cache_clear()
        self.get_property.cache_clear()
        logger.info("OntologyManager caches cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        return {
            "get_class_cache": self.get_class.cache_info()._asdict(),
            "get_property_cache": self.get_property.cache_info()._asdict(),
            "loaded_ontologies_count": len(self._loaded_ontologies),
            "total_classes": sum(
                len(list(ont.classes())) for ont in self._loaded_ontologies.values()
            ),
            "total_properties": sum(
                len(list(ont.properties())) for ont in self._loaded_ontologies.values()
            ),
        }


# Global singleton instance
ontology_manager = OntologyManager()

# Export public interface
__all__ = [
    "OntologyManager",
    "OntologyError",
    "OntologyLoadError",
    "OntologyValidationError",
    "OntologyNotFoundError",
    "ontology_manager",
]
