"""
In-Memory RDF Graph Construction System
Implements OMR-P1-005: Build In-Memory RDF Graph Construction System

This module provides the GraphBuilder class for constructing and managing
large RDF graphs in memory, with efficient batch operations, SPARQL querying,
and memory monitoring capabilities.
"""

import logging
import psutil
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
import time

import rdflib
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.query import ResultRow
from rdflib.plugins.stores.memory import Memory

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    In-memory RDF graph construction and management system.

    Provides efficient batch operations, SPARQL querying, serialization,
    and memory monitoring for large RDF graphs.
    """

    def __init__(self, base_uri: Optional[str] = None):
        """
        Initialize the GraphBuilder with an in-memory RDF graph.

        Args:
            base_uri: Optional base URI for the graph namespace
        """
        # Initialize in-memory graph with Memory store
        self.graph = Graph(store=Memory())
        self.base_uri = base_uri or "http://mosaic.dev/graph/"
        self.namespace = Namespace(self.base_uri)

        # Bind common namespaces
        self._bind_namespaces()

        # Performance metrics
        self.triple_count = 0
        self.batch_operations = 0
        self.query_count = 0

        logger.info(f"GraphBuilder initialized with base URI: {self.base_uri}")

    def _bind_namespaces(self) -> None:
        """Bind common RDF namespaces to the graph."""
        namespaces = {
            "mosaic": self.namespace,
            "rdf": rdflib.RDF,
            "rdfs": rdflib.RDFS,
            "owl": rdflib.OWL,
            "xsd": rdflib.XSD,
            "skos": rdflib.SKOS,
            "dcterms": rdflib.DCTERMS,
            "foaf": rdflib.FOAF,
        }

        for prefix, namespace in namespaces.items():
            self.graph.bind(prefix, namespace)

    def add_triples(
        self, triples: List[Tuple[Any, Any, Any]], batch_size: int = 1000
    ) -> None:
        """
        Add triples to the graph using efficient batch operations.

        Args:
            triples: List of (subject, predicate, object) tuples
            batch_size: Number of triples to process in each batch
        """
        if not triples:
            logger.warning("No triples provided to add_triples")
            return

        start_time = time.time()
        processed = 0

        try:
            # Process in batches for memory efficiency
            for i in range(0, len(triples), batch_size):
                batch = triples[i : i + batch_size]

                # Convert to RDF terms if needed
                rdf_batch = []
                for subject, predicate, obj in batch:
                    s = self._to_rdf_term(subject)
                    p = self._to_rdf_term(predicate)
                    o = self._to_rdf_term(obj)
                    rdf_batch.append((s, p, o))

                # Use addN for efficient batch addition
                self.graph.addN([(s, p, o, self.graph) for s, p, o in rdf_batch])
                processed += len(batch)

                if processed % 5000 == 0:
                    logger.debug(f"Processed {processed}/{len(triples)} triples")

            self.triple_count += len(triples)
            self.batch_operations += 1

            elapsed = time.time() - start_time
            triples_per_sec = len(triples) / elapsed if elapsed > 0 else len(triples)
            logger.info(
                f"Added {len(triples)} triples in {elapsed:.2f}s "
                f"({triples_per_sec:.0f} triples/sec)"
            )

        except Exception as e:
            logger.error(f"Error adding triples: {e}")
            raise

    def _to_rdf_term(self, term: Any) -> Union[URIRef, Literal, BNode]:
        """
        Convert a term to appropriate RDF term type.

        Args:
            term: Input term (string, URI, etc.)

        Returns:
            Appropriate RDF term (URIRef, Literal, or BNode)
        """
        if isinstance(term, (URIRef, Literal, BNode)):
            return term
        elif isinstance(term, str):
            # Check if it looks like a URI
            if term.startswith(("http://", "https://", "file://", "urn:")):
                return URIRef(term)
            elif term.startswith(self.base_uri):
                return URIRef(term)
            else:
                # Treat as literal
                return Literal(term)
        else:
            # Convert other types to literals
            return Literal(str(term))

    def query(self, sparql_query: str) -> List[ResultRow]:
        """
        Execute a SPARQL query against the graph.

        Args:
            sparql_query: SPARQL query string

        Returns:
            List of query result rows
        """
        try:
            start_time = time.time()
            results = list(self.graph.query(sparql_query))
            elapsed = time.time() - start_time

            self.query_count += 1
            logger.debug(f"Query executed in {elapsed:.3f}s, {len(results)} results")

            return results

        except Exception as e:
            logger.error(f"SPARQL query error: {e}")
            logger.debug(f"Query: {sparql_query}")
            raise

    def serialize(
        self, format: str = "turtle", destination: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Serialize the graph to various RDF formats.

        Args:
            format: Output format ('turtle', 'nt', 'xml', 'json-ld')
            destination: Optional file path to write to

        Returns:
            Serialized graph as string
        """
        try:
            serialized = self.graph.serialize(format=format)

            if destination:
                output_path = Path(destination)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                if isinstance(serialized, bytes):
                    output_path.write_bytes(serialized)
                else:
                    output_path.write_text(serialized, encoding="utf-8")

                logger.info(f"Graph serialized to {output_path} ({format} format)")

            return (
                serialized
                if isinstance(serialized, str)
                else serialized.decode("utf-8")
            )

        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.

        Returns:
            Dictionary with memory usage metrics
        """
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "memory_mb": memory_info.rss / 1024 / 1024,
            "memory_percent": process.memory_percent(),
            "triple_count": len(self.graph),
            "triples_per_mb": (
                len(self.graph) / (memory_info.rss / 1024 / 1024)
                if memory_info.rss > 0
                else 0
            ),
        }

    def clear(self) -> None:
        """Clear all triples from the graph."""
        triple_count = len(self.graph)
        self.graph = Graph(store=Memory())
        self._bind_namespaces()

        self.triple_count = 0
        logger.info(f"Graph cleared ({triple_count} triples removed)")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive graph statistics.

        Returns:
            Dictionary with graph and performance statistics
        """
        memory_stats = self.get_memory_usage()

        return {
            "triple_count": len(self.graph),
            "batch_operations": self.batch_operations,
            "query_count": self.query_count,
            "memory_usage_mb": memory_stats["memory_mb"],
            "memory_percent": memory_stats["memory_percent"],
            "triples_per_mb": memory_stats["triples_per_mb"],
            "base_uri": self.base_uri,
            "bound_namespaces": list(self.graph.namespaces()),
        }

    def add_from_triple_generator(
        self, triple_generator_output: List[Dict[str, Any]]
    ) -> None:
        """
        Add triples from TripleGenerator output format.

        Args:
            triple_generator_output: List of triple dictionaries from TripleGenerator
        """
        triples = []

        for triple_dict in triple_generator_output:
            try:
                subject = triple_dict.get("subject")
                predicate = triple_dict.get("predicate")
                obj = triple_dict.get("object")

                if subject and predicate and obj:
                    triples.append((subject, predicate, obj))
                else:
                    logger.warning(f"Incomplete triple: {triple_dict}")

            except Exception as e:
                logger.error(f"Error processing triple from generator: {e}")
                continue

        if triples:
            self.add_triples(triples)
            logger.info(f"Added {len(triples)} triples from TripleGenerator output")
        else:
            logger.warning("No valid triples found in TripleGenerator output")


class GraphBuilderError(Exception):
    """Custom exception for GraphBuilder operations."""

    pass
