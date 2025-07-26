"""
GraphSynchronizer for maintaining RDF graphs synchronized with Cosmos DB data.

Provides efficient bulk and incremental synchronization of RDF triples from Cosmos DB
to in-memory RDF graphs, ensuring query consistency and performance for the SPARQL system.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

# Azure imports with error handling for missing dependencies
try:
    from azure.cosmos.aio import CosmosClient
    from azure.cosmos import PartitionKey
    from azure.identity.aio import DefaultAzureCredential

    AZURE_AVAILABLE = True
except ImportError:
    # Mock classes for development/testing without Azure SDK
    class CosmosClient:
        pass

    class PartitionKey:
        pass

    class DefaultAzureCredential:
        pass

    AZURE_AVAILABLE = False

from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS
import psutil

# Settings import with fallback
try:
    from ..config.settings import get_settings
except ImportError:
    # Fallback for testing
    def get_settings():
        class MockSettings:
            AZURE_COSMOS_DB_ACCOUNT_NAME = "test-account"

        return MockSettings()


@dataclass
class SyncMetrics:
    """Metrics for synchronization operations."""

    total_documents: int = 0
    processed_documents: int = 0
    total_triples: int = 0
    added_triples: int = 0
    errors: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    memory_usage_mb: float = 0.0

    @property
    def duration_seconds(self) -> float:
        """Calculate duration in seconds."""
        return (
            self.end_time - self.start_time if self.end_time > self.start_time else 0.0
        )

    @property
    def triples_per_second(self) -> float:
        """Calculate processing rate."""
        duration = self.duration_seconds
        return self.added_triples / duration if duration > 0 else 0.0


@dataclass
class SyncState:
    """Persistent state for incremental synchronization."""

    last_sync_timestamp: Optional[str] = None
    continuation_token: Optional[str] = None
    last_processed_id: Optional[str] = None
    total_triples_synced: int = 0
    sync_errors: Optional[List[str]] = None

    def __post_init__(self):
        if self.sync_errors is None:
            self.sync_errors = []


class GraphSynchronizerError(Exception):
    """Base exception for graph synchronization errors."""

    pass


class ConsistencyValidationError(GraphSynchronizerError):
    """Exception for graph consistency validation failures."""

    pass


class GraphSynchronizer:
    """
    Synchronizes RDF graphs with Cosmos DB data using change feed and bulk operations.

    Provides efficient synchronization for large datasets with consistency validation,
    performance monitoring, and incremental updates using Azure Cosmos DB change feed.
    """

    def __init__(self, cosmos_client: Optional[CosmosClient] = None):
        """
        Initialize the GraphSynchronizer.

        Args:
            cosmos_client: Optional Cosmos client. If None, creates a new client.
        """
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)

        # Azure Cosmos DB configuration
        self.cosmos_client = cosmos_client
        self.database_name = "mosaic-knowledge"
        self.container_name = "code-entities"

        # RDF Graph management
        self.graph = Graph()
        self.graph.bind("mosaic", "http://mosaic.ai/graph#")
        self.graph.bind("code", "http://mosaic.ai/ontology/code_base#")
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)

        # Synchronization state
        self.sync_state = SyncState()
        self.state_file_path = Path("graph_sync_state.json")

        # Performance configuration
        self.batch_size = 1000  # Documents per batch
        self.max_memory_mb = 2048  # Maximum memory usage
        self.consistency_check_interval = 10000  # Triples between consistency checks

        # Monitoring
        self.metrics = SyncMetrics()
        self.is_syncing = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize_cosmos_client()
        self._load_sync_state()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.cosmos_client:
            await self.cosmos_client.close()

    async def _initialize_cosmos_client(self):
        """Initialize Azure Cosmos DB client if not provided."""
        if self.cosmos_client is None:
            credential = DefaultAzureCredential()
            endpoint = f"https://{self.settings.AZURE_COSMOS_DB_ACCOUNT_NAME}.documents.azure.com:443/"
            self.cosmos_client = CosmosClient(endpoint, credential=credential)

        # Get database and container references
        self.database = self.cosmos_client.get_database_client(self.database_name)
        self.container = self.database.get_container_client(self.container_name)

    def _load_sync_state(self):
        """Load synchronization state from persistent storage."""
        if self.state_file_path.exists():
            try:
                with open(self.state_file_path, "r") as f:
                    state_data = json.load(f)
                    self.sync_state = SyncState(**state_data)
                self.logger.info(
                    f"Loaded sync state: {self.sync_state.total_triples_synced} triples synced"
                )
            except Exception as e:
                self.logger.error(f"Failed to load sync state: {e}")
                self.sync_state = SyncState()

    def _save_sync_state(self):
        """Save synchronization state to persistent storage."""
        try:
            state_data = {
                "last_sync_timestamp": self.sync_state.last_sync_timestamp,
                "continuation_token": self.sync_state.continuation_token,
                "last_processed_id": self.sync_state.last_processed_id,
                "total_triples_synced": self.sync_state.total_triples_synced,
                "sync_errors": (self.sync_state.sync_errors or [])[
                    -10:
                ],  # Keep last 10 errors
            }
            with open(self.state_file_path, "w") as f:
                json.dump(state_data, f, indent=2)
            self.logger.debug("Saved sync state")
        except Exception as e:
            self.logger.error(f"Failed to save sync state: {e}")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def _check_memory_limit(self):
        """Check if memory usage exceeds limit."""
        current_memory = self._get_memory_usage()
        if current_memory > self.max_memory_mb:
            raise GraphSynchronizerError(
                f"Memory usage {current_memory:.1f}MB exceeds limit {self.max_memory_mb}MB"
            )

    def _parse_rdf_triple_document(
        self, document: Dict[str, Any]
    ) -> Optional[Tuple[URIRef, URIRef, Any]]:
        """
        Parse RDF triple from Cosmos DB document.

        Args:
            document: Cosmos DB document containing RDF triple data

        Returns:
            Tuple of (subject, predicate, object) or None if invalid
        """
        try:
            if document.get("type") != "rdf_triple":
                return None

            subject_str = document.get("subject")
            predicate_str = document.get("predicate")
            object_str = document.get("object")

            if not all([subject_str, predicate_str, object_str]):
                return None

            # Parse subject (always URI)
            subject = URIRef(subject_str)

            # Parse predicate (always URI)
            predicate = URIRef(predicate_str)

            # Parse object (URI or Literal)
            if object_str.startswith("http://") or object_str.startswith("https://"):
                obj = URIRef(object_str)
            else:
                # Handle typed literals
                if "^^" in object_str:
                    value, datatype = object_str.split("^^", 1)
                    obj = Literal(value.strip('"'), datatype=URIRef(datatype))
                else:
                    obj = Literal(object_str.strip('"'))

            return (subject, predicate, obj)

        except Exception as e:
            self.logger.warning(
                f"Failed to parse RDF triple from document {document.get('id', 'unknown')}: {e}"
            )
            return None

    async def _process_document_batch(self, documents: List[Dict[str, Any]]) -> int:
        """
        Process a batch of documents and add RDF triples to graph.

        Args:
            documents: List of Cosmos DB documents

        Returns:
            Number of triples added
        """
        triples_added = 0

        for document in documents:
            try:
                triple = self._parse_rdf_triple_document(document)
                if triple:
                    self.graph.add(triple)
                    triples_added += 1

                    # Update processed document ID
                    self.sync_state.last_processed_id = document.get("id")

                self.metrics.processed_documents += 1

                # Check memory usage periodically
                if self.metrics.processed_documents % 100 == 0:
                    self._check_memory_limit()

            except Exception as e:
                self.metrics.errors += 1
                error_msg = (
                    f"Error processing document {document.get('id', 'unknown')}: {e}"
                )
                self.logger.error(error_msg)
                if self.sync_state.sync_errors is not None:
                    self.sync_state.sync_errors.append(error_msg)

        return triples_added

    async def sync_from_cosmos(
        self, max_documents: Optional[int] = None
    ) -> SyncMetrics:
        """
        Perform bulk synchronization of RDF triples from Cosmos DB.

        Args:
            max_documents: Maximum number of documents to process (None for all)

        Returns:
            Synchronization metrics
        """
        if self.is_syncing:
            raise GraphSynchronizerError("Synchronization already in progress")

        self.is_syncing = True
        self.metrics = SyncMetrics()
        self.metrics.start_time = time.time()

        try:
            self.logger.info("Starting bulk synchronization from Cosmos DB")

            # Query for RDF triple documents
            query = "SELECT * FROM c WHERE c.type = 'rdf_triple' ORDER BY c._ts"
            query_items = self.container.query_items(
                query=query,
                enable_cross_partition_query=True,
                max_item_count=self.batch_size,
            )

            documents_batch = []
            total_processed = 0

            async for document in query_items:
                documents_batch.append(document)
                self.metrics.total_documents += 1

                # Process batch when full
                if len(documents_batch) >= self.batch_size:
                    triples_added = await self._process_document_batch(documents_batch)
                    self.metrics.added_triples += triples_added

                    self.logger.info(
                        f"Processed batch: {len(documents_batch)} documents, "
                        f"{triples_added} triples added, "
                        f"total: {self.metrics.processed_documents}"
                    )

                    documents_batch = []
                    total_processed += self.batch_size

                    # Check limits
                    if max_documents and total_processed >= max_documents:
                        break

                    # Periodic consistency check
                    if (
                        self.metrics.added_triples % self.consistency_check_interval
                        == 0
                    ):
                        await self._validate_graph_consistency()

            # Process remaining documents
            if documents_batch:
                triples_added = await self._process_document_batch(documents_batch)
                self.metrics.added_triples += triples_added

            # Update sync state
            self.sync_state.last_sync_timestamp = datetime.now(timezone.utc).isoformat()
            self.sync_state.total_triples_synced += self.metrics.added_triples
            self._save_sync_state()

            # Final consistency validation
            await self._validate_graph_consistency()

            self.metrics.end_time = time.time()
            self.metrics.memory_usage_mb = self._get_memory_usage()
            self.metrics.total_triples = len(self.graph)

            self.logger.info(
                f"Bulk synchronization completed: {self.metrics.processed_documents} documents, "
                f"{self.metrics.added_triples} triples added, "
                f"{self.metrics.triples_per_second:.1f} triples/sec, "
                f"{self.metrics.memory_usage_mb:.1f}MB memory"
            )

            return self.metrics

        except Exception as e:
            self.metrics.errors += 1
            error_msg = f"Bulk synchronization failed: {e}"
            self.logger.error(error_msg)
            if self.sync_state.sync_errors is not None:
                self.sync_state.sync_errors.append(error_msg)
            raise GraphSynchronizerError(error_msg) from e

        finally:
            self.is_syncing = False

    async def incremental_sync(self) -> SyncMetrics:
        """
        Perform incremental synchronization using Cosmos DB change feed.

        Returns:
            Synchronization metrics
        """
        if self.is_syncing:
            raise GraphSynchronizerError("Synchronization already in progress")

        self.is_syncing = True
        self.metrics = SyncMetrics()
        self.metrics.start_time = time.time()

        try:
            self.logger.info("Starting incremental synchronization using change feed")

            # Determine starting point for change feed
            start_time = self.sync_state.last_sync_timestamp or "Beginning"
            continuation = self.sync_state.continuation_token

            # Configure change feed iterator
            if continuation:
                response_iterator = self.container.query_items_change_feed(
                    continuation=continuation
                )
            else:
                response_iterator = self.container.query_items_change_feed(
                    start_time=start_time
                )

            documents_processed = 0

            async for document in response_iterator:
                # Filter for RDF triple documents only
                if document.get("type") == "rdf_triple":
                    triple = self._parse_rdf_triple_document(document)
                    if triple:
                        self.graph.add(triple)
                        self.metrics.added_triples += 1

                documents_processed += 1
                self.metrics.processed_documents += 1

                # Save continuation token periodically
                if documents_processed % 100 == 0:
                    self.sync_state.continuation_token = (
                        self.container.client_connection.last_response_headers.get(
                            "etag"
                        )
                    )
                    self._save_sync_state()

                    self.logger.debug(f"Processed {documents_processed} changes")

                # Check memory usage
                if documents_processed % 100 == 0:
                    self._check_memory_limit()

            # Update sync state
            self.sync_state.continuation_token = (
                self.container.client_connection.last_response_headers.get("etag")
            )
            self.sync_state.last_sync_timestamp = datetime.now(timezone.utc).isoformat()
            self.sync_state.total_triples_synced += self.metrics.added_triples
            self._save_sync_state()

            self.metrics.end_time = time.time()
            self.metrics.memory_usage_mb = self._get_memory_usage()
            self.metrics.total_triples = len(self.graph)

            self.logger.info(
                f"Incremental synchronization completed: {self.metrics.processed_documents} changes, "
                f"{self.metrics.added_triples} triples added, "
                f"{self.metrics.memory_usage_mb:.1f}MB memory"
            )

            return self.metrics

        except Exception as e:
            self.metrics.errors += 1
            error_msg = f"Incremental synchronization failed: {e}"
            self.logger.error(error_msg)
            if self.sync_state.sync_errors is not None:
                self.sync_state.sync_errors.append(error_msg)
            raise GraphSynchronizerError(error_msg) from e

        finally:
            self.is_syncing = False

    async def _validate_graph_consistency(self) -> bool:
        """
        Validate graph consistency and repair if necessary.

        Returns:
            True if graph is consistent, False if repairs were needed
        """
        try:
            # Check basic graph integrity
            triple_count = len(self.graph)
            if triple_count == 0:
                self.logger.warning("Graph is empty")
                return True

            # Validate namespace bindings
            expected_namespaces = ["mosaic", "code", "rdf", "rdfs"]
            for prefix in expected_namespaces:
                if prefix not in dict(self.graph.namespaces()):
                    self.logger.warning(f"Missing namespace binding: {prefix}")

            # Run basic SPARQL validation queries
            validation_queries = [
                # Check for orphaned subjects
                "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o . FILTER NOT EXISTS { ?s a ?type } }",
                # Check for invalid URIs (basic validation)
                "SELECT ?s WHERE { ?s ?p ?o . FILTER(!isURI(?s) && !isBlank(?s)) }",
                # Check for malformed predicates
                "SELECT ?p WHERE { ?s ?p ?o . FILTER(!isURI(?p)) }",
            ]

            for query in validation_queries:
                try:
                    results = list(self.graph.query(query))
                    if query.startswith("SELECT (COUNT(*)"):
                        count = int(results[0][0]) if results else 0
                        if count > 0:
                            self.logger.warning(
                                f"Found {count} potential consistency issues"
                            )
                    elif results:
                        self.logger.warning(f"Found {len(results)} validation issues")
                except Exception as e:
                    self.logger.error(f"Validation query failed: {e}")

            self.logger.debug(
                f"Graph consistency check completed: {triple_count} triples"
            )
            return True

        except Exception as e:
            self.logger.error(f"Graph consistency validation failed: {e}")
            raise ConsistencyValidationError(
                f"Consistency validation failed: {e}"
            ) from e

    def get_sync_status(self) -> Dict[str, Any]:
        """
        Get current synchronization status and metrics.

        Returns:
            Dictionary containing sync status information
        """
        return {
            "is_syncing": self.is_syncing,
            "total_triples": len(self.graph),
            "last_sync_timestamp": self.sync_state.last_sync_timestamp,
            "total_triples_synced": self.sync_state.total_triples_synced,
            "recent_errors": (
                self.sync_state.sync_errors[-5:] if self.sync_state.sync_errors else []
            ),
            "memory_usage_mb": self._get_memory_usage(),
            "metrics": {
                "processed_documents": self.metrics.processed_documents,
                "added_triples": self.metrics.added_triples,
                "errors": self.metrics.errors,
                "duration_seconds": self.metrics.duration_seconds,
                "triples_per_second": self.metrics.triples_per_second,
            },
        }

    def clear_graph(self):
        """Clear the RDF graph and reset sync state."""
        self.graph = Graph()
        self.graph.bind("mosaic", "http://mosaic.ai/graph#")
        self.graph.bind("code", "http://mosaic.ai/ontology/code_base#")
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)

        self.sync_state = SyncState()
        self._save_sync_state()

        self.logger.info("Graph cleared and sync state reset")

    async def repair_graph_consistency(self) -> int:
        """
        Repair graph consistency issues.

        Returns:
            Number of issues repaired
        """
        repairs_made = 0

        try:
            # Remove duplicate triples (shouldn't happen with RDFLib but check anyway)
            len(self.graph)

            # Remove triples with invalid subjects, predicates, or objects
            invalid_triples = []

            for s, p, o in self.graph:
                # Check for invalid subject
                if not isinstance(s, (URIRef, BNode)):
                    invalid_triples.append((s, p, o))
                    continue

                # Check for invalid predicate (must be URI)
                if not isinstance(p, URIRef):
                    invalid_triples.append((s, p, o))
                    continue

                # Check for blank string literals
                if isinstance(o, Literal) and str(o).strip() == "":
                    invalid_triples.append((s, p, o))
                    continue

            # Remove invalid triples
            for triple in invalid_triples:
                self.graph.remove(triple)
                repairs_made += 1

            if repairs_made > 0:
                self.logger.info(f"Repaired {repairs_made} invalid triples")

            # Re-add namespace bindings if missing
            namespace_bindings = {
                "mosaic": "http://mosaic.ai/graph#",
                "code": "http://mosaic.ai/ontology/code_base#",
                "rdf": str(RDF),
                "rdfs": str(RDFS),
            }

            current_namespaces = dict(self.graph.namespaces())
            for prefix, namespace in namespace_bindings.items():
                if prefix not in current_namespaces:
                    self.graph.bind(prefix, namespace)
                    self.logger.debug(f"Re-added namespace binding: {prefix}")

            return repairs_made

        except Exception as e:
            self.logger.error(f"Graph repair failed: {e}")
            raise ConsistencyValidationError(f"Graph repair failed: {e}") from e

    def export_graph(
        self, format: str = "turtle", filepath: Optional[str] = None
    ) -> str:
        """
        Export graph to file or return as string.

        Args:
            format: RDF serialization format (turtle, n3, xml, json-ld)
            filepath: Optional file path to save to

        Returns:
            Serialized graph as string
        """
        try:
            serialized = self.graph.serialize(format=format)

            if filepath:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(serialized)
                self.logger.info(f"Graph exported to {filepath} ({format} format)")

            return serialized

        except Exception as e:
            self.logger.error(f"Graph export failed: {e}")
            raise GraphSynchronizerError(f"Graph export failed: {e}") from e

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get detailed performance metrics for monitoring.

        Returns:
            Dictionary containing performance metrics
        """
        try:
            # Run performance test query
            test_query = "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }"
            start_time = time.time()
            results = list(self.graph.query(test_query))
            query_time = time.time() - start_time

            triple_count = int(results[0][0]) if results else 0

            return {
                "graph_size": {
                    "total_triples": triple_count,
                    "memory_usage_mb": self._get_memory_usage(),
                    "storage_efficiency_triples_per_mb": triple_count
                    / max(self._get_memory_usage(), 1),
                },
                "query_performance": {
                    "simple_count_query_seconds": query_time,
                    "estimated_queries_per_second": 1 / max(query_time, 0.001),
                },
                "sync_metrics": {
                    "total_synced": self.sync_state.total_triples_synced,
                    "last_sync": self.sync_state.last_sync_timestamp,
                    "error_count": len(self.sync_state.sync_errors or []),
                    "is_healthy": len(self.sync_state.sync_errors or []) == 0,
                },
            }

        except Exception as e:
            self.logger.error(f"Performance metrics collection failed: {e}")
            return {"error": str(e)}


# Factory function for dependency injection
async def create_graph_synchronizer(
    cosmos_client: Optional[Any] = None,
) -> "GraphSynchronizer":
    """
    Factory function to create and initialize a GraphSynchronizer.

    Args:
        cosmos_client: Optional Cosmos client

    Returns:
        Initialized GraphSynchronizer instance
    """
    synchronizer = GraphSynchronizer(cosmos_client)
    await synchronizer._initialize_cosmos_client()
    return synchronizer
