"""
RetrievalPlugin for Mosaic Query Server

Implements FR-5 (Hybrid Search), FR-6 (Graph-Based Code Analysis), and FR-7 (Candidate Aggregation)
using the OmniRAG pattern with unified Azure Cosmos DB backend.

ARCHITECTURAL NOTE: This plugin has been refactored to focus ONLY on querying operations.
Heavy ingestion operations have been moved to the separate Ingestion Service.

This plugin provides:
- Hybrid search combining vector and keyword search (QUERY ONLY)
- Graph-based code dependency analysis using embedded JSON (QUERY ONLY)
- Result aggregation and deduplication (QUERY ONLY)
- Real-time, lightweight operations for MCP server responsiveness
"""

import logging
from typing import List, Dict, Any, Optional

from semantic_kernel.plugin_definition import sk_function, sk_function_context_parameter
from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from azure.identity import DefaultAzureCredential
from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding

from ..config.settings import MosaicSettings
from ..models.base import Document, LibraryNode


logger = logging.getLogger(__name__)


class RetrievalPlugin:
    """
    Semantic Kernel plugin for QUERY-ONLY retrieval operations using OmniRAG pattern.

    Architectural Separation: This plugin handles ONLY lightweight query operations.
    Heavy ingestion operations are handled by the separate Ingestion Service.

    Implements unified retrieval from Azure Cosmos DB combining:
    - Vector search for semantic similarity (QUERY ONLY)
    - Keyword search for lexical matching (QUERY ONLY)
    - Graph traversal for dependency analysis (QUERY ONLY)
    - Result aggregation and deduplication (QUERY ONLY)

    Performance: Optimized for real-time MCP response times.
    """

    def __init__(self, settings: MosaicSettings):
        """Initialize the RetrievalPlugin."""
        self.settings = settings
        self.cosmos_client: Optional[CosmosClient] = None
        self.database = None
        self.knowledge_container = None
        self.embedding_service: Optional[AzureTextEmbedding] = None

    async def initialize(self) -> None:
        """Initialize Azure services and connections."""
        try:
            # Initialize Cosmos DB client with managed identity
            await self._initialize_cosmos()

            # Initialize embedding service
            await self._initialize_embedding_service()

            logger.info("RetrievalPlugin initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RetrievalPlugin: {e}")
            raise

    async def _initialize_cosmos(self) -> None:
        """Initialize Azure Cosmos DB connection with managed identity."""
        cosmos_config = self.settings.get_cosmos_config()

        # Always use managed identity
        credential = DefaultAzureCredential()
        self.cosmos_client = CosmosClient(cosmos_config["endpoint"], credential)

        # Get database and container references
        self.database = self.cosmos_client.get_database_client(
            cosmos_config["database_name"]
        )
        self.knowledge_container = self.database.get_container_client(
            cosmos_config["container_name"]
        )

    async def _initialize_embedding_service(self) -> None:
        """Initialize Azure embedding service with managed identity."""
        self.embedding_service = AzureTextEmbedding(
            deployment_name=self.settings.azure_openai_text_embedding_deployment_name,
            endpoint=self.settings.azure_openai_endpoint,
            service_id="retrieval_embedding",
        )

    @sk_function(
        description="Perform hybrid search using vector and keyword search",
        name="hybrid_search",
    )
    @sk_function_context_parameter(
        name="query", description="Search query string", type_="str"
    )
    async def hybrid_search(self, query: str) -> List[Document]:
        """
        Perform hybrid search combining vector and keyword search (FR-5).

        Args:
            query: Search query string

        Returns:
            List of relevant documents
        """
        try:
            # Generate embedding for the query
            query_embedding = await self._generate_embedding(query)

            # Perform vector search
            vector_results = await self._vector_search(query_embedding)

            # Perform keyword search
            keyword_results = await self._keyword_search(query)

            # Aggregate and deduplicate results (FR-7)
            aggregated_results = await self._aggregate_results(
                vector_results, keyword_results
            )

            logger.info(f"Hybrid search returned {len(aggregated_results)} results")
            return aggregated_results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise

    @sk_function(
        description="Query code graph relationships using embedded JSON pattern",
        name="query_code_graph",
    )
    @sk_function_context_parameter(
        name="library_id", description="Library identifier to query", type_="str"
    )
    @sk_function_context_parameter(
        name="relationship_type",
        description="Type of relationship (dependencies, dependents, etc.)",
        type_="str",
    )
    async def query_code_graph(
        self, library_id: str, relationship_type: str
    ) -> List[LibraryNode]:
        """
        Query code graph relationships using OmniRAG embedded JSON pattern (FR-6).

        Args:
            library_id: Identifier for the code library
            relationship_type: Type of relationship to query

        Returns:
            List of related library nodes
        """
        try:
            # Query the library node
            library_node = await self._get_library_node(library_id)

            if not library_node:
                return []

            # Get related nodes based on relationship type
            related_nodes = await self._get_related_nodes(
                library_node, relationship_type
            )

            logger.info(f"Code graph query returned {len(related_nodes)} related nodes")
            return related_nodes

        except Exception as e:
            logger.error(f"Code graph query failed: {e}")
            raise

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Azure embedding service."""
        try:
            result = await self.embedding_service.generate_embeddings([text])
            return result[0]

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    async def _vector_search(self, query_embedding: List[float]) -> List[Document]:
        """Perform vector similarity search."""
        try:
            # Cosmos DB vector search query
            vector_query = {
                "query": "SELECT * FROM c ORDER BY VectorDistance(c.embedding, @embedding) OFFSET 0 LIMIT @limit",
                "parameters": [
                    {"name": "@embedding", "value": query_embedding},
                    {"name": "@limit", "value": self.settings.max_search_results // 2},
                ],
            }

            items = list(
                self.knowledge_container.query_items(
                    query=vector_query["query"],
                    parameters=vector_query["parameters"],
                    enable_cross_partition_query=True,
                )
            )

            return [self._item_to_document(item) for item in items]

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise

    async def _keyword_search(self, query: str) -> List[Document]:
        """Perform keyword search using SQL query."""
        try:
            # Simple keyword search using SQL CONTAINS or LIKE
            keyword_query = {
                "query": "SELECT * FROM c WHERE CONTAINS(c.content, @query) OFFSET 0 LIMIT @limit",
                "parameters": [
                    {"name": "@query", "value": query},
                    {"name": "@limit", "value": self.settings.max_search_results // 2},
                ],
            }

            items = list(
                self.knowledge_container.query_items(
                    query=keyword_query["query"],
                    parameters=keyword_query["parameters"],
                    enable_cross_partition_query=True,
                )
            )

            return [self._item_to_document(item) for item in items]

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            raise

    async def _aggregate_results(
        self, vector_results: List[Document], keyword_results: List[Document]
    ) -> List[Document]:
        """Aggregate and deduplicate search results (FR-7)."""
        # Use document ID for deduplication
        seen_ids = set()
        aggregated = []

        # Process vector results first (typically higher quality)
        for doc in vector_results:
            if doc.id not in seen_ids:
                doc.metadata["source"] = "vector_search"
                aggregated.append(doc)
                seen_ids.add(doc.id)

        # Add keyword results that weren't already found
        for doc in keyword_results:
            if doc.id not in seen_ids:
                doc.metadata["source"] = "keyword_search"
                aggregated.append(doc)
                seen_ids.add(doc.id)

        # Sort by score if available
        aggregated.sort(key=lambda x: x.score or 0, reverse=True)

        return aggregated[: self.settings.max_search_results]

    async def _get_library_node(self, library_id: str) -> Optional[LibraryNode]:
        """Get library node by ID."""
        try:
            item = self.knowledge_container.read_item(
                item=library_id, partition_key=library_id
            )
            return self._item_to_library_node(item)

        except CosmosResourceNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Failed to get library node {library_id}: {e}")
            raise

    async def _get_related_nodes(
        self, library_node: LibraryNode, relationship_type: str
    ) -> List[LibraryNode]:
        """Get related nodes based on relationship type."""
        related_ids = []

        if relationship_type == "dependencies":
            related_ids = library_node.dependency_ids
        elif relationship_type == "dependents":
            related_ids = library_node.used_by_lib
        elif relationship_type == "all":
            related_ids = library_node.dependency_ids + library_node.used_by_lib

        # Fetch related nodes
        related_nodes = []
        for node_id in related_ids:
            node = await self._get_library_node(node_id)
            if node:
                related_nodes.append(node)

        return related_nodes

    def _item_to_document(self, item: Dict[str, Any]) -> Document:
        """Convert Cosmos DB item to Document model."""
        return Document(
            id=item.get("id", ""),
            content=item.get("content", ""),
            embedding=item.get("embedding"),
            metadata=item.get("metadata", {}),
            score=item.get("_score"),  # Cosmos DB similarity score
            source=item.get("source"),
            timestamp=item.get("timestamp"),
        )

    def _item_to_library_node(self, item: Dict[str, Any]) -> LibraryNode:
        """Convert Cosmos DB item to LibraryNode model."""
        return LibraryNode(
            id=item.get("id", ""),
            libtype=item.get("libtype", ""),
            libname=item.get("libname", ""),
            developers=item.get("developers", []),
            dependency_ids=item.get("dependency_ids", []),
            used_by_lib=item.get("used_by_lib", []),
            embedding=item.get("embedding"),
            metadata=item.get("metadata", {}),
        )

    async def get_status(self) -> Dict[str, Any]:
        """Get plugin status information."""
        return {
            "status": "active",
            "cosmos_connected": self.cosmos_client is not None,
            "embedding_service_connected": self.embedding_service is not None,
        }

    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        # Cosmos client cleanup is handled automatically
        # Embedding service cleanup is handled automatically
        logger.info("RetrievalPlugin cleanup completed")
