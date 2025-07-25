"""
Unified VectorSearchPlugin for Mosaic Query Server

Implements hierarchical vector search using Azure Cosmos DB's native VectorDistance function
and enhanced GoldenNode hierarchical relationships. Follows Microsoft's CosmosAIGraph patterns
and OmniRAG approach for unified retrieval.

Key Features:
- Native Azure Cosmos DB VectorDistance queries with quantizedFlat indexing
- Hierarchical filtering and traversal operations
- Support for parent-child relationships with materialized paths
- Integration with Semantic Kernel plugin architecture
- Real-time query operations optimized for MCP server responsiveness

Based on research findings:
- Azure Cosmos DB supports quantizedFlat indexing for 1536-dimension embeddings
- Vector paths must be excluded from general indexing for optimal performance
- Hierarchical queries can be efficiently implemented using composite indexes
- CosmosAIGraph OmniRAG pattern provides proven architecture for unified retrieval
"""

import logging
from typing import List, Dict, Any, Optional
from uuid import UUID

from semantic_kernel.plugin_definition import sk_function, sk_function_context_parameter
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding

from ..config.settings import MosaicSettings


logger = logging.getLogger(__name__)


class VectorSearchPlugin:
    """
    Unified Vector Search Plugin implementing hierarchical relationships and native Azure Cosmos DB vector search.

    Combines vector similarity search with hierarchical traversal operations using:
    - Native VectorDistance() function for optimal performance
    - Hierarchical parent-child relationships with UUID references
    - Materialized paths for efficient hierarchy traversal
    - Filtered vector search with hierarchical constraints
    - OmniRAG pattern for dynamic retrieval strategy selection

    Follows Microsoft's best practices for Azure Cosmos DB vector search:
    - quantizedFlat indexing for 1536-dimension Azure OpenAI embeddings
    - Vector paths excluded from general indexing
    - Composite indexes on (parent_id, hierarchy_level)
    - Native VectorDistance queries for optimal RU consumption
    """

    def __init__(self, settings: MosaicSettings):
        """Initialize the VectorSearchPlugin."""
        self.settings = settings
        self.cosmos_client: Optional[CosmosClient] = None
        self.database = None
        self.golden_nodes_container = None
        self.embedding_service: Optional[AzureTextEmbedding] = None

    async def initialize(self) -> None:
        """Initialize Azure services and connections."""
        try:
            # Initialize Cosmos DB client with managed identity
            await self._initialize_cosmos()

            # Initialize embedding service
            await self._initialize_embedding_service()

            logger.info("VectorSearchPlugin initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize VectorSearchPlugin: {e}")
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
        self.golden_nodes_container = self.database.get_container_client(
            "golden_nodes"  # Assuming Golden Nodes are stored in their own container
        )

    async def _initialize_embedding_service(self) -> None:
        """Initialize Azure embedding service with managed identity."""
        self.embedding_service = AzureTextEmbedding(
            deployment_name=self.settings.azure_openai_text_embedding_deployment_name,
            endpoint=self.settings.azure_openai_endpoint,
            service_id="vector_search_embedding",
        )

    @sk_function(
        description="Perform vector similarity search using native Azure Cosmos DB VectorDistance function",
        name="vector_similarity_search",
    )
    @sk_function_context_parameter(
        name="query",
        description="Text query to search for similar code entities",
    )
    @sk_function_context_parameter(
        name="limit",
        description="Maximum number of results to return (default: 10)",
    )
    @sk_function_context_parameter(
        name="similarity_threshold",
        description="Minimum similarity score threshold (default: 0.7)",
    )
    async def vector_similarity_search(
        self, query: str, limit: int = 10, similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using Azure Cosmos DB native VectorDistance function.

        Uses quantizedFlat indexing for optimal performance with 1536-dimension embeddings.
        Follows Microsoft's best practices for vector search queries.
        """
        try:
            # Generate embedding for query
            query_embedding = await self._generate_embedding(query)

            # Native VectorDistance query following Microsoft documentation patterns
            query_sql = """
            SELECT TOP @limit 
                c.id,
                c.code_entity.name,
                c.code_entity.entity_type,
                c.code_entity.language,
                c.code_entity.parent_id,
                c.code_entity.hierarchy_level,
                c.code_entity.hierarchy_path,
                c.file_context.file_path,
                c.ai_enrichment.summary,
                VectorDistance(c.embedding, @query_vector) AS similarity_score
            FROM c
            WHERE c.document_type = 'golden_node'
                AND c.embedding != null
                AND VectorDistance(c.embedding, @query_vector) > @similarity_threshold
            ORDER BY VectorDistance(c.embedding, @query_vector)
            """

            parameters = [
                {"name": "@limit", "value": limit},
                {"name": "@query_vector", "value": query_embedding},
                {"name": "@similarity_threshold", "value": similarity_threshold},
            ]

            # Execute query
            items = list(
                self.golden_nodes_container.query_items(
                    query=query_sql,
                    parameters=parameters,
                    enable_cross_partition_query=True,
                )
            )

            logger.info(
                f"Vector search returned {len(items)} results for query: {query}"
            )
            return items

        except Exception as e:
            logger.error(f"Vector similarity search failed: {e}")
            raise

    @sk_function(
        description="Search for similar entities within a specific hierarchical context",
        name="hierarchical_vector_search",
    )
    @sk_function_context_parameter(
        name="query",
        description="Text query to search for similar code entities",
    )
    @sk_function_context_parameter(
        name="parent_id",
        description="Parent entity UUID to limit search scope (optional)",
    )
    @sk_function_context_parameter(
        name="max_depth",
        description="Maximum hierarchy depth to search within (optional)",
    )
    @sk_function_context_parameter(
        name="entity_type",
        description="Filter by specific entity type (optional)",
    )
    async def hierarchical_vector_search(
        self,
        query: str,
        parent_id: Optional[str] = None,
        max_depth: Optional[int] = None,
        entity_type: Optional[str] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector search with hierarchical filtering constraints.

        Combines vector similarity with hierarchical relationships for context-aware search.
        Uses materialized paths and hierarchy levels for efficient filtering.
        """
        try:
            # Generate embedding for query
            query_embedding = await self._generate_embedding(query)

            # Build hierarchical filter conditions
            hierarchy_conditions = []
            parameters = [
                {"name": "@limit", "value": limit},
                {"name": "@query_vector", "value": query_embedding},
                {"name": "@similarity_threshold", "value": similarity_threshold},
            ]

            if parent_id:
                # Validate UUID format
                try:
                    UUID(parent_id)
                    hierarchy_conditions.append(
                        "(c.code_entity.parent_id = @parent_id OR ARRAY_CONTAINS(c.code_entity.hierarchy_path, @parent_id))"
                    )
                    parameters.append({"name": "@parent_id", "value": parent_id})
                except ValueError:
                    raise ValueError(f"Invalid parent_id UUID format: {parent_id}")

            if max_depth is not None:
                hierarchy_conditions.append(
                    "c.code_entity.hierarchy_level <= @max_depth"
                )
                parameters.append({"name": "@max_depth", "value": max_depth})

            if entity_type:
                hierarchy_conditions.append("c.code_entity.entity_type = @entity_type")
                parameters.append({"name": "@entity_type", "value": entity_type})

            # Build complete query with hierarchical filters
            hierarchy_filter = (
                " AND ".join(hierarchy_conditions) if hierarchy_conditions else "1=1"
            )

            query_sql = f"""
            SELECT TOP @limit 
                c.id,
                c.code_entity.name,
                c.code_entity.entity_type,
                c.code_entity.language,
                c.code_entity.parent_id,
                c.code_entity.hierarchy_level,
                c.code_entity.hierarchy_path,
                c.code_entity.full_hierarchy_path,
                c.file_context.file_path,
                c.ai_enrichment.summary,
                VectorDistance(c.embedding, @query_vector) AS similarity_score
            FROM c
            WHERE c.document_type = 'golden_node'
                AND c.embedding != null
                AND VectorDistance(c.embedding, @query_vector) > @similarity_threshold
                AND {hierarchy_filter}
            ORDER BY VectorDistance(c.embedding, @query_vector)
            """

            # Execute query
            items = list(
                self.golden_nodes_container.query_items(
                    query=query_sql,
                    parameters=parameters,
                    enable_cross_partition_query=True,
                )
            )

            logger.info(f"Hierarchical vector search returned {len(items)} results")
            return items

        except Exception as e:
            logger.error(f"Hierarchical vector search failed: {e}")
            raise

    @sk_function(
        description="Get all child entities of a specific node",
        name="get_children",
    )
    @sk_function_context_parameter(
        name="node_id",
        description="UUID of the parent node",
    )
    @sk_function_context_parameter(
        name="max_depth",
        description="Maximum depth to traverse (optional, default: 1 for direct children)",
    )
    async def get_children(
        self, node_id: str, max_depth: Optional[int] = 1
    ) -> List[Dict[str, Any]]:
        """
        Get all child entities of a specific node using hierarchical relationships.

        Uses parent_id references and hierarchy_path for efficient traversal.
        Supports both direct children (max_depth=1) and deep traversal.
        """
        try:
            # Validate UUID format
            UUID(node_id)

            # Query for children using parent_id or hierarchy_path
            if max_depth == 1:
                # Direct children only - more efficient query
                query_sql = """
                SELECT *
                FROM c
                WHERE c.document_type = 'golden_node'
                    AND c.code_entity.parent_id = @node_id
                ORDER BY c.code_entity.name
                """
                parameters = [{"name": "@node_id", "value": node_id}]
            else:
                # Get current node's hierarchy info first
                parent_info = await self._get_node_hierarchy_info(node_id)
                if not parent_info:
                    return []

                max_hierarchy_level = parent_info["hierarchy_level"] + (
                    max_depth or 999
                )

                query_sql = """
                SELECT *
                FROM c
                WHERE c.document_type = 'golden_node'
                    AND (c.code_entity.parent_id = @node_id 
                         OR ARRAY_CONTAINS(c.code_entity.hierarchy_path, @node_id))
                    AND c.code_entity.hierarchy_level <= @max_hierarchy_level
                ORDER BY c.code_entity.hierarchy_level, c.code_entity.name
                """
                parameters = [
                    {"name": "@node_id", "value": node_id},
                    {"name": "@max_hierarchy_level", "value": max_hierarchy_level},
                ]

            items = list(
                self.golden_nodes_container.query_items(
                    query=query_sql,
                    parameters=parameters,
                    enable_cross_partition_query=True,
                )
            )

            logger.info(f"Found {len(items)} children for node {node_id}")
            return items

        except ValueError as e:
            logger.error(f"Invalid node_id format: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to get children for node {node_id}: {e}")
            raise

    @sk_function(
        description="Get all parent entities of a specific node",
        name="get_parents",
    )
    @sk_function_context_parameter(
        name="node_id",
        description="UUID of the child node",
    )
    @sk_function_context_parameter(
        name="max_depth",
        description="Maximum depth to traverse up the hierarchy (optional)",
    )
    async def get_parents(
        self, node_id: str, max_depth: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all parent entities of a specific node using materialized paths.

        Uses hierarchy_path for efficient upward traversal.
        Returns parents in order from immediate parent to root.
        """
        try:
            # Get current node's hierarchy info
            node_info = await self._get_node_hierarchy_info(node_id)
            if not node_info or not node_info.get("hierarchy_path"):
                return []

            hierarchy_path = node_info["hierarchy_path"]

            # Limit depth if specified
            if max_depth is not None:
                hierarchy_path = hierarchy_path[-max_depth:]

            # Query for all parents in the hierarchy path
            query_sql = """
            SELECT *
            FROM c
            WHERE c.document_type = 'golden_node'
                AND ARRAY_CONTAINS(@hierarchy_path, c.id)
            ORDER BY c.code_entity.hierarchy_level DESC
            """

            parameters = [{"name": "@hierarchy_path", "value": hierarchy_path}]

            items = list(
                self.golden_nodes_container.query_items(
                    query=query_sql,
                    parameters=parameters,
                    enable_cross_partition_query=True,
                )
            )

            logger.info(f"Found {len(items)} parents for node {node_id}")
            return items

        except Exception as e:
            logger.error(f"Failed to get parents for node {node_id}: {e}")
            raise

    @sk_function(
        description="Get sibling entities at the same hierarchy level",
        name="get_siblings",
    )
    @sk_function_context_parameter(
        name="node_id",
        description="UUID of the node to find siblings for",
    )
    async def get_siblings(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Get all sibling entities at the same hierarchy level.

        Uses parent_id and hierarchy_level for efficient sibling identification.
        """
        try:
            # Get current node's hierarchy info
            node_info = await self._get_node_hierarchy_info(node_id)
            if not node_info:
                return []

            parent_id = node_info.get("parent_id")
            hierarchy_level = node_info.get("hierarchy_level", 0)

            if parent_id:
                # Has parent - find siblings with same parent
                query_sql = """
                SELECT *
                FROM c
                WHERE c.document_type = 'golden_node'
                    AND c.code_entity.parent_id = @parent_id
                    AND c.id != @node_id
                ORDER BY c.code_entity.name
                """
                parameters = [
                    {"name": "@parent_id", "value": parent_id},
                    {"name": "@node_id", "value": node_id},
                ]
            else:
                # Root level - find other root nodes
                query_sql = """
                SELECT *
                FROM c
                WHERE c.document_type = 'golden_node'
                    AND c.code_entity.hierarchy_level = @hierarchy_level
                    AND c.code_entity.parent_id = null
                    AND c.id != @node_id
                ORDER BY c.code_entity.name
                """
                parameters = [
                    {"name": "@hierarchy_level", "value": hierarchy_level},
                    {"name": "@node_id", "value": node_id},
                ]

            items = list(
                self.golden_nodes_container.query_items(
                    query=query_sql,
                    parameters=parameters,
                    enable_cross_partition_query=True,
                )
            )

            logger.info(f"Found {len(items)} siblings for node {node_id}")
            return items

        except Exception as e:
            logger.error(f"Failed to get siblings for node {node_id}: {e}")
            raise

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for the given text using Azure OpenAI."""
        try:
            if not self.embedding_service:
                raise RuntimeError("Embedding service not initialized")

            # Generate embedding using Semantic Kernel
            embedding_result = await self.embedding_service.generate_embeddings_async(
                [text]
            )
            if not embedding_result or len(embedding_result) == 0:
                raise RuntimeError("Failed to generate embedding")

            embedding = embedding_result[0]

            # Validate embedding dimensions (Azure OpenAI should return 1536 dimensions)
            if len(embedding) != 1536:
                logger.warning(
                    f"Unexpected embedding dimension: {len(embedding)}, expected 1536"
                )

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    async def _get_node_hierarchy_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get hierarchy information for a specific node."""
        try:
            query_sql = """
            SELECT 
                c.code_entity.parent_id,
                c.code_entity.hierarchy_level,
                c.code_entity.hierarchy_path
            FROM c
            WHERE c.document_type = 'golden_node' AND c.id = @node_id
            """

            parameters = [{"name": "@node_id", "value": node_id}]

            items = list(
                self.golden_nodes_container.query_items(
                    query=query_sql,
                    parameters=parameters,
                    enable_cross_partition_query=True,
                )
            )

            if items:
                return items[0]
            return None

        except Exception as e:
            logger.error(f"Failed to get hierarchy info for node {node_id}: {e}")
            return None
