"""
OmniRAG Semantic Kernel Plugin for Mosaic MCP Tool

This plugin implements the OmniRAG pattern with Azure Cosmos DB:
- Database RAG: Direct queries for specific entities
- Vector RAG: Semantic similarity search
- Graph RAG: Relationship traversal and dependency analysis

Based on Microsoft's CosmosAIGraph OmniRAG approach.
"""

import logging
import json
from typing import List, Dict, Any
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding

logger = logging.getLogger(__name__)


class OmniRAGPlugin:
    """
    OmniRAG Plugin for intelligent query routing across:
    - Database queries (direct entity lookup)
    - Vector search (semantic similarity)
    - Graph traversal (relationship analysis)
    """

    def __init__(self, settings):
        """Initialize OmniRAG plugin with Azure Cosmos DB and embedding service."""
        self.settings = settings
        self.cosmos_client = None
        self.database = None
        self.embedding_service = None
        self._initialize_services()

    def _initialize_services(self):
        """Initialize Azure Cosmos DB and embedding services."""
        try:
            # Initialize Cosmos DB client with DefaultAzureCredential
            if self.settings.azure_cosmos_endpoint:
                credential = DefaultAzureCredential()
                self.cosmos_client = CosmosClient(
                    url=self.settings.azure_cosmos_endpoint, credential=credential
                )
                self.database = self.cosmos_client.get_database_client(
                    self.settings.azure_cosmos_database_name
                )
                logger.info("✅ Connected to Azure Cosmos DB")
            else:
                logger.warning(
                    "⚠️ No Cosmos DB endpoint configured - using simulated mode"
                )

            # Initialize Azure OpenAI embedding service
            if self.settings.azure_openai_endpoint:
                self.embedding_service = AzureTextEmbedding(
                    service_id="omnirag_embeddings",
                    deployment_name=self.settings.azure_openai_text_embedding_deployment_name,
                    endpoint=self.settings.azure_openai_endpoint,
                    api_version=self.settings.azure_openai_api_version,
                )
                logger.info("✅ Connected to Azure OpenAI embeddings")
            else:
                logger.warning("⚠️ No Azure OpenAI endpoint configured")

        except Exception as e:
            logger.error(f"❌ Failed to initialize OmniRAG services: {e}")

    def _determine_query_strategy(self, query: str) -> str:
        """
        Determine optimal RAG strategy based on query intent.

        OmniRAG Strategy Selection:
        - Database RAG: Direct entity queries ("what is Flask", "show me React")
        - Graph RAG: Relationship queries ("dependencies of X", "who uses Y")
        - Vector RAG: Semantic queries ("similar to X", "find related concepts")
        """
        query_lower = query.lower()

        # Graph RAG indicators
        graph_keywords = [
            "dependencies",
            "depends on",
            "uses",
            "used by",
            "relationships",
            "connects to",
            "imports",
            "imported by",
            "related to",
            "hierarchy",
            "children",
            "parents",
            "graph",
            "network",
            "architecture",
        ]

        # Database RAG indicators
        db_keywords = [
            "what is",
            "show me",
            "describe",
            "definition",
            "details about",
            "information about",
            "tell me about",
            "find entity",
            "get",
        ]

        # Vector RAG indicators
        vector_keywords = [
            "similar",
            "like",
            "resembles",
            "related concepts",
            "semantic search",
            "find similar",
            "similar to",
            "comparable",
            "equivalent",
        ]

        if any(keyword in query_lower for keyword in graph_keywords):
            return "graph_rag"
        elif any(keyword in query_lower for keyword in vector_keywords):
            return "vector_rag"
        else:
            return "database_rag"

    @kernel_function(
        description="Intelligent query routing using OmniRAG pattern",
        name="omnirag_query",
    )
    async def omnirag_query(self, query: str) -> str:
        """
        Main OmniRAG entry point - intelligently routes queries to optimal retrieval method.

        Args:
            query: User question to answer using OmniRAG pattern

        Returns:
            JSON response with results and strategy used
        """
        try:
            strategy = self._determine_query_strategy(query)
            logger.info(f"🧠 OmniRAG Strategy: {strategy} for query: '{query}'")

            if strategy == "database_rag":
                results = await self._database_rag(query)
            elif strategy == "vector_rag":
                results = await self._vector_rag(query)
            elif strategy == "graph_rag":
                results = await self._graph_rag(query)
            else:
                results = {"error": f"Unknown strategy: {strategy}"}

            return json.dumps(
                {
                    "query": query,
                    "strategy": strategy,
                    "results": results,
                    "omnirag_version": "1.0",
                },
                indent=2,
            )

        except Exception as e:
            logger.error(f"❌ OmniRAG query failed: {e}")
            return json.dumps({"error": str(e), "query": query, "strategy": "error"})

    async def _database_rag(self, query: str) -> Dict[str, Any]:
        """Database RAG: Direct entity lookup and retrieval."""
        if not self.cosmos_client:
            return self._simulated_database_response(query)

        try:
            # Query code entities container
            container = self.database.get_container_client("code_entities")

            # Extract potential entity names from query
            entity_keywords = self._extract_entity_keywords(query)

            entities = []
            for keyword in entity_keywords:
                query_sql = """
                SELECT * FROM c 
                WHERE CONTAINS(LOWER(c.name), @keyword) 
                OR CONTAINS(LOWER(c.content), @keyword)
                ORDER BY c.timestamp DESC
                OFFSET 0 LIMIT 10
                """

                items = list(
                    container.query_items(
                        query=query_sql,
                        parameters=[{"name": "@keyword", "value": keyword.lower()}],
                        enable_cross_partition_query=True,
                    )
                )
                entities.extend(items)

            return {
                "method": "database_rag",
                "entities_found": len(entities),
                "entities": entities[:5],  # Limit response size
                "keywords_searched": entity_keywords,
            }

        except Exception as e:
            logger.error(f"Database RAG failed: {e}")
            return {"error": str(e), "method": "database_rag"}

    async def _vector_rag(self, query: str) -> Dict[str, Any]:
        """Vector RAG: Semantic similarity search using embeddings."""
        if not self.cosmos_client or not self.embedding_service:
            return self._simulated_vector_response(query)

        try:
            # Generate embedding for query
            query_embedding = await self.embedding_service.generate_embeddings([query])

            # Vector search in Cosmos DB
            container = self.database.get_container_client("code_entities")

            # Use VectorDistance function for semantic search
            query_sql = """
            SELECT TOP 10 c.name, c.entity_type, c.content, c.file_path,
                   VectorDistance(c.embedding, @query_vector) AS similarity_score
            FROM c
            WHERE c.embedding != null
            ORDER BY VectorDistance(c.embedding, @query_vector)
            """

            items = list(
                container.query_items(
                    query=query_sql,
                    parameters=[{"name": "@query_vector", "value": query_embedding[0]}],
                    enable_cross_partition_query=True,
                )
            )

            return {
                "method": "vector_rag",
                "semantic_matches": len(items),
                "top_matches": items,
                "query_embedding_dims": len(query_embedding[0])
                if query_embedding
                else 0,
            }

        except Exception as e:
            logger.error(f"Vector RAG failed: {e}")
            return {"error": str(e), "method": "vector_rag"}

    async def _graph_rag(self, query: str) -> Dict[str, Any]:
        """Graph RAG: Relationship traversal and dependency analysis."""
        if not self.cosmos_client:
            return self._simulated_graph_response(query)

        try:
            # Query relationships container
            relationships_container = self.database.get_container_client(
                "code_relationships"
            )
            entities_container = self.database.get_container_client("code_entities")

            # Extract entity from query for relationship traversal
            entity_keywords = self._extract_entity_keywords(query)

            graph_data = []
            for keyword in entity_keywords:
                # Find relationships involving this entity
                rel_query = """
                SELECT * FROM r 
                WHERE CONTAINS(LOWER(r.source_name), @keyword) 
                OR CONTAINS(LOWER(r.target_name), @keyword)
                ORDER BY r.relationship_strength DESC
                """

                relationships = list(
                    relationships_container.query_items(
                        query=rel_query,
                        parameters=[{"name": "@keyword", "value": keyword.lower()}],
                        enable_cross_partition_query=True,
                    )
                )

                graph_data.extend(relationships)

            # Get related entities
            related_entities = set()
            for rel in graph_data:
                related_entities.add(rel.get("source_name", ""))
                related_entities.add(rel.get("target_name", ""))

            return {
                "method": "graph_rag",
                "relationships_found": len(graph_data),
                "related_entities": list(related_entities),
                "graph_data": graph_data[:10],  # Limit response size
                "keywords_searched": entity_keywords,
            }

        except Exception as e:
            logger.error(f"Graph RAG failed: {e}")
            return {"error": str(e), "method": "graph_rag"}

    def _extract_entity_keywords(self, query: str) -> List[str]:
        """Extract potential entity names from query text."""
        # Simple keyword extraction - could be enhanced with NLP
        import re

        # Remove common words and extract meaningful terms
        stop_words = {
            "what",
            "is",
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "show",
            "me",
            "find",
            "get",
            "tell",
            "about",
        }

        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]*\b", query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        return keywords[:3]  # Limit to top 3 keywords

    def _simulated_database_response(self, query: str) -> Dict[str, Any]:
        """Simulated database response when real Cosmos DB is not available."""
        return {
            "method": "database_rag",
            "mode": "simulated",
            "message": "Database RAG would search code entities for direct matches",
            "example_entities": [
                {"name": "semantic_kernel", "type": "module", "language": "python"},
                {"name": "FastMCP", "type": "class", "language": "python"},
                {"name": "OmniRAG", "type": "function", "language": "python"},
            ],
            "query": query,
        }

    def _simulated_vector_response(self, query: str) -> Dict[str, Any]:
        """Simulated vector response when real services are not available."""
        return {
            "method": "vector_rag",
            "mode": "simulated",
            "message": "Vector RAG would perform semantic similarity search",
            "example_matches": [
                {"name": "similar_concept_1", "similarity": 0.85},
                {"name": "related_function_2", "similarity": 0.78},
                {"name": "comparable_class_3", "similarity": 0.72},
            ],
            "query": query,
        }

    def _simulated_graph_response(self, query: str) -> Dict[str, Any]:
        """Simulated graph response when real Cosmos DB is not available."""
        return {
            "method": "graph_rag",
            "mode": "simulated",
            "message": "Graph RAG would traverse relationships and dependencies",
            "example_relationships": [
                {
                    "source": "FastMCP",
                    "target": "semantic_kernel",
                    "type": "depends_on",
                },
                {"source": "OmniRAG", "target": "Azure_Cosmos_DB", "type": "uses"},
                {"source": "Streamlit", "target": "OmniRAG", "type": "imports"},
            ],
            "query": query,
        }

    @kernel_function(
        description="Get filtered graph data based on query results",
        name="get_filtered_graph",
    )
    async def get_filtered_graph(self, query: str) -> str:
        """
        Get graph visualization data filtered by OmniRAG query results.

        Args:
            query: Query to filter graph nodes and edges

        Returns:
            JSON graph data for visualization
        """
        try:
            # First get OmniRAG results
            omnirag_result = await self.omnirag_query(query)
            result_data = json.loads(omnirag_result)

            # Extract relevant entities from results
            relevant_entities = set()

            if result_data.get("strategy") == "database_rag":
                for entity in result_data.get("results", {}).get("entities", []):
                    relevant_entities.add(entity.get("name", ""))

            elif result_data.get("strategy") == "graph_rag":
                relevant_entities.update(
                    result_data.get("results", {}).get("related_entities", [])
                )

            elif result_data.get("strategy") == "vector_rag":
                for match in result_data.get("results", {}).get("top_matches", []):
                    relevant_entities.add(match.get("name", ""))

            # Generate filtered graph data
            filtered_graph = self._generate_filtered_graph_data(relevant_entities)

            return json.dumps(filtered_graph, indent=2)

        except Exception as e:
            logger.error(f"Failed to get filtered graph: {e}")
            return json.dumps({"error": str(e)})

    def _generate_filtered_graph_data(self, relevant_entities: set) -> Dict[str, Any]:
        """Generate graph visualization data filtered by relevant entities."""
        if not relevant_entities:
            # Return full graph if no specific entities found
            return {
                "nodes": [],
                "links": [],
                "message": "No specific entities found - would show full graph",
                "filter_applied": False,
            }

        # This would normally query the actual graph database
        # For now, return simulated filtered data
        filtered_nodes = []
        filtered_links = []

        for entity in list(relevant_entities)[:10]:  # Limit nodes
            filtered_nodes.append(
                {"id": entity, "name": entity, "type": "filtered_entity", "group": 1}
            )

        # Add some connections between filtered nodes
        entities_list = list(relevant_entities)
        for i in range(min(len(entities_list) - 1, 5)):
            filtered_links.append(
                {
                    "source": entities_list[i],
                    "target": entities_list[i + 1],
                    "type": "filtered_relationship",
                }
            )

        return {
            "nodes": filtered_nodes,
            "links": filtered_links,
            "total_nodes": len(filtered_nodes),
            "total_links": len(filtered_links),
            "filter_applied": True,
            "filtered_entities": list(relevant_entities),
        }
