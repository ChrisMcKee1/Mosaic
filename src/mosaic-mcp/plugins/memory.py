"""
MemoryPlugin for Mosaic MCP Tool

Implements FR-9 (Unified Memory Interface), FR-10 (Multi-Layered Storage),
and FR-11 (LLM-Powered Consolidation) using OmniRAG pattern.

This plugin provides:
- Unified memory interface (save, retrieve, clear)
- Multi-layered storage (Redis for short-term, Cosmos DB for long-term)
- Automatic memory consolidation and importance scoring
"""

import logging
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
import asyncio

from semantic_kernel.plugin_definition import sk_function, sk_function_context_parameter
from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding
from azure.cosmos import CosmosClient
import redis.asyncio as redis
from azure.identity import DefaultAzureCredential

from ..config.settings import MosaicSettings
from ..models.base import MemoryEntry


logger = logging.getLogger(__name__)


class MemoryPlugin:
    """
    Semantic Kernel plugin for multi-layered memory management.

    Implements hybrid memory system with:
    - Redis for short-term conversational memory
    - Cosmos DB for long-term persistent memory
    - Automatic consolidation and importance scoring
    - Vector similarity search for memory retrieval
    """

    def __init__(self, settings: MosaicSettings):
        """Initialize the MemoryPlugin."""
        self.settings = settings
        self.credential = DefaultAzureCredential()

        # Azure service clients
        self.cosmos_client: Optional[CosmosClient] = None
        self.memory_container = None
        self.redis_client: Optional[redis.Redis] = None
        self.embedding_service: Optional[AzureTextEmbedding] = None

    async def initialize(self) -> None:
        """Initialize memory storage services."""
        try:
            # Initialize Cosmos DB for long-term memory
            await self._initialize_cosmos()

            # Initialize Redis for short-term memory
            await self._initialize_redis()

            # Initialize embedding service
            await self._initialize_embedding_service()

            logger.info("MemoryPlugin initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MemoryPlugin: {e}")
            raise

    async def _initialize_cosmos(self) -> None:
        """Initialize Azure Cosmos DB connection."""
        cosmos_config = self.settings.get_cosmos_config()

        self.cosmos_client = CosmosClient(cosmos_config["endpoint"], self.credential)

        database = self.cosmos_client.get_database_client(
            cosmos_config["database_name"]
        )
        self.memory_container = database.get_container_client(
            cosmos_config["memory_container"]
        )

    async def _initialize_redis(self) -> None:
        """Initialize Azure Redis connection."""
        redis_config = self.settings.get_redis_config()

        # Extract host from endpoint (remove protocol and port)
        endpoint = redis_config["endpoint"]
        if "://" in endpoint:
            endpoint = endpoint.split("://")[1]
        host = endpoint.split(":")[0]

        self.redis_client = redis.Redis(
            host=host,
            port=redis_config["port"],
            ssl=redis_config["ssl"],
            ssl_cert_reqs=None,
            decode_responses=True,
            socket_timeout=30,
            socket_connect_timeout=30,
            retry_on_timeout=True,
        )

        # Test connection
        await self.redis_client.ping()

    async def _initialize_embedding_service(self) -> None:
        """Initialize Azure embedding service."""
        self.embedding_service = AzureTextEmbedding(
            deployment_name=self.settings.azure_openai_text_embedding_deployment_name,
            endpoint=self.settings.azure_openai_endpoint,
            service_id="memory_embedding",
        )

    @sk_function(description="Save memory to multi-layered storage system", name="save")
    @sk_function_context_parameter(
        name="session_id", description="Session identifier", type_="str"
    )
    @sk_function_context_parameter(
        name="content", description="Memory content to save", type_="str"
    )
    @sk_function_context_parameter(
        name="memory_type",
        description="Type of memory (episodic, semantic, procedural)",
        type_="str",
    )
    async def save(
        self, session_id: str, content: str, memory_type: str
    ) -> Dict[str, Any]:
        """
        Save memory using multi-layered storage (FR-9, FR-10).

        Args:
            session_id: Session identifier
            content: Memory content to save
            memory_type: Type of memory (episodic, semantic, procedural)

        Returns:
            Dictionary with memory ID and confirmation
        """
        try:
            # Generate memory ID
            memory_id = self._generate_memory_id(session_id, content)

            # Generate embedding for the content
            embedding = await self._generate_embedding(content)

            # Calculate importance score
            importance_score = await self._calculate_importance(content, memory_type)

            # Create memory entry
            memory_entry = MemoryEntry(
                id=memory_id,
                session_id=session_id,
                type=memory_type,
                content=content,
                embedding=embedding,
                importance_score=importance_score,
                metadata={"source": "user_input", "storage_layer": "both"},
            )

            # Save to both Redis (short-term) and Cosmos DB (long-term)
            await self._save_to_redis(memory_entry)
            await self._save_to_cosmos(memory_entry)

            logger.info(f"Memory saved: {memory_id} (type: {memory_type})")

            return {
                "memory_id": memory_id,
                "session_id": session_id,
                "type": memory_type,
                "importance_score": importance_score,
                "timestamp": memory_entry.timestamp.isoformat(),
                "status": "saved",
            }

        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
            raise

    @sk_function(
        description="Retrieve relevant memories using hybrid search", name="retrieve"
    )
    @sk_function_context_parameter(
        name="session_id", description="Session identifier", type_="str"
    )
    @sk_function_context_parameter(
        name="query", description="Memory retrieval query", type_="str"
    )
    @sk_function_context_parameter(
        name="limit", description="Maximum number of memories to return", type_="int"
    )
    async def retrieve(
        self, session_id: str, query: str, limit: int = 10
    ) -> List[MemoryEntry]:
        """
        Retrieve relevant memories using hybrid search (FR-9, FR-10).

        Args:
            session_id: Session identifier
            query: Memory retrieval query
            limit: Maximum number of memories to return

        Returns:
            List of relevant memory entries
        """
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)

            # Search both Redis and Cosmos DB
            redis_memories = await self._search_redis(session_id, query, limit // 2)
            cosmos_memories = await self._search_cosmos(
                session_id, query_embedding, limit // 2
            )

            # Combine and deduplicate results
            all_memories = self._deduplicate_memories(redis_memories + cosmos_memories)

            # Sort by relevance and importance
            sorted_memories = sorted(
                all_memories,
                key=lambda m: (
                    m.importance_score,
                    self._calculate_text_similarity(query, m.content),
                ),
                reverse=True,
            )

            result = sorted_memories[:limit]

            logger.info(f"Retrieved {len(result)} memories for session {session_id}")
            return result

        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []

    @sk_function(description="Clear all memories for a session", name="clear")
    @sk_function_context_parameter(
        name="session_id", description="Session identifier", type_="str"
    )
    async def clear(self, session_id: str) -> Dict[str, Any]:
        """
        Clear all memories for a session (FR-9).

        Args:
            session_id: Session identifier

        Returns:
            Confirmation message
        """
        try:
            # Clear from Redis
            redis_cleared = await self._clear_redis_session(session_id)

            # Clear from Cosmos DB
            cosmos_cleared = await self._clear_cosmos_session(session_id)

            logger.info(f"Cleared memories for session {session_id}")

            return {
                "session_id": session_id,
                "redis_memories_cleared": redis_cleared,
                "cosmos_memories_cleared": cosmos_cleared,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "cleared",
            }

        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")
            raise

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text content."""
        result = await self.embedding_service.generate_embeddings([text])
        return result[0]

    def _generate_memory_id(self, session_id: str, content: str) -> str:
        """Generate unique memory ID."""
        data = f"{session_id}:{content}:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    async def _calculate_importance(self, content: str, memory_type: str) -> float:
        """Calculate importance score for memory content."""
        base_score = 0.5

        # Type-based scoring
        type_scores = {
            "semantic": 0.8,  # Generally important for long-term retention
            "procedural": 0.7,  # Important for task completion
            "episodic": 0.6,  # Context-dependent importance
        }

        type_score = type_scores.get(memory_type, base_score)

        # Content-based scoring (simple heuristics)
        content_lower = content.lower()

        # Keywords that suggest importance
        important_keywords = [
            "important",
            "critical",
            "key",
            "essential",
            "remember",
            "decision",
            "agreed",
            "confirmed",
            "final",
            "conclusion",
        ]

        keyword_bonus = sum(
            0.1 for keyword in important_keywords if keyword in content_lower
        )
        keyword_bonus = min(keyword_bonus, 0.3)  # Cap at 0.3

        # Length bonus (longer content might be more important)
        length_bonus = min(len(content) / 1000, 0.1)  # Up to 0.1 bonus

        final_score = min(type_score + keyword_bonus + length_bonus, 1.0)
        return final_score

    async def _save_to_redis(self, memory_entry: MemoryEntry) -> None:
        """Save memory to Redis for short-term storage."""
        key = f"memory:{memory_entry.session_id}:{memory_entry.id}"
        value = json.dumps(
            {
                "id": memory_entry.id,
                "content": memory_entry.content,
                "type": memory_entry.type,
                "importance_score": memory_entry.importance_score,
                "timestamp": memory_entry.timestamp.isoformat(),
                "metadata": memory_entry.metadata,
            }
        )

        # Set with TTL
        await self.redis_client.setex(key, self.settings.short_term_memory_ttl, value)

    async def _save_to_cosmos(self, memory_entry: MemoryEntry) -> None:
        """Save memory to Cosmos DB for long-term storage."""
        item = {
            "id": memory_entry.id,
            "sessionId": memory_entry.session_id,
            "type": memory_entry.type,
            "content": memory_entry.content,
            "embedding": memory_entry.embedding,
            "importanceScore": memory_entry.importance_score,
            "timestamp": memory_entry.timestamp.isoformat(),
            "metadata": memory_entry.metadata,
        }

        await asyncio.to_thread(self.memory_container.create_item, body=item)

    async def _search_redis(
        self, session_id: str, query: str, limit: int
    ) -> List[MemoryEntry]:
        """Search Redis for recent memories."""
        try:
            pattern = f"memory:{session_id}:*"
            keys = await self.redis_client.keys(pattern)

            memories = []
            for key in keys[: limit * 2]:  # Get more to filter
                value = await self.redis_client.get(key)
                if value:
                    data = json.loads(value)

                    # Simple text matching
                    if self._text_matches(query, data["content"]):
                        memory = MemoryEntry(
                            id=data["id"],
                            session_id=session_id,
                            type=data["type"],
                            content=data["content"],
                            importance_score=data["importance_score"],
                            timestamp=datetime.fromisoformat(data["timestamp"]),
                            metadata=data["metadata"],
                        )
                        memories.append(memory)

            return memories[:limit]

        except Exception as e:
            logger.error(f"Redis search failed: {e}")
            return []

    async def _search_cosmos(
        self, session_id: str, query_embedding: List[float], limit: int
    ) -> List[MemoryEntry]:
        """Search Cosmos DB using vector similarity."""
        try:
            query = {
                "query": """
                SELECT * FROM c 
                WHERE c.sessionId = @sessionId 
                ORDER BY VectorDistance(c.embedding, @embedding) 
                OFFSET 0 LIMIT @limit
                """,
                "parameters": [
                    {"name": "@sessionId", "value": session_id},
                    {"name": "@embedding", "value": query_embedding},
                    {"name": "@limit", "value": limit},
                ],
            }

            items = list(
                await asyncio.to_thread(
                    lambda: list(
                        self.memory_container.query_items(
                            query=query["query"],
                            parameters=query["parameters"],
                            enable_cross_partition_query=True,
                        )
                    )
                )
            )

            memories = []
            for item in items:
                memory = MemoryEntry(
                    id=item["id"],
                    session_id=item["sessionId"],
                    type=item["type"],
                    content=item["content"],
                    embedding=item.get("embedding"),
                    importance_score=item["importanceScore"],
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                    metadata=item.get("metadata", {}),
                )
                memories.append(memory)

            return memories

        except Exception as e:
            logger.error(f"Cosmos search failed: {e}")
            return []

    def _text_matches(self, query: str, content: str) -> bool:
        """Simple text matching for Redis search."""
        query_terms = set(query.lower().split())
        content_terms = set(content.lower().split())
        overlap = len(query_terms.intersection(content_terms))
        return overlap > 0

    def _calculate_text_similarity(self, query: str, content: str) -> float:
        """Calculate simple text similarity score."""
        query_terms = set(query.lower().split())
        content_terms = set(content.lower().split())

        if not query_terms:
            return 0.0

        overlap = len(query_terms.intersection(content_terms))
        return overlap / len(query_terms)

    def _deduplicate_memories(self, memories: List[MemoryEntry]) -> List[MemoryEntry]:
        """Remove duplicate memories by ID."""
        seen_ids = set()
        unique_memories = []

        for memory in memories:
            if memory.id not in seen_ids:
                unique_memories.append(memory)
                seen_ids.add(memory.id)

        return unique_memories

    async def _clear_redis_session(self, session_id: str) -> int:
        """Clear all Redis memories for a session."""
        pattern = f"memory:{session_id}:*"
        keys = await self.redis_client.keys(pattern)

        if keys:
            return await self.redis_client.delete(*keys)
        return 0

    async def _clear_cosmos_session(self, session_id: str) -> int:
        """Clear all Cosmos DB memories for a session."""
        query = "SELECT c.id FROM c WHERE c.sessionId = @sessionId"
        parameters = [{"name": "@sessionId", "value": session_id}]

        items = list(
            await asyncio.to_thread(
                lambda: list(
                    self.memory_container.query_items(
                        query=query,
                        parameters=parameters,
                        enable_cross_partition_query=True,
                    )
                )
            )
        )

        count = 0
        for item in items:
            try:
                await asyncio.to_thread(
                    self.memory_container.delete_item,
                    item=item["id"],
                    partition_key=item["id"],
                )
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete memory {item['id']}: {e}")

        return count

    async def get_status(self) -> Dict[str, Any]:
        """Get plugin status information."""
        redis_connected = False
        cosmos_connected = False

        try:
            if self.redis_client:
                await self.redis_client.ping()
                redis_connected = True
        except Exception:
            pass

        try:
            if self.memory_container:
                # Simple test query
                list(
                    self.memory_container.query_items(
                        query="SELECT TOP 1 c.id FROM c",
                        enable_cross_partition_query=True,
                    )
                )
                cosmos_connected = True
        except Exception:
            pass

        return {
            "status": "active",
            "redis_connected": redis_connected,
            "cosmos_connected": cosmos_connected,
            "embedding_service_connected": self.embedding_service is not None,
            "short_term_ttl": self.settings.short_term_memory_ttl,
            "importance_threshold": self.settings.memory_importance_threshold,
        }

    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        if self.redis_client:
            await self.redis_client.aclose()
            self.redis_client = None

        # Cosmos client cleanup is handled automatically
        logger.info("MemoryPlugin cleanup completed")
