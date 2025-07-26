"""
Specific Repository Implementations for CRUD-005

Implements branch-aware repositories for different Cosmos DB containers:
- KnowledgeRepository: For code entities with branch isolation
- RepositoryStateRepository: For commit state with branch support
- MemoryRepository: For memory storage with branch context

Each repository provides optimized operations for its specific data type
while maintaining consistent branch-aware partitioning and TTL management.

Author: Mosaic MCP Tool - CRUD-005 Implementation
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from azure.cosmos import CosmosClient
from .branch_aware_repository import BranchAwareRepository, TTLConfiguration
from ..models.golden_node import GoldenNode


logger = logging.getLogger(__name__)


class KnowledgeRepository(BranchAwareRepository):
    """
    Repository for code entities (Golden Nodes) with branch-aware partitioning.
    
    Handles:
    - Golden Node storage with branch isolation
    - Cross-branch conflict detection for merges
    - Entity relationship queries across branches
    - AI enrichment data with branch context
    """
    
    def get_entity_type(self) -> str:
        """Entity type for knowledge container."""
        return "golden_node"
    
    async def upsert_golden_node(
        self,
        golden_node: GoldenNode,
        ttl_override: Optional[int] = None
    ) -> GoldenNode:
        """
        Upsert Golden Node with branch-aware partitioning.
        
        Args:
            golden_node: Golden Node to upsert
            ttl_override: Override TTL for this node
            
        Returns:
            Updated Golden Node from Cosmos DB
        """
        try:
            # Convert to Cosmos document format
            doc = golden_node.to_cosmos_document()
            
            # Upsert with branch-aware partitioning
            result = await self.upsert_item(
                item=doc,
                repository_url=golden_node.git_context.repository_url,
                branch_name=golden_node.git_context.branch_name,
                ttl_override=ttl_override
            )
            
            # Convert back to Golden Node
            return GoldenNode.from_cosmos_document(result)
            
        except Exception as e:
            logger.error(f"Error upserting Golden Node {golden_node.id}: {e}")
            raise
    
    async def get_golden_node(
        self,
        node_id: str,
        repository_url: str,
        branch_name: str
    ) -> Optional[GoldenNode]:
        """
        Get Golden Node by ID with branch-aware partitioning.
        
        Args:
            node_id: Golden Node ID
            repository_url: Repository URL
            branch_name: Branch name
            
        Returns:
            Golden Node if found, None otherwise
        """
        try:
            doc = await self.get_item(node_id, repository_url, branch_name)
            return GoldenNode.from_cosmos_document(doc) if doc else None
            
        except Exception as e:
            logger.error(f"Error getting Golden Node {node_id}: {e}")
            raise
    
    async def query_entities_by_file(
        self,
        repository_url: str,
        branch_name: str,
        file_path: str
    ) -> List[GoldenNode]:
        """
        Query all entities in a specific file within a branch.
        
        Args:
            repository_url: Repository URL
            branch_name: Branch name
            file_path: File path to query
            
        Returns:
            List of Golden Nodes in the file
        """
        try:
            items = await self.query_branch_items(
                repository_url=repository_url,
                branch_name=branch_name,
                additional_filter="c.file_context.file_path = @file_path",
                parameters=[{"name": "@file_path", "value": file_path}]
            )
            
            return [GoldenNode.from_cosmos_document(item) for item in items]
            
        except Exception as e:
            logger.error(f"Error querying entities by file {file_path}: {e}")
            raise
    
    async def find_merge_conflicts(
        self,
        repository_url: str,
        source_branch: str,
        target_branch: str,
        file_paths: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find potential merge conflicts between branches.
        
        Args:
            repository_url: Repository URL
            source_branch: Source branch name
            target_branch: Target branch name
            file_paths: Specific file paths to check (optional)
            
        Returns:
            List of conflict information
        """
        try:
            # Build query for entities that exist in both branches
            additional_filter = """
                c.code_entity.name IN (
                    SELECT VALUE s.code_entity.name 
                    FROM s 
                    WHERE s.repository_url = @repository_url 
                    AND s.branch_name = @source_branch
                    AND s.entity_type = @entity_type
                )
            """
            
            # Add file path filter if specified
            if file_paths:
                file_placeholders = ", ".join([f"@file_{i}" for i in range(len(file_paths))])
                additional_filter += f" AND c.file_context.file_path IN ({file_placeholders})"
            
            # Query target branch for potential conflicts
            parameters = [
                {"name": "@repository_url", "value": repository_url},
                {"name": "@source_branch", "value": source_branch}
            ]
            
            if file_paths:
                for i, file_path in enumerate(file_paths):
                    parameters.append({"name": f"@file_{i}", "value": file_path})
            
            target_items = await self.query_branch_items(
                repository_url=repository_url,
                branch_name=target_branch,
                additional_filter=additional_filter,
                parameters=parameters
            )
            
            # Get corresponding source items
            source_items = await self.query_branch_items(
                repository_url=repository_url,
                branch_name=source_branch
            )
            
            # Build conflict detection logic
            conflicts = []
            source_by_name = {
                item["code_entity"]["name"]: item 
                for item in source_items
            }
            
            for target_item in target_items:
                entity_name = target_item["code_entity"]["name"]
                source_item = source_by_name.get(entity_name)
                
                if source_item:
                    # Check for content differences
                    if source_item["code_entity"]["content"] != target_item["code_entity"]["content"]:
                        conflicts.append({
                            "entity_name": entity_name,
                            "file_path": target_item["file_context"]["file_path"],
                            "conflict_type": "content_divergence",
                            "source_branch": source_branch,
                            "target_branch": target_branch,
                            "source_updated": source_item.get("updated_at"),
                            "target_updated": target_item.get("updated_at")
                        })
            
            logger.info(
                f"Found {len(conflicts)} potential merge conflicts between "
                f"{source_branch} and {target_branch}"
            )
            
            return conflicts
            
        except Exception as e:
            logger.error(f"Error finding merge conflicts: {e}")
            raise
    
    async def get_entity_relationships(
        self,
        repository_url: str,
        branch_name: str,
        entity_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all relationships for an entity within a branch.
        
        Args:
            repository_url: Repository URL
            branch_name: Branch name
            entity_id: Entity ID
            
        Returns:
            List of entity relationships
        """
        try:
            items = await self.query_branch_items(
                repository_url=repository_url,
                branch_name=branch_name,
                additional_filter="""
                    ARRAY_LENGTH(c.relationships) > 0 
                    AND (
                        ARRAY_CONTAINS(c.relationships, {"source_entity_id": @entity_id}, true)
                        OR ARRAY_CONTAINS(c.relationships, {"target_entity_id": @entity_id}, true)
                    )
                """,
                parameters=[{"name": "@entity_id", "value": entity_id}]
            )
            
            # Extract relevant relationships
            relationships = []
            for item in items:
                for relationship in item.get("relationships", []):
                    if (relationship.get("source_entity_id") == entity_id or 
                        relationship.get("target_entity_id") == entity_id):
                        relationships.append({
                            **relationship,
                            "parent_entity_id": item["id"]
                        })
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error getting entity relationships: {e}")
            raise


class RepositoryStateRepository(BranchAwareRepository):
    """
    Repository for commit state and repository metadata with branch support.
    
    Handles:
    - Commit state tracking per branch
    - Repository metadata with branch context
    - Branch creation and deletion tracking
    """
    
    def get_entity_type(self) -> str:
        """Entity type for repository state."""
        return "repository_state"
    
    async def upsert_commit_state(
        self,
        repository_url: str,
        branch_name: str,
        commit_sha: str,
        commit_message: str,
        author_name: str,
        author_email: str,
        commit_timestamp: datetime,
        processing_status: str = "completed"
    ) -> Dict[str, Any]:
        """
        Upsert commit state with branch-aware partitioning.
        
        Args:
            repository_url: Repository URL
            branch_name: Branch name
            commit_sha: Commit SHA
            commit_message: Commit message
            author_name: Author name
            author_email: Author email
            commit_timestamp: Commit timestamp
            processing_status: Processing status
            
        Returns:
            Upserted commit state document
        """
        try:
            # Generate consistent ID for commit state
            import hashlib
            state_id = hashlib.sha256(f"{repository_url}#{branch_name}".encode()).hexdigest()
            
            commit_state = {
                "id": state_id,
                "repository_url": repository_url,
                "branch_name": branch_name,
                "last_commit_sha": commit_sha,
                "last_commit_message": commit_message,
                "author_name": author_name,
                "author_email": author_email,
                "last_commit_timestamp": commit_timestamp.isoformat(),
                "processing_status": processing_status,
                "version": "2.0"  # Updated for CRUD-005
            }
            
            result = await self.upsert_item(
                item=commit_state,
                repository_url=repository_url,
                branch_name=branch_name
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error upserting commit state: {e}")
            raise
    
    async def get_commit_state(
        self,
        repository_url: str,
        branch_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get commit state for a specific branch.
        
        Args:
            repository_url: Repository URL
            branch_name: Branch name
            
        Returns:
            Commit state if found, None otherwise
        """
        try:
            import hashlib
            state_id = hashlib.sha256(f"{repository_url}#{branch_name}".encode()).hexdigest()
            
            return await self.get_item(state_id, repository_url, branch_name)
            
        except Exception as e:
            logger.error(f"Error getting commit state: {e}")
            raise
    
    async def list_repository_branches(
        self,
        repository_url: str
    ) -> List[Dict[str, Any]]:
        """
        List all branches for a repository.
        
        Args:
            repository_url: Repository URL
            
        Returns:
            List of branch information
        """
        try:
            items = await self.query_cross_branch_items(
                repository_url=repository_url
            )
            
            return items
            
        except Exception as e:
            logger.error(f"Error listing repository branches: {e}")
            raise


class MemoryRepository(BranchAwareRepository):
    """
    Repository for memory storage with branch context.
    
    Handles:
    - Memory consolidation per branch
    - Branch-aware memory retrieval
    - Cross-branch memory sharing for related entities
    """
    
    def get_entity_type(self) -> str:
        """Entity type for memory storage."""
        return "memory"
    
    async def store_memory(
        self,
        memory_id: str,
        repository_url: str,
        branch_name: str,
        memory_content: Dict[str, Any],
        importance_score: float = 0.5,
        ttl_override: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Store memory with branch-aware partitioning.
        
        Args:
            memory_id: Memory ID
            repository_url: Repository URL
            branch_name: Branch name
            memory_content: Memory content
            importance_score: Importance score (0.0-1.0)
            ttl_override: Override TTL
            
        Returns:
            Stored memory document
        """
        try:
            memory_doc = {
                "id": memory_id,
                "memory_content": memory_content,
                "importance_score": importance_score,
                "memory_type": memory_content.get("type", "generic"),
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            result = await self.upsert_item(
                item=memory_doc,
                repository_url=repository_url,
                branch_name=branch_name,
                ttl_override=ttl_override
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise
    
    async def retrieve_branch_memories(
        self,
        repository_url: str,
        branch_name: str,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories for a specific branch.
        
        Args:
            repository_url: Repository URL
            branch_name: Branch name
            memory_type: Filter by memory type
            min_importance: Minimum importance score
            
        Returns:
            List of memory documents
        """
        try:
            additional_filter = f"c.importance_score >= @min_importance"
            parameters = [{"name": "@min_importance", "value": min_importance}]
            
            if memory_type:
                additional_filter += " AND c.memory_type = @memory_type"
                parameters.append({"name": "@memory_type", "value": memory_type})
            
            items = await self.query_branch_items(
                repository_url=repository_url,
                branch_name=branch_name,
                additional_filter=additional_filter,
                parameters=parameters
            )
            
            return items
            
        except Exception as e:
            logger.error(f"Error retrieving branch memories: {e}")
            raise


# Repository Factory for creating repository instances

class RepositoryFactory:
    """Factory for creating branch-aware repository instances."""
    
    def __init__(
        self,
        cosmos_client: CosmosClient,
        database_name: str,
        ttl_config: Optional[TTLConfiguration] = None
    ):
        """
        Initialize repository factory.
        
        Args:
            cosmos_client: Cosmos DB client
            database_name: Database name
            ttl_config: TTL configuration
        """
        self.cosmos_client = cosmos_client
        self.database_name = database_name
        self.ttl_config = ttl_config or TTLConfiguration()
    
    def create_knowledge_repository(self, container_name: str = "knowledge") -> KnowledgeRepository:
        """Create knowledge repository instance."""
        return KnowledgeRepository(
            cosmos_client=self.cosmos_client,
            database_name=self.database_name,
            container_name=container_name,
            ttl_config=self.ttl_config
        )
    
    def create_repository_state_repository(self, container_name: str = "repositories") -> RepositoryStateRepository:
        """Create repository state repository instance."""
        return RepositoryStateRepository(
            cosmos_client=self.cosmos_client,
            database_name=self.database_name,
            container_name=container_name,
            ttl_config=self.ttl_config
        )
    
    def create_memory_repository(self, container_name: str = "memory") -> MemoryRepository:
        """Create memory repository instance."""
        return MemoryRepository(
            cosmos_client=self.cosmos_client,
            database_name=self.database_name,
            container_name=container_name,
            ttl_config=self.ttl_config
        )
