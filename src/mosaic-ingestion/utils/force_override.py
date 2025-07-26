"""
Force Override Utility for Mosaic Ingestion Service

This module provides functionality to force clear all memory/nodes for a specific
repository and branch combination while preserving data from other branches and repositories.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

try:
    from azure.cosmos import CosmosClient, exceptions as cosmos_exceptions
    from azure.identity import DefaultAzureCredential
    AZURE_SDK_AVAILABLE = True
except ImportError:
    AZURE_SDK_AVAILABLE = False

logger = logging.getLogger(__name__)


class ForceOverrideManager:
    """
    Manages force override operations for clearing repository+branch specific data.
    
    Supports both Azure Cosmos DB and local development scenarios.
    """
    
    def __init__(self, cosmos_client: Optional[CosmosClient] = None, database_name: str = "MosaicKnowledge"):
        """Initialize the force override manager."""
        self.cosmos_client = cosmos_client
        self.database_name = database_name
        self.database = None
        self.containers = {}
        
        # Containers that may contain repository-specific data
        self.target_containers = [
            "knowledge",
            "memory", 
            "golden_nodes",
            "diagrams",
            "code_entities",
            "code_relationships",
            "repositories"
        ]
    
    async def initialize(self) -> bool:
        """Initialize Cosmos DB connection if not provided."""
        try:
            if not self.cosmos_client and AZURE_SDK_AVAILABLE:
                logger.info("🔗 Initializing Azure Cosmos DB connection...")
                credential = DefaultAzureCredential()
                endpoint = os.environ.get("AZURE_COSMOS_ENDPOINT")
                if not endpoint:
                    logger.error("❌ AZURE_COSMOS_ENDPOINT environment variable not set")
                    return False
                    
                self.cosmos_client = CosmosClient(endpoint, credential)
            
            if self.cosmos_client:
                self.database = self.cosmos_client.get_database_client(self.database_name)
                
                # Initialize container references
                for container_name in self.target_containers:
                    try:
                        self.containers[container_name] = self.database.get_container_client(container_name)
                        logger.debug(f"✅ Connected to container: {container_name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not connect to container {container_name}: {e}")
                
                logger.info(f"✅ Initialized {len(self.containers)} containers for force override")
                return True
            else:
                logger.warning("⚠️ No Cosmos DB client available - force override will be skipped")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize force override manager: {e}")
            return False
    
    async def confirm_force_override(self, repository_url: str, branch: str) -> bool:
        """
        Get user confirmation for force override operation.
        
        Args:
            repository_url: Repository URL that will be cleared
            branch: Branch name that will be cleared
            
        Returns:
            True if user confirms, False otherwise
        """
        print("\n" + "="*70)
        print("⚠️  FORCE OVERRIDE CONFIRMATION REQUIRED")
        print("="*70)
        print(f"🗂️  Repository: {repository_url}")
        print(f"🌿 Branch: {branch}")
        print("\n🔥 This operation will PERMANENTLY DELETE all data matching:")
        print(f"   - Repository URL: {repository_url}")
        print(f"   - Branch Name: {branch}")
        print("\n✅ Data that will be PRESERVED:")
        print(f"   - Same repository, different branches")
        print(f"   - Different repositories, same branch name")
        print("\n❌ Data that will be DELETED:")
        print(f"   - All entities, relationships, and memory for this exact repo+branch")
        print("="*70)
        
        try:
            response = input("\n❓ Type 'DELETE' to confirm force override: ").strip()
            if response == "DELETE":
                print("✅ Force override confirmed")
                return True
            else:
                print("❌ Force override cancelled")
                return False
        except (EOFError, KeyboardInterrupt):
            print("\n❌ Force override cancelled")
            return False
    
    async def execute_force_override(self, repository_url: str, branch: str, auto_confirm: bool = False) -> Dict[str, Any]:
        """
        Execute force override operation to clear all data for repository+branch.
        
        Args:
            repository_url: Repository URL to clear
            branch: Branch name to clear
            auto_confirm: Skip confirmation prompt (for automated scenarios)
            
        Returns:
            Summary of force override operation
        """
        start_time = datetime.utcnow()
        
        # Get confirmation unless auto-confirmed
        if not auto_confirm:
            confirmed = await self.confirm_force_override(repository_url, branch)
            if not confirmed:
                return {
                    "status": "cancelled",
                    "repository_url": repository_url,
                    "branch": branch,
                    "timestamp": start_time.isoformat(),
                    "reason": "User cancelled force override"
                }
        
        logger.info(f"🔥 Starting force override for {repository_url} (branch: {branch})")
        
        deletion_summary = {
            "repository_url": repository_url,
            "branch": branch,
            "containers_processed": 0,
            "total_items_deleted": 0,
            "container_details": {},
            "errors": [],
            "start_time": start_time.isoformat(),
            "status": "in_progress"
        }
        
        try:
            # Process each container
            for container_name, container_client in self.containers.items():
                try:
                    container_result = await self._clear_container(
                        container_client, container_name, repository_url, branch
                    )
                    
                    deletion_summary["containers_processed"] += 1
                    deletion_summary["total_items_deleted"] += container_result["items_deleted"]
                    deletion_summary["container_details"][container_name] = container_result
                    
                    logger.info(f"✅ Cleared {container_result['items_deleted']} items from {container_name}")
                    
                except Exception as container_error:
                    error_msg = f"Failed to clear container {container_name}: {container_error}"
                    logger.error(f"❌ {error_msg}")
                    deletion_summary["errors"].append(error_msg)
            
            # Finalize summary
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            deletion_summary.update({
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "status": "completed" if not deletion_summary["errors"] else "completed_with_errors"
            })
            
            logger.info(f"🎯 Force override completed in {duration:.2f}s")
            logger.info(f"📊 Total items deleted: {deletion_summary['total_items_deleted']}")
            
            return deletion_summary
            
        except Exception as e:
            error_msg = f"Force override failed: {e}"
            logger.error(f"❌ {error_msg}")
            deletion_summary.update({
                "status": "failed",
                "error": error_msg,
                "end_time": datetime.utcnow().isoformat()
            })
            return deletion_summary
    
    async def _clear_container(self, container_client, container_name: str, repository_url: str, branch: str) -> Dict[str, Any]:
        """
        Clear all items from a specific container matching repository+branch.
        
        Args:
            container_client: Cosmos DB container client
            container_name: Name of the container
            repository_url: Repository URL to match
            branch: Branch name to match
            
        Returns:
            Summary of container clearing operation
        """
        items_deleted = 0
        items_queried = 0
        
        try:
            # Build query to find matching items
            # Different containers may use different property names
            possible_queries = [
                # Standard repository and branch properties
                f"SELECT * FROM c WHERE c.repository_url = '{repository_url}' AND c.branch = '{branch}'",
                f"SELECT * FROM c WHERE c.repository_url = '{repository_url}' AND c.branch_name = '{branch}'",
                # Source repository variations
                f"SELECT * FROM c WHERE c.source_repository = '{repository_url}' AND c.branch = '{branch}'",
                f"SELECT * FROM c WHERE c.source_repository = '{repository_url}' AND c.branch_name = '{branch}'",
                # Nested repository information
                f"SELECT * FROM c WHERE c.repository.url = '{repository_url}' AND c.repository.branch = '{branch}'",
                # Golden nodes with repository context
                f"SELECT * FROM c WHERE c.content.repository_url = '{repository_url}' AND c.content.branch = '{branch}'"
            ]
            
            items_to_delete = []
            
            # Try each query pattern
            for query in possible_queries:
                try:
                    items = list(container_client.query_items(query=query, enable_cross_partition_query=True))
                    items_queried += len(items)
                    items_to_delete.extend(items)
                    
                    if items:
                        logger.debug(f"📋 Found {len(items)} items in {container_name} with query pattern")
                        
                except Exception as query_error:
                    logger.debug(f"🔍 Query pattern failed for {container_name}: {query_error}")
                    continue
            
            # Remove duplicates based on id
            unique_items = {item['id']: item for item in items_to_delete}
            items_to_delete = list(unique_items.values())
            
            # Delete items in batches
            batch_size = 100
            for i in range(0, len(items_to_delete), batch_size):
                batch = items_to_delete[i:i + batch_size]
                
                for item in batch:
                    try:
                        container_client.delete_item(item=item['id'], partition_key=item.get('id'))
                        items_deleted += 1
                        
                    except cosmos_exceptions.CosmosResourceNotFoundError:
                        logger.debug(f"⚠️ Item {item['id']} already deleted")
                    except Exception as delete_error:
                        logger.warning(f"⚠️ Failed to delete item {item['id']}: {delete_error}")
                
                # Log progress for large batches
                if len(items_to_delete) > batch_size:
                    logger.debug(f"🔄 Deleted {min(i + batch_size, len(items_to_delete))}/{len(items_to_delete)} items from {container_name}")
            
            return {
                "container_name": container_name,
                "items_queried": items_queried,
                "items_deleted": items_deleted,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Error clearing container {container_name}: {e}")
            return {
                "container_name": container_name,
                "items_queried": items_queried,
                "items_deleted": items_deleted,
                "success": False,
                "error": str(e)
            }


# Convenience function for direct usage
async def force_clear_repository_branch(repository_url: str, branch: str, cosmos_client: Optional[CosmosClient] = None, auto_confirm: bool = False) -> Dict[str, Any]:
    """
    Convenience function to force clear all data for a repository+branch combination.
    
    Args:
        repository_url: Repository URL to clear
        branch: Branch name to clear  
        cosmos_client: Optional Cosmos DB client (will create if None)
        auto_confirm: Skip confirmation prompt
        
    Returns:
        Summary of force override operation
    """
    manager = ForceOverrideManager(cosmos_client)
    
    if not await manager.initialize():
        return {
            "status": "failed",
            "error": "Could not initialize force override manager",
            "repository_url": repository_url,
            "branch": branch
        }
    
    return await manager.execute_force_override(repository_url, branch, auto_confirm)
