"""
Branch Lifecycle Management System

Implements comprehensive branch management operations for Git repository processing.
Provides branch creation, switching, deletion, metadata tracking, and conflict detection.

This addresses CRUD-004 acceptance criteria:
- Branch lifecycle management (create, switch, delete, merge/conflict detection)
- Branch metadata tracking (creation timestamp, parent commit, author)
- Integration with existing OntologyManager and ingestion workflows
- Support for branch isolation scenarios
- Comprehensive testing for branch lifecycle scenarios

Architecture:
- BranchLifecycleManager coordinates all branch operations
- Integrates with CommitStateManager for persistence
- Uses GitPython for all Git operations
- Stores branch metadata in Cosmos DB
- Provides hooks for ingestion workflow integration

Author: Mosaic MCP Tool - CRUD-004 Implementation
"""

import logging
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

import git
from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from azure.identity import DefaultAzureCredential

from .commit_state_manager import CommitStateManager, CommitState


logger = logging.getLogger(__name__)


@dataclass
class BranchMetadata:
    """Represents comprehensive metadata for a Git branch."""
    
    branch_name: str
    repository_url: str
    creation_timestamp: datetime
    parent_commit_sha: str
    author_name: str
    author_email: str
    current_commit_sha: str
    is_active: bool = False
    is_merged: bool = False
    merge_target: Optional[str] = None
    conflict_status: str = "none"  # none, detected, resolved
    processing_status: str = "active"  # active, archived, deleted


@dataclass
class BranchConflict:
    """Represents a detected branch conflict."""
    
    conflict_type: str  # merge, rebase, content
    conflicted_files: List[str]
    description: str
    severity: str  # low, medium, high, critical
    detected_at: datetime


class BranchLifecycleManager:
    """
    Manages Git branch lifecycle operations for Mosaic ingestion workflows.
    
    Features:
    - Complete branch CRUD operations (create, read, update, delete)
    - Branch metadata tracking and persistence
    - Conflict detection and reporting
    - Integration with CommitStateManager for audit trails
    - Support for branch isolation scenarios
    - GitPython-based branch operations
    """
    
    def __init__(
        self,
        commit_state_manager: CommitStateManager,
        cosmos_client: Optional[CosmosClient] = None,
        database_name: Optional[str] = None,
        branches_container_name: str = "branches"
    ):
        """
        Initialize BranchLifecycleManager.
        
        Args:
            commit_state_manager: CommitStateManager for persistence operations
            cosmos_client: Azure Cosmos DB client (optional, will use from commit_state_manager)
            database_name: Cosmos DB database name (optional, will use from commit_state_manager)
            branches_container_name: Container name for branch metadata
        """
        self.commit_state_manager = commit_state_manager
        
        # Use Cosmos DB configuration from CommitStateManager
        self.cosmos_client = cosmos_client or commit_state_manager.cosmos_client
        self.database_name = database_name or commit_state_manager.database_name
        self.branches_container_name = branches_container_name
        
        # Get database and container references
        self.database = self.cosmos_client.get_database_client(self.database_name)
        self.branches_container = self.database.get_container_client(
            branches_container_name
        )
        
        logger.info(
            f"BranchLifecycleManager initialized for database '{self.database_name}', "
            f"container '{branches_container_name}'"
        )
    
    @classmethod
    def from_settings(cls, settings) -> "BranchLifecycleManager":
        """
        Create BranchLifecycleManager from Mosaic settings configuration.
        
        Args:
            settings: MosaicSettings instance with Cosmos DB configuration
            
        Returns:
            Configured BranchLifecycleManager instance
        """
        # Create CommitStateManager first
        commit_state_manager = CommitStateManager.from_settings(settings)
        
        return cls(
            commit_state_manager=commit_state_manager
        )
    
    def _generate_branch_id(self, repository_url: str, branch_name: str) -> str:
        """
        Generate unique branch ID for Cosmos DB document.
        
        Args:
            repository_url: Git repository URL
            branch_name: Git branch name
            
        Returns:
            SHA-256 hash of repository_url + branch_name
        """
        content = f"branch#{repository_url}#{branch_name}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def create_branch(
        self,
        repo: git.Repo,
        repository_url: str,
        branch_name: str,
        parent_commit: Optional[Union[str, git.Commit]] = None,
        switch_to_branch: bool = True
    ) -> BranchMetadata:
        """
        Create a new Git branch with metadata tracking.
        
        Args:
            repo: GitPython repository object
            repository_url: Repository URL for tracking
            branch_name: Name for the new branch
            parent_commit: Parent commit (SHA or Commit object), defaults to HEAD
            switch_to_branch: Whether to switch to the new branch after creation
            
        Returns:
            BranchMetadata for the created branch
        """
        try:
            logger.info(f"Creating branch '{branch_name}' in {repository_url}")
            
            # Validate branch name doesn't already exist
            if branch_name in [head.name for head in repo.heads]:
                raise ValueError(f"Branch '{branch_name}' already exists")
            
            # Determine parent commit
            if parent_commit is None:
                parent_commit = repo.head.commit
            elif isinstance(parent_commit, str):
                parent_commit = repo.commit(parent_commit)
            
            # Create the branch using GitPython
            new_branch = repo.create_head(branch_name, parent_commit)
            
            # Switch to the branch if requested
            if switch_to_branch:
                repo.head.reference = new_branch
                logger.info(f"Switched to branch '{branch_name}'")
            
            # Create branch metadata
            now = datetime.now(timezone.utc)
            branch_metadata = BranchMetadata(
                branch_name=branch_name,
                repository_url=repository_url,
                creation_timestamp=now,
                parent_commit_sha=parent_commit.hexsha,
                author_name=repo.config_reader().get_value("user", "name", fallback="unknown"),
                author_email=repo.config_reader().get_value("user", "email", fallback="unknown"),
                current_commit_sha=new_branch.commit.hexsha,
                is_active=switch_to_branch,
                processing_status="active"
            )
            
            # Persist metadata to Cosmos DB
            await self._persist_branch_metadata(branch_metadata)
            
            # Update commit state tracking for the new branch
            await self.commit_state_manager.update_commit_state(
                repository_url=repository_url,
                branch_name=branch_name,
                current_commit_sha=new_branch.commit.hexsha,
                processing_status="active"
            )
            
            logger.info(
                f"Branch '{branch_name}' created successfully with metadata tracking"
            )
            
            return branch_metadata
            
        except Exception as e:
            logger.error(f"Error creating branch '{branch_name}': {e}")
            raise
    
    async def switch_branch(
        self,
        repo: git.Repo,
        repository_url: str,
        target_branch: str,
        create_if_missing: bool = False
    ) -> BranchMetadata:
        """
        Switch to a different Git branch.
        
        Args:
            repo: GitPython repository object
            repository_url: Repository URL for tracking
            target_branch: Branch name to switch to
            create_if_missing: Create branch if it doesn't exist
            
        Returns:
            BranchMetadata for the target branch
        """
        try:
            logger.info(f"Switching to branch '{target_branch}' in {repository_url}")
            
            # Check if target branch exists
            if target_branch not in [head.name for head in repo.heads]:
                if create_if_missing:
                    logger.info(f"Branch '{target_branch}' not found, creating it")
                    return await self.create_branch(
                        repo=repo,
                        repository_url=repository_url,
                        branch_name=target_branch,
                        switch_to_branch=True
                    )
                else:
                    raise ValueError(f"Branch '{target_branch}' does not exist")
            
            # Get current branch for metadata update
            current_branch = repo.active_branch.name if hasattr(repo.active_branch, 'name') else "HEAD"
            
            # Check for uncommitted changes
            if repo.is_dirty():
                logger.warning(f"Repository has uncommitted changes, switching may cause issues")
            
            # Switch to the target branch
            repo.head.reference = repo.heads[target_branch]
            repo.head.reset(index=True, working_tree=True)
            
            # Update branch metadata - mark old branch as inactive, new as active
            if current_branch != target_branch:
                await self._update_branch_active_status(repository_url, current_branch, False)
            
            await self._update_branch_active_status(repository_url, target_branch, True)
            
            # Get updated metadata
            branch_metadata = await self.get_branch_metadata(repository_url, target_branch)
            
            if not branch_metadata:
                # Create metadata for existing branch
                branch_metadata = await self._create_metadata_for_existing_branch(
                    repo, repository_url, target_branch
                )
            
            logger.info(f"Successfully switched to branch '{target_branch}'")
            
            return branch_metadata
            
        except Exception as e:
            logger.error(f"Error switching to branch '{target_branch}': {e}")
            raise
    
    async def delete_branch(
        self,
        repo: git.Repo,
        repository_url: str,
        branch_name: str,
        force: bool = False,
        archive_metadata: bool = True
    ) -> bool:
        """
        Delete a Git branch with metadata cleanup.
        
        Args:
            repo: GitPython repository object
            repository_url: Repository URL for tracking
            branch_name: Branch name to delete
            force: Force deletion even if not merged
            archive_metadata: Archive metadata instead of deleting
            
        Returns:
            True if deletion was successful
        """
        try:
            logger.info(f"Deleting branch '{branch_name}' in {repository_url}")
            
            # Validate branch exists
            if branch_name not in [head.name for head in repo.heads]:
                raise ValueError(f"Branch '{branch_name}' does not exist")
            
            # Prevent deletion of current branch
            if repo.active_branch.name == branch_name:
                raise ValueError(f"Cannot delete currently active branch '{branch_name}'")
            
            # Check if branch is merged (unless force=True)
            if not force:
                # Simple merge check - in production, more sophisticated checks may be needed
                logger.warning(f"Merge check for '{branch_name}' - using force deletion for now")
            
            # Delete the Git branch
            branch_to_delete = repo.heads[branch_name]
            repo.delete_head(branch_to_delete, force=force)
            
            # Update metadata
            if archive_metadata:
                await self._archive_branch_metadata(repository_url, branch_name)
            else:
                await self._delete_branch_metadata(repository_url, branch_name)
            
            logger.info(f"Branch '{branch_name}' deleted successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting branch '{branch_name}': {e}")
            raise
    
    async def detect_conflicts(
        self,
        repo: git.Repo,
        repository_url: str,
        source_branch: str,
        target_branch: str = "main"
    ) -> Optional[BranchConflict]:
        """
        Detect potential merge conflicts between branches.
        
        Args:
            repo: GitPython repository object
            repository_url: Repository URL for tracking
            source_branch: Source branch to merge from
            target_branch: Target branch to merge into
            
        Returns:
            BranchConflict object if conflicts detected, None otherwise
        """
        try:
            logger.info(
                f"Detecting conflicts between '{source_branch}' and '{target_branch}' in {repository_url}"
            )
            
            # Validate branches exist
            if source_branch not in [head.name for head in repo.heads]:
                raise ValueError(f"Source branch '{source_branch}' does not exist")
            if target_branch not in [head.name for head in repo.heads]:
                raise ValueError(f"Target branch '{target_branch}' does not exist")
            
            # Get commits from both branches
            source_commit = repo.heads[source_branch].commit
            target_commit = repo.heads[target_branch].commit
            
            # Find merge base
            merge_base = repo.merge_base(source_commit, target_commit)
            if not merge_base:
                return BranchConflict(
                    conflict_type="merge",
                    conflicted_files=[],
                    description=f"No common ancestor found between '{source_branch}' and '{target_branch}'",
                    severity="critical",
                    detected_at=datetime.now(timezone.utc)
                )
            
            # Get diffs from merge base to both branches
            source_diff = list(merge_base[0].diff(source_commit))
            target_diff = list(merge_base[0].diff(target_commit))
            
            # Find files modified in both branches
            source_files = {diff.b_path for diff in source_diff if diff.b_path}
            target_files = {diff.b_path for diff in target_diff if diff.b_path}
            
            conflicted_files = list(source_files.intersection(target_files))
            
            if conflicted_files:
                severity = "high" if len(conflicted_files) > 10 else "medium" if len(conflicted_files) > 3 else "low"
                
                return BranchConflict(
                    conflict_type="merge",
                    conflicted_files=conflicted_files,
                    description=f"Potential conflicts in {len(conflicted_files)} files between '{source_branch}' and '{target_branch}'",
                    severity=severity,
                    detected_at=datetime.now(timezone.utc)
                )
            
            logger.info(f"No conflicts detected between '{source_branch}' and '{target_branch}'")
            return None
            
        except Exception as e:
            logger.error(f"Error detecting conflicts: {e}")
            # Return error as critical conflict
            return BranchConflict(
                conflict_type="error",
                conflicted_files=[],
                description=f"Error during conflict detection: {str(e)}",
                severity="critical",
                detected_at=datetime.now(timezone.utc)
            )
    
    async def list_branches(
        self,
        repo: git.Repo,
        repository_url: str,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List all branches with optional metadata.
        
        Args:
            repo: GitPython repository object
            repository_url: Repository URL for tracking
            include_metadata: Whether to include stored metadata
            
        Returns:
            List of branch information dictionaries
        """
        try:
            logger.info(f"Listing branches for {repository_url}")
            
            branches = []
            current_branch = repo.active_branch.name if hasattr(repo.active_branch, 'name') else None
            
            for head in repo.heads:
                branch_info = {
                    "name": head.name,
                    "commit_sha": head.commit.hexsha,
                    "commit_message": head.commit.message.strip(),
                    "commit_date": head.commit.committed_datetime,
                    "is_current": head.name == current_branch
                }
                
                if include_metadata:
                    metadata = await self.get_branch_metadata(repository_url, head.name)
                    if metadata:
                        branch_info.update({
                            "creation_timestamp": metadata.creation_timestamp,
                            "parent_commit_sha": metadata.parent_commit_sha,
                            "author_name": metadata.author_name,
                            "is_active": metadata.is_active,
                            "processing_status": metadata.processing_status
                        })
                
                branches.append(branch_info)
            
            logger.info(f"Found {len(branches)} branches")
            return branches
            
        except Exception as e:
            logger.error(f"Error listing branches: {e}")
            raise
    
    async def get_branch_metadata(
        self,
        repository_url: str,
        branch_name: str
    ) -> Optional[BranchMetadata]:
        """
        Get stored metadata for a specific branch.
        
        Args:
            repository_url: Repository URL
            branch_name: Branch name
            
        Returns:
            BranchMetadata if exists, None otherwise
        """
        branch_id = self._generate_branch_id(repository_url, branch_name)
        
        try:
            logger.debug(f"Retrieving metadata for branch {repository_url}#{branch_name}")
            
            # Query Cosmos DB for branch metadata
            item = await self._get_branch_item(branch_id)
            
            if not item:
                logger.debug(f"No metadata found for branch {repository_url}#{branch_name}")
                return None
            
            metadata = BranchMetadata(
                branch_name=item["branch_name"],
                repository_url=item["repository_url"],
                creation_timestamp=datetime.fromisoformat(item["creation_timestamp"]),
                parent_commit_sha=item["parent_commit_sha"],
                author_name=item["author_name"],
                author_email=item["author_email"],
                current_commit_sha=item["current_commit_sha"],
                is_active=item.get("is_active", False),
                is_merged=item.get("is_merged", False),
                merge_target=item.get("merge_target"),
                conflict_status=item.get("conflict_status", "none"),
                processing_status=item.get("processing_status", "active")
            )
            
            logger.debug(f"Retrieved metadata for branch {branch_name}")
            return metadata
            
        except CosmosResourceNotFoundError:
            logger.debug(f"No metadata found for branch {repository_url}#{branch_name}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving branch metadata: {e}")
            raise
    
    # Private helper methods
    
    async def _persist_branch_metadata(self, metadata: BranchMetadata) -> None:
        """Persist branch metadata to Cosmos DB."""
        branch_id = self._generate_branch_id(metadata.repository_url, metadata.branch_name)
        
        try:
            branch_doc = {
                "id": branch_id,
                "branch_name": metadata.branch_name,
                "repository_url": metadata.repository_url,
                "creation_timestamp": metadata.creation_timestamp.isoformat(),
                "parent_commit_sha": metadata.parent_commit_sha,
                "author_name": metadata.author_name,
                "author_email": metadata.author_email,
                "current_commit_sha": metadata.current_commit_sha,
                "is_active": metadata.is_active,
                "is_merged": metadata.is_merged,
                "merge_target": metadata.merge_target,
                "conflict_status": metadata.conflict_status,
                "processing_status": metadata.processing_status,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                # Metadata for Cosmos DB partitioning and queries
                "partition_key": branch_id,
                "entity_type": "branch_metadata",
                "version": "1.0"
            }
            
            await self._upsert_branch_item(branch_doc)
            
            logger.debug(f"Persisted metadata for branch {metadata.branch_name}")
            
        except Exception as e:
            logger.error(f"Error persisting branch metadata: {e}")
            raise
    
    async def _update_branch_active_status(
        self,
        repository_url: str,
        branch_name: str,
        is_active: bool
    ) -> None:
        """Update the active status of a branch."""
        metadata = await self.get_branch_metadata(repository_url, branch_name)
        
        if metadata:
            metadata.is_active = is_active
            await self._persist_branch_metadata(metadata)
    
    async def _create_metadata_for_existing_branch(
        self,
        repo: git.Repo,
        repository_url: str,
        branch_name: str
    ) -> BranchMetadata:
        """Create metadata for an existing branch that doesn't have metadata."""
        try:
            branch_head = repo.heads[branch_name]
            commit = branch_head.commit
            
            # Try to find parent commit (first parent)
            parent_commit_sha = commit.parents[0].hexsha if commit.parents else commit.hexsha
            
            metadata = BranchMetadata(
                branch_name=branch_name,
                repository_url=repository_url,
                creation_timestamp=datetime.now(timezone.utc),  # Approximate
                parent_commit_sha=parent_commit_sha,
                author_name=commit.author.name,
                author_email=commit.author.email,
                current_commit_sha=commit.hexsha,
                is_active=True,
                processing_status="active"
            )
            
            await self._persist_branch_metadata(metadata)
            
            logger.info(f"Created metadata for existing branch '{branch_name}'")
            return metadata
            
        except Exception as e:
            logger.error(f"Error creating metadata for existing branch: {e}")
            raise
    
    async def _archive_branch_metadata(self, repository_url: str, branch_name: str) -> None:
        """Archive branch metadata instead of deleting."""
        metadata = await self.get_branch_metadata(repository_url, branch_name)
        
        if metadata:
            metadata.processing_status = "archived"
            metadata.is_active = False
            await self._persist_branch_metadata(metadata)
            
            logger.info(f"Archived metadata for branch '{branch_name}'")
    
    async def _delete_branch_metadata(self, repository_url: str, branch_name: str) -> None:
        """Delete branch metadata from Cosmos DB."""
        branch_id = self._generate_branch_id(repository_url, branch_name)
        
        try:
            await self._delete_branch_item(branch_id)
            logger.info(f"Deleted metadata for branch '{branch_name}'")
            
        except CosmosResourceNotFoundError:
            logger.warning(f"No metadata found to delete for branch '{branch_name}'")
        except Exception as e:
            logger.error(f"Error deleting branch metadata: {e}")
            raise
    
    async def _get_branch_item(self, branch_id: str) -> Optional[Dict[str, Any]]:
        """Get branch item from Cosmos DB."""
        try:
            response = self.branches_container.read_item(
                item=branch_id,
                partition_key=branch_id
            )
            return response
        except CosmosResourceNotFoundError:
            return None
    
    async def _upsert_branch_item(self, item: Dict[str, Any]) -> None:
        """Upsert branch item to Cosmos DB."""
        self.branches_container.upsert_item(body=item)
    
    async def _delete_branch_item(self, branch_id: str) -> None:
        """Delete branch item from Cosmos DB."""
        self.branches_container.delete_item(
            item=branch_id,
            partition_key=branch_id
        )
    
    # Integration methods for ingestion workflows
    
    async def get_branch_for_ingestion(
        self,
        repository_url: str,
        branch_name: str
    ) -> Tuple[Optional[BranchMetadata], Optional[CommitState]]:
        """
        Get branch and commit state information for ingestion workflows.
        
        Args:
            repository_url: Repository URL
            branch_name: Branch name
            
        Returns:
            Tuple of (BranchMetadata, CommitState) or (None, None) if not found
        """
        try:
            # Get branch metadata
            branch_metadata = await self.get_branch_metadata(repository_url, branch_name)
            
            # Get commit state
            commit_state = await self.commit_state_manager.get_last_commit(
                repository_url, branch_name
            )
            
            return branch_metadata, commit_state
            
        except Exception as e:
            logger.error(f"Error getting branch information for ingestion: {e}")
            raise
    
    async def setup_branch_for_processing(
        self,
        repo: git.Repo,
        repository_url: str,
        branch_name: str,
        create_if_missing: bool = True
    ) -> Tuple[BranchMetadata, CommitState]:
        """
        Setup a branch for processing - ensure it exists and has metadata.
        
        Args:
            repo: GitPython repository object
            repository_url: Repository URL
            branch_name: Branch name
            create_if_missing: Create branch if it doesn't exist
            
        Returns:
            Tuple of (BranchMetadata, CommitState)
        """
        try:
            logger.info(f"Setting up branch '{branch_name}' for processing")
            
            # Check if branch exists
            if branch_name not in [head.name for head in repo.heads]:
                if create_if_missing:
                    # Create the branch
                    branch_metadata = await self.create_branch(
                        repo=repo,
                        repository_url=repository_url,
                        branch_name=branch_name,
                        switch_to_branch=True
                    )
                else:
                    raise ValueError(f"Branch '{branch_name}' does not exist")
            else:
                # Switch to existing branch
                branch_metadata = await self.switch_branch(
                    repo=repo,
                    repository_url=repository_url,
                    target_branch=branch_name
                )
            
            # Ensure commit state exists
            commit_state = await self.commit_state_manager.get_last_commit(
                repository_url, branch_name
            )
            
            if not commit_state:
                # Initialize commit state for the branch
                current_commit = repo.head.commit
                commit_state = await self.commit_state_manager.update_commit_state(
                    repository_url=repository_url,
                    branch_name=branch_name,
                    current_commit_sha=current_commit.hexsha,
                    processing_status="active"
                )
            
            logger.info(f"Branch '{branch_name}' ready for processing")
            return branch_metadata, commit_state
            
        except Exception as e:
            logger.error(f"Error setting up branch for processing: {e}")
            raise
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get BranchLifecycleManager status information.
        
        Returns:
            Status dictionary with connection and container information
        """
        try:
            # Test Cosmos DB connection
            database_properties = self.database.read()
            container_properties = self.branches_container.read()
            
            # Get commit state manager status
            commit_state_status = await self.commit_state_manager.get_status()
            
            return {
                "status": "healthy",
                "database_name": self.database_name,
                "container_name": self.branches_container_name,
                "database_id": database_properties["id"],
                "container_id": container_properties["id"],
                "commit_state_manager_status": commit_state_status,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
