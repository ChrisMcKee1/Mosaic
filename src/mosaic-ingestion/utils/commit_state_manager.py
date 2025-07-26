"""
Git Commit State Tracking System

Implements commit state persistence and tracking for CRUD-based ingestion operations.
Enables incremental updates by tracking the last processed commit for each repository/branch.

This addresses CRUD-001 acceptance criteria:
- Commit state persistence to Cosmos DB
- GitPython integration for commit traversal and diff operations  
- Full git history access (replaces shallow clone limitation)
- get_last_commit() and update_commit_state() methods

Architecture:
- Standalone utility class used by both local_main.py and ingestion.py
- Uses existing 'repositories' Cosmos DB container
- Integrates with GitPython for commit operations
- Supports branch-aware state tracking

Author: Mosaic MCP Tool - CRUD-001 Implementation
"""

import logging
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

import git
from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from azure.identity import DefaultAzureCredential


logger = logging.getLogger(__name__)


@dataclass
class CommitState:
    """Represents the commit state for a repository/branch combination."""
    
    repository_id: str
    repository_url: str
    branch_name: str
    last_commit_sha: str
    last_processed_timestamp: datetime
    commit_count: int = 0
    processing_status: str = "completed"  # completed, processing, failed


class CommitStateManager:
    """
    Manages git commit state tracking for incremental CRUD operations.
    
    Features:
    - Persist commit state to Cosmos DB repositories container
    - Track last processed commit SHA per repository/branch
    - Enable git diff-based change detection
    - Support full commit history traversal with GitPython
    - Replace shallow clone limitation with configurable depth
    """
    
    def __init__(
        self,
        cosmos_client: CosmosClient,
        database_name: str,
        repositories_container_name: str = "repositories"
    ):
        """
        Initialize CommitStateManager.
        
        Args:
            cosmos_client: Azure Cosmos DB client instance
            database_name: Cosmos DB database name  
            repositories_container_name: Container name for repository metadata
        """
        self.cosmos_client = cosmos_client
        self.database_name = database_name
        self.repositories_container_name = repositories_container_name
        
        # Get database and container references
        self.database = self.cosmos_client.get_database_client(database_name)
        self.repositories_container = self.database.get_container_client(
            repositories_container_name
        )
        
        logger.info(
            f"CommitStateManager initialized for database '{database_name}', "
            f"container '{repositories_container_name}'"
        )
    
    @classmethod
    def from_settings(cls, settings) -> "CommitStateManager":
        """
        Create CommitStateManager from Mosaic settings configuration.
        
        Args:
            settings: MosaicSettings instance with Cosmos DB configuration
            
        Returns:
            Configured CommitStateManager instance
        """
        cosmos_config = settings.get_cosmos_config()
        
        credential = DefaultAzureCredential()
        cosmos_client = CosmosClient(cosmos_config["endpoint"], credential)
        
        return cls(
            cosmos_client=cosmos_client,
            database_name=cosmos_config["database_name"],
            repositories_container_name=cosmos_config["repositories_container"]
        )
    
    def _generate_repository_id(self, repository_url: str, branch_name: str) -> str:
        """
        Generate unique repository ID for Cosmos DB document.
        
        Args:
            repository_url: Git repository URL
            branch_name: Git branch name
            
        Returns:
            SHA-256 hash of repository_url + branch_name
        """
        content = f"{repository_url}#{branch_name}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def get_last_commit(self, repository_url: str, branch_name: str) -> Optional[CommitState]:
        """
        Get the last processed commit state for a repository/branch.
        
        Args:
            repository_url: Git repository URL
            branch_name: Git branch name
            
        Returns:
            CommitState if exists, None if first-time processing
        """
        repository_id = self._generate_repository_id(repository_url, branch_name)
        
        try:
            logger.debug(f"Retrieving commit state for {repository_url}#{branch_name}")
            
            # Query Cosmos DB for existing commit state
            item = await self._get_repository_item(repository_id)
            
            if not item:
                logger.info(f"No previous commit state found for {repository_url}#{branch_name}")
                return None
            
            commit_state = CommitState(
                repository_id=item["id"],
                repository_url=item["repository_url"],
                branch_name=item["branch_name"],
                last_commit_sha=item["last_commit_sha"],
                last_processed_timestamp=datetime.fromisoformat(
                    item["last_processed_timestamp"]
                ),
                commit_count=item.get("commit_count", 0),
                processing_status=item.get("processing_status", "completed")
            )
            
            logger.info(
                f"Found previous commit state: {commit_state.last_commit_sha[:8]} "
                f"from {commit_state.last_processed_timestamp}"
            )
            
            return commit_state
            
        except CosmosResourceNotFoundError:
            logger.info(f"No commit state found for {repository_url}#{branch_name}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving commit state: {e}")
            raise
    
    async def update_commit_state(
        self,
        repository_url: str,
        branch_name: str,
        current_commit_sha: str,
        commit_count: Optional[int] = None,
        processing_status: str = "completed"
    ) -> CommitState:
        """
        Update the commit state for a repository/branch.
        
        Args:
            repository_url: Git repository URL
            branch_name: Git branch name
            current_commit_sha: Current HEAD commit SHA
            commit_count: Total number of commits processed (optional)
            processing_status: Processing status (completed, processing, failed)
            
        Returns:
            Updated CommitState
        """
        repository_id = self._generate_repository_id(repository_url, branch_name)
        now = datetime.now(timezone.utc)
        
        try:
            logger.info(
                f"Updating commit state for {repository_url}#{branch_name} "
                f"to {current_commit_sha[:8]}"
            )
            
            # Create or update commit state document
            commit_state_doc = {
                "id": repository_id,
                "repository_url": repository_url,
                "branch_name": branch_name,
                "last_commit_sha": current_commit_sha,
                "last_processed_timestamp": now.isoformat(),
                "processing_status": processing_status,
                "updated_at": now.isoformat(),
                # Metadata for Cosmos DB partitioning and queries
                "partition_key": repository_id,
                "entity_type": "commit_state",
                "version": "1.0"
            }
            
            # Add commit count if provided
            if commit_count is not None:
                commit_state_doc["commit_count"] = commit_count
            
            # Upsert to Cosmos DB
            await self._upsert_repository_item(commit_state_doc)
            
            commit_state = CommitState(
                repository_id=repository_id,
                repository_url=repository_url,
                branch_name=branch_name,
                last_commit_sha=current_commit_sha,
                last_processed_timestamp=now,
                commit_count=commit_count or 0,
                processing_status=processing_status
            )
            
            logger.info(f"Commit state updated successfully: {current_commit_sha[:8]}")
            return commit_state
            
        except Exception as e:
            logger.error(f"Error updating commit state: {e}")
            raise
    
    def iter_commits_since_last(
        self,
        repo: git.Repo,
        repository_url: str,
        branch_name: str,
        last_commit_sha: Optional[str] = None
    ) -> List[git.Commit]:
        """
        Get list of commits since the last processed commit.
        
        Uses GitPython repo.iter_commits() for commit traversal as specified
        in CRUD-001 acceptance criteria.
        
        Args:
            repo: GitPython repository object
            repository_url: Repository URL for logging
            branch_name: Branch name to traverse
            last_commit_sha: Last processed commit SHA (if None, returns all commits)
            
        Returns:
            List of commits since last_commit_sha (chronological order, oldest first)
        """
        try:
            logger.info(
                f"Traversing commits for {repository_url}#{branch_name} "
                f"since {last_commit_sha[:8] if last_commit_sha else 'beginning'}"
            )
            
            # Get all commits from current HEAD
            all_commits = list(repo.iter_commits(branch_name))
            
            if not last_commit_sha:
                # First-time processing: return all commits
                commits_to_process = all_commits
                logger.info(f"First-time processing: found {len(commits_to_process)} total commits")
            else:
                # Find the last processed commit and get everything after it
                last_commit_index = None
                for i, commit in enumerate(all_commits):
                    if commit.hexsha == last_commit_sha:
                        last_commit_index = i
                        break
                
                if last_commit_index is None:
                    logger.warning(
                        f"Last commit {last_commit_sha[:8]} not found in current branch. "
                        f"Performing full re-processing."
                    )
                    commits_to_process = all_commits
                else:
                    # Get commits after the last processed one
                    commits_to_process = all_commits[:last_commit_index]
                    logger.info(
                        f"Found {len(commits_to_process)} new commits since {last_commit_sha[:8]}"
                    )
            
            # Reverse to get chronological order (oldest first)
            commits_to_process.reverse()
            
            return commits_to_process
            
        except Exception as e:
            logger.error(f"Error traversing commits: {e}")
            raise
    
    def get_commit_diff(
        self,
        repo: git.Repo,
        commit: git.Commit,
        previous_commit: Optional[git.Commit] = None
    ) -> List[git.Diff]:
        """
        Get file changes for a specific commit using GitPython commit.diff().
        
        Implements commit diff operations as specified in CRUD-001 acceptance criteria.
        
        Args:
            repo: GitPython repository object
            commit: Commit to analyze
            previous_commit: Previous commit for comparison (if None, uses parent)
            
        Returns:
            List of git.Diff objects representing file changes
        """
        try:
            if previous_commit is None:
                # Use the first parent commit for comparison
                if commit.parents:
                    previous_commit = commit.parents[0]
                else:
                    # Initial commit: compare against empty tree
                    logger.debug(f"Analyzing initial commit {commit.hexsha[:8]}")
                    return list(commit.diff(git.NULL_TREE))
            
            logger.debug(
                f"Getting diff between {previous_commit.hexsha[:8]} and {commit.hexsha[:8]}"
            )
            
            # Get diff between commits
            diff_list = list(previous_commit.diff(commit))
            
            logger.debug(f"Found {len(diff_list)} changed files in commit {commit.hexsha[:8]}")
            
            return diff_list
            
        except Exception as e:
            logger.error(f"Error getting commit diff: {e}")
            raise
    
    def clone_with_full_history(
        self,
        repository_url: str,
        target_directory: str,
        branch: str = "main",
        clone_depth: Optional[int] = None
    ) -> git.Repo:
        """
        Clone repository with full history access (replaces shallow clone).
        
        Addresses CRUD-001 requirement to "Add git clone depth configuration 
        to enable commit history access (replace shallow clone)".
        
        Args:
            repository_url: Git repository URL
            target_directory: Local directory for clone
            branch: Git branch to clone
            clone_depth: Clone depth (None for full history, int for partial)
            
        Returns:
            GitPython repository object with full history access
        """
        try:
            logger.info(
                f"Cloning {repository_url} (branch: {branch}) with "
                f"{'full history' if clone_depth is None else f'depth {clone_depth}'}"
            )
            
            clone_kwargs = {
                "url": repository_url,
                "to_path": target_directory,
                "branch": branch
            }
            
            # Add depth only if specified (None = full history)
            if clone_depth is not None:
                clone_kwargs["depth"] = clone_depth
            
            repo = git.Repo.clone_from(**clone_kwargs)
            
            commit_count = len(list(repo.iter_commits()))
            logger.info(
                f"Clone successful: {commit_count} commits available for analysis "
                f"(HEAD: {repo.head.commit.hexsha[:8]})"
            )
            
            return repo
            
        except Exception as e:
            logger.error(f"Error cloning repository with full history: {e}")
            raise
    
    async def _get_repository_item(self, repository_id: str) -> Optional[Dict[str, Any]]:
        """Get repository item from Cosmos DB."""
        try:
            response = self.repositories_container.read_item(
                item=repository_id,
                partition_key=repository_id
            )
            return response
        except CosmosResourceNotFoundError:
            return None
    
    async def _upsert_repository_item(self, item: Dict[str, Any]) -> None:
        """Upsert repository item to Cosmos DB."""
        self.repositories_container.upsert_item(body=item)
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get CommitStateManager status information.
        
        Returns:
            Status dictionary with connection and container information
        """
        try:
            # Test Cosmos DB connection
            database_properties = self.database.read()
            container_properties = self.repositories_container.read()
            
            return {
                "status": "healthy",
                "database_name": self.database_name,
                "container_name": self.repositories_container_name,
                "database_id": database_properties["id"],
                "container_id": container_properties["id"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
