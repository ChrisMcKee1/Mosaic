"""
CRUD-004 Test Suite: Branch Lifecycle Management

Comprehensive test suite for BranchLifecycleManager functionality.
Tests all branch lifecycle scenarios as specified in CRUD-004 acceptance criteria.

Test Categories:
1. Branch Creation and Metadata Tracking
2. Branch Switching and State Management
3. Branch Deletion with Archive Options
4. Conflict Detection and Resolution
5. Integration with Ingestion Workflows
6. Branch Isolation Scenarios
7. Error Handling and Edge Cases

Author: Mosaic MCP Tool - CRUD-004 Implementation
"""

import pytest
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

import git
from azure.cosmos.exceptions import CosmosResourceNotFoundError

# Import the classes we're testing
from utils.branch_lifecycle_manager import (
    BranchLifecycleManager,
    BranchMetadata,
    BranchConflict,
)
from utils.commit_state_manager import CommitStateManager, CommitState


class TestBranchLifecycleManagerInitialization:
    """Test BranchLifecycleManager initialization and setup."""

    @pytest.fixture
    def mock_commit_state_manager(self):
        """Mock CommitStateManager for testing."""
        mock_manager = MagicMock()
        mock_manager.cosmos_client = MagicMock()
        mock_manager.database_name = "test_database"
        return mock_manager

    @pytest.fixture
    def mock_cosmos_client(self):
        """Mock Cosmos DB client."""
        mock_client = MagicMock()
        mock_database = MagicMock()
        mock_container = MagicMock()
        
        mock_client.get_database_client.return_value = mock_database
        mock_database.get_container_client.return_value = mock_container
        
        return mock_client

    def test_branch_lifecycle_manager_initialization(self, mock_commit_state_manager, mock_cosmos_client):
        """Test BranchLifecycleManager initialization with required dependencies."""
        mock_commit_state_manager.cosmos_client = mock_cosmos_client
        
        manager = BranchLifecycleManager(
            commit_state_manager=mock_commit_state_manager
        )
        
        assert manager.commit_state_manager == mock_commit_state_manager
        assert manager.database_name == "test_database"
        assert manager.branches_container_name == "branches"

    def test_branch_id_generation(self, mock_commit_state_manager):
        """Test unique branch ID generation."""
        manager = BranchLifecycleManager(
            commit_state_manager=mock_commit_state_manager
        )
        
        repo_url = "https://github.com/test/repo"
        branch_name = "feature/test"
        
        branch_id = manager._generate_branch_id(repo_url, branch_name)
        
        # Should be deterministic and unique
        assert isinstance(branch_id, str)
        assert len(branch_id) == 64  # SHA-256 hex string
        
        # Same inputs should produce same ID
        branch_id2 = manager._generate_branch_id(repo_url, branch_name)
        assert branch_id == branch_id2
        
        # Different inputs should produce different IDs
        branch_id3 = manager._generate_branch_id(repo_url, "different_branch")
        assert branch_id != branch_id3


class TestBranchCreation:
    """Test branch creation functionality."""

    @pytest.fixture
    async def temp_git_repo(self):
        """Create a temporary Git repository for testing."""
        temp_dir = tempfile.mkdtemp()
        repo = git.Repo.init(temp_dir)
        
        # Configure user
        repo.config_writer().set_value("user", "name", "Test User").release()
        repo.config_writer().set_value("user", "email", "test@example.com").release()
        
        # Create initial commit
        test_file = Path(temp_dir) / "README.md"
        test_file.write_text("# Test Repository")
        repo.index.add(["README.md"])
        repo.index.commit("Initial commit")
        
        yield repo, temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_branch_manager(self, mock_commit_state_manager):
        """Create BranchLifecycleManager with mocked dependencies."""
        with patch('utils.branch_lifecycle_manager.CosmosClient'):
            manager = BranchLifecycleManager(
                commit_state_manager=mock_commit_state_manager
            )
            
            # Mock the persistence methods
            manager._persist_branch_metadata = AsyncMock()
            manager._get_branch_item = AsyncMock(return_value=None)
            manager._upsert_branch_item = AsyncMock()
            mock_commit_state_manager.update_commit_state = AsyncMock()
            
            return manager

    @pytest.mark.asyncio
    async def test_create_branch_success(self, temp_git_repo, mock_branch_manager):
        """Test successful branch creation with metadata tracking."""
        repo, temp_dir = temp_git_repo
        repository_url = "https://github.com/test/repo"
        branch_name = "feature/new-feature"
        
        # Create branch
        result = await mock_branch_manager.create_branch(
            repo=repo,
            repository_url=repository_url,
            branch_name=branch_name,
            switch_to_branch=True
        )
        
        # Verify branch was created in Git
        assert branch_name in [head.name for head in repo.heads]
        assert repo.active_branch.name == branch_name
        
        # Verify metadata
        assert isinstance(result, BranchMetadata)
        assert result.branch_name == branch_name
        assert result.repository_url == repository_url
        assert result.is_active == True
        assert result.processing_status == "active"
        assert result.author_name == "Test User"
        assert result.author_email == "test@example.com"
        
        # Verify persistence was called
        mock_branch_manager._persist_branch_metadata.assert_called_once()
        mock_branch_manager.commit_state_manager.update_commit_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_branch_duplicate_name(self, temp_git_repo, mock_branch_manager):
        """Test creating branch with duplicate name raises error."""
        repo, temp_dir = temp_git_repo
        repository_url = "https://github.com/test/repo"
        branch_name = "feature/duplicate"
        
        # Create branch first time
        repo.create_head(branch_name)
        
        # Try to create same branch again
        with pytest.raises(ValueError, match="already exists"):
            await mock_branch_manager.create_branch(
                repo=repo,
                repository_url=repository_url,
                branch_name=branch_name
            )

    @pytest.mark.asyncio
    async def test_create_branch_with_parent_commit(self, temp_git_repo, mock_branch_manager):
        """Test creating branch from specific parent commit."""
        repo, temp_dir = temp_git_repo
        repository_url = "https://github.com/test/repo"
        branch_name = "feature/from-commit"
        
        # Create additional commit
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("test content")
        repo.index.add(["test.txt"])
        second_commit = repo.index.commit("Second commit")
        
        # Create branch from first commit (parent of current HEAD)
        parent_commit = repo.head.commit.parents[0]
        
        result = await mock_branch_manager.create_branch(
            repo=repo,
            repository_url=repository_url,
            branch_name=branch_name,
            parent_commit=parent_commit,
            switch_to_branch=False
        )
        
        # Verify branch points to parent commit
        assert result.parent_commit_sha == parent_commit.hexsha
        assert result.is_active == False  # Not switched to
        
        # Verify Git branch
        created_branch = repo.heads[branch_name]
        assert created_branch.commit.hexsha == parent_commit.hexsha


class TestBranchSwitching:
    """Test branch switching functionality."""

    @pytest.fixture
    async def temp_git_repo_with_branches(self):
        """Create repository with multiple branches."""
        temp_dir = tempfile.mkdtemp()
        repo = git.Repo.init(temp_dir)
        
        # Configure user
        repo.config_writer().set_value("user", "name", "Test User").release()
        repo.config_writer().set_value("user", "email", "test@example.com").release()
        
        # Create initial commit on main
        test_file = Path(temp_dir) / "README.md"
        test_file.write_text("# Test Repository")
        repo.index.add(["README.md"])
        repo.index.commit("Initial commit")
        
        # Create feature branch
        feature_branch = repo.create_head("feature/test")
        
        yield repo, temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_switch_to_existing_branch(self, temp_git_repo_with_branches, mock_branch_manager):
        """Test switching to an existing branch."""
        repo, temp_dir = temp_git_repo_with_branches
        repository_url = "https://github.com/test/repo"
        target_branch = "feature/test"
        
        # Mock metadata retrieval
        mock_metadata = BranchMetadata(
            branch_name=target_branch,
            repository_url=repository_url,
            creation_timestamp=datetime.now(timezone.utc),
            parent_commit_sha=repo.head.commit.hexsha,
            author_name="Test User",
            author_email="test@example.com",
            current_commit_sha=repo.heads[target_branch].commit.hexsha,
            is_active=False
        )
        mock_branch_manager.get_branch_metadata = AsyncMock(return_value=mock_metadata)
        mock_branch_manager._update_branch_active_status = AsyncMock()
        
        # Switch to branch
        result = await mock_branch_manager.switch_branch(
            repo=repo,
            repository_url=repository_url,
            target_branch=target_branch
        )
        
        # Verify Git switch
        assert repo.active_branch.name == target_branch
        
        # Verify metadata update calls
        mock_branch_manager._update_branch_active_status.assert_any_call(
            repository_url, "main", False
        )
        mock_branch_manager._update_branch_active_status.assert_any_call(
            repository_url, target_branch, True
        )

    @pytest.mark.asyncio
    async def test_switch_to_nonexistent_branch_create_if_missing(self, temp_git_repo_with_branches, mock_branch_manager):
        """Test switching to non-existent branch with create_if_missing=True."""
        repo, temp_dir = temp_git_repo_with_branches
        repository_url = "https://github.com/test/repo"
        target_branch = "feature/new-branch"
        
        # Mock create_branch to return metadata
        mock_metadata = BranchMetadata(
            branch_name=target_branch,
            repository_url=repository_url,
            creation_timestamp=datetime.now(timezone.utc),
            parent_commit_sha=repo.head.commit.hexsha,
            author_name="Test User",
            author_email="test@example.com",
            current_commit_sha=repo.head.commit.hexsha,
            is_active=True
        )
        mock_branch_manager.create_branch = AsyncMock(return_value=mock_metadata)
        
        # Switch to non-existent branch with create_if_missing=True
        result = await mock_branch_manager.switch_branch(
            repo=repo,
            repository_url=repository_url,
            target_branch=target_branch,
            create_if_missing=True
        )
        
        # Verify create_branch was called
        mock_branch_manager.create_branch.assert_called_once_with(
            repo=repo,
            repository_url=repository_url,
            branch_name=target_branch,
            switch_to_branch=True
        )
        
        assert result == mock_metadata

    @pytest.mark.asyncio
    async def test_switch_to_nonexistent_branch_no_create(self, temp_git_repo_with_branches, mock_branch_manager):
        """Test switching to non-existent branch with create_if_missing=False raises error."""
        repo, temp_dir = temp_git_repo_with_branches
        repository_url = "https://github.com/test/repo"
        target_branch = "feature/nonexistent"
        
        # Try to switch to non-existent branch
        with pytest.raises(ValueError, match="does not exist"):
            await mock_branch_manager.switch_branch(
                repo=repo,
                repository_url=repository_url,
                target_branch=target_branch,
                create_if_missing=False
            )


class TestBranchDeletion:
    """Test branch deletion functionality."""

    @pytest.mark.asyncio
    async def test_delete_branch_success(self, temp_git_repo_with_branches, mock_branch_manager):
        """Test successful branch deletion with archiving."""
        repo, temp_dir = temp_git_repo_with_branches
        repository_url = "https://github.com/test/repo"
        branch_to_delete = "feature/test"
        
        # Ensure we're not on the branch to delete
        assert repo.active_branch.name != branch_to_delete
        
        # Mock archiving
        mock_branch_manager._archive_branch_metadata = AsyncMock()
        
        # Delete branch
        result = await mock_branch_manager.delete_branch(
            repo=repo,
            repository_url=repository_url,
            branch_name=branch_to_delete,
            archive_metadata=True
        )
        
        # Verify deletion
        assert result == True
        assert branch_to_delete not in [head.name for head in repo.heads]
        
        # Verify archiving was called
        mock_branch_manager._archive_branch_metadata.assert_called_once_with(
            repository_url, branch_to_delete
        )

    @pytest.mark.asyncio
    async def test_delete_active_branch_raises_error(self, temp_git_repo_with_branches, mock_branch_manager):
        """Test deleting currently active branch raises error."""
        repo, temp_dir = temp_git_repo_with_branches
        repository_url = "https://github.com/test/repo"
        
        # Get currently active branch
        active_branch = repo.active_branch.name
        
        # Try to delete active branch
        with pytest.raises(ValueError, match="Cannot delete currently active branch"):
            await mock_branch_manager.delete_branch(
                repo=repo,
                repository_url=repository_url,
                branch_name=active_branch
            )

    @pytest.mark.asyncio
    async def test_delete_nonexistent_branch_raises_error(self, temp_git_repo_with_branches, mock_branch_manager):
        """Test deleting non-existent branch raises error."""
        repo, temp_dir = temp_git_repo_with_branches
        repository_url = "https://github.com/test/repo"
        branch_name = "feature/nonexistent"
        
        # Try to delete non-existent branch
        with pytest.raises(ValueError, match="does not exist"):
            await mock_branch_manager.delete_branch(
                repo=repo,
                repository_url=repository_url,
                branch_name=branch_name
            )


class TestConflictDetection:
    """Test branch conflict detection functionality."""

    @pytest.fixture
    async def temp_git_repo_with_conflicts(self):
        """Create repository with potential conflicts."""
        temp_dir = tempfile.mkdtemp()
        repo = git.Repo.init(temp_dir)
        
        # Configure user
        repo.config_writer().set_value("user", "name", "Test User").release()
        repo.config_writer().set_value("user", "email", "test@example.com").release()
        
        # Create initial commit
        test_file = Path(temp_dir) / "shared.txt"
        test_file.write_text("original content")
        repo.index.add(["shared.txt"])
        initial_commit = repo.index.commit("Initial commit")
        
        # Create main branch changes
        test_file.write_text("main branch content")
        repo.index.add(["shared.txt"])
        repo.index.commit("Main branch change")
        
        # Create feature branch from initial commit
        feature_branch = repo.create_head("feature/conflict", initial_commit)
        repo.head.reference = feature_branch
        repo.head.reset(index=True, working_tree=True)
        
        # Make conflicting change in feature branch
        test_file.write_text("feature branch content")
        repo.index.add(["shared.txt"])
        repo.index.commit("Feature branch change")
        
        # Switch back to main
        repo.head.reference = repo.heads.main
        repo.head.reset(index=True, working_tree=True)
        
        yield repo, temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_detect_conflicts_found(self, temp_git_repo_with_conflicts, mock_branch_manager):
        """Test conflict detection when conflicts exist."""
        repo, temp_dir = temp_git_repo_with_conflicts
        repository_url = "https://github.com/test/repo"
        
        # Detect conflicts between feature and main
        conflict = await mock_branch_manager.detect_conflicts(
            repo=repo,
            repository_url=repository_url,
            source_branch="feature/conflict",
            target_branch="main"
        )
        
        # Verify conflict detected
        assert conflict is not None
        assert isinstance(conflict, BranchConflict)
        assert conflict.conflict_type == "merge"
        assert "shared.txt" in conflict.conflicted_files
        assert conflict.severity in ["low", "medium", "high"]
        assert "conflict" in conflict.description.lower()

    @pytest.mark.asyncio
    async def test_detect_conflicts_none_found(self, temp_git_repo_with_branches, mock_branch_manager):
        """Test conflict detection when no conflicts exist."""
        repo, temp_dir = temp_git_repo_with_branches
        repository_url = "https://github.com/test/repo"
        
        # Detect conflicts between identical branches
        conflict = await mock_branch_manager.detect_conflicts(
            repo=repo,
            repository_url=repository_url,
            source_branch="feature/test",
            target_branch="main"
        )
        
        # Verify no conflicts
        assert conflict is None

    @pytest.mark.asyncio
    async def test_detect_conflicts_nonexistent_branch(self, temp_git_repo_with_branches, mock_branch_manager):
        """Test conflict detection with non-existent branch raises error."""
        repo, temp_dir = temp_git_repo_with_branches
        repository_url = "https://github.com/test/repo"
        
        # Try to detect conflicts with non-existent branch
        with pytest.raises(ValueError, match="does not exist"):
            await mock_branch_manager.detect_conflicts(
                repo=repo,
                repository_url=repository_url,
                source_branch="feature/nonexistent",
                target_branch="main"
            )


class TestBranchListing:
    """Test branch listing functionality."""

    @pytest.mark.asyncio
    async def test_list_branches_without_metadata(self, temp_git_repo_with_branches, mock_branch_manager):
        """Test listing branches without metadata."""
        repo, temp_dir = temp_git_repo_with_branches
        repository_url = "https://github.com/test/repo"
        
        # List branches without metadata
        branches = await mock_branch_manager.list_branches(
            repo=repo,
            repository_url=repository_url,
            include_metadata=False
        )
        
        # Verify basic branch information
        assert len(branches) == 2  # main and feature/test
        branch_names = [b["name"] for b in branches]
        assert "main" in branch_names
        assert "feature/test" in branch_names
        
        # Verify current branch detection
        current_branches = [b for b in branches if b["is_current"]]
        assert len(current_branches) == 1
        assert current_branches[0]["name"] == repo.active_branch.name

    @pytest.mark.asyncio
    async def test_list_branches_with_metadata(self, temp_git_repo_with_branches, mock_branch_manager):
        """Test listing branches with metadata."""
        repo, temp_dir = temp_git_repo_with_branches
        repository_url = "https://github.com/test/repo"
        
        # Mock metadata for feature branch
        mock_metadata = BranchMetadata(
            branch_name="feature/test",
            repository_url=repository_url,
            creation_timestamp=datetime.now(timezone.utc),
            parent_commit_sha=repo.head.commit.hexsha,
            author_name="Test User",
            author_email="test@example.com",
            current_commit_sha=repo.heads["feature/test"].commit.hexsha,
            is_active=False,
            processing_status="active"
        )
        
        # Mock get_branch_metadata to return metadata for feature branch only
        async def mock_get_metadata(repo_url, branch_name):
            if branch_name == "feature/test":
                return mock_metadata
            return None
        
        mock_branch_manager.get_branch_metadata = mock_get_metadata
        
        # List branches with metadata
        branches = await mock_branch_manager.list_branches(
            repo=repo,
            repository_url=repository_url,
            include_metadata=True
        )
        
        # Find feature branch
        feature_branch = next(b for b in branches if b["name"] == "feature/test")
        
        # Verify metadata is included
        assert "creation_timestamp" in feature_branch
        assert "author_name" in feature_branch
        assert feature_branch["author_name"] == "Test User"
        assert feature_branch["processing_status"] == "active"


class TestIngestionIntegration:
    """Test integration with ingestion workflows."""

    @pytest.mark.asyncio
    async def test_setup_branch_for_processing_existing(self, temp_git_repo_with_branches, mock_branch_manager):
        """Test setting up existing branch for processing."""
        repo, temp_dir = temp_git_repo_with_branches
        repository_url = "https://github.com/test/repo"
        branch_name = "feature/test"
        
        # Mock dependencies
        mock_branch_metadata = BranchMetadata(
            branch_name=branch_name,
            repository_url=repository_url,
            creation_timestamp=datetime.now(timezone.utc),
            parent_commit_sha=repo.head.commit.hexsha,
            author_name="Test User",
            author_email="test@example.com",
            current_commit_sha=repo.heads[branch_name].commit.hexsha,
            is_active=True
        )
        
        mock_commit_state = CommitState(
            repository_id="test_id",
            repository_url=repository_url,
            branch_name=branch_name,
            last_commit_sha=repo.heads[branch_name].commit.hexsha,
            last_processed_timestamp=datetime.now(timezone.utc)
        )
        
        mock_branch_manager.switch_branch = AsyncMock(return_value=mock_branch_metadata)
        mock_branch_manager.commit_state_manager.get_last_commit = AsyncMock(return_value=mock_commit_state)
        
        # Setup branch for processing
        branch_metadata, commit_state = await mock_branch_manager.setup_branch_for_processing(
            repo=repo,
            repository_url=repository_url,
            branch_name=branch_name,
            create_if_missing=True
        )
        
        # Verify setup
        assert branch_metadata == mock_branch_metadata
        assert commit_state == mock_commit_state
        mock_branch_manager.switch_branch.assert_called_once()

    @pytest.mark.asyncio
    async def test_setup_branch_for_processing_create_missing(self, temp_git_repo_with_branches, mock_branch_manager):
        """Test setting up non-existent branch for processing with create_if_missing=True."""
        repo, temp_dir = temp_git_repo_with_branches
        repository_url = "https://github.com/test/repo"
        branch_name = "feature/new-branch"
        
        # Mock dependencies
        mock_branch_metadata = BranchMetadata(
            branch_name=branch_name,
            repository_url=repository_url,
            creation_timestamp=datetime.now(timezone.utc),
            parent_commit_sha=repo.head.commit.hexsha,
            author_name="Test User",
            author_email="test@example.com",
            current_commit_sha=repo.head.commit.hexsha,
            is_active=True
        )
        
        mock_commit_state = CommitState(
            repository_id="test_id",
            repository_url=repository_url,
            branch_name=branch_name,
            last_commit_sha=repo.head.commit.hexsha,
            last_processed_timestamp=datetime.now(timezone.utc)
        )
        
        mock_branch_manager.create_branch = AsyncMock(return_value=mock_branch_metadata)
        mock_branch_manager.commit_state_manager.get_last_commit = AsyncMock(return_value=mock_commit_state)
        
        # Setup non-existent branch for processing
        branch_metadata, commit_state = await mock_branch_manager.setup_branch_for_processing(
            repo=repo,
            repository_url=repository_url,
            branch_name=branch_name,
            create_if_missing=True
        )
        
        # Verify creation and setup
        assert branch_metadata == mock_branch_metadata
        assert commit_state == mock_commit_state
        mock_branch_manager.create_branch.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_branch_for_ingestion(self, mock_branch_manager):
        """Test getting branch information for ingestion workflows."""
        repository_url = "https://github.com/test/repo"
        branch_name = "feature/test"
        
        # Mock dependencies
        mock_branch_metadata = BranchMetadata(
            branch_name=branch_name,
            repository_url=repository_url,
            creation_timestamp=datetime.now(timezone.utc),
            parent_commit_sha="abc123",
            author_name="Test User",
            author_email="test@example.com",
            current_commit_sha="def456",
            is_active=True
        )
        
        mock_commit_state = CommitState(
            repository_id="test_id",
            repository_url=repository_url,
            branch_name=branch_name,
            last_commit_sha="def456",
            last_processed_timestamp=datetime.now(timezone.utc)
        )
        
        mock_branch_manager.get_branch_metadata = AsyncMock(return_value=mock_branch_metadata)
        mock_branch_manager.commit_state_manager.get_last_commit = AsyncMock(return_value=mock_commit_state)
        
        # Get branch information
        branch_metadata, commit_state = await mock_branch_manager.get_branch_for_ingestion(
            repository_url=repository_url,
            branch_name=branch_name
        )
        
        # Verify results
        assert branch_metadata == mock_branch_metadata
        assert commit_state == mock_commit_state


class TestBranchMetadataPersistence:
    """Test branch metadata persistence operations."""

    @pytest.mark.asyncio
    async def test_persist_branch_metadata(self, mock_branch_manager):
        """Test persisting branch metadata to Cosmos DB."""
        metadata = BranchMetadata(
            branch_name="feature/test",
            repository_url="https://github.com/test/repo",
            creation_timestamp=datetime.now(timezone.utc),
            parent_commit_sha="abc123",
            author_name="Test User",
            author_email="test@example.com",
            current_commit_sha="def456",
            is_active=True,
            processing_status="active"
        )
        
        # Test persistence
        await mock_branch_manager._persist_branch_metadata(metadata)
        
        # Verify upsert was called
        mock_branch_manager._upsert_branch_item.assert_called_once()
        
        # Verify document structure
        call_args = mock_branch_manager._upsert_branch_item.call_args[0][0]
        assert call_args["branch_name"] == metadata.branch_name
        assert call_args["repository_url"] == metadata.repository_url
        assert call_args["entity_type"] == "branch_metadata"
        assert call_args["version"] == "1.0"

    @pytest.mark.asyncio
    async def test_get_branch_metadata_found(self, mock_branch_manager):
        """Test retrieving existing branch metadata."""
        repository_url = "https://github.com/test/repo"
        branch_name = "feature/test"
        
        # Mock Cosmos DB response
        mock_item = {
            "branch_name": branch_name,
            "repository_url": repository_url,
            "creation_timestamp": datetime.now(timezone.utc).isoformat(),
            "parent_commit_sha": "abc123",
            "author_name": "Test User",
            "author_email": "test@example.com",
            "current_commit_sha": "def456",
            "is_active": True,
            "is_merged": False,
            "processing_status": "active"
        }
        
        mock_branch_manager._get_branch_item = AsyncMock(return_value=mock_item)
        
        # Get metadata
        metadata = await mock_branch_manager.get_branch_metadata(repository_url, branch_name)
        
        # Verify metadata
        assert metadata is not None
        assert isinstance(metadata, BranchMetadata)
        assert metadata.branch_name == branch_name
        assert metadata.repository_url == repository_url
        assert metadata.author_name == "Test User"

    @pytest.mark.asyncio
    async def test_get_branch_metadata_not_found(self, mock_branch_manager):
        """Test retrieving non-existent branch metadata."""
        repository_url = "https://github.com/test/repo"
        branch_name = "feature/nonexistent"
        
        # Mock Cosmos DB response (not found)
        mock_branch_manager._get_branch_item = AsyncMock(return_value=None)
        
        # Get metadata
        metadata = await mock_branch_manager.get_branch_metadata(repository_url, branch_name)
        
        # Verify not found
        assert metadata is None


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_cosmos_db_connection_error(self, mock_branch_manager):
        """Test handling Cosmos DB connection errors."""
        mock_branch_manager._get_branch_item = AsyncMock(side_effect=Exception("Connection failed"))
        
        # Should propagate exception
        with pytest.raises(Exception, match="Connection failed"):
            await mock_branch_manager.get_branch_metadata("test_url", "test_branch")

    @pytest.mark.asyncio
    async def test_git_operation_error(self, temp_git_repo_with_branches, mock_branch_manager):
        """Test handling Git operation errors."""
        repo, temp_dir = temp_git_repo_with_branches
        
        # Force Git error by corrupting repository
        shutil.rmtree(Path(temp_dir) / ".git", ignore_errors=True)
        
        # Should raise exception
        with pytest.raises(Exception):
            await mock_branch_manager.create_branch(
                repo=repo,
                repository_url="test_url",
                branch_name="test_branch"
            )

    @pytest.mark.asyncio
    async def test_status_healthy(self, mock_branch_manager):
        """Test status reporting when healthy."""
        # Mock successful operations
        mock_branch_manager.database.read = MagicMock(return_value={"id": "test_db"})
        mock_branch_manager.branches_container.read = MagicMock(return_value={"id": "test_container"})
        mock_branch_manager.commit_state_manager.get_status = AsyncMock(return_value={"status": "healthy"})
        
        status = await mock_branch_manager.get_status()
        
        assert status["status"] == "healthy"
        assert status["database_name"] == mock_branch_manager.database_name
        assert "commit_state_manager_status" in status

    @pytest.mark.asyncio
    async def test_status_unhealthy(self, mock_branch_manager):
        """Test status reporting when unhealthy."""
        # Mock failed operation
        mock_branch_manager.database.read = MagicMock(side_effect=Exception("Connection failed"))
        
        status = await mock_branch_manager.get_status()
        
        assert status["status"] == "unhealthy"
        assert "error" in status
        assert "Connection failed" in status["error"]


# Test configuration
@pytest.fixture(scope="session")
def mock_commit_state_manager():
    """Session-scoped mock for CommitStateManager."""
    mock_manager = MagicMock()
    mock_manager.cosmos_client = MagicMock()
    mock_manager.database_name = "test_database"
    mock_manager.update_commit_state = AsyncMock()
    mock_manager.get_last_commit = AsyncMock()
    return mock_manager


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
