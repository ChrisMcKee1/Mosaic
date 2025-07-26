"""
Unit tests for CommitStateManager (CRUD-001).

Comprehensive test suite covering all functionality of the CommitStateManager
system with >80% code coverage as required by acceptance criteria.

Test Categories:
- Initialization and configuration
- Commit state persistence (get_last_commit, update_commit_state)  
- GitPython integration (iter_commits_since_last, get_commit_diff)
- Repository cloning with full history access
- Error handling and edge cases
- Cosmos DB integration patterns

Author: Mosaic MCP Tool CRUD-001 Test Suite
"""

import pytest
import asyncio
import tempfile
import shutil
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, Optional

import git
from azure.cosmos.exceptions import CosmosResourceNotFoundError

# Import the modules under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.commit_state_manager import CommitStateManager, CommitState


class TestCommitState:
    """Test CommitState dataclass."""
    
    def test_commit_state_creation(self):
        """Test CommitState dataclass initialization."""
        now = datetime.now(timezone.utc)
        
        state = CommitState(
            repository_id="test_id",
            repository_url="https://github.com/test/repo",
            branch_name="main",
            last_commit_sha="abc123",
            last_processed_timestamp=now,
            commit_count=42,
            processing_status="completed"
        )
        
        assert state.repository_id == "test_id"
        assert state.repository_url == "https://github.com/test/repo"
        assert state.branch_name == "main"
        assert state.last_commit_sha == "abc123"
        assert state.last_processed_timestamp == now
        assert state.commit_count == 42
        assert state.processing_status == "completed"
    
    def test_commit_state_defaults(self):
        """Test CommitState default values."""
        now = datetime.now(timezone.utc)
        
        state = CommitState(
            repository_id="test_id",
            repository_url="https://github.com/test/repo",
            branch_name="main",
            last_commit_sha="abc123",
            last_processed_timestamp=now
        )
        
        assert state.commit_count == 0
        assert state.processing_status == "completed"


class TestCommitStateManagerInitialization:
    """Test CommitStateManager initialization and configuration."""
    
    @pytest.fixture
    def mock_cosmos_client(self):
        """Create mock Cosmos DB client."""
        mock_client = Mock()
        mock_database = Mock()
        mock_container = Mock()
        
        mock_client.get_database_client.return_value = mock_database
        mock_database.get_container_client.return_value = mock_container
        
        return mock_client, mock_database, mock_container
    
    def test_init_with_cosmos_client(self, mock_cosmos_client):
        """Test initialization with Cosmos DB client."""
        mock_client, mock_database, mock_container = mock_cosmos_client
        
        manager = CommitStateManager(
            cosmos_client=mock_client,
            database_name="test_db",
            repositories_container_name="test_repos"
        )
        
        assert manager.cosmos_client == mock_client
        assert manager.database_name == "test_db"
        assert manager.repositories_container_name == "test_repos"
        assert manager.database == mock_database
        assert manager.repositories_container == mock_container
    
    @patch('utils.commit_state_manager.DefaultAzureCredential')
    @patch('utils.commit_state_manager.CosmosClient')
    def test_from_settings(self, mock_cosmos_client_class, mock_credential):
        """Test initialization from settings."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.get_cosmos_config.return_value = {
            "endpoint": "https://test.documents.azure.com:443/",
            "database_name": "mosaic",
            "repositories_container": "repositories"
        }
        
        # Mock cosmos client
        mock_client = Mock()
        mock_cosmos_client_class.return_value = mock_client
        
        manager = CommitStateManager.from_settings(mock_settings)
        
        mock_credential.assert_called_once()
        mock_cosmos_client_class.assert_called_once()
        assert manager.database_name == "mosaic"
        assert manager.repositories_container_name == "repositories"
    
    def test_generate_repository_id(self, mock_cosmos_client):
        """Test repository ID generation."""
        mock_client, _, _ = mock_cosmos_client
        
        manager = CommitStateManager(
            cosmos_client=mock_client,
            database_name="test_db"
        )
        
        repo_url = "https://github.com/test/repo"
        branch = "main"
        
        repo_id = manager._generate_repository_id(repo_url, branch)
        
        # Should be SHA-256 hash
        expected_content = f"{repo_url}#{branch}"
        expected_hash = hashlib.sha256(expected_content.encode()).hexdigest()
        
        assert repo_id == expected_hash
        assert len(repo_id) == 64  # SHA-256 hex length
    
    def test_generate_repository_id_consistency(self, mock_cosmos_client):
        """Test repository ID generation is consistent."""
        mock_client, _, _ = mock_cosmos_client
        
        manager = CommitStateManager(
            cosmos_client=mock_client,
            database_name="test_db"
        )
        
        repo_url = "https://github.com/test/repo"
        branch = "main"
        
        id1 = manager._generate_repository_id(repo_url, branch)
        id2 = manager._generate_repository_id(repo_url, branch)
        
        assert id1 == id2


class TestCommitStatePersistence:
    """Test commit state persistence operations."""
    
    @pytest.fixture
    def mock_cosmos_manager(self):
        """Create CommitStateManager with mocked Cosmos operations."""
        mock_client = Mock()
        mock_database = Mock()
        mock_container = Mock()
        
        mock_client.get_database_client.return_value = mock_database
        mock_database.get_container_client.return_value = mock_container
        
        manager = CommitStateManager(
            cosmos_client=mock_client,
            database_name="test_db"
        )
        
        return manager, mock_container
    
    @pytest.mark.asyncio
    async def test_get_last_commit_found(self, mock_cosmos_manager):
        """Test retrieving existing commit state."""
        manager, mock_container = mock_cosmos_manager
        
        # Mock Cosmos DB response
        mock_item = {
            "id": "test_repo_id",
            "repository_url": "https://github.com/test/repo",
            "branch_name": "main",
            "last_commit_sha": "abc123def456",
            "last_processed_timestamp": "2025-01-27T12:00:00+00:00",
            "commit_count": 42,
            "processing_status": "completed"
        }
        
        mock_container.read_item.return_value = mock_item
        
        result = await manager.get_last_commit(
            "https://github.com/test/repo", "main"
        )
        
        assert result is not None
        assert result.repository_url == "https://github.com/test/repo"
        assert result.branch_name == "main"
        assert result.last_commit_sha == "abc123def456"
        assert result.commit_count == 42
        assert result.processing_status == "completed"
    
    @pytest.mark.asyncio
    async def test_get_last_commit_not_found(self, mock_cosmos_manager):
        """Test retrieving non-existent commit state."""
        manager, mock_container = mock_cosmos_manager
        
        mock_container.read_item.side_effect = CosmosResourceNotFoundError(
            message="Not found"
        )
        
        result = await manager.get_last_commit(
            "https://github.com/test/repo", "main"
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_update_commit_state(self, mock_cosmos_manager):
        """Test updating commit state."""
        manager, mock_container = mock_cosmos_manager
        
        result = await manager.update_commit_state(
            repository_url="https://github.com/test/repo",
            branch_name="main",
            current_commit_sha="new_commit_sha",
            commit_count=50,
            processing_status="completed"
        )
        
        # Verify upsert was called
        mock_container.upsert_item.assert_called_once()
        call_args = mock_container.upsert_item.call_args[1]
        
        assert call_args["body"]["repository_url"] == "https://github.com/test/repo"
        assert call_args["body"]["branch_name"] == "main"
        assert call_args["body"]["last_commit_sha"] == "new_commit_sha"
        assert call_args["body"]["commit_count"] == 50
        assert call_args["body"]["processing_status"] == "completed"
        
        # Verify returned CommitState
        assert result.repository_url == "https://github.com/test/repo"
        assert result.branch_name == "main"
        assert result.last_commit_sha == "new_commit_sha"
        assert result.commit_count == 50
        assert result.processing_status == "completed"
    
    @pytest.mark.asyncio
    async def test_update_commit_state_minimal(self, mock_cosmos_manager):
        """Test updating commit state with minimal parameters."""
        manager, mock_container = mock_cosmos_manager
        
        result = await manager.update_commit_state(
            repository_url="https://github.com/test/repo",
            branch_name="main",
            current_commit_sha="minimal_commit"
        )
        
        # Verify default values
        call_args = mock_container.upsert_item.call_args[1]
        assert call_args["body"]["processing_status"] == "completed"
        assert "commit_count" not in call_args["body"]  # Should not be set when None
    
    @pytest.mark.asyncio
    async def test_get_status_healthy(self, mock_cosmos_manager):
        """Test status check when healthy."""
        manager, mock_container = mock_cosmos_manager
        
        # Mock successful database operations
        manager.database.read.return_value = {"id": "test_db"}
        mock_container.read.return_value = {"id": "repositories"}
        
        status = await manager.get_status()
        
        assert status["status"] == "healthy"
        assert status["database_name"] == "test_db"
        assert status["container_name"] == "repositories"
        assert "timestamp" in status
    
    @pytest.mark.asyncio
    async def test_get_status_unhealthy(self, mock_cosmos_manager):
        """Test status check when unhealthy."""
        manager, mock_container = mock_cosmos_manager
        
        # Mock database error
        manager.database.read.side_effect = Exception("Connection failed")
        
        status = await manager.get_status()
        
        assert status["status"] == "unhealthy"
        assert "error" in status
        assert "timestamp" in status


class TestGitPythonIntegration:
    """Test GitPython integration for commit operations."""
    
    @pytest.fixture
    def mock_repo(self):
        """Create mock GitPython repository."""
        mock_repo = Mock(spec=git.Repo)
        
        # Create mock commits
        commits = []
        commit_shas = ["commit1", "commit2", "commit3", "commit4"]
        
        for i, sha in enumerate(commit_shas):
            mock_commit = Mock(spec=git.Commit)
            mock_commit.hexsha = sha
            mock_commit.parents = commits[:i] if i > 0 else []
            commits.append(mock_commit)
        
        mock_repo.iter_commits.return_value = commits
        mock_repo.head.commit = commits[-1]  # HEAD is latest commit
        
        return mock_repo, commits
    
    @pytest.fixture
    def commit_manager(self):
        """Create CommitStateManager with mocked Cosmos."""
        mock_client = Mock()
        mock_database = Mock()
        mock_container = Mock()
        
        mock_client.get_database_client.return_value = mock_database
        mock_database.get_container_client.return_value = mock_container
        
        return CommitStateManager(
            cosmos_client=mock_client,
            database_name="test_db"
        )
    
    def test_iter_commits_since_last_first_time(self, commit_manager, mock_repo):
        """Test commit iteration for first-time processing."""
        mock_repo_obj, commits = mock_repo
        
        result = commit_manager.iter_commits_since_last(
            repo=mock_repo_obj,
            repository_url="https://github.com/test/repo",
            branch_name="main",
            last_commit_sha=None
        )
        
        # Should return all commits in chronological order
        assert len(result) == 4
        assert result == list(reversed(commits))  # Oldest first
    
    def test_iter_commits_since_last_incremental(self, commit_manager, mock_repo):
        """Test commit iteration for incremental processing."""
        mock_repo_obj, commits = mock_repo
        
        # Process since commit2
        result = commit_manager.iter_commits_since_last(
            repo=mock_repo_obj,
            repository_url="https://github.com/test/repo",
            branch_name="main",
            last_commit_sha="commit2"
        )
        
        # Should return commits after commit2 (commit3, commit4)
        # Note: iter_commits_since_last returns new commits in chronological order (oldest first)
        expected = [commits[2], commits[3]]  # commit3, commit4
        assert len(result) == 2
        assert result[0].hexsha == "commit3"
        assert result[1].hexsha == "commit4"
    
    def test_iter_commits_since_last_not_found(self, commit_manager, mock_repo):
        """Test commit iteration when last commit not found."""
        mock_repo_obj, commits = mock_repo
        
        result = commit_manager.iter_commits_since_last(
            repo=mock_repo_obj,
            repository_url="https://github.com/test/repo",
            branch_name="main",
            last_commit_sha="nonexistent_commit"
        )
        
        # Should return all commits when last commit not found
        assert len(result) == 4
        assert result == list(reversed(commits))
    
    def test_get_commit_diff_with_parent(self, commit_manager, mock_repo):
        """Test getting commit diff with parent commit."""
        mock_repo_obj, commits = mock_repo
        
        # Mock diff between commits
        mock_diff = [Mock(spec=git.Diff)]
        commits[1].diff.return_value = iter(mock_diff)  # Make it iterable
        
        result = commit_manager.get_commit_diff(
            repo=mock_repo_obj,
            commit=commits[2],  # commit3
            previous_commit=commits[1]  # commit2
        )
        
        assert result == mock_diff
        commits[1].diff.assert_called_once_with(commits[2])
    
    def test_get_commit_diff_auto_parent(self, commit_manager, mock_repo):
        """Test getting commit diff with automatic parent detection."""
        mock_repo_obj, commits = mock_repo
        
        # Mock diff with first parent
        mock_diff = [Mock(spec=git.Diff)]
        commits[1].diff.return_value = iter(mock_diff)  # Make it iterable
        
        result = commit_manager.get_commit_diff(
            repo=mock_repo_obj,
            commit=commits[2]  # commit3 (has commit2 as parent)
        )
        
        assert result == mock_diff
        commits[1].diff.assert_called_once_with(commits[2])
    
    def test_get_commit_diff_initial_commit(self, commit_manager, mock_repo):
        """Test getting diff for initial commit."""
        mock_repo_obj, commits = mock_repo
        
        # Initial commit has no parents
        commits[0].parents = []
        commits[0].hexsha = "commit1"
        mock_diff = [Mock(spec=git.Diff)]
        commits[0].diff.return_value = iter(mock_diff)  # Make it iterable
        
        result = commit_manager.get_commit_diff(
            repo=mock_repo_obj,
            commit=commits[0]
        )
        
        assert result == mock_diff
        commits[0].diff.assert_called_once_with(git.NULL_TREE)


class TestRepositoryCloning:
    """Test repository cloning with full history access."""
    
    @pytest.fixture
    def commit_manager(self):
        """Create CommitStateManager for testing."""
        mock_client = Mock()
        mock_database = Mock()
        mock_container = Mock()
        
        mock_client.get_database_client.return_value = mock_database
        mock_database.get_container_client.return_value = mock_container
        
        return CommitStateManager(
            cosmos_client=mock_client,
            database_name="test_db"
        )
    
    @patch('utils.commit_state_manager.git.Repo.clone_from')
    def test_clone_with_full_history(self, mock_clone, commit_manager):
        """Test cloning with full history (no depth limitation)."""
        # Mock repository
        mock_repo = Mock(spec=git.Repo)
        mock_commits = [Mock() for _ in range(100)]  # 100 commits
        mock_repo.iter_commits.return_value = iter(mock_commits)  # Make it iterable
        mock_repo.head.commit.hexsha = "latest_commit"
        
        mock_clone.return_value = mock_repo
        
        result = commit_manager.clone_with_full_history(
            repository_url="https://github.com/test/repo",
            target_directory="/tmp/test_repo",
            branch="main"
        )
        
        # Verify clone was called without depth parameter
        mock_clone.assert_called_once_with(
            url="https://github.com/test/repo",
            to_path="/tmp/test_repo",
            branch="main"
        )
        
        assert result == mock_repo
    
    @patch('utils.commit_state_manager.git.Repo.clone_from')
    def test_clone_with_specific_depth(self, mock_clone, commit_manager):
        """Test cloning with specified depth."""
        mock_repo = Mock(spec=git.Repo)
        mock_repo.iter_commits.return_value = iter([Mock() for _ in range(10)])  # Make it iterable
        mock_clone.return_value = mock_repo
        
        result = commit_manager.clone_with_full_history(
            repository_url="https://github.com/test/repo",
            target_directory="/tmp/test_repo",
            branch="main",
            clone_depth=10
        )
        
        # Verify clone was called with depth parameter
        mock_clone.assert_called_once_with(
            url="https://github.com/test/repo",
            to_path="/tmp/test_repo",
            branch="main",
            depth=10
        )
        
        assert result == mock_repo
    
    @patch('utils.commit_state_manager.git.Repo.clone_from')
    def test_clone_error_handling(self, mock_clone, commit_manager):
        """Test clone error handling."""
        mock_clone.side_effect = git.exc.GitCommandError(
            "git clone", "Authentication failed"
        )
        
        with pytest.raises(git.exc.GitCommandError):
            commit_manager.clone_with_full_history(
                repository_url="https://github.com/test/repo",
                target_directory="/tmp/test_repo",
                branch="main"
            )


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def commit_manager(self):
        """Create CommitStateManager for testing."""
        mock_client = Mock()
        mock_database = Mock()
        mock_container = Mock()
        
        mock_client.get_database_client.return_value = mock_database
        mock_database.get_container_client.return_value = mock_container
        
        return CommitStateManager(
            cosmos_client=mock_client,
            database_name="test_db"
        ), mock_container
    
    @pytest.mark.asyncio
    async def test_get_last_commit_cosmos_error(self, commit_manager):
        """Test get_last_commit with Cosmos DB error."""
        manager, mock_container = commit_manager
        
        mock_container.read_item.side_effect = Exception("Cosmos DB error")
        
        with pytest.raises(Exception, match="Cosmos DB error"):
            await manager.get_last_commit(
                "https://github.com/test/repo", "main"
            )
    
    @pytest.mark.asyncio
    async def test_update_commit_state_cosmos_error(self, commit_manager):
        """Test update_commit_state with Cosmos DB error."""
        manager, mock_container = commit_manager
        
        mock_container.upsert_item.side_effect = Exception("Upsert failed")
        
        with pytest.raises(Exception, match="Upsert failed"):
            await manager.update_commit_state(
                repository_url="https://github.com/test/repo",
                branch_name="main",
                current_commit_sha="test_commit"
            )
    
    def test_iter_commits_git_error(self, commit_manager):
        """Test iter_commits_since_last with GitPython error."""
        manager, _ = commit_manager
        
        mock_repo = Mock(spec=git.Repo)
        mock_repo.iter_commits.side_effect = git.exc.GitCommandError(
            "git log", "Branch not found"
        )
        
        with pytest.raises(git.exc.GitCommandError):
            manager.iter_commits_since_last(
                repo=mock_repo,
                repository_url="https://github.com/test/repo",
                branch_name="nonexistent_branch"
            )
    
    def test_get_commit_diff_error(self, commit_manager):
        """Test get_commit_diff with GitPython error."""
        manager, _ = commit_manager
        
        mock_repo = Mock(spec=git.Repo)
        mock_commit = Mock(spec=git.Commit)
        mock_commit.parents = []
        mock_commit.hexsha = "test_commit"  # Add hexsha attribute
        mock_commit.diff.side_effect = git.exc.GitCommandError(
            "git diff", "Diff failed"
        )
        
        with pytest.raises(git.exc.GitCommandError):
            manager.get_commit_diff(
                repo=mock_repo,
                commit=mock_commit
            )


class TestIntegrationScenarios:
    """Test integration scenarios and workflows."""
    
    @pytest.fixture
    def full_mock_manager(self):
        """Create fully mocked CommitStateManager for integration tests."""
        mock_client = Mock()
        mock_database = Mock()
        mock_container = Mock()
        
        mock_client.get_database_client.return_value = mock_database
        mock_database.get_container_client.return_value = mock_container
        
        manager = CommitStateManager(
            cosmos_client=mock_client,
            database_name="test_db"
        )
        
        return manager, mock_container
    
    @pytest.mark.asyncio
    async def test_first_time_ingestion_workflow(self, full_mock_manager):
        """Test complete workflow for first-time repository ingestion."""
        manager, mock_container = full_mock_manager
        
        # 1. Check for existing state (should be None)
        mock_container.read_item.side_effect = CosmosResourceNotFoundError(
            status_code=404, message="Not found"
        )
        
        last_state = await manager.get_last_commit(
            "https://github.com/test/repo", "main"
        )
        assert last_state is None
        
        # 2. After processing, update state
        result = await manager.update_commit_state(
            repository_url="https://github.com/test/repo",
            branch_name="main",
            current_commit_sha="first_commit_sha",
            commit_count=1,
            processing_status="completed"
        )
        
        assert result.last_commit_sha == "first_commit_sha"
        assert result.commit_count == 1
        assert result.processing_status == "completed"
    
    @pytest.mark.asyncio
    async def test_incremental_update_workflow(self, full_mock_manager):
        """Test complete workflow for incremental repository update."""
        manager, mock_container = full_mock_manager
        
        # 1. Check for existing state
        mock_container.read_item.return_value = {
            "id": "test_repo_id",
            "repository_url": "https://github.com/test/repo",
            "branch_name": "main",
            "last_commit_sha": "old_commit_sha",
            "last_processed_timestamp": "2025-01-27T10:00:00+00:00",
            "commit_count": 5,
            "processing_status": "completed"
        }
        
        last_state = await manager.get_last_commit(
            "https://github.com/test/repo", "main"
        )
        
        assert last_state is not None
        assert last_state.last_commit_sha == "old_commit_sha"
        assert last_state.commit_count == 5
        
        # 2. After processing new commits, update state
        result = await manager.update_commit_state(
            repository_url="https://github.com/test/repo",
            branch_name="main",
            current_commit_sha="new_commit_sha",
            commit_count=7,
            processing_status="completed"
        )
        
        assert result.last_commit_sha == "new_commit_sha"
        assert result.commit_count == 7


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=utils.commit_state_manager",
        "--cov-report=term-missing",
        "--cov-fail-under=80",
    ])
