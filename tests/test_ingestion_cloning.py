"""
Comprehensive test suite for repository cloning functionality.

Tests the enhanced _clone_repository method with various authentication
patterns, error conditions, and security scenarios.
"""

import os
import tempfile
import shutil
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import git
from git.exc import GitCommandError, InvalidGitRepositoryError

# Import the ingestion plugin
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from ingestion_service.plugins.ingestion import IngestionPlugin


class TestRepositoryCloning:
    """Test suite for repository cloning functionality."""
    
    @pytest.fixture
    def mock_ingestion_plugin(self):
        """Create a mock ingestion plugin for testing."""
        plugin = Mock(spec=IngestionPlugin)
        
        # Mock the actual method we're testing
        from ingestion_service.plugins.ingestion import IngestionPlugin
        actual_plugin = IngestionPlugin()
        plugin._clone_repository = actual_plugin._clone_repository.__get__(plugin)
        
        return plugin
    
    @pytest.fixture
    def clean_env(self):
        """Clean environment variables before each test."""
        original_env = {}
        env_vars = ["GITHUB_TOKEN", "GIT_USERNAME", "GIT_PASSWORD", "GIT_TOKEN"]
        
        # Store original values
        for var in env_vars:
            original_env[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]
        
        yield
        
        # Restore original values
        for var, value in original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]
    
    @pytest.mark.asyncio
    async def test_successful_public_repository_clone(self, mock_ingestion_plugin, clean_env):
        """Test successful cloning of a public repository."""
        mock_repo = Mock()
        mock_repo.heads = [Mock()]  # Mock branch existence
        mock_repo.head.commit.hexsha = "1234567890abcdef"
        
        with patch('git.Repo.clone_from', return_value=mock_repo) as mock_clone, \
             patch('tempfile.mkdtemp', return_value='/tmp/test_repo') as mock_temp, \
             patch('os.chmod') as mock_chmod:
            
            result = await mock_ingestion_plugin._clone_repository(
                "https://github.com/user/public-repo.git",
                "main"
            )
            
            assert result == '/tmp/test_repo'
            mock_clone.assert_called_once()
            mock_chmod.assert_called_once_with('/tmp/test_repo', 0o700)
    
    @pytest.mark.asyncio
    async def test_github_token_authentication(self, mock_ingestion_plugin, clean_env):
        """Test GitHub token authentication."""
        os.environ["GITHUB_TOKEN"] = "test_github_token"
        
        mock_repo = Mock()
        mock_repo.heads = [Mock()]
        mock_repo.head.commit.hexsha = "1234567890abcdef"
        
        with patch('git.Repo.clone_from', return_value=mock_repo) as mock_clone, \
             patch('tempfile.mkdtemp', return_value='/tmp/test_repo'), \
             patch('os.chmod'):
            
            await mock_ingestion_plugin._clone_repository(
                "https://github.com/user/private-repo.git",
                "main"
            )
            
            # Verify clone was called with modified URL and headers
            call_args = mock_clone.call_args
            assert "test_github_token@github.com" in call_args[0][0]
            assert "Authorization: token test_github_token" in call_args[1]['env']['GIT_HTTP_EXTRAHEADER']
    
    @pytest.mark.asyncio
    async def test_username_password_authentication(self, mock_ingestion_plugin, clean_env):
        """Test username/password authentication."""
        os.environ["GIT_USERNAME"] = "testuser"
        os.environ["GIT_PASSWORD"] = "testpass"
        
        mock_repo = Mock()
        mock_repo.heads = [Mock()]
        mock_repo.head.commit.hexsha = "1234567890abcdef"
        
        with patch('git.Repo.clone_from', return_value=mock_repo) as mock_clone, \
             patch('tempfile.mkdtemp', return_value='/tmp/test_repo'), \
             patch('os.chmod'):
            
            await mock_ingestion_plugin._clone_repository(
                "https://gitlab.com/user/private-repo.git",
                "main"
            )
            
            # Verify clone was called with credentials in URL
            call_args = mock_clone.call_args
            assert "testuser:testpass@gitlab.com" in call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_legacy_git_token_authentication(self, mock_ingestion_plugin, clean_env):
        """Test legacy GIT_TOKEN authentication."""
        os.environ["GIT_TOKEN"] = "legacy_token_base64"
        
        mock_repo = Mock()
        mock_repo.heads = [Mock()]
        mock_repo.head.commit.hexsha = "1234567890abcdef"
        
        with patch('git.Repo.clone_from', return_value=mock_repo) as mock_clone, \
             patch('tempfile.mkdtemp', return_value='/tmp/test_repo'), \
             patch('os.chmod'):
            
            await mock_ingestion_plugin._clone_repository(
                "https://dev.azure.com/org/project/_git/repo",
                "main"
            )
            
            # Verify clone was called with authorization header
            call_args = mock_clone.call_args
            assert "Authorization: Basic legacy_token_base64" in call_args[1]['env']['GIT_HTTP_EXTRAHEADER']
    
    @pytest.mark.asyncio
    async def test_authentication_failure_error_handling(self, mock_ingestion_plugin, clean_env):
        """Test handling of authentication failures."""
        with patch('git.Repo.clone_from') as mock_clone, \
             patch('tempfile.mkdtemp', return_value='/tmp/test_repo'), \
             patch('os.chmod'), \
             patch('os.path.exists', return_value=True), \
             patch('shutil.rmtree') as mock_cleanup:
            
            # Simulate authentication failure
            mock_clone.side_effect = GitCommandError(
                ['git', 'clone'], 128, "Authentication failed"
            )
            
            with pytest.raises(GitCommandError) as exc_info:
                await mock_ingestion_plugin._clone_repository(
                    "https://github.com/user/private-repo.git",
                    "main"
                )
            
            # Verify error message includes helpful guidance
            assert "Authentication failed" in str(exc_info.value)
            assert "GITHUB_TOKEN" in str(exc_info.value)
            # Verify cleanup was called
            mock_cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_repository_not_found_error_handling(self, mock_ingestion_plugin, clean_env):
        """Test handling of repository not found errors."""
        with patch('git.Repo.clone_from') as mock_clone, \
             patch('tempfile.mkdtemp', return_value='/tmp/test_repo'), \
             patch('os.chmod'), \
             patch('os.path.exists', return_value=True), \
             patch('shutil.rmtree') as mock_cleanup:
            
            # Simulate repository not found
            mock_clone.side_effect = GitCommandError(
                ['git', 'clone'], 128, "repository does not exist"
            )
            
            with pytest.raises(GitCommandError) as exc_info:
                await mock_ingestion_plugin._clone_repository(
                    "https://github.com/user/nonexistent-repo.git",
                    "main"
                )
            
            # Verify error message is helpful
            assert "Repository not found" in str(exc_info.value)
            assert "Verify URL and access permissions" in str(exc_info.value)
            mock_cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_network_timeout_error_handling(self, mock_ingestion_plugin, clean_env):
        """Test handling of network timeout errors."""
        with patch('git.Repo.clone_from') as mock_clone, \
             patch('tempfile.mkdtemp', return_value='/tmp/test_repo'), \
             patch('os.chmod'), \
             patch('os.path.exists', return_value=True), \
             patch('shutil.rmtree') as mock_cleanup:
            
            # Simulate network timeout
            mock_clone.side_effect = GitCommandError(
                ['git', 'clone'], 128, "network timeout occurred"
            )
            
            with pytest.raises(TimeoutError) as exc_info:
                await mock_ingestion_plugin._clone_repository(
                    "https://github.com/user/large-repo.git",
                    "main"
                )
            
            # Verify error message is helpful
            assert "Network timeout" in str(exc_info.value)
            assert "Check network connectivity" in str(exc_info.value)
            mock_cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_invalid_repository_error_handling(self, mock_ingestion_plugin, clean_env):
        """Test handling of invalid repository (no branches) errors."""
        mock_repo = Mock()
        mock_repo.heads = []  # No branches found
        
        with patch('git.Repo.clone_from', return_value=mock_repo), \
             patch('tempfile.mkdtemp', return_value='/tmp/test_repo'), \
             patch('os.chmod'), \
             patch('os.path.exists', return_value=True), \
             patch('shutil.rmtree') as mock_cleanup:
            
            with pytest.raises(InvalidGitRepositoryError) as exc_info:
                await mock_ingestion_plugin._clone_repository(
                    "https://github.com/user/empty-repo.git",
                    "main"
                )
            
            assert "No branches found" in str(exc_info.value)
            mock_cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_on_unexpected_error(self, mock_ingestion_plugin, clean_env):
        """Test cleanup occurs on unexpected errors."""
        with patch('git.Repo.clone_from') as mock_clone, \
             patch('tempfile.mkdtemp', return_value='/tmp/test_repo'), \
             patch('os.chmod'), \
             patch('os.path.exists', return_value=True), \
             patch('shutil.rmtree') as mock_cleanup:
            
            # Simulate unexpected error
            mock_clone.side_effect = RuntimeError("Unexpected system error")
            
            with pytest.raises(OSError) as exc_info:
                await mock_ingestion_plugin._clone_repository(
                    "https://github.com/user/repo.git",
                    "main"
                )
            
            # Verify error wrapping and cleanup
            assert "Unexpected error cloning" in str(exc_info.value)
            mock_cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_failure_handling(self, mock_ingestion_plugin, clean_env):
        """Test graceful handling of cleanup failures."""
        with patch('git.Repo.clone_from') as mock_clone, \
             patch('tempfile.mkdtemp', return_value='/tmp/test_repo'), \
             patch('os.chmod'), \
             patch('os.path.exists', return_value=True), \
             patch('shutil.rmtree') as mock_cleanup, \
             patch('logging.Logger.warning') as mock_log_warning:
            
            # Simulate clone failure and cleanup failure
            mock_clone.side_effect = GitCommandError(['git', 'clone'], 128, "Clone failed")
            mock_cleanup.side_effect = OSError("Permission denied")
            
            with pytest.raises(GitCommandError):
                await mock_ingestion_plugin._clone_repository(
                    "https://github.com/user/repo.git",
                    "main"
                )
            
            # Verify cleanup was attempted and warning was logged
            mock_cleanup.assert_called_once()
            mock_log_warning.assert_called()
    
    @pytest.mark.asyncio
    async def test_secure_temporary_directory_permissions(self, mock_ingestion_plugin, clean_env):
        """Test that temporary directory has secure permissions."""
        mock_repo = Mock()
        mock_repo.heads = [Mock()]
        mock_repo.head.commit.hexsha = "1234567890abcdef"
        
        with patch('git.Repo.clone_from', return_value=mock_repo), \
             patch('tempfile.mkdtemp', return_value='/tmp/test_repo') as mock_temp, \
             patch('os.chmod') as mock_chmod:
            
            await mock_ingestion_plugin._clone_repository(
                "https://github.com/user/repo.git",
                "main"
            )
            
            # Verify temporary directory has owner-only permissions
            mock_temp.assert_called_once_with(prefix="mosaic_repo_", suffix="_clone")
            mock_chmod.assert_called_once_with('/tmp/test_repo', 0o700)
    
    @pytest.mark.asyncio
    async def test_timeout_configuration(self, mock_ingestion_plugin, clean_env):
        """Test that clone operation includes timeout configuration."""
        mock_repo = Mock()
        mock_repo.heads = [Mock()]
        mock_repo.head.commit.hexsha = "1234567890abcdef"
        
        with patch('git.Repo.clone_from', return_value=mock_repo) as mock_clone, \
             patch('tempfile.mkdtemp', return_value='/tmp/test_repo'), \
             patch('os.chmod'):
            
            await mock_ingestion_plugin._clone_repository(
                "https://github.com/user/repo.git",
                "main"
            )
            
            # Verify timeout was set
            call_args = mock_clone.call_args
            assert call_args[1]['timeout'] == 300  # 5 minutes
    
    @pytest.mark.asyncio 
    async def test_shallow_clone_configuration(self, mock_ingestion_plugin, clean_env):
        """Test that shallow clone is configured for performance."""
        mock_repo = Mock()
        mock_repo.heads = [Mock()]
        mock_repo.head.commit.hexsha = "1234567890abcdef"
        
        with patch('git.Repo.clone_from', return_value=mock_repo) as mock_clone, \
             patch('tempfile.mkdtemp', return_value='/tmp/test_repo'), \
             patch('os.chmod'):
            
            await mock_ingestion_plugin._clone_repository(
                "https://github.com/user/repo.git",
                "main"
            )
            
            # Verify shallow clone configuration
            call_args = mock_clone.call_args
            assert call_args[1]['depth'] == 1
            assert call_args[1]['branch'] == "main"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])