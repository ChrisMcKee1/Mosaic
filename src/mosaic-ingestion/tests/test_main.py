"""
Unit tests for main entry points.

Tests the main ingestion service entry points including both production
and local development versions.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

# Import main modules
from main import IngestionService, main as production_main
from local_main import LocalIngestionService, main as local_main


class TestIngestionService:
    """Test cases for production IngestionService."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = Mock()
        settings.azure_openai_endpoint = "https://test.openai.azure.com/"
        settings.azure_cosmos_endpoint = "https://test.cosmos.azure.com/"
        return settings

    @pytest.fixture
    def ingestion_service(self, mock_settings):
        """Create IngestionService instance."""
        return IngestionService(mock_settings)

    def test_ingestion_service_initialization(self, ingestion_service, mock_settings):
        """Test IngestionService initialization."""
        assert ingestion_service.settings == mock_settings
        assert ingestion_service.orchestrator is None

    @pytest.mark.asyncio
    async def test_initialize_success(self, ingestion_service):
        """Test successful initialization."""
        with patch("main.MosaicMagenticOrchestrator") as mock_orchestrator_class:
            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            await ingestion_service.initialize()

            assert ingestion_service.orchestrator == mock_orchestrator
            mock_orchestrator_class.assert_called_once_with(ingestion_service.settings)

    @pytest.mark.asyncio
    async def test_initialize_failure(self, ingestion_service):
        """Test initialization failure."""
        with patch(
            "main.MosaicMagenticOrchestrator", side_effect=Exception("Init failed")
        ):
            with pytest.raises(Exception, match="Init failed"):
                await ingestion_service.initialize()

    @pytest.mark.asyncio
    async def test_ingest_repository_success(self, ingestion_service):
        """Test successful repository ingestion."""
        # Setup mock orchestrator
        mock_orchestrator = AsyncMock()
        mock_orchestrator.orchestrate_repository_ingestion.return_value = {
            "repository_url": "https://github.com/test/repo.git",
            "branch": "main",
            "status": "completed",
            "entities_extracted": 25,
        }
        ingestion_service.orchestrator = mock_orchestrator

        result = await ingestion_service.ingest_repository(
            "https://github.com/test/repo.git", "main"
        )

        assert result["status"] == "completed"
        assert result["entities_extracted"] == 25
        mock_orchestrator.orchestrate_repository_ingestion.assert_called_once_with(
            "https://github.com/test/repo.git", "main"
        )

    @pytest.mark.asyncio
    async def test_ingest_repository_not_initialized(self, ingestion_service):
        """Test repository ingestion without initialization."""
        with pytest.raises(RuntimeError, match="Magentic orchestrator not initialized"):
            await ingestion_service.ingest_repository(
                "https://github.com/test/repo.git", "main"
            )

    @pytest.mark.asyncio
    async def test_cleanup(self, ingestion_service):
        """Test service cleanup."""
        mock_orchestrator = AsyncMock()
        ingestion_service.orchestrator = mock_orchestrator

        await ingestion_service.cleanup()

        mock_orchestrator.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_no_orchestrator(self, ingestion_service):
        """Test cleanup with no orchestrator."""
        # Should not raise exception
        await ingestion_service.cleanup()


class TestLocalIngestionService:
    """Test cases for local development LocalIngestionService."""

    @pytest.fixture
    def local_service(self):
        """Create LocalIngestionService instance."""
        return LocalIngestionService()

    def test_local_ingestion_service_initialization(self, local_service):
        """Test LocalIngestionService initialization."""
        assert local_service.supported_extensions is not None
        assert ".py" in local_service.supported_extensions
        assert ".js" in local_service.supported_extensions

        # Verify initial stats
        assert local_service.stats["files_processed"] == 0
        assert local_service.stats["lines_of_code"] == 0
        assert local_service.stats["languages_detected"] == set()

    @pytest.mark.asyncio
    async def test_local_ingest_repository_success(self, local_service):
        """Test successful local repository ingestion."""
        repository_url = "https://github.com/test/simple-repo.git"
        branch = "main"

        with (
            patch.object(
                local_service, "_clone_repository", return_value="/tmp/test_repo"
            ) as mock_clone,
            patch.object(local_service, "_scan_repository") as mock_scan,
            patch("shutil.rmtree"),
        ):
            # Mock stats after scanning
            local_service.stats = {
                "files_processed": 5,
                "lines_of_code": 200,
                "languages_detected": {"python", "javascript"},
                "entities_found": 15,
            }

            result = await local_service.ingest_repository(repository_url, branch)

            mock_clone.assert_called_once_with(repository_url, branch)
            mock_scan.assert_called_once_with("/tmp/test_repo")

            # Verify result structure
            assert result["repository_url"] == repository_url
            assert result["branch"] == branch
            assert result["status"] == "completed"
            assert result["mode"] == "local_development"
            assert result["files_processed"] == 5
            assert result["lines_of_code"] == 200
            assert "python" in result["languages_detected"]
            assert "javascript" in result["languages_detected"]

    @pytest.mark.asyncio
    async def test_local_clone_repository_success(self, local_service):
        """Test successful local repository cloning."""
        repository_url = "https://github.com/test/repo.git"
        branch = "main"

        with (
            patch("git.Repo.clone_from") as mock_clone,
            patch("tempfile.mkdtemp", return_value="/tmp/local_test"),
        ):
            mock_repo = Mock()
            mock_repo.heads = [Mock()]
            mock_repo.head.commit.hexsha = "abc123def"
            mock_clone.return_value = mock_repo

            result = await local_service._clone_repository(repository_url, branch)

            assert result == "/tmp/local_test"
            mock_clone.assert_called_once_with(
                repository_url, "/tmp/local_test", branch=branch, depth=1, env=None
            )

    @pytest.mark.asyncio
    async def test_local_clone_with_github_token(self, local_service):
        """Test local cloning with GitHub token authentication."""
        repository_url = "https://github.com/private/repo.git"
        branch = "main"

        with (
            patch("git.Repo.clone_from") as mock_clone,
            patch("tempfile.mkdtemp", return_value="/tmp/auth_test"),
            patch("os.getenv", return_value="test_github_token"),
        ):
            mock_repo = Mock()
            mock_repo.heads = [Mock()]
            mock_clone.return_value = mock_repo

            await local_service._clone_repository(repository_url, branch)

            # Verify authentication header was set
            call_args = mock_clone.call_args
            assert "env" in call_args.kwargs
            env = call_args.kwargs["env"]
            assert "GIT_HTTP_EXTRAHEADER" in env

    @pytest.mark.asyncio
    async def test_local_scan_repository(self, local_service):
        """Test local repository scanning."""
        repo_path = "/tmp/test_repo"

        # Mock file system with realistic Python project
        from pathlib import Path

        mock_files = [
            Path("/tmp/test_repo/src/main.py"),
            Path("/tmp/test_repo/src/utils.py"),
            Path("/tmp/test_repo/tests/test_main.py"),
            Path("/tmp/test_repo/README.md"),
            Path("/tmp/test_repo/.git/config"),  # Should be skipped
        ]

        with (
            patch("pathlib.Path.rglob", return_value=mock_files),
            patch("pathlib.Path.is_file", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
            patch.object(local_service, "_process_file") as mock_process,
        ):
            mock_stat_result = Mock()
            mock_stat_result.st_size = 1000
            mock_stat.return_value = mock_stat_result

            await local_service._scan_repository(repo_path)

            # Should process Python files but skip .git
            assert mock_process.call_count == 3  # Only Python files

            # Verify stats were updated
            assert local_service.stats["languages_detected"] == {"python"}

    @pytest.mark.asyncio
    async def test_local_process_file_python(self, local_service):
        """Test processing individual Python file."""
        from pathlib import Path

        file_path = Path("/tmp/repo/src/example.py")
        language = "python"

        python_content = """
def hello_world():
    '''Simple greeting function.'''
    return "Hello, World!"

class Calculator:
    def add(self, a, b):
        return a + b
        
    def multiply(self, a, b):
        return a * b

# This is a comment
CONSTANT_VALUE = 42
"""

        with (
            patch("pathlib.Path.read_text", return_value=python_content),
            patch("pathlib.Path.stat") as mock_stat,
        ):
            mock_stat_result = Mock()
            mock_stat_result.st_size = len(python_content)
            mock_stat.return_value = mock_stat_result

            await local_service._process_file(file_path, language)

            # Verify stats were updated
            assert local_service.stats["files_processed"] == 1
            assert local_service.stats["lines_of_code"] > 0
            assert (
                local_service.stats["entities_found"] > 0
            )  # Should find function and class

    def test_local_detect_simple_entities_python(self, local_service):
        """Test simple entity detection for Python code."""
        python_content = """
def calculate_average(numbers):
    return sum(numbers) / len(numbers)

class DataProcessor:
    def __init__(self):
        pass
        
    async def process_data(self, data):
        return [x * 2 for x in data]
"""

        entities = local_service._detect_simple_entities(python_content, "python")

        # Should find function, class, and async function
        assert len(entities) >= 3

        entity_types = [e["type"] for e in entities]
        assert "function" in entity_types
        assert "class" in entity_types
        assert "async_function" in entity_types

    def test_local_detect_simple_entities_javascript(self, local_service):
        """Test simple entity detection for JavaScript code."""
        js_content = """
function greetUser(name) {
    return `Hello, ${name}!`;
}

class UserService {
    constructor(apiUrl) {
        this.apiUrl = apiUrl;
    }
    
    async fetchUser(id) {
        return await fetch(`${this.apiUrl}/users/${id}`);
    }
}

const helper = (data) => {
    return data.map(x => x * 2);
};
"""

        entities = local_service._detect_simple_entities(js_content, "javascript")

        # Should find various JavaScript constructs
        assert len(entities) >= 3

        entity_types = [e["type"] for e in entities]
        assert "function" in entity_types
        assert "class" in entity_types


class TestMainFunctions:
    """Test main entry point functions."""

    @pytest.mark.asyncio
    async def test_production_main_success(self):
        """Test production main function success path."""
        test_args = [
            "main.py",
            "--repository-url",
            "https://github.com/test/repo.git",
            "--branch",
            "main",
        ]

        with (
            patch("sys.argv", test_args),
            patch("main.MosaicSettings") as mock_settings_class,
            patch("main.IngestionService") as mock_service_class,
        ):
            mock_settings = Mock()
            mock_settings_class.return_value = mock_settings

            mock_service = AsyncMock()
            mock_service.ingest_repository.return_value = {
                "repository_url": "https://github.com/test/repo.git",
                "branch": "main",
                "status": "completed",
                "agents_executed": 5,
                "processing_time_seconds": 45.2,
            }
            mock_service_class.return_value = mock_service

            # Run main function
            await production_main()

            # Verify service calls
            mock_service.initialize.assert_called_once()
            mock_service.ingest_repository.assert_called_once_with(
                "https://github.com/test/repo.git", "main"
            )
            mock_service.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_local_main_success(self):
        """Test local main function success path."""
        test_args = [
            "local_main.py",
            "--repository-url",
            "https://github.com/test/simple-repo.git",
            "--branch",
            "develop",
            "--debug",
        ]

        with (
            patch("sys.argv", test_args),
            patch("local_main.LocalIngestionService") as mock_service_class,
        ):
            mock_service = AsyncMock()
            mock_service.ingest_repository.return_value = {
                "repository_url": "https://github.com/test/simple-repo.git",
                "branch": "develop",
                "status": "completed",
                "mode": "local_development",
                "files_processed": 12,
                "lines_of_code": 1500,
            }
            mock_service_class.return_value = mock_service

            # Capture logging output
            with patch("logging.getLogger") as mock_logger:
                await local_main()

                # Verify service calls
                mock_service.ingest_repository.assert_called_once_with(
                    "https://github.com/test/simple-repo.git", "develop"
                )

    @pytest.mark.asyncio
    async def test_main_keyboard_interrupt(self):
        """Test graceful handling of keyboard interrupt."""
        test_args = ["main.py", "--repository-url", "https://github.com/test/repo.git"]

        with (
            patch("sys.argv", test_args),
            patch("main.IngestionService") as mock_service_class,
        ):
            mock_service = AsyncMock()
            mock_service.initialize.side_effect = KeyboardInterrupt()
            mock_service_class.return_value = mock_service

            # Should handle KeyboardInterrupt gracefully
            await production_main()

            # Cleanup should still be called
            mock_service.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_with_invalid_args(self):
        """Test main function with invalid arguments."""
        test_args = ["main.py"]  # Missing required --repository-url

        with patch("sys.argv", test_args), patch("sys.exit") as mock_exit:
            await production_main()

            # Should exit with error code
            mock_exit.assert_called_once_with(1)


class TestMainIntegration:
    """Integration tests for main entry points."""

    @pytest.mark.asyncio
    async def test_end_to_end_local_ingestion(self):
        """Test complete end-to-end local ingestion workflow."""
        # This test would be run with actual git repository in real scenario
        repository_url = "https://github.com/octocat/Hello-World.git"

        local_service = LocalIngestionService()

        # Mock only the git operations, let everything else run
        with (
            patch.object(local_service, "_clone_repository") as mock_clone,
            patch("shutil.rmtree"),
        ):
            # Mock a simple repository structure
            mock_clone.return_value = "/tmp/mock_hello_world"

            # Mock file system scanning
            with (
                patch("pathlib.Path.rglob") as mock_rglob,
                patch("pathlib.Path.is_file", return_value=True),
                patch("pathlib.Path.stat") as mock_stat,
                patch(
                    "pathlib.Path.read_text",
                    return_value="# Hello World\nprint('Hello, World!')",
                ),
            ):
                mock_rglob.return_value = [Path("/tmp/mock_hello_world/hello_world.py")]
                mock_stat_result = Mock()
                mock_stat_result.st_size = 50
                mock_stat.return_value = mock_stat_result

                result = await local_service.ingest_repository(repository_url, "main")

                # Verify end-to-end result
                assert result["status"] == "completed"
                assert result["repository_url"] == repository_url
                assert result["mode"] == "local_development"
                assert result["files_processed"] >= 0
                assert isinstance(result["timestamp"], str)
