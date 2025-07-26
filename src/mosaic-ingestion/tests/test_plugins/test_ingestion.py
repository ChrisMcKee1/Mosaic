"""
Unit tests for IngestionPlugin.

Tests the core ingestion functionality including GitPython repository access,
tree-sitter AST parsing, entity extraction, and Azure Cosmos DB integration.
"""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

from plugins.ingestion import IngestionPlugin


class TestIngestionPlugin:
    """Test cases for IngestionPlugin class."""

    @pytest.fixture
    def mock_cosmos_client(self):
        """Create mock Cosmos DB client."""
        mock_client = Mock()
        mock_database = Mock()
        mock_container = Mock()

        mock_client.get_database_client.return_value = mock_database
        mock_database.get_container_client.return_value = mock_container

        return mock_client

    @pytest.fixture
    def mock_kernel(self):
        """Create mock Semantic Kernel."""
        return Mock()

    @pytest.fixture
    def ingestion_plugin(self, mock_cosmos_client, mock_kernel):
        """Create IngestionPlugin instance with mocked dependencies."""
        return IngestionPlugin(
            cosmos_client=mock_cosmos_client,
            kernel=mock_kernel,
            database_name="test-db",
            container_name="test-container",
        )

    @pytest.fixture
    def sample_repository_files(self):
        """Sample repository file structure."""
        return [
            {
                "path": "/tmp/repo/src/main.py",
                "relative_path": "src/main.py",
                "size": 2000,
                "language": "python",
            },
            {
                "path": "/tmp/repo/src/utils.py",
                "relative_path": "src/utils.py",
                "size": 1500,
                "language": "python",
            },
            {
                "path": "/tmp/repo/tests/test_main.py",
                "relative_path": "tests/test_main.py",
                "size": 1000,
                "language": "python",
            },
            {
                "path": "/tmp/repo/frontend/app.js",
                "relative_path": "frontend/app.js",
                "size": 3000,
                "language": "javascript",
            },
        ]

    def test_ingestion_plugin_initialization(
        self, ingestion_plugin, mock_cosmos_client, mock_kernel
    ):
        """Test IngestionPlugin initialization."""
        assert ingestion_plugin.cosmos_client == mock_cosmos_client
        assert ingestion_plugin.kernel == mock_kernel
        assert ingestion_plugin.database_name == "test-db"
        assert ingestion_plugin.container_name == "test-container"
        assert ingestion_plugin.supported_languages == {
            "python",
            "javascript",
            "typescript",
            "java",
            "go",
            "rust",
            "c",
            "cpp",
            "csharp",
            "html",
            "css",
        }

    def test_detect_language_from_extension(self, ingestion_plugin):
        """Test language detection from file extensions."""
        test_cases = [
            ("main.py", "python"),
            ("app.js", "javascript"),
            ("component.tsx", "typescript"),
            ("Application.java", "java"),
            ("main.go", "go"),
            ("lib.rs", "rust"),
            ("helper.c", "c"),
            ("algorithm.cpp", "cpp"),
            ("Service.cs", "csharp"),
            ("index.html", "html"),
            ("styles.css", "css"),
            ("unknown.xyz", None),
        ]

        for filename, expected_language in test_cases:
            result = ingestion_plugin._detect_language_from_extension(filename)
            assert result == expected_language

    @pytest.mark.asyncio
    async def test_ingest_repository_success(
        self, ingestion_plugin, sample_repository_files
    ):
        """Test successful repository ingestion."""
        repository_url = "https://github.com/test/repo.git"
        branch = "main"

        # Mock repository cloning
        with (
            patch.object(
                ingestion_plugin, "_clone_repository", return_value="/tmp/repo"
            ) as mock_clone,
            patch.object(
                ingestion_plugin,
                "_scan_repository_files",
                return_value=sample_repository_files,
            ) as mock_scan,
            patch.object(
                ingestion_plugin,
                "_process_files",
                return_value={"entities_extracted": 15},
            ) as mock_process,
            patch.object(ingestion_plugin, "_cleanup_temp_directory") as mock_cleanup,
        ):
            result = await ingestion_plugin.ingest_repository(repository_url, branch)

            # Verify method calls
            mock_clone.assert_called_once_with(repository_url, branch)
            mock_scan.assert_called_once_with("/tmp/repo")
            mock_process.assert_called_once_with(sample_repository_files)
            mock_cleanup.assert_called_once_with("/tmp/repo")

            # Verify result structure
            assert result["repository_url"] == repository_url
            assert result["branch"] == branch
            assert result["status"] == "completed"
            assert result["files_processed"] == len(sample_repository_files)
            assert result["entities_extracted"] == 15

    @pytest.mark.asyncio
    async def test_ingest_repository_clone_failure(self, ingestion_plugin):
        """Test repository ingestion with clone failure."""
        repository_url = "https://github.com/nonexistent/repo.git"
        branch = "main"

        with patch.object(
            ingestion_plugin, "_clone_repository", side_effect=Exception("Clone failed")
        ):
            with pytest.raises(Exception, match="Clone failed"):
                await ingestion_plugin.ingest_repository(repository_url, branch)

    @pytest.mark.asyncio
    async def test_clone_repository_success(self, ingestion_plugin):
        """Test successful repository cloning."""
        repository_url = "https://github.com/test/repo.git"
        branch = "main"

        with (
            patch("git.Repo.clone_from") as mock_clone,
            patch("tempfile.mkdtemp", return_value="/tmp/test_clone"),
        ):
            mock_repo = Mock()
            mock_repo.heads = [Mock()]
            mock_repo.head.commit.hexsha = "abc123"
            mock_clone.return_value = mock_repo

            result = await ingestion_plugin._clone_repository(repository_url, branch)

            assert result == "/tmp/test_clone"
            mock_clone.assert_called_once_with(
                repository_url, "/tmp/test_clone", branch=branch, depth=1, env=None
            )

    @pytest.mark.asyncio
    async def test_clone_repository_with_authentication(self, ingestion_plugin):
        """Test repository cloning with GitHub token authentication."""
        repository_url = "https://github.com/private/repo.git"
        branch = "main"

        with (
            patch("git.Repo.clone_from") as mock_clone,
            patch("tempfile.mkdtemp", return_value="/tmp/test_clone"),
            patch.dict("os.environ", {"GITHUB_TOKEN": "test_token"}),
        ):
            mock_repo = Mock()
            mock_repo.heads = [Mock()]
            mock_clone.return_value = mock_repo

            await ingestion_plugin._clone_repository(repository_url, branch)

            # Verify authentication was configured
            call_args = mock_clone.call_args
            assert call_args[1]["env"] is not None
            assert "GIT_HTTP_EXTRAHEADER" in call_args[1]["env"]

    @pytest.mark.asyncio
    async def test_scan_repository_files(self, ingestion_plugin):
        """Test repository file scanning."""
        repo_path = "/tmp/test_repo"

        # Mock file system structure
        mock_files = [
            Path("/tmp/test_repo/src/main.py"),
            Path("/tmp/test_repo/src/utils.py"),
            Path("/tmp/test_repo/tests/test_main.py"),
            Path("/tmp/test_repo/README.md"),
            Path("/tmp/test_repo/.git/config"),  # Should be ignored
            Path("/tmp/test_repo/node_modules/package.js"),  # Should be ignored
        ]

        with (
            patch("pathlib.Path.rglob") as mock_rglob,
            patch("pathlib.Path.is_file", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
        ):
            # Filter to return only relevant files
            mock_rglob.return_value = [
                f
                for f in mock_files
                if not any(skip_dir in str(f) for skip_dir in [".git", "node_modules"])
            ]

            mock_stat_result = Mock()
            mock_stat_result.st_size = 1000
            mock_stat.return_value = mock_stat_result

            result = await ingestion_plugin._scan_repository_files(repo_path)

            assert len(result) == 4  # Excluding .git and node_modules files

            # Verify file structure
            py_files = [f for f in result if f["language"] == "python"]
            assert len(py_files) == 3

            md_files = [f for f in result if f["relative_path"] == "README.md"]
            assert len(md_files) == 1

    @pytest.mark.asyncio
    async def test_process_files_with_tree_sitter(
        self, ingestion_plugin, sample_repository_files
    ):
        """Test file processing with tree-sitter parsing."""
        # Mock tree-sitter parser setup
        with (
            patch.object(ingestion_plugin, "_setup_tree_sitter_parser") as mock_setup,
            patch.object(
                ingestion_plugin,
                "_parse_file_ast",
                return_value=[
                    {
                        "entity_type": "function",
                        "name": "test_function",
                        "start_line": 10,
                        "end_line": 15,
                        "content": "def test_function(): pass",
                    }
                ],
            ) as mock_parse,
            patch.object(ingestion_plugin, "_store_entities_in_cosmos") as mock_store,
        ):
            mock_parser = Mock()
            mock_setup.return_value = mock_parser

            result = await ingestion_plugin._process_files(sample_repository_files)

            # Verify parser setup for each language
            setup_calls = mock_setup.call_args_list
            languages_setup = {call[0][0] for call in setup_calls}
            assert "python" in languages_setup
            assert "javascript" in languages_setup

            # Verify file parsing
            assert mock_parse.call_count == len(sample_repository_files)

            # Verify result structure
            assert "entities_extracted" in result
            assert "files_processed" in result
            assert result["files_processed"] == len(sample_repository_files)

    def test_setup_tree_sitter_parser_python(self, ingestion_plugin):
        """Test tree-sitter parser setup for Python."""
        with (
            patch("tree_sitter.Language") as mock_language,
            patch("tree_sitter.Parser") as mock_parser_class,
        ):
            mock_lang = Mock()
            mock_language.return_value = mock_lang

            mock_parser = Mock()
            mock_parser_class.return_value = mock_parser

            result = ingestion_plugin._setup_tree_sitter_parser("python")

            assert result == mock_parser
            mock_parser.set_language.assert_called_once_with(mock_lang)

    def test_setup_tree_sitter_parser_unsupported(self, ingestion_plugin):
        """Test tree-sitter parser setup for unsupported language."""
        result = ingestion_plugin._setup_tree_sitter_parser("unsupported")
        assert result is None

    @pytest.mark.asyncio
    async def test_parse_file_ast_python(self, ingestion_plugin):
        """Test AST parsing for Python file."""
        file_info = {
            "path": "/tmp/repo/src/main.py",
            "relative_path": "src/main.py",
            "language": "python",
        }

        python_code = """
def hello_world():
    '''A simple greeting function.'''
    return "Hello, World!"

class Calculator:
    def add(self, a, b):
        return a + b
        
    def multiply(self, a, b):
        return a * b
"""

        # Mock tree-sitter parsing
        with (
            patch("builtins.open", mock_open(read_data=python_code)),
            patch.object(ingestion_plugin, "_setup_tree_sitter_parser") as mock_setup,
        ):
            # Mock parser and tree structure
            mock_parser = Mock()
            mock_tree = Mock()

            # Mock AST nodes
            function_node = Mock()
            function_node.type = "function_definition"
            function_node.start_point = (1, 0)
            function_node.end_point = (3, 25)
            function_node.children = [
                Mock(type="identifier", text=b"hello_world"),
                Mock(type="parameters", children=[]),
                Mock(type="block", children=[]),
            ]

            class_node = Mock()
            class_node.type = "class_definition"
            class_node.start_point = (5, 0)
            class_node.end_point = (11, 20)
            class_node.children = [
                Mock(type="identifier", text=b"Calculator"),
                Mock(type="superclasses", children=[]),
                Mock(type="block", children=[]),
            ]

            mock_tree.root_node.children = [function_node, class_node]
            mock_parser.parse.return_value = mock_tree
            mock_setup.return_value = mock_parser

            result = await ingestion_plugin._parse_file_ast(file_info)

            assert len(result) == 2  # Function and class

            # Check function entity
            func_entity = next(e for e in result if e["entity_type"] == "function")
            assert func_entity["name"] == "hello_world"
            assert func_entity["start_line"] == 2  # 1-indexed
            assert func_entity["end_line"] == 4

            # Check class entity
            class_entity = next(e for e in result if e["entity_type"] == "class")
            assert class_entity["name"] == "Calculator"
            assert class_entity["start_line"] == 6

    @pytest.mark.asyncio
    async def test_parse_file_ast_javascript(self, ingestion_plugin):
        """Test AST parsing for JavaScript file."""
        file_info = {
            "path": "/tmp/repo/src/app.js",
            "relative_path": "src/app.js",
            "language": "javascript",
        }

        js_code = """
function greetUser(name) {
    return `Hello, ${name}!`;
}

class UserService {
    constructor(apiUrl) {
        this.apiUrl = apiUrl;
    }
    
    async fetchUser(id) {
        const response = await fetch(`${this.apiUrl}/users/${id}`);
        return response.json();
    }
}
"""

        with (
            patch("builtins.open", mock_open(read_data=js_code)),
            patch.object(ingestion_plugin, "_setup_tree_sitter_parser") as mock_setup,
        ):
            mock_parser = Mock()
            mock_tree = Mock()

            # Mock JavaScript AST nodes
            function_node = Mock()
            function_node.type = "function_declaration"
            function_node.start_point = (1, 0)
            function_node.end_point = (3, 1)
            function_node.children = [Mock(type="identifier", text=b"greetUser")]

            class_node = Mock()
            class_node.type = "class_declaration"
            class_node.start_point = (5, 0)
            class_node.end_point = (14, 1)
            class_node.children = [Mock(type="identifier", text=b"UserService")]

            mock_tree.root_node.children = [function_node, class_node]
            mock_parser.parse.return_value = mock_tree
            mock_setup.return_value = mock_parser

            result = await ingestion_plugin._parse_file_ast(file_info)

            assert len(result) == 2

            # Verify entities
            func_entity = next(e for e in result if e["entity_type"] == "function")
            assert func_entity["name"] == "greetUser"

            class_entity = next(e for e in result if e["entity_type"] == "class")
            assert class_entity["name"] == "UserService"

    @pytest.mark.asyncio
    async def test_extract_node_name(self, ingestion_plugin):
        """Test extracting entity names from AST nodes."""
        # Mock node with identifier child
        mock_node = Mock()
        mock_identifier = Mock()
        mock_identifier.type = "identifier"
        mock_identifier.text = b"test_function"
        mock_node.children = [mock_identifier]

        result = ingestion_plugin._extract_node_name(mock_node)
        assert result == "test_function"

        # Test node without identifier
        mock_node_no_id = Mock()
        mock_node_no_id.children = []

        result_no_id = ingestion_plugin._extract_node_name(mock_node_no_id)
        assert result_no_id is None

    @pytest.mark.asyncio
    async def test_store_entities_in_cosmos(self, ingestion_plugin):
        """Test storing extracted entities in Cosmos DB."""
        entities = [
            {
                "id": "entity_1",
                "entity_type": "function",
                "name": "test_function",
                "content": "def test_function(): pass",
                "file_context": {
                    "file_path": "/tmp/repo/src/main.py",
                    "language": "python",
                },
            },
            {
                "id": "entity_2",
                "entity_type": "class",
                "name": "TestClass",
                "content": "class TestClass: pass",
                "file_context": {
                    "file_path": "/tmp/repo/src/main.py",
                    "language": "python",
                },
            },
        ]

        # Mock successful Cosmos DB operations
        mock_container = (
            ingestion_plugin.cosmos_client.get_database_client().get_container_client()
        )
        mock_container.upsert_item.return_value = {"id": "success"}

        result = await ingestion_plugin._store_entities_in_cosmos(entities)

        # Verify all entities were stored
        assert mock_container.upsert_item.call_count == len(entities)
        assert result["entities_stored"] == len(entities)
        assert result["errors"] == 0

    @pytest.mark.asyncio
    async def test_store_entities_in_cosmos_with_errors(self, ingestion_plugin):
        """Test storing entities with some Cosmos DB errors."""
        entities = [
            {"id": "entity_1", "name": "valid_entity"},
            {"id": "entity_2", "name": "error_entity"},
        ]

        # Mock mixed success/failure responses
        mock_container = (
            ingestion_plugin.cosmos_client.get_database_client().get_container_client()
        )
        mock_container.upsert_item.side_effect = [
            {"id": "success"},  # First entity succeeds
            Exception("Cosmos DB error"),  # Second entity fails
        ]

        result = await ingestion_plugin._store_entities_in_cosmos(entities)

        assert result["entities_stored"] == 1
        assert result["errors"] == 1

    def test_cleanup_temp_directory(self, ingestion_plugin):
        """Test cleanup of temporary directories."""
        with (
            patch("shutil.rmtree") as mock_rmtree,
            patch("pathlib.Path.exists", return_value=True),
        ):
            ingestion_plugin._cleanup_temp_directory("/tmp/test_dir")

            mock_rmtree.assert_called_once_with("/tmp/test_dir", ignore_errors=True)

    def test_cleanup_temp_directory_not_exists(self, ingestion_plugin):
        """Test cleanup when directory doesn't exist."""
        with (
            patch("shutil.rmtree") as mock_rmtree,
            patch("pathlib.Path.exists", return_value=False),
        ):
            ingestion_plugin._cleanup_temp_directory("/tmp/nonexistent")

            mock_rmtree.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_entity_id(self, ingestion_plugin):
        """Test entity ID generation."""
        file_path = "/tmp/repo/src/main.py"
        entity_name = "test_function"
        line_number = 10

        entity_id = ingestion_plugin._generate_entity_id(
            file_path, entity_name, line_number
        )

        assert isinstance(entity_id, str)
        assert len(entity_id) > 0
        # ID should be deterministic for same inputs
        entity_id_2 = ingestion_plugin._generate_entity_id(
            file_path, entity_name, line_number
        )
        assert entity_id == entity_id_2

    @pytest.mark.asyncio
    async def test_create_golden_node(self, ingestion_plugin):
        """Test GoldenNode creation from extracted entity."""
        entity_data = {
            "entity_type": "function",
            "name": "calculate_metrics",
            "start_line": 10,
            "end_line": 25,
            "content": "def calculate_metrics(data):\n    return sum(data) / len(data)",
        }

        file_info = {
            "path": "/tmp/repo/src/analytics.py",
            "relative_path": "src/analytics.py",
            "size": 2000,
            "language": "python",
        }

        git_info = {
            "repository_url": "https://github.com/test/repo.git",
            "branch": "main",
            "commit_hash": "abc123",
        }

        golden_node = ingestion_plugin._create_golden_node(
            entity_data, file_info, git_info
        )

        # Verify GoldenNode structure
        assert golden_node["entity_type"] == "function"
        assert golden_node["name"] == "calculate_metrics"
        assert golden_node["start_line"] == 10
        assert golden_node["end_line"] == 25
        assert golden_node["content"] == entity_data["content"]

        # Verify file context
        assert golden_node["file_context"]["file_path"] == "/tmp/repo/src/analytics.py"
        assert golden_node["file_context"]["language"] == "python"
        assert golden_node["file_context"]["file_size"] == 2000

        # Verify git context
        assert (
            golden_node["git_context"]["repository_url"]
            == "https://github.com/test/repo.git"
        )
        assert golden_node["git_context"]["branch"] == "main"
        assert golden_node["git_context"]["commit_hash"] == "abc123"

    @pytest.mark.asyncio
    async def test_enhanced_entity_extraction(self, ingestion_plugin):
        """Test enhanced entity extraction with AI descriptions."""
        entity = {
            "id": "test_entity",
            "entity_type": "function",
            "name": "process_data",
            "content": "def process_data(input_data):\n    return [x * 2 for x in input_data]",
        }

        # Mock Semantic Kernel AI enhancement
        with patch.object(
            ingestion_plugin,
            "_enhance_entity_with_ai",
            return_value={
                "ai_description": "Processes input data by doubling each value in the list",
                "ai_summary": "Data transformation function",
                "complexity_score": 2,
            },
        ) as mock_enhance:
            enhanced_entity = await ingestion_plugin._enhance_entity_with_ai(entity)

            mock_enhance.assert_called_once_with(entity)
            assert (
                enhanced_entity["ai_description"]
                == "Processes input data by doubling each value in the list"
            )
            assert enhanced_entity["ai_summary"] == "Data transformation function"
            assert enhanced_entity["complexity_score"] == 2


class TestIngestionPluginIntegration:
    """Integration tests for IngestionPlugin with realistic scenarios."""

    @pytest.mark.asyncio
    async def test_complete_python_project_ingestion(self):
        """Test complete ingestion workflow for a Python project."""
        # Create realistic mock setup
        mock_cosmos_client = Mock()
        mock_kernel = Mock()

        plugin = IngestionPlugin(
            cosmos_client=mock_cosmos_client,
            kernel=mock_kernel,
            database_name="test-knowledge",
            container_name="golden-nodes",
        )

        # Mock realistic Python project structure
        python_project_files = [
            {
                "path": "/tmp/repo/src/__init__.py",
                "relative_path": "src/__init__.py",
                "size": 0,
                "language": "python",
            },
            {
                "path": "/tmp/repo/src/analytics.py",
                "relative_path": "src/analytics.py",
                "size": 3500,
                "language": "python",
            },
            {
                "path": "/tmp/repo/src/models.py",
                "relative_path": "src/models.py",
                "size": 2800,
                "language": "python",
            },
            {
                "path": "/tmp/repo/tests/test_analytics.py",
                "relative_path": "tests/test_analytics.py",
                "size": 4200,
                "language": "python",
            },
        ]

        # Mock the full ingestion pipeline
        with (
            patch.object(plugin, "_clone_repository", return_value="/tmp/repo"),
            patch.object(
                plugin, "_scan_repository_files", return_value=python_project_files
            ),
            patch.object(
                plugin,
                "_process_files",
                return_value={
                    "entities_extracted": 28,
                    "functions_found": 18,
                    "classes_found": 6,
                    "modules_found": 4,
                },
            ),
            patch.object(plugin, "_cleanup_temp_directory"),
        ):
            result = await plugin.ingest_repository(
                "https://github.com/company/analytics-service.git", "main"
            )

            # Verify comprehensive result
            assert result["status"] == "completed"
            assert result["files_processed"] == 4
            assert result["entities_extracted"] == 28
            assert "processing_time_seconds" in result

            # Verify specific entity counts
            assert result["functions_found"] == 18
            assert result["classes_found"] == 6
            assert result["modules_found"] == 4

    @pytest.mark.asyncio
    async def test_multi_language_project_ingestion(self):
        """Test ingestion of project with multiple languages."""
        mock_cosmos_client = Mock()
        mock_kernel = Mock()

        plugin = IngestionPlugin(
            cosmos_client=mock_cosmos_client,
            kernel=mock_kernel,
            database_name="test-knowledge",
            container_name="golden-nodes",
        )

        # Mixed language project
        mixed_files = [
            {"path": "/tmp/repo/backend/main.py", "language": "python", "size": 2000},
            {"path": "/tmp/repo/backend/models.py", "language": "python", "size": 1500},
            {
                "path": "/tmp/repo/frontend/app.js",
                "language": "javascript",
                "size": 3000,
            },
            {
                "path": "/tmp/repo/frontend/components.jsx",
                "language": "javascript",
                "size": 2500,
            },
            {"path": "/tmp/repo/api/service.go", "language": "go", "size": 1800},
            {"path": "/tmp/repo/web/index.html", "language": "html", "size": 800},
            {"path": "/tmp/repo/web/styles.css", "language": "css", "size": 600},
        ]

        with (
            patch.object(plugin, "_clone_repository", return_value="/tmp/repo"),
            patch.object(plugin, "_scan_repository_files", return_value=mixed_files),
            patch.object(
                plugin,
                "_process_files",
                return_value={
                    "entities_extracted": 45,
                    "languages_processed": [
                        "python",
                        "javascript",
                        "go",
                        "html",
                        "css",
                    ],
                    "python_entities": 15,
                    "javascript_entities": 20,
                    "go_entities": 8,
                    "html_entities": 1,
                    "css_entities": 1,
                },
            ),
            patch.object(plugin, "_cleanup_temp_directory"),
        ):
            result = await plugin.ingest_repository(
                "https://github.com/company/fullstack-app.git", "develop"
            )

            # Verify multi-language processing
            assert result["status"] == "completed"
            assert result["files_processed"] == 7
            assert result["entities_extracted"] == 45

            # Verify language-specific processing
            assert len(result["languages_processed"]) == 5
            assert "python" in result["languages_processed"]
            assert "javascript" in result["languages_processed"]
            assert "go" in result["languages_processed"]
