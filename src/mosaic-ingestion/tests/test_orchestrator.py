"""
Unit tests for MosaicMagenticOrchestrator.

Tests the AI agent orchestration system that coordinates specialized agents
for comprehensive repository ingestion and knowledge graph construction.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import asyncio

from orchestrator import MosaicMagenticOrchestrator


class TestMosaicMagenticOrchestrator:
    """Test cases for MosaicMagenticOrchestrator class."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock()
        settings.azure_openai_endpoint = "https://test.openai.azure.com/"
        settings.azure_cosmos_endpoint = "https://test.cosmos.azure.com/"
        settings.database_name = "test-database"
        settings.container_name = "test-container"
        return settings

    @pytest.fixture
    def orchestrator(self, mock_settings):
        """Create orchestrator instance with mocked dependencies."""
        return MosaicMagenticOrchestrator(mock_settings)

    @pytest.fixture
    def sample_repository_data(self):
        """Sample repository data for testing."""
        return {
            "repository_url": "https://github.com/test/repo.git",
            "branch": "main",
            "total_files": 50,
            "languages": ["python", "javascript"],
            "file_list": [
                {"path": "src/main.py", "size": 1000},
                {"path": "src/utils.py", "size": 500},
                {"path": "tests/test_main.py", "size": 800},
            ],
        }

    def test_orchestrator_initialization(self, orchestrator, mock_settings):
        """Test orchestrator initialization."""
        assert orchestrator.settings == mock_settings
        assert orchestrator.agents_executed == 0
        assert orchestrator.processing_start_time is None
        assert orchestrator.orchestration_result is None

    @pytest.mark.asyncio
    async def test_orchestrate_repository_ingestion_success(
        self, orchestrator, sample_repository_data
    ):
        """Test successful repository ingestion orchestration."""
        repository_url = sample_repository_data["repository_url"]
        branch = sample_repository_data["branch"]

        # Mock the agent execution methods
        with (
            patch.object(
                orchestrator,
                "_execute_git_sleuth_agent",
                return_value=sample_repository_data,
            ) as mock_git_sleuth,
            patch.object(
                orchestrator,
                "_execute_code_parser_agent",
                return_value={"entities_parsed": 25},
            ) as mock_code_parser,
            patch.object(
                orchestrator,
                "_execute_graph_architect_agent",
                return_value={"relationships_mapped": 50},
            ) as mock_graph_architect,
            patch.object(
                orchestrator,
                "_execute_docu_writer_agent",
                return_value={"docs_enhanced": 25},
            ) as mock_docu_writer,
            patch.object(
                orchestrator,
                "_execute_graph_auditor_agent",
                return_value={"quality_score": 0.95},
            ) as mock_graph_auditor,
        ):
            result = await orchestrator.orchestrate_repository_ingestion(
                repository_url, branch
            )

            # Verify all agents were called
            mock_git_sleuth.assert_called_once_with(repository_url, branch)
            mock_code_parser.assert_called_once_with(sample_repository_data)
            mock_graph_architect.assert_called_once()
            mock_docu_writer.assert_called_once()
            mock_graph_auditor.assert_called_once()

            # Verify result structure
            assert result["repository_url"] == repository_url
            assert result["branch"] == branch
            assert result["status"] == "completed"
            assert result["agents_executed"] == 5
            assert "processing_time_seconds" in result
            assert "orchestration_result" in result

    @pytest.mark.asyncio
    async def test_orchestrate_repository_ingestion_failure(self, orchestrator):
        """Test repository ingestion orchestration with failure."""
        repository_url = "https://github.com/test/repo.git"
        branch = "main"

        # Mock git sleuth agent to raise exception
        with patch.object(
            orchestrator,
            "_execute_git_sleuth_agent",
            side_effect=Exception("Git clone failed"),
        ):
            with pytest.raises(Exception, match="Git clone failed"):
                await orchestrator.orchestrate_repository_ingestion(
                    repository_url, branch
                )

    @pytest.mark.asyncio
    async def test_execute_git_sleuth_agent(self, orchestrator):
        """Test GitSleuth agent execution."""
        repository_url = "https://github.com/test/repo.git"
        branch = "main"

        # Mock the underlying git operations
        with (
            patch("git.Repo.clone_from") as mock_clone,
            patch("tempfile.mkdtemp", return_value="/tmp/test_repo"),
            patch("shutil.rmtree"),
            patch(
                "os.walk",
                return_value=[
                    ("/tmp/test_repo", [], ["file1.py", "file2.js"]),
                    ("/tmp/test_repo/src", [], ["main.py"]),
                ],
            ),
        ):
            mock_repo = Mock()
            mock_repo.heads = [Mock()]
            mock_repo.head.commit.hexsha = "abc123def456"
            mock_clone.return_value = mock_repo

            result = await orchestrator._execute_git_sleuth_agent(
                repository_url, branch
            )

            assert result["repository_url"] == repository_url
            assert result["branch"] == branch
            assert "total_files" in result
            assert "file_list" in result
            assert isinstance(result["file_list"], list)

    @pytest.mark.asyncio
    async def test_execute_code_parser_agent(
        self, orchestrator, sample_repository_data
    ):
        """Test CodeParser agent execution."""
        # Mock tree-sitter parsing operations
        with patch.object(
            orchestrator,
            "_parse_files_with_tree_sitter",
            return_value={
                "entities_found": 30,
                "functions_parsed": 15,
                "classes_parsed": 8,
                "modules_parsed": 5,
                "golden_nodes": [],
            },
        ) as mock_parser:
            result = await orchestrator._execute_code_parser_agent(
                sample_repository_data
            )

            mock_parser.assert_called_once_with(sample_repository_data["file_list"])
            assert result["entities_found"] == 30
            assert result["functions_parsed"] == 15
            assert result["classes_parsed"] == 8

    @pytest.mark.asyncio
    async def test_execute_graph_architect_agent(self, orchestrator):
        """Test GraphArchitect agent execution."""
        # Mock golden nodes data
        orchestrator.golden_nodes = [
            {"id": "node1", "entity_type": "function", "name": "func1"},
            {"id": "node2", "entity_type": "function", "name": "func2"},
        ]

        with patch.object(
            orchestrator,
            "_analyze_relationships",
            return_value={
                "relationships_found": 10,
                "dependency_links": 5,
                "call_relationships": 5,
            },
        ) as mock_analyze:
            result = await orchestrator._execute_graph_architect_agent()

            mock_analyze.assert_called_once_with(orchestrator.golden_nodes)
            assert result["relationships_found"] == 10
            assert result["dependency_links"] == 5

    @pytest.mark.asyncio
    async def test_execute_docu_writer_agent(self, orchestrator):
        """Test DocuWriter agent execution."""
        # Mock golden nodes data
        orchestrator.golden_nodes = [
            {
                "id": "node1",
                "entity_type": "function",
                "name": "func1",
                "content": "def func1(): pass",
            },
        ]

        with patch.object(
            orchestrator,
            "_enhance_with_ai_descriptions",
            return_value={
                "nodes_enhanced": 1,
                "descriptions_added": 1,
                "summaries_generated": 1,
            },
        ) as mock_enhance:
            result = await orchestrator._execute_docu_writer_agent()

            mock_enhance.assert_called_once_with(orchestrator.golden_nodes)
            assert result["nodes_enhanced"] == 1
            assert result["descriptions_added"] == 1

    @pytest.mark.asyncio
    async def test_execute_graph_auditor_agent(self, orchestrator):
        """Test GraphAuditor agent execution."""
        # Mock golden nodes and relationships
        orchestrator.golden_nodes = [
            {"id": "node1", "entity_type": "function", "name": "func1"},
            {"id": "node2", "entity_type": "function", "name": "func2"},
        ]

        with patch.object(
            orchestrator,
            "_validate_graph_quality",
            return_value={
                "quality_score": 0.92,
                "completeness_score": 0.88,
                "accuracy_score": 0.95,
                "issues_found": [],
                "nodes_validated": 2,
            },
        ) as mock_validate:
            result = await orchestrator._execute_graph_auditor_agent()

            mock_validate.assert_called_once_with(orchestrator.golden_nodes)
            assert result["quality_score"] == 0.92
            assert result["completeness_score"] == 0.88
            assert result["nodes_validated"] == 2

    @pytest.mark.asyncio
    async def test_parse_files_with_tree_sitter(self, orchestrator):
        """Test tree-sitter parsing functionality."""
        file_list = [
            {"path": "/tmp/repo/src/main.py", "size": 1000},
            {"path": "/tmp/repo/src/utils.js", "size": 500},
        ]

        # Mock tree-sitter operations
        with (
            patch("tree_sitter.Language"),
            patch("tree_sitter.Parser") as mock_parser_class,
        ):
            mock_parser = Mock()
            mock_parser_class.return_value = mock_parser

            # Mock parsed tree
            mock_tree = Mock()
            mock_node = Mock()
            mock_node.type = "function_definition"
            mock_node.start_point = (10, 0)
            mock_node.end_point = (15, 0)
            mock_tree.root_node.children = [mock_node]
            mock_parser.parse.return_value = mock_tree

            # Mock file reading
            with patch("builtins.open", mock_open=True) as mock_file:
                mock_file.return_value.read.return_value = "def test(): pass"

                result = await orchestrator._parse_files_with_tree_sitter(file_list)

                assert "entities_found" in result
                assert "golden_nodes" in result
                assert isinstance(result["golden_nodes"], list)

    @pytest.mark.asyncio
    async def test_analyze_relationships(self, orchestrator):
        """Test relationship analysis functionality."""
        golden_nodes = [
            {
                "id": "node1",
                "entity_type": "function",
                "name": "caller_func",
                "content": "def caller_func():\n    helper_func()",
                "relationships": [],
            },
            {
                "id": "node2",
                "entity_type": "function",
                "name": "helper_func",
                "content": "def helper_func():\n    pass",
                "relationships": [],
            },
        ]

        result = await orchestrator._analyze_relationships(golden_nodes)

        assert "relationships_found" in result
        assert isinstance(result["relationships_found"], int)
        assert result["relationships_found"] >= 0

    @pytest.mark.asyncio
    async def test_enhance_with_ai_descriptions(self, orchestrator):
        """Test AI description enhancement."""
        golden_nodes = [
            {
                "id": "node1",
                "entity_type": "function",
                "name": "calculate_metrics",
                "content": "def calculate_metrics(data):\n    return sum(data) / len(data)",
                "ai_description": None,
                "ai_summary": None,
            }
        ]

        # Mock Azure OpenAI calls
        with (
            patch.object(
                orchestrator,
                "_generate_ai_description",
                return_value="Calculates average of data values",
            ) as mock_desc,
            patch.object(
                orchestrator,
                "_generate_ai_summary",
                return_value="Data aggregation function",
            ) as mock_summary,
        ):
            result = await orchestrator._enhance_with_ai_descriptions(golden_nodes)

            mock_desc.assert_called()
            mock_summary.assert_called()
            assert result["nodes_enhanced"] == 1
            assert result["descriptions_added"] == 1

    @pytest.mark.asyncio
    async def test_validate_graph_quality(self, orchestrator):
        """Test graph quality validation."""
        golden_nodes = [
            {
                "id": "node1",
                "entity_type": "function",
                "name": "valid_func",
                "content": "def valid_func(): pass",
                "relationships": [],
                "ai_description": "A valid function",
            },
            {
                "id": "node2",
                "entity_type": "function",
                "name": "incomplete_func",
                "content": "def incomplete_func(): pass",
                "relationships": [],
                "ai_description": None,  # Missing AI description
            },
        ]

        result = await orchestrator._validate_graph_quality(golden_nodes)

        assert "quality_score" in result
        assert "completeness_score" in result
        assert "accuracy_score" in result
        assert "issues_found" in result
        assert "nodes_validated" in result

        assert 0.0 <= result["quality_score"] <= 1.0
        assert result["nodes_validated"] == 2
        # Should find issue with incomplete node
        assert len(result["issues_found"]) > 0

    @pytest.mark.asyncio
    async def test_cleanup(self, orchestrator):
        """Test orchestrator cleanup."""
        # Set up some state to clean
        orchestrator.golden_nodes = [{"test": "data"}]
        orchestrator.agents_executed = 3
        orchestrator.processing_start_time = datetime.now()

        await orchestrator.cleanup()

        # Verify cleanup occurred
        assert orchestrator.golden_nodes == []
        assert orchestrator.agents_executed == 0
        assert orchestrator.processing_start_time is None
        assert orchestrator.orchestration_result is None

    def test_performance_tracking(self, orchestrator):
        """Test performance tracking functionality."""
        # Test that performance metrics are tracked
        orchestrator.processing_start_time = datetime.now()
        orchestrator.agents_executed = 3

        # Simulate getting processing time
        processing_time = orchestrator._get_processing_time_seconds()

        assert isinstance(processing_time, float)
        assert processing_time >= 0.0

    @pytest.mark.asyncio
    async def test_error_handling_in_orchestration(self, orchestrator):
        """Test error handling during orchestration."""
        repository_url = "https://github.com/test/repo.git"
        branch = "main"

        # Mock git sleuth to succeed but code parser to fail
        with (
            patch.object(
                orchestrator,
                "_execute_git_sleuth_agent",
                return_value={"success": True},
            ) as mock_git,
            patch.object(
                orchestrator,
                "_execute_code_parser_agent",
                side_effect=Exception("Parsing failed"),
            ) as mock_parser,
        ):
            with pytest.raises(Exception, match="Parsing failed"):
                await orchestrator.orchestrate_repository_ingestion(
                    repository_url, branch
                )

            # Verify git sleuth was called but orchestration failed
            mock_git.assert_called_once()
            mock_parser.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_agent_execution(
        self, orchestrator, sample_repository_data
    ):
        """Test that agents can be executed concurrently where appropriate."""

        # Mock all agents with different execution times
        async def slow_agent(*args, **kwargs):
            await asyncio.sleep(0.1)
            return {"result": "slow"}

        async def fast_agent(*args, **kwargs):
            await asyncio.sleep(0.05)
            return {"result": "fast"}

        with (
            patch.object(
                orchestrator,
                "_execute_git_sleuth_agent",
                return_value=sample_repository_data,
            ),
            patch.object(
                orchestrator, "_execute_code_parser_agent", side_effect=slow_agent
            ),
            patch.object(
                orchestrator, "_execute_graph_architect_agent", side_effect=fast_agent
            ),
            patch.object(
                orchestrator, "_execute_docu_writer_agent", side_effect=fast_agent
            ),
            patch.object(
                orchestrator, "_execute_graph_auditor_agent", side_effect=fast_agent
            ),
        ):
            start_time = datetime.now()
            result = await orchestrator.orchestrate_repository_ingestion(
                "https://github.com/test/repo.git", "main"
            )
            end_time = datetime.now()

            # Verify execution completed
            assert result["status"] == "completed"

            # Execution time should be reasonable (not sum of all agent times)
            execution_time = (end_time - start_time).total_seconds()
            assert execution_time < 1.0  # Should complete quickly with mocked agents


class TestMosaicMagenticOrchestratorIntegration:
    """Integration tests for MosaicMagenticOrchestrator with realistic scenarios."""

    @pytest.fixture
    def realistic_orchestrator(self):
        """Create orchestrator with realistic settings."""
        mock_settings = Mock()
        mock_settings.azure_openai_endpoint = "https://company.openai.azure.com/"
        mock_settings.azure_cosmos_endpoint = "https://company.cosmos.azure.com/"
        mock_settings.database_name = "mosaic-knowledge"
        mock_settings.container_name = "golden-nodes"
        mock_settings.openai_api_version = "2024-02-01"
        mock_settings.embedding_model = "text-embedding-3-small"
        mock_settings.chat_model = "gpt-4"
        return MosaicMagenticOrchestrator(mock_settings)

    @pytest.mark.asyncio
    async def test_full_python_project_ingestion(self, realistic_orchestrator):
        """Test complete ingestion workflow for a Python project."""
        repository_url = "https://github.com/company/analytics-service.git"
        branch = "main"

        # Mock realistic project structure
        project_files = [
            {"path": "/tmp/repo/src/__init__.py", "size": 0},
            {"path": "/tmp/repo/src/analytics.py", "size": 2500},
            {"path": "/tmp/repo/src/models.py", "size": 1800},
            {"path": "/tmp/repo/src/utils.py", "size": 1200},
            {"path": "/tmp/repo/tests/test_analytics.py", "size": 3000},
            {"path": "/tmp/repo/requirements.txt", "size": 400},
            {"path": "/tmp/repo/README.md", "size": 1500},
        ]

        # Mock complete orchestration chain
        with (
            patch.object(
                realistic_orchestrator,
                "_execute_git_sleuth_agent",
                return_value={
                    "repository_url": repository_url,
                    "branch": branch,
                    "total_files": len(project_files),
                    "file_list": project_files,
                    "languages": ["python"],
                },
            ),
            patch.object(
                realistic_orchestrator,
                "_execute_code_parser_agent",
                return_value={
                    "entities_found": 25,
                    "functions_parsed": 15,
                    "classes_parsed": 5,
                    "modules_parsed": 4,
                    "golden_nodes": [
                        {
                            "id": "func_analytics_calculate",
                            "entity_type": "function",
                            "name": "calculate_metrics",
                            "content": "def calculate_metrics(data): return sum(data)",
                            "file_context": {"file_path": "/tmp/repo/src/analytics.py"},
                        }
                    ],
                },
            ),
            patch.object(
                realistic_orchestrator,
                "_execute_graph_architect_agent",
                return_value={
                    "relationships_found": 12,
                    "dependency_links": 8,
                    "call_relationships": 4,
                },
            ),
            patch.object(
                realistic_orchestrator,
                "_execute_docu_writer_agent",
                return_value={
                    "nodes_enhanced": 25,
                    "descriptions_added": 25,
                    "summaries_generated": 25,
                },
            ),
            patch.object(
                realistic_orchestrator,
                "_execute_graph_auditor_agent",
                return_value={
                    "quality_score": 0.93,
                    "completeness_score": 0.91,
                    "accuracy_score": 0.95,
                    "issues_found": [],
                    "nodes_validated": 25,
                },
            ),
        ):
            result = await realistic_orchestrator.orchestrate_repository_ingestion(
                repository_url, branch
            )

            # Verify comprehensive result
            assert result["repository_url"] == repository_url
            assert result["branch"] == branch
            assert result["status"] == "completed"
            assert result["agents_executed"] == 5
            assert result["orchestration_result"]["entities_found"] == 25
            assert result["orchestration_result"]["quality_score"] == 0.93

            # Verify processing metrics
            assert "processing_time_seconds" in result
            assert isinstance(result["processing_time_seconds"], float)
