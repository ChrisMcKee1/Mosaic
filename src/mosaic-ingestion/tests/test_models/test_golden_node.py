"""
Unit tests for GoldenNode Pydantic models.

Tests the core data models used throughout the mosaic-ingestion service,
ensuring proper validation, serialization, and Azure Cosmos DB compatibility.
"""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import ValidationError

from models.golden_node import (
    EntityType,
    LanguageType,
    ProcessingStatus,
    AgentType,
    FileContext,
    GitContext,
    Dependency,
    Relationship,
    AgentResult,
    GoldenNode,
)


class TestEntityType:
    """Test EntityType enum."""

    def test_entity_type_values(self):
        """Test all entity type enum values."""
        assert EntityType.FUNCTION == "function"
        assert EntityType.CLASS == "class"
        assert EntityType.MODULE == "module"
        assert EntityType.INTERFACE == "interface"
        assert EntityType.HTML_ELEMENT == "html_element"
        assert EntityType.CSS_RULE == "css_rule"

    def test_entity_type_membership(self):
        """Test entity type membership checks."""
        assert "function" in EntityType
        assert "invalid_type" not in EntityType


class TestLanguageType:
    """Test LanguageType enum."""

    def test_language_type_values(self):
        """Test all language type enum values."""
        assert LanguageType.PYTHON == "python"
        assert LanguageType.JAVASCRIPT == "javascript"
        assert LanguageType.TYPESCRIPT == "typescript"
        assert LanguageType.JAVA == "java"
        assert LanguageType.GO == "go"
        assert LanguageType.RUST == "rust"
        assert LanguageType.C == "c"
        assert LanguageType.CPP == "cpp"
        assert LanguageType.CSHARP == "csharp"
        assert LanguageType.HTML == "html"
        assert LanguageType.CSS == "css"


class TestProcessingStatus:
    """Test ProcessingStatus enum."""

    def test_processing_status_values(self):
        """Test all processing status enum values."""
        assert ProcessingStatus.PENDING == "pending"
        assert ProcessingStatus.IN_PROGRESS == "in_progress"
        assert ProcessingStatus.COMPLETED == "completed"
        assert ProcessingStatus.FAILED == "failed"
        assert ProcessingStatus.SKIPPED == "skipped"


class TestAgentType:
    """Test AgentType enum."""

    def test_agent_type_values(self):
        """Test all agent type enum values."""
        assert AgentType.GIT_SLEUTH == "git_sleuth"
        assert AgentType.CODE_PARSER == "code_parser"
        assert AgentType.GRAPH_ARCHITECT == "graph_architect"
        assert AgentType.DOCU_WRITER == "docu_writer"
        assert AgentType.GRAPH_AUDITOR == "graph_auditor"


class TestFileContext:
    """Test FileContext model."""

    @pytest.fixture
    def valid_file_context_data(self):
        """Valid FileContext data for testing."""
        return {
            "file_path": "/src/utils/helper.py",
            "relative_path": "src/utils/helper.py",
            "file_size": 1024,
            "line_count": 50,
            "language": LanguageType.PYTHON,
            "encoding": "utf-8",
            "file_hash": "abc123def456",
        }

    def test_file_context_creation(self, valid_file_context_data):
        """Test FileContext model creation with valid data."""
        file_context = FileContext(**valid_file_context_data)

        assert file_context.file_path == "/src/utils/helper.py"
        assert file_context.relative_path == "src/utils/helper.py"
        assert file_context.file_size == 1024
        assert file_context.line_count == 50
        assert file_context.language == LanguageType.PYTHON
        assert file_context.encoding == "utf-8"
        assert file_context.file_hash == "abc123def456"

    def test_file_context_validation_errors(self):
        """Test FileContext validation errors."""
        # Test invalid file_size (must be positive)
        with pytest.raises(ValidationError):
            FileContext(
                file_path="/test.py",
                relative_path="test.py",
                file_size=-1,
                line_count=10,
                language=LanguageType.PYTHON,
                encoding="utf-8",
                file_hash="hash123",
            )

        # Test invalid line_count (must be positive)
        with pytest.raises(ValidationError):
            FileContext(
                file_path="/test.py",
                relative_path="test.py",
                file_size=100,
                line_count=0,
                language=LanguageType.PYTHON,
                encoding="utf-8",
                file_hash="hash123",
            )

    def test_file_context_serialization(self, valid_file_context_data):
        """Test FileContext JSON serialization."""
        file_context = FileContext(**valid_file_context_data)
        json_data = file_context.model_dump()

        assert json_data["file_path"] == "/src/utils/helper.py"
        assert json_data["language"] == "python"
        assert isinstance(json_data["file_size"], int)


class TestGitContext:
    """Test GitContext model."""

    @pytest.fixture
    def valid_git_context_data(self):
        """Valid GitContext data for testing."""
        return {
            "repository_url": "https://github.com/user/repo.git",
            "branch": "main",
            "commit_hash": "abc123def456789",
            "commit_message": "Add new feature",
            "author_name": "John Doe",
            "author_email": "john@example.com",
            "commit_timestamp": datetime.now(timezone.utc),
        }

    def test_git_context_creation(self, valid_git_context_data):
        """Test GitContext model creation with valid data."""
        git_context = GitContext(**valid_git_context_data)

        assert git_context.repository_url == "https://github.com/user/repo.git"
        assert git_context.branch == "main"
        assert git_context.commit_hash == "abc123def456789"
        assert git_context.commit_message == "Add new feature"
        assert git_context.author_name == "John Doe"
        assert git_context.author_email == "john@example.com"
        assert isinstance(git_context.commit_timestamp, datetime)

    def test_git_context_optional_fields(self):
        """Test GitContext with minimal required fields."""
        git_context = GitContext(
            repository_url="https://github.com/user/repo.git",
            branch="main",
            commit_hash="abc123",
        )

        assert git_context.repository_url == "https://github.com/user/repo.git"
        assert git_context.commit_message is None
        assert git_context.author_name is None


class TestDependency:
    """Test Dependency model."""

    def test_dependency_creation(self):
        """Test Dependency model creation."""
        dependency = Dependency(
            name="requests",
            version="2.31.0",
            dependency_type="runtime",
            source="pypi",
        )

        assert dependency.name == "requests"
        assert dependency.version == "2.31.0"
        assert dependency.dependency_type == "runtime"
        assert dependency.source == "pypi"

    def test_dependency_optional_fields(self):
        """Test Dependency with minimal fields."""
        dependency = Dependency(name="numpy")

        assert dependency.name == "numpy"
        assert dependency.version is None
        assert dependency.dependency_type is None


class TestRelationship:
    """Test Relationship model."""

    def test_relationship_creation(self):
        """Test Relationship model creation."""
        relationship = Relationship(
            relationship_type="calls",
            target_entity_id="func_456",
            source_line=25,
            metadata={"context": "error_handling"},
        )

        assert relationship.relationship_type == "calls"
        assert relationship.target_entity_id == "func_456"
        assert relationship.source_line == 25
        assert relationship.metadata["context"] == "error_handling"

    def test_relationship_validation(self):
        """Test Relationship validation."""
        # Test positive line number validation
        with pytest.raises(ValidationError):
            Relationship(
                relationship_type="calls",
                target_entity_id="func_456",
                source_line=0,  # Must be positive
            )


class TestAgentResult:
    """Test AgentResult model."""

    def test_agent_result_creation(self):
        """Test AgentResult model creation."""
        agent_result = AgentResult(
            agent_type=AgentType.CODE_PARSER,
            status=ProcessingStatus.COMPLETED,
            processing_time_ms=1500,
            result_data={"functions_found": 5, "classes_found": 2},
            error_message=None,
        )

        assert agent_result.agent_type == AgentType.CODE_PARSER
        assert agent_result.status == ProcessingStatus.COMPLETED
        assert agent_result.processing_time_ms == 1500
        assert agent_result.result_data["functions_found"] == 5
        assert agent_result.error_message is None

    def test_agent_result_with_error(self):
        """Test AgentResult with error status."""
        agent_result = AgentResult(
            agent_type=AgentType.GIT_SLEUTH,
            status=ProcessingStatus.FAILED,
            processing_time_ms=500,
            error_message="Repository not found",
        )

        assert agent_result.status == ProcessingStatus.FAILED
        assert agent_result.error_message == "Repository not found"
        assert agent_result.result_data is None


class TestGoldenNode:
    """Test GoldenNode model (main model)."""

    @pytest.fixture
    def valid_golden_node_data(self):
        """Valid GoldenNode data for testing."""
        return {
            "id": str(uuid4()),
            "entity_type": EntityType.FUNCTION,
            "name": "calculate_metrics",
            "content": "def calculate_metrics(data):\n    return sum(data)",
            "file_context": {
                "file_path": "/src/analytics.py",
                "relative_path": "src/analytics.py",
                "file_size": 2048,
                "line_count": 80,
                "language": LanguageType.PYTHON,
                "encoding": "utf-8",
                "file_hash": "def456ghi789",
            },
            "git_context": {
                "repository_url": "https://github.com/user/analytics.git",
                "branch": "main",
                "commit_hash": "xyz789abc123",
                "commit_message": "Add analytics functions",
                "author_name": "Jane Smith",
                "author_email": "jane@example.com",
                "commit_timestamp": datetime.now(timezone.utc),
            },
            "start_line": 10,
            "end_line": 15,
            "start_column": 0,
            "end_column": 25,
        }

    def test_golden_node_creation(self, valid_golden_node_data):
        """Test GoldenNode model creation with valid data."""
        golden_node = GoldenNode(**valid_golden_node_data)

        assert golden_node.entity_type == EntityType.FUNCTION
        assert golden_node.name == "calculate_metrics"
        assert golden_node.content.startswith("def calculate_metrics")
        assert golden_node.file_context.language == LanguageType.PYTHON
        assert golden_node.git_context.branch == "main"
        assert golden_node.start_line == 10
        assert golden_node.end_line == 15

    def test_golden_node_with_relationships(self, valid_golden_node_data):
        """Test GoldenNode with relationships."""
        valid_golden_node_data["relationships"] = [
            {
                "relationship_type": "calls",
                "target_entity_id": "helper_func_123",
                "source_line": 12,
                "metadata": {"call_type": "direct"},
            }
        ]

        golden_node = GoldenNode(**valid_golden_node_data)

        assert len(golden_node.relationships) == 1
        assert golden_node.relationships[0].relationship_type == "calls"
        assert golden_node.relationships[0].target_entity_id == "helper_func_123"

    def test_golden_node_with_dependencies(self, valid_golden_node_data):
        """Test GoldenNode with dependencies."""
        valid_golden_node_data["dependencies"] = [
            {
                "name": "pandas",
                "version": "2.0.0",
                "dependency_type": "runtime",
                "source": "pypi",
            }
        ]

        golden_node = GoldenNode(**valid_golden_node_data)

        assert len(golden_node.dependencies) == 1
        assert golden_node.dependencies[0].name == "pandas"
        assert golden_node.dependencies[0].version == "2.0.0"

    def test_golden_node_with_agent_results(self, valid_golden_node_data):
        """Test GoldenNode with agent processing results."""
        valid_golden_node_data["agent_results"] = [
            {
                "agent_type": AgentType.CODE_PARSER,
                "status": ProcessingStatus.COMPLETED,
                "processing_time_ms": 1200,
                "result_data": {"complexity_score": 3, "test_coverage": 85},
            },
            {
                "agent_type": AgentType.DOCU_WRITER,
                "status": ProcessingStatus.COMPLETED,
                "processing_time_ms": 800,
                "result_data": {"documentation_added": True},
            },
        ]

        golden_node = GoldenNode(**valid_golden_node_data)

        assert len(golden_node.agent_results) == 2
        assert golden_node.agent_results[0].agent_type == AgentType.CODE_PARSER
        assert golden_node.agent_results[1].agent_type == AgentType.DOCU_WRITER
        assert golden_node.agent_results[0].result_data["complexity_score"] == 3

    def test_golden_node_validation_errors(self, valid_golden_node_data):
        """Test GoldenNode validation errors."""
        # Test invalid line numbers
        invalid_data = valid_golden_node_data.copy()
        invalid_data["start_line"] = 0  # Must be positive

        with pytest.raises(ValidationError):
            GoldenNode(**invalid_data)

        # Test end_line less than start_line
        invalid_data = valid_golden_node_data.copy()
        invalid_data["start_line"] = 20
        invalid_data["end_line"] = 10

        with pytest.raises(ValidationError):
            GoldenNode(**invalid_data)

    def test_golden_node_serialization(self, valid_golden_node_data):
        """Test GoldenNode JSON serialization for Cosmos DB."""
        golden_node = GoldenNode(**valid_golden_node_data)
        json_data = golden_node.model_dump()

        # Verify structure suitable for Cosmos DB
        assert "id" in json_data
        assert "entity_type" in json_data
        assert "file_context" in json_data
        assert "git_context" in json_data
        assert json_data["entity_type"] == "function"
        assert json_data["file_context"]["language"] == "python"

        # Verify datetime serialization
        assert isinstance(json_data["git_context"]["commit_timestamp"], str)

    def test_golden_node_cosmos_db_compatibility(self, valid_golden_node_data):
        """Test GoldenNode Cosmos DB document compatibility."""
        golden_node = GoldenNode(**valid_golden_node_data)

        # Add computed fields that would be added for Cosmos DB
        cosmos_doc = golden_node.model_dump()
        cosmos_doc["_ts"] = int(datetime.now().timestamp())
        cosmos_doc["_etag"] = "test_etag"

        # Verify document structure
        assert "id" in cosmos_doc
        assert "_ts" in cosmos_doc
        assert "_etag" in cosmos_doc

        # Verify embedded structure (following Microsoft docs guidance)
        assert isinstance(cosmos_doc["file_context"], dict)
        assert isinstance(cosmos_doc["git_context"], dict)

        # Verify arrays are properly structured
        if "relationships" in cosmos_doc:
            assert isinstance(cosmos_doc["relationships"], list)
        if "dependencies" in cosmos_doc:
            assert isinstance(cosmos_doc["dependencies"], list)

    def test_golden_node_minimal_creation(self):
        """Test GoldenNode creation with minimal required fields."""
        minimal_data = {
            "id": str(uuid4()),
            "entity_type": EntityType.CLASS,
            "name": "TestClass",
            "content": "class TestClass:\n    pass",
            "file_context": {
                "file_path": "/test.py",
                "relative_path": "test.py",
                "file_size": 100,
                "line_count": 5,
                "language": LanguageType.PYTHON,
                "encoding": "utf-8",
                "file_hash": "test123",
            },
            "start_line": 1,
            "end_line": 2,
        }

        golden_node = GoldenNode(**minimal_data)

        assert golden_node.entity_type == EntityType.CLASS
        assert golden_node.name == "TestClass"
        assert golden_node.git_context is None
        assert golden_node.dependencies == []
        assert golden_node.relationships == []
        assert golden_node.agent_results == []


class TestGoldenNodeIntegration:
    """Integration tests for GoldenNode with realistic data."""

    def test_complete_python_function_node(self):
        """Test complete GoldenNode for a Python function with all fields."""
        node_data = {
            "id": str(uuid4()),
            "entity_type": EntityType.FUNCTION,
            "name": "process_user_data",
            "content": """def process_user_data(user_id: int, data: Dict[str, Any]) -> UserProfile:
    '''Process user data and return profile.'''
    if not data:
        raise ValueError("Data cannot be empty")
    
    profile = UserProfile(user_id=user_id)
    profile.update_from_dict(data)
    return profile""",
            "file_context": {
                "file_path": "/src/user_service/processors.py",
                "relative_path": "src/user_service/processors.py",
                "file_size": 5120,
                "line_count": 150,
                "language": LanguageType.PYTHON,
                "encoding": "utf-8",
                "file_hash": "sha256:abc123def456...",
            },
            "git_context": {
                "repository_url": "https://github.com/company/user-service.git",
                "branch": "feature/data-processing",
                "commit_hash": "commit123abc456def",
                "commit_message": "Implement user data processing with validation",
                "author_name": "Developer One",
                "author_email": "dev1@company.com",
                "commit_timestamp": datetime.now(timezone.utc),
            },
            "start_line": 45,
            "end_line": 53,
            "start_column": 0,
            "end_column": 19,
            "dependencies": [
                {
                    "name": "typing",
                    "dependency_type": "builtin",
                    "source": "python_stdlib",
                },
                {
                    "name": "user_service.models",
                    "dependency_type": "internal",
                    "source": "local_module",
                },
            ],
            "relationships": [
                {
                    "relationship_type": "calls",
                    "target_entity_id": "UserProfile.__init__",
                    "source_line": 49,
                    "metadata": {"call_type": "constructor"},
                },
                {
                    "relationship_type": "calls",
                    "target_entity_id": "UserProfile.update_from_dict",
                    "source_line": 50,
                    "metadata": {"call_type": "method"},
                },
            ],
            "agent_results": [
                {
                    "agent_type": AgentType.CODE_PARSER,
                    "status": ProcessingStatus.COMPLETED,
                    "processing_time_ms": 850,
                    "result_data": {
                        "complexity_score": 4,
                        "parameters_count": 2,
                        "return_type": "UserProfile",
                        "has_docstring": True,
                        "has_type_hints": True,
                    },
                },
                {
                    "agent_type": AgentType.DOCU_WRITER,
                    "status": ProcessingStatus.COMPLETED,
                    "processing_time_ms": 600,
                    "result_data": {
                        "documentation_quality": "high",
                        "missing_docs": [],
                        "suggestions": [],
                    },
                },
                {
                    "agent_type": AgentType.GRAPH_ARCHITECT,
                    "status": ProcessingStatus.COMPLETED,
                    "processing_time_ms": 300,
                    "result_data": {
                        "relationships_mapped": 2,
                        "dependencies_analyzed": 2,
                        "graph_depth": 3,
                    },
                },
            ],
            "ai_description": "Processes user data input and creates a UserProfile object with validation.",
            "ai_summary": "Data processing function with error handling and type safety.",
            "metadata": {
                "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
                "extraction_version": "1.0.0",
                "tree_sitter_version": "0.20.0",
                "confidence_score": 0.95,
            },
        }

        golden_node = GoldenNode(**node_data)

        # Verify complete node structure
        assert golden_node.entity_type == EntityType.FUNCTION
        assert golden_node.name == "process_user_data"
        assert len(golden_node.dependencies) == 2
        assert len(golden_node.relationships) == 2
        assert len(golden_node.agent_results) == 3

        # Verify agent results
        code_parser_result = next(
            r
            for r in golden_node.agent_results
            if r.agent_type == AgentType.CODE_PARSER
        )
        assert code_parser_result.result_data["complexity_score"] == 4
        assert code_parser_result.result_data["has_type_hints"] is True

        # Verify relationships
        constructor_call = next(
            r
            for r in golden_node.relationships
            if r.target_entity_id == "UserProfile.__init__"
        )
        assert constructor_call.metadata["call_type"] == "constructor"

        # Verify serialization for Cosmos DB
        cosmos_doc = golden_node.model_dump()
        assert isinstance(cosmos_doc, dict)
        assert len(str(cosmos_doc)) < 2000000  # Within Cosmos DB document size limit
