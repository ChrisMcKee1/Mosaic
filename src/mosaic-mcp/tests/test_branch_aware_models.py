"""
Comprehensive Unit Tests for Branch-Aware Entity Models

Tests all aspects of the branch-aware entity models implementing the Identity vs State 
separation pattern for git-aware knowledge graph entities (CRUD-002).

Coverage includes:
- Model instantiation and validation
- Partition key generation functions
- ID generation utilities  
- Enum validations
- Relationships between models
- Edge cases and error handling
- JSON serialization/deserialization
- Business logic validation
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any
import json

from mosaic_mcp.models.branch_aware_models import (
    BranchState,
    FileChangeType,
    PartitionStrategy,
    Branch,
    FileIdentity,
    FileVersion,
    EntityVersion,
    BranchAwarePartitionConfig,
    generate_branch_partition_key,
    generate_file_identity_id,
    generate_file_version_id,
    generate_entity_version_id,
)
from mosaic_mcp.models.base import EntityType


class TestEnums:
    """Test enum classes and their values."""
    
    def test_branch_state_enum(self):
        """Test BranchState enum values."""
        assert BranchState.ACTIVE == "active"
        assert BranchState.MERGED == "merged"
        assert BranchState.DELETED == "deleted"
        assert BranchState.STALE == "stale"
        
        # Test all values are valid
        all_states = [BranchState.ACTIVE, BranchState.MERGED, BranchState.DELETED, BranchState.STALE]
        assert len(all_states) == 4
    
    def test_file_change_type_enum(self):
        """Test FileChangeType enum values match git diff output."""
        assert FileChangeType.ADDED == "A"
        assert FileChangeType.DELETED == "D"
        assert FileChangeType.MODIFIED == "M"
        assert FileChangeType.RENAMED == "R"
        assert FileChangeType.COPIED == "C"
        assert FileChangeType.TYPECHANGE == "T"
    
    def test_partition_strategy_enum(self):
        """Test PartitionStrategy enum values."""
        assert PartitionStrategy.BRANCH_NAME == "branch_name"
        assert PartitionStrategy.REPOSITORY_BRANCH == "repository_branch"
        assert PartitionStrategy.FILE_IDENTITY == "file_identity"


class TestUtilityFunctions:
    """Test utility functions for ID and partition key generation."""
    
    def test_generate_branch_partition_key(self):
        """Test branch partition key generation."""
        repo_id = "repo-123"
        branch_name = "feature/crud-implementation"
        
        result = generate_branch_partition_key(repo_id, branch_name)
        
        assert result == "repo-123:feature/crud-implementation"
        assert isinstance(result, str)
    
    def test_generate_file_identity_id(self):
        """Test file identity ID generation with path normalization."""
        repo_id = "repo-123"
        
        # Test forward slash path
        file_path1 = "src/models/base.py"
        id1 = generate_file_identity_id(repo_id, file_path1)
        
        # Test backslash path (Windows)
        file_path2 = "src\\models\\base.py"
        id2 = generate_file_identity_id(repo_id, file_path2)
        
        # Test leading slash path
        file_path3 = "/src/models/base.py"
        id3 = generate_file_identity_id(repo_id, file_path3)
        
        # All should generate the same ID due to normalization
        assert id1 == id2 == id3
        assert len(id1) == 32  # MD5 hash length
        assert isinstance(id1, str)
    
    def test_generate_file_version_id(self):
        """Test file version ID generation."""
        file_identity_id = "abc123def456"
        branch_name = "main"
        commit_sha = "1234567890abcdef1234567890abcdef12345678"
        
        result = generate_file_version_id(file_identity_id, branch_name, commit_sha)
        
        # Format: file_identity_id:branch_name:commit_sha[:8]
        expected = f"{file_identity_id}:{branch_name}:12345678"
        assert result == expected
    
    def test_generate_entity_version_id(self):
        """Test entity version ID generation."""
        file_version_id = "abc123:main:12345678"
        entity_name = "MyClass"
        start_line = 42
        
        result = generate_entity_version_id(file_version_id, entity_name, start_line)
        
        assert isinstance(result, str)
        assert len(result) == 32  # MD5 hash length
        
        # Same inputs should generate same ID
        result2 = generate_entity_version_id(file_version_id, entity_name, start_line)
        assert result == result2


class TestBranchModel:
    """Test Branch model functionality."""
    
    def test_branch_creation_minimal(self):
        """Test creating a Branch with minimal required fields."""
        branch = Branch(
            id="repo-123:main",
            repository_id="repo-123",
            branch_name="main",
            head_commit_sha="abc123def456",
            partition_key="repo-123:main"
        )
        
        assert branch.id == "repo-123:main"
        assert branch.repository_id == "repo-123"
        assert branch.branch_name == "main"
        assert branch.state == BranchState.ACTIVE  # default
        assert branch.is_default is False  # default
        assert isinstance(branch.created_at, datetime)
        assert isinstance(branch.updated_at, datetime)
    
    def test_branch_creation_full(self):
        """Test creating a Branch with all fields."""
        now = datetime.utcnow()
        
        branch = Branch(
            id="repo-123:feature-branch",
            repository_id="repo-123",
            branch_name="feature-branch",
            state=BranchState.ACTIVE,
            created_at=now,
            updated_at=now,
            head_commit_sha="def456abc123",
            base_commit_sha="abc123def456",
            merge_target="main",
            author="developer@example.com",
            description="Feature implementation branch",
            is_default=False,
            is_protected=True,
            partition_key="repo-123:feature-branch",
            metadata={"priority": "high", "reviewer": "senior-dev"}
        )
        
        assert branch.state == BranchState.ACTIVE
        assert branch.author == "developer@example.com"
        assert branch.is_protected is True
        assert branch.metadata["priority"] == "high"
    
    def test_branch_json_serialization(self):
        """Test Branch JSON serialization includes datetime ISO format."""
        branch = Branch(
            id="repo-123:main",
            repository_id="repo-123",
            branch_name="main",
            head_commit_sha="abc123",
            partition_key="repo-123:main"
        )
        
        json_data = branch.model_dump()
        json_str = json.dumps(json_data, default=str)
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["id"] == "repo-123:main"
        assert "created_at" in parsed


class TestFileIdentityModel:
    """Test FileIdentity model functionality."""
    
    def test_file_identity_creation(self):
        """Test creating a FileIdentity."""
        file_id = generate_file_identity_id("repo-123", "src/models/base.py")
        
        identity = FileIdentity(
            id=file_id,
            repository_id="repo-123",
            file_path="src/models/base.py",
            file_name="base.py",
            file_extension=".py",
            directory_path="src/models",
            partition_key="repo-123"
        )
        
        assert identity.id == file_id
        assert identity.file_name == "base.py"
        assert identity.file_extension == ".py"
        assert identity.language is None  # default
        assert identity.is_test_file is False  # default
        assert identity.total_versions == 0  # default
        assert identity.active_versions == []  # default
    
    def test_file_identity_with_classification(self):
        """Test FileIdentity with file type classification."""
        file_id = generate_file_identity_id("repo-123", "tests/test_models.py")
        
        identity = FileIdentity(
            id=file_id,
            repository_id="repo-123", 
            file_path="tests/test_models.py",
            file_name="test_models.py",
            file_extension=".py",
            directory_path="tests",
            language="python",
            is_test_file=True,
            is_config_file=False,
            is_documentation=False,
            partition_key="repo-123",
            active_versions=["v1", "v2"],
            total_versions=2
        )
        
        assert identity.language == "python"
        assert identity.is_test_file is True
        assert identity.total_versions == 2
        assert len(identity.active_versions) == 2


class TestFileVersionModel:
    """Test FileVersion model functionality."""
    
    def test_file_version_creation_minimal(self):
        """Test creating a FileVersion with minimal fields."""
        file_identity_id = generate_file_identity_id("repo-123", "src/main.py")
        version_id = generate_file_version_id(file_identity_id, "main", "abc123def456")
        
        version = FileVersion(
            id=version_id,
            file_identity_id=file_identity_id,
            repository_id="repo-123",
            branch_name="main",
            commit_sha="abc123def456",
            content="def hello():\n    return 'Hello World'",
            content_hash="hash123",
            file_size_bytes=1024,
            line_count=10,
            language="python",
            change_type=FileChangeType.ADDED,
            partition_key="repo-123:main"
        )
        
        assert version.file_identity_id == file_identity_id
        assert version.branch_name == "main"
        assert version.change_type == FileChangeType.ADDED
        assert version.is_active is True  # default
        assert version.is_head is True  # default
        assert version.entities_extracted == 0  # default
    
    def test_file_version_with_ai_analysis(self):
        """Test FileVersion with AI analysis data."""
        file_identity_id = generate_file_identity_id("repo-123", "src/complex.py")
        version_id = generate_file_version_id(file_identity_id, "feature", "def456abc123")
        
        version = FileVersion(
            id=version_id,
            file_identity_id=file_identity_id,
            repository_id="repo-123",
            branch_name="feature",
            commit_sha="def456abc123",
            content="class ComplexClass:\n    def complex_method(self):\n        pass",
            content_hash="hash456",
            file_size_bytes=2048,
            line_count=20,
            language="python",
            complexity_score=7.5,
            change_type=FileChangeType.MODIFIED,
            previous_version_id="prev_version_123",
            entities_extracted=2,
            entity_types=["class", "method"],
            ai_summary="Complex class with advanced functionality",
            ai_complexity=8,
            ai_dependencies=["numpy", "pandas"],
            ai_tags=["complex", "data-processing"],
            has_ai_analyst=True,
            ai_analyst_timestamp="2025-01-21T12:00:00Z",
            embedding=[0.1, 0.2, 0.3],
            embedding_model="text-embedding-ada-002",
            rdf_triples_count=5,
            partition_key="repo-123:feature"
        )
        
        assert version.complexity_score == 7.5
        assert version.ai_complexity == 8
        assert version.has_ai_analyst is True
        assert len(version.ai_dependencies) == 2
        assert len(version.ai_tags) == 2
        assert len(version.embedding) == 3
        assert version.rdf_triples_count == 5
    
    def test_file_version_change_types(self):
        """Test FileVersion with different change types."""
        for change_type in FileChangeType:
            file_identity_id = f"file_{change_type.value}"
            version_id = f"version_{change_type.value}"
            
            version = FileVersion(
                id=version_id,
                file_identity_id=file_identity_id,
                repository_id="repo-123",
                branch_name="test",
                commit_sha="test123",
                content="test content",
                content_hash="testhash",
                file_size_bytes=100,
                line_count=5,
                language="python",
                change_type=change_type,
                partition_key="test"
            )
            
            assert version.change_type == change_type


class TestEntityVersionModel:
    """Test EntityVersion model functionality."""
    
    def test_entity_version_creation(self):
        """Test creating an EntityVersion."""
        file_version_id = "file_v1:main:abc123"
        entity_id = generate_entity_version_id(file_version_id, "MyFunction", 10)
        
        entity = EntityVersion(
            id=entity_id,
            entity_name="MyFunction",
            entity_type=EntityType.FUNCTION,
            file_version_id=file_version_id,
            file_identity_id="file_identity_123",
            branch_name="main",
            commit_sha="abc123def456",
            start_line=10,
            end_line=25,
            content="def my_function():\n    return True",
            partition_key="repo-123:main"
        )
        
        assert entity.entity_name == "MyFunction"
        assert entity.entity_type == EntityType.FUNCTION
        assert entity.start_line == 10
        assert entity.end_line == 25
        assert entity.hierarchy_level == 0  # default
        assert entity.is_exported is False  # default
        assert entity.scope == "public"  # default
    
    def test_entity_version_with_hierarchy(self):
        """Test EntityVersion with hierarchy information."""
        file_version_id = "file_v1:main:abc123"
        entity_id = generate_entity_version_id(file_version_id, "MyClass.method", 50)
        
        entity = EntityVersion(
            id=entity_id,
            entity_name="method",
            entity_type=EntityType.FUNCTION,
            file_version_id=file_version_id,
            file_identity_id="file_identity_123",
            branch_name="main",
            commit_sha="abc123def456",
            start_line=50,
            end_line=60,
            content="def method(self):\n    pass",
            signature="def method(self) -> None:",
            parent_entity_id="parent_class_123",
            hierarchy_level=1,
            hierarchy_path=["MyClass", "method"],
            complexity_score=3.2,
            is_exported=True,
            scope="public",
            calls=["helper_function", "another_method"],
            imports=["os", "sys"],
            relationships=["rel1", "rel2"],
            embedding=[0.5, 0.6, 0.7],
            partition_key="repo-123:main"
        )
        
        assert entity.hierarchy_level == 1
        assert entity.hierarchy_path == ["MyClass", "method"]
        assert entity.parent_entity_id == "parent_class_123"
        assert entity.complexity_score == 3.2
        assert entity.is_exported is True
        assert len(entity.calls) == 2
        assert len(entity.imports) == 2
        assert len(entity.relationships) == 2


class TestBranchAwarePartitionConfig:
    """Test BranchAwarePartitionConfig model."""
    
    def test_config_creation_defaults(self):
        """Test creating config with defaults."""
        config = BranchAwarePartitionConfig(
            strategy=PartitionStrategy.BRANCH_NAME
        )
        
        assert config.strategy == PartitionStrategy.BRANCH_NAME
        assert config.branch_partition_pattern == "/branch_name"
        assert config.enable_ttl is True
        assert config.default_ttl_seconds == 7776000  # 90 days
        assert config.deleted_branch_ttl_seconds == 86400  # 1 day
        assert config.max_degree_of_parallelism == 10
        assert config.enable_cross_partition_query is True
        assert config.index_branch_name is True
    
    def test_config_creation_custom(self):
        """Test creating config with custom values."""
        config = BranchAwarePartitionConfig(
            strategy=PartitionStrategy.REPOSITORY_BRANCH,
            repository_branch_pattern="/repo_branch",
            enable_ttl=False,
            default_ttl_seconds=3600,
            max_degree_of_parallelism=20,
            index_branch_name=False,
            index_repository_id=True
        )
        
        assert config.strategy == PartitionStrategy.REPOSITORY_BRANCH
        assert config.repository_branch_pattern == "/repo_branch"
        assert config.enable_ttl is False
        assert config.default_ttl_seconds == 3600
        assert config.max_degree_of_parallelism == 20
        assert config.index_branch_name is False
        assert config.index_repository_id is True


class TestModelRelationships:
    """Test relationships between branch-aware models."""
    
    def test_complete_entity_hierarchy(self):
        """Test complete hierarchy: Branch -> FileIdentity -> FileVersion -> EntityVersion."""
        # Create Branch
        branch = Branch(
            id="repo-123:feature",
            repository_id="repo-123",
            branch_name="feature",
            head_commit_sha="abc123",
            partition_key="repo-123:feature"
        )
        
        # Create FileIdentity
        file_identity_id = generate_file_identity_id("repo-123", "src/example.py")
        file_identity = FileIdentity(
            id=file_identity_id,
            repository_id="repo-123",
            file_path="src/example.py",
            file_name="example.py",
            file_extension=".py",
            directory_path="src",
            partition_key="repo-123"
        )
        
        # Create FileVersion
        file_version_id = generate_file_version_id(file_identity_id, "feature", "abc123")
        file_version = FileVersion(
            id=file_version_id,
            file_identity_id=file_identity_id,
            repository_id="repo-123",
            branch_name="feature",
            commit_sha="abc123",
            content="def example(): pass",
            content_hash="hash123",
            file_size_bytes=100,
            line_count=1,
            language="python",
            change_type=FileChangeType.ADDED,
            partition_key="repo-123:feature"
        )
        
        # Create EntityVersion
        entity_id = generate_entity_version_id(file_version_id, "example", 1)
        entity_version = EntityVersion(
            id=entity_id,
            entity_name="example",
            entity_type=EntityType.FUNCTION,
            file_version_id=file_version_id,
            file_identity_id=file_identity_id,
            branch_name="feature",
            commit_sha="abc123",
            start_line=1,
            end_line=1,
            content="def example(): pass",
            partition_key="repo-123:feature"
        )
        
        # Verify relationships
        assert branch.branch_name == file_version.branch_name == entity_version.branch_name
        assert file_identity.id == file_version.file_identity_id == entity_version.file_identity_id
        assert file_version.id == entity_version.file_version_id
        assert branch.partition_key == file_version.partition_key == entity_version.partition_key
    
    def test_partition_key_consistency(self):
        """Test partition key consistency across related models."""
        repo_id = "repo-123"
        branch_name = "main"
        partition_key = generate_branch_partition_key(repo_id, branch_name)
        
        # All models should use the same partition key for branch isolation
        branch = Branch(
            id=f"{repo_id}:{branch_name}",
            repository_id=repo_id,
            branch_name=branch_name,
            head_commit_sha="abc123",
            partition_key=partition_key
        )
        
        file_version = FileVersion(
            id="file_v1",
            file_identity_id="file_id_1",
            repository_id=repo_id,
            branch_name=branch_name,
            commit_sha="abc123",
            content="test",
            content_hash="hash",
            file_size_bytes=10,
            line_count=1,
            language="python",
            change_type=FileChangeType.ADDED,
            partition_key=partition_key
        )
        
        entity = EntityVersion(
            id="entity_1",
            entity_name="test_func",
            entity_type=EntityType.FUNCTION,
            file_version_id="file_v1",
            file_identity_id="file_id_1",
            branch_name=branch_name,
            commit_sha="abc123",
            start_line=1,
            end_line=1,
            content="def test_func(): pass",
            partition_key=partition_key
        )
        
        # All partition keys should match
        assert branch.partition_key == file_version.partition_key == entity.partition_key
        assert partition_key == f"{repo_id}:{branch_name}"


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_required_fields(self):
        """Test that missing required fields raise validation errors."""
        with pytest.raises(ValueError):
            Branch()  # Missing required fields
        
        with pytest.raises(ValueError):
            FileIdentity()  # Missing required fields
        
        with pytest.raises(ValueError):
            FileVersion()  # Missing required fields
    
    def test_invalid_enum_values(self):
        """Test that invalid enum values raise validation errors."""
        with pytest.raises(ValueError):
            Branch(
                id="test",
                repository_id="repo",
                branch_name="main", 
                state="invalid_state",  # Invalid enum value
                head_commit_sha="abc123",
                partition_key="test"
            )
    
    def test_empty_strings(self):
        """Test handling of empty string values."""
        # Should allow empty strings where appropriate
        branch = Branch(
            id="repo:main",
            repository_id="repo",
            branch_name="main",
            head_commit_sha="abc123",
            partition_key="repo:main",
            description=""  # Empty string should be allowed
        )
        
        assert branch.description == ""


class TestSerializationDeserialization:
    """Test JSON serialization and deserialization."""
    
    def test_branch_serialization_roundtrip(self):
        """Test Branch serialization and deserialization."""
        original = Branch(
            id="repo:main",
            repository_id="repo",
            branch_name="main",
            head_commit_sha="abc123",
            partition_key="repo:main",
            metadata={"key": "value"}
        )
        
        # Serialize to JSON
        json_data = original.model_dump()
        json_str = json.dumps(json_data, default=str)
        
        # Deserialize from JSON
        parsed_data = json.loads(json_str)
        reconstructed = Branch(**parsed_data)
        
        assert reconstructed.id == original.id
        assert reconstructed.repository_id == original.repository_id
        assert reconstructed.metadata == original.metadata
    
    def test_file_version_complex_serialization(self):
        """Test FileVersion with complex data serialization."""
        original = FileVersion(
            id="file_v1",
            file_identity_id="file_1",
            repository_id="repo",
            branch_name="main",
            commit_sha="abc123",
            content="def test(): pass",
            content_hash="hash123",
            file_size_bytes=100,
            line_count=1,
            language="python",
            change_type=FileChangeType.ADDED,
            ai_dependencies=["numpy", "pandas"],
            ai_tags=["ml", "data"],
            embedding=[0.1, 0.2, 0.3],
            partition_key="repo:main"
        )
        
        # Serialize
        json_data = original.model_dump()
        json_str = json.dumps(json_data, default=str)
        
        # Deserialize
        parsed_data = json.loads(json_str)
        reconstructed = FileVersion(**parsed_data)
        
        assert reconstructed.ai_dependencies == original.ai_dependencies
        assert reconstructed.ai_tags == original.ai_tags
        assert reconstructed.embedding == original.embedding
        assert reconstructed.change_type == original.change_type


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--cov=mosaic_mcp.models.branch_aware_models", "--cov-report=term-missing"])
