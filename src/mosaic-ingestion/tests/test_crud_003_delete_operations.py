"""
Unit tests for CRUD-003 DELETE operations with Git diff change detection.

Tests the enhanced DELETE functionality including:
- Git diff change_type detection for 'D' (deleted) files
- Cascade delete for FileVersion -> related entities
- Soft delete with deleted_at timestamps
- Audit logging for deletion operations
- Integration with branch-aware entity model

Test Categories:
- Git diff change type detection
- Soft delete operations
- Cascade delete operations
- Audit logging
- Error handling
- Integration with existing deletion logic

Author: Mosaic MCP Tool Test Suite
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch


# Import the module under test
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from plugins.ingestion import IngestionPlugin


class TestGitDiffChangeDetection:
    """Test Git diff change_type detection functionality."""

    @pytest.fixture
    async def ingestion_plugin(self):
        """Create IngestionPlugin instance with mocked dependencies."""
        plugin = IngestionPlugin()

        # Mock dependencies
        plugin.knowledge_container = AsyncMock()
        plugin.commit_state_manager = AsyncMock()
        plugin.kernel = AsyncMock()

        return plugin

    @pytest.fixture
    def mock_git_repo(self):
        """Create mock git repository with diff items."""
        repo = MagicMock()

        # Mock commits
        main_commit = MagicMock()
        current_commit = MagicMock()

        # Mock branches
        main_branch = MagicMock()
        main_branch.commit = main_commit
        repo.heads = {"main": main_branch}
        repo.head.commit = current_commit

        return repo, main_commit, current_commit

    async def test_get_changed_files_detects_deleted_files(
        self, ingestion_plugin, mock_git_repo
    ):
        """Test that _get_changed_files correctly identifies deleted files using change_type."""
        repo, main_commit, current_commit = mock_git_repo

        # Create mock diff items with different change types
        deleted_item = MagicMock()
        deleted_item.change_type = "D"
        deleted_item.a_path = "deleted_file.py"
        deleted_item.b_path = None

        modified_item = MagicMock()
        modified_item.change_type = "M"
        modified_item.a_path = "modified_file.py"
        modified_item.b_path = "modified_file.py"

        added_item = MagicMock()
        added_item.change_type = "A"
        added_item.a_path = None
        added_item.b_path = "added_file.py"

        renamed_item = MagicMock()
        renamed_item.change_type = "R"
        renamed_item.a_path = "old_name.py"
        renamed_item.b_path = "new_name.py"

        # Mock diff
        main_commit.diff.return_value = [
            deleted_item,
            modified_item,
            added_item,
            renamed_item,
        ]

        # Set up language extensions
        ingestion_plugin.language_extensions = {"python": [".py"]}

        # Call method
        changed_files = await ingestion_plugin._get_changed_files(
            repo, "feature-branch"
        )

        # Verify results
        assert "deleted_file.py" in changed_files
        assert "modified_file.py" in changed_files
        assert "added_file.py" in changed_files
        assert "old_name.py" in changed_files
        assert "new_name.py" in changed_files

        # Verify change metadata is stored
        assert hasattr(ingestion_plugin, "_change_metadata")
        metadata = ingestion_plugin._change_metadata
        assert "deleted_file.py" in metadata["deleted_files"]
        assert ("old_name.py", "new_name.py") in metadata["renamed_files"]

    async def test_get_changed_files_filters_unsupported_extensions(
        self, ingestion_plugin, mock_git_repo
    ):
        """Test that only supported file extensions are included."""
        repo, main_commit, current_commit = mock_git_repo

        # Create diff items with mixed extensions
        py_item = MagicMock()
        py_item.change_type = "M"
        py_item.b_path = "script.py"

        txt_item = MagicMock()
        txt_item.change_type = "M"
        txt_item.b_path = "readme.txt"

        main_commit.diff.return_value = [py_item, txt_item]

        # Only support Python files
        ingestion_plugin.language_extensions = {"python": [".py"]}

        changed_files = await ingestion_plugin._get_changed_files(
            repo, "feature-branch"
        )

        # Only .py file should be included
        assert "script.py" in changed_files
        assert "readme.txt" not in changed_files

    async def test_get_changed_files_handles_no_main_branch(
        self, ingestion_plugin, mock_git_repo
    ):
        """Test fallback when no main branch is found."""
        repo, main_commit, current_commit = mock_git_repo

        # Mock missing main branches
        repo.heads = {}  # No main, master, or develop

        changed_files = await ingestion_plugin._get_changed_files(
            repo, "feature-branch"
        )

        # Should return empty list and log warning
        assert changed_files == []


class TestSoftDeleteOperations:
    """Test soft delete functionality."""

    @pytest.fixture
    async def ingestion_plugin(self):
        """Create IngestionPlugin instance with mocked dependencies."""
        plugin = IngestionPlugin()
        plugin.knowledge_container = AsyncMock()
        return plugin

    async def test_soft_delete_entity_adds_timestamps(self, ingestion_plugin):
        """Test that soft delete adds proper timestamp and metadata."""
        # Mock entity
        entity = {
            "id": "test_entity_123",
            "type": "FileVersion",
            "name": "test_file.py",
            "content": "test content",
        }

        stats = {"soft_deletions": 0, "audit_entries": [], "errors": []}

        # Mock container operations
        ingestion_plugin.knowledge_container.upsert_item = AsyncMock()
        ingestion_plugin.knowledge_container.create_item = AsyncMock()

        # Call soft delete
        await ingestion_plugin._soft_delete_entity(entity, "test_file.py", stats)

        # Verify entity was updated with soft delete fields
        assert "deleted_at" in entity
        assert entity["deletion_reason"] == "File deleted: test_file.py"
        assert entity["deletion_type"] == "soft_delete"

        # Verify timestamp format
        deleted_at = datetime.fromisoformat(entity["deleted_at"].replace("Z", "+00:00"))
        assert deleted_at.tzinfo is not None

        # Verify stats updated
        assert stats["soft_deletions"] == 1
        assert len(stats["audit_entries"]) == 1

        # Verify container calls
        ingestion_plugin.knowledge_container.upsert_item.assert_called_once_with(entity)
        ingestion_plugin.knowledge_container.create_item.assert_called_once()

    async def test_soft_delete_entity_creates_audit_entry(self, ingestion_plugin):
        """Test that soft delete creates proper audit log entry."""
        entity = {"id": "test_entity_123", "type": "CodeEntity", "name": "TestClass"}

        stats = {"soft_deletions": 0, "audit_entries": [], "errors": []}

        # Mock container
        ingestion_plugin.knowledge_container.upsert_item = AsyncMock()
        ingestion_plugin.knowledge_container.create_item = AsyncMock()

        await ingestion_plugin._soft_delete_entity(entity, "src/test.py", stats)

        # Get the audit entry that was created
        audit_call = ingestion_plugin.knowledge_container.create_item.call_args[0][0]

        # Verify audit entry structure
        assert audit_call["type"] == "audit_log"
        assert audit_call["operation"] == "soft_delete"
        assert audit_call["entity_id"] == "test_entity_123"
        assert audit_call["entity_type"] == "CodeEntity"
        assert audit_call["file_path"] == "src/test.py"
        assert audit_call["reason"] == "File deletion detected via git diff"

        # Verify metadata
        metadata = audit_call["metadata"]
        assert metadata["original_entity_type"] == "CodeEntity"
        assert metadata["file_extension"] == ".py"
        assert metadata["deletion_method"] == "git_diff_change_type"

    async def test_soft_delete_entity_handles_errors(self, ingestion_plugin):
        """Test error handling in soft delete operations."""
        entity = {"id": "test_123", "type": "FileVersion"}
        stats = {"soft_deletions": 0, "audit_entries": [], "errors": []}

        # Mock container to raise exception
        ingestion_plugin.knowledge_container.upsert_item = AsyncMock(
            side_effect=Exception("Database error")
        )

        await ingestion_plugin._soft_delete_entity(entity, "test.py", stats)

        # Verify error was recorded
        assert len(stats["errors"]) == 1
        assert "Soft delete failed for test_123" in stats["errors"][0]
        assert stats["soft_deletions"] == 0


class TestCascadeDeleteOperations:
    """Test cascade delete functionality."""

    @pytest.fixture
    async def ingestion_plugin(self):
        """Create IngestionPlugin instance with mocked dependencies."""
        plugin = IngestionPlugin()
        plugin.knowledge_container = AsyncMock()
        return plugin

    async def test_cascade_delete_fileversion_entities(self, ingestion_plugin):
        """Test cascade delete of CodeEntity and Relationship entities for FileVersion."""
        stats = {"cascade_deletions": 0, "errors": []}

        # Mock related entities
        code_entity1 = {
            "id": "code_1",
            "type": "CodeEntity",
            "file_version_id": "file_123",
        }
        code_entity2 = {
            "id": "code_2",
            "type": "CodeEntity",
            "file_version_id": "file_123",
        }
        relationship1 = {
            "id": "rel_1",
            "type": "Relationship",
            "source_id": "file_123",
            "target_id": "other",
        }

        # Mock query results
        def mock_query(query, parameters, enable_cross_partition_query):
            if "CodeEntity" in query:
                return [code_entity1, code_entity2]
            elif "Relationship" in query:
                return [relationship1]
            return []

        ingestion_plugin.knowledge_container.query_items = mock_query
        ingestion_plugin.knowledge_container.upsert_item = AsyncMock()

        # Call cascade delete
        await ingestion_plugin._cascade_delete_related_entities(
            "file_123", "FileVersion", stats
        )

        # Verify all related entities were soft deleted
        assert stats["cascade_deletions"] == 3

        # Verify upsert was called for each entity
        assert ingestion_plugin.knowledge_container.upsert_item.call_count == 3

        # Verify entities have cascade delete markers
        calls = ingestion_plugin.knowledge_container.upsert_item.call_args_list
        for call in calls:
            entity = call[0][0]
            assert "deleted_at" in entity
            assert entity["deletion_type"] == "cascade_delete"
            assert (
                "Cascade delete from FileVersion file_123" in entity["deletion_reason"]
            )

    async def test_cascade_delete_non_fileversion_entity(self, ingestion_plugin):
        """Test that cascade delete only works for FileVersion entities."""
        stats = {"cascade_deletions": 0, "errors": []}

        # Call with non-FileVersion entity
        await ingestion_plugin._cascade_delete_related_entities(
            "code_123", "CodeEntity", stats
        )

        # Should not perform any deletions
        assert stats["cascade_deletions"] == 0

    async def test_cascade_delete_handles_query_errors(self, ingestion_plugin):
        """Test error handling in cascade delete queries."""
        stats = {"cascade_deletions": 0, "errors": []}

        # Mock query to raise exception
        ingestion_plugin.knowledge_container.query_items = MagicMock(
            side_effect=Exception("Query failed")
        )

        await ingestion_plugin._cascade_delete_related_entities(
            "file_123", "FileVersion", stats
        )

        # Verify error was recorded
        assert len(stats["errors"]) == 1
        assert "Cascade delete failed for file_123" in stats["errors"][0]


class TestDeleteEntitiesByFilepath:
    """Test enhanced _delete_entities_by_filepath method."""

    @pytest.fixture
    async def ingestion_plugin(self):
        """Create IngestionPlugin instance with mocked dependencies."""
        plugin = IngestionPlugin()
        plugin.knowledge_container = AsyncMock()

        # Mock change metadata
        plugin._change_metadata = {
            "deleted_files": ["deleted_file.py"],
            "renamed_files": [("old_name.py", "new_name.py")],
            "all_changed": [
                "deleted_file.py",
                "modified_file.py",
                "old_name.py",
                "new_name.py",
            ],
        }

        return plugin

    async def test_delete_entities_distinguishes_operations(self, ingestion_plugin):
        """Test that method distinguishes between file deletions and modifications."""
        # Mock entities
        entities = [
            {"id": "entity_1", "type": "FileVersion", "file_path": "deleted_file.py"},
            {"id": "entity_2", "type": "FileVersion", "file_path": "modified_file.py"},
        ]

        ingestion_plugin.knowledge_container.query_items = MagicMock(
            return_value=entities
        )
        ingestion_plugin.knowledge_container.delete_item = AsyncMock()

        # Mock soft delete and cascade delete methods
        ingestion_plugin._soft_delete_entity = AsyncMock()
        ingestion_plugin._cascade_delete_related_entities = AsyncMock()

        changed_files = ["deleted_file.py", "modified_file.py"]

        # Call method
        stats = await ingestion_plugin._delete_entities_by_filepath(
            changed_files, "update"
        )

        # Verify soft delete was called for deleted file
        ingestion_plugin._soft_delete_entity.assert_called()

        # Verify cascade delete was called for deleted file
        ingestion_plugin._cascade_delete_related_entities.assert_called()

        # Verify hard delete was called for modified file
        ingestion_plugin.knowledge_container.delete_item.assert_called()

        # Verify statistics
        assert stats["files_processed"] == 2
        assert stats["entities_deleted"] == 2

    async def test_delete_entities_returns_statistics(self, ingestion_plugin):
        """Test that method returns comprehensive deletion statistics."""
        # Mock empty query results
        ingestion_plugin.knowledge_container.query_items = MagicMock(return_value=[])

        changed_files = ["test_file.py"]

        stats = await ingestion_plugin._delete_entities_by_filepath(
            changed_files, "update"
        )

        # Verify statistics structure
        expected_keys = [
            "files_processed",
            "entities_deleted",
            "cascade_deletions",
            "soft_deletions",
            "errors",
            "audit_entries",
        ]

        for key in expected_keys:
            assert key in stats

        assert stats["files_processed"] == 1
        assert isinstance(stats["errors"], list)
        assert isinstance(stats["audit_entries"], list)

    async def test_delete_entities_handles_query_errors(self, ingestion_plugin):
        """Test error handling when entity queries fail."""
        # Mock query to raise exception
        ingestion_plugin.knowledge_container.query_items = MagicMock(
            side_effect=Exception("Query failed")
        )

        changed_files = ["test_file.py"]

        stats = await ingestion_plugin._delete_entities_by_filepath(
            changed_files, "update"
        )

        # Verify error was recorded and method didn't crash
        assert len(stats["errors"]) == 1
        assert "Error during entity cleanup" in stats["errors"][0]
        assert stats["files_processed"] == 0


class TestIncrementalUpdateIntegration:
    """Test integration of enhanced DELETE with incremental updates."""

    @pytest.fixture
    async def ingestion_plugin(self):
        """Create IngestionPlugin instance with mocked dependencies."""
        plugin = IngestionPlugin()
        plugin.knowledge_container = AsyncMock()
        plugin.commit_state_manager = AsyncMock()

        # Mock other required methods
        plugin._extract_entities_from_file = AsyncMock(return_value=[])
        plugin._extract_file_relationships = AsyncMock(return_value=[])
        plugin._store_entities_batch = AsyncMock()
        plugin._store_relationships_batch = AsyncMock()
        plugin._ai_enhance_entities = AsyncMock(return_value=[])

        return plugin

    async def test_process_incremental_update_includes_deletion_stats(
        self, ingestion_plugin
    ):
        """Test that incremental update includes deletion statistics in summary."""
        # Mock deletion statistics
        deletion_stats = {
            "files_processed": 2,
            "entities_deleted": 5,
            "cascade_deletions": 3,
            "soft_deletions": 2,
            "audit_entries": ["audit_1", "audit_2"],
            "errors": [],
        }

        ingestion_plugin._delete_entities_by_filepath = AsyncMock(
            return_value=deletion_stats
        )

        # Mock repo
        mock_repo = MagicMock()
        mock_repo.head.commit.hexsha = "abc123def456"

        with patch("git.Repo", return_value=mock_repo):
            summary = await ingestion_plugin._process_incremental_update(
                "/tmp/repo", ["test_file.py"], "https://github.com/test/repo", "main"
            )

        # Verify deletion stats are included in summary
        assert "deletion_stats" in summary
        assert summary["entities_deleted"] == 5
        assert summary["soft_deletions"] == 2
        assert summary["cascade_deletions"] == 3
        assert summary["audit_entries_created"] == 2
        assert summary["enhancement_version"] == "3.0_crud_003_delete_operations"


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    @pytest.fixture
    async def ingestion_plugin(self):
        """Create IngestionPlugin instance with mocked dependencies."""
        plugin = IngestionPlugin()
        plugin.knowledge_container = AsyncMock()
        return plugin

    async def test_soft_delete_with_missing_entity_fields(self, ingestion_plugin):
        """Test soft delete with entities missing expected fields."""
        entity = {}  # Empty entity
        stats = {"soft_deletions": 0, "audit_entries": [], "errors": []}

        ingestion_plugin.knowledge_container.upsert_item = AsyncMock()
        ingestion_plugin.knowledge_container.create_item = AsyncMock()

        # Should not crash
        await ingestion_plugin._soft_delete_entity(entity, "test.py", stats)

        # Verify operation completed despite missing fields
        assert stats["soft_deletions"] == 1
        ingestion_plugin.knowledge_container.upsert_item.assert_called_once()

    async def test_cascade_delete_with_empty_query_results(self, ingestion_plugin):
        """Test cascade delete when no related entities are found."""
        stats = {"cascade_deletions": 0, "errors": []}

        # Mock empty query results
        ingestion_plugin.knowledge_container.query_items = MagicMock(return_value=[])

        await ingestion_plugin._cascade_delete_related_entities(
            "file_123", "FileVersion", stats
        )

        # Should complete without errors
        assert stats["cascade_deletions"] == 0
        assert len(stats["errors"]) == 0

    async def test_delete_entities_with_no_change_metadata(self, ingestion_plugin):
        """Test delete entities when change metadata is not available."""
        # Remove change metadata
        if hasattr(ingestion_plugin, "_change_metadata"):
            delattr(ingestion_plugin, "_change_metadata")

        ingestion_plugin.knowledge_container.query_items = MagicMock(return_value=[])

        changed_files = ["test_file.py"]

        # Should not crash and should default to update operations
        stats = await ingestion_plugin._delete_entities_by_filepath(
            changed_files, "update"
        )

        assert stats["files_processed"] == 1
        assert len(stats["errors"]) == 0


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main(
        [
            __file__,
            "-v",
            "--cov=plugins.ingestion",
            "--cov-report=term-missing",
            "--cov-fail-under=80",
        ]
    )
