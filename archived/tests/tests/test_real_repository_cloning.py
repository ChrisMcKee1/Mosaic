"""
Real-world test for repository cloning using the actual Mosaic repository.

This test validates the enhanced _clone_repository method against the actual
Mosaic repository, providing practical validation of the implementation.
"""

import os
import shutil
import pytest

# Import the ingestion plugin
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


class TestRealRepositoryCloning:
    """Test repository cloning with the actual Mosaic repository."""

    @pytest.fixture
    def ingestion_plugin(self):
        """Create actual ingestion plugin instance."""
        from ingestion_service.plugins.ingestion import IngestionPlugin

        return IngestionPlugin()

    @pytest.mark.asyncio
    async def test_clone_mosaic_repository_main_branch(self, ingestion_plugin):
        """Test cloning the actual Mosaic repository main branch."""
        repository_url = "https://github.com/ChrisMcKee1/Mosaic.git"
        branch = "main"

        try:
            # Clone the repository
            repo_path = await ingestion_plugin._clone_repository(repository_url, branch)

            # Verify the clone was successful
            assert os.path.exists(repo_path)
            assert os.path.isdir(repo_path)

            # Verify key files exist in the cloned repository
            expected_files = [
                "README.md",
                "CLAUDE.md",
                "requirements.txt",
                "azure.yaml",
                "pyproject.toml",
            ]

            for expected_file in expected_files:
                file_path = os.path.join(repo_path, expected_file)
                assert os.path.exists(file_path), (
                    f"Expected file {expected_file} not found"
                )

            # Verify directory structure
            expected_dirs = [
                "src",
                "src/mosaic",
                "src/ingestion_service",
                "infra",
                "docs",
            ]

            for expected_dir in expected_dirs:
                dir_path = os.path.join(repo_path, expected_dir)
                assert os.path.exists(dir_path), (
                    f"Expected directory {expected_dir} not found"
                )
                assert os.path.isdir(dir_path), (
                    f"Expected {expected_dir} to be a directory"
                )

            # Verify the ingestion plugin file exists and has our enhanced method
            ingestion_file = os.path.join(
                repo_path, "src", "ingestion_service", "plugins", "ingestion.py"
            )
            assert os.path.exists(ingestion_file)

            # Read the file and verify our enhanced method is present
            with open(ingestion_file, "r", encoding="utf-8") as f:
                content = f.read()
                assert "enterprise security patterns" in content
                assert "GITHUB_TOKEN" in content
                assert "GIT_USERNAME" in content
                assert "timeout=300" in content

            print(f"✅ Successfully cloned Mosaic repository to: {repo_path}")
            print(
                f"📁 Repository contains {len(os.listdir(repo_path))} top-level items"
            )

            # Get some basic repository statistics
            src_files = []
            for root, dirs, files in os.walk(os.path.join(repo_path, "src")):
                for file in files:
                    if file.endswith((".py", ".yaml", ".yml", ".json")):
                        src_files.append(os.path.join(root, file))

            print(f"🐍 Found {len(src_files)} Python/config files in src/ directory")

        finally:
            # Clean up the cloned repository
            if "repo_path" in locals() and os.path.exists(repo_path):
                shutil.rmtree(repo_path, ignore_errors=True)
                print(f"🧹 Cleaned up temporary repository: {repo_path}")

    @pytest.mark.asyncio
    async def test_clone_with_different_branch_fallback(self, ingestion_plugin):
        """Test cloning with a non-existent branch to verify error handling."""
        repository_url = "https://github.com/ChrisMcKee1/Mosaic.git"
        branch = "nonexistent-feature-branch"

        # This should fail gracefully with our enhanced error handling
        with pytest.raises(Exception) as exc_info:
            await ingestion_plugin._clone_repository(repository_url, branch)

        # Verify the error message is helpful
        error_msg = str(exc_info.value)
        print(f"📝 Error message for nonexistent branch: {error_msg}")

        # Should mention the branch or repository issue
        assert any(
            keyword in error_msg.lower()
            for keyword in ["branch", "not found", "does not exist", "repository"]
        )

    @pytest.mark.asyncio
    async def test_repository_analysis_preview(self, ingestion_plugin):
        """Test basic repository analysis on the cloned Mosaic repo."""
        repository_url = "https://github.com/ChrisMcKee1/Mosaic.git"
        branch = "main"

        try:
            # Clone the repository
            repo_path = await ingestion_plugin._clone_repository(repository_url, branch)

            # Perform basic analysis similar to what the ingestion service would do
            python_files = []
            yaml_files = []
            markdown_files = []

            for root, dirs, files in os.walk(repo_path):
                # Skip hidden directories and __pycache__
                dirs[:] = [
                    d for d in dirs if not d.startswith(".") and d != "__pycache__"
                ]

                for file in files:
                    file_path = os.path.join(root, file)
                    if file.endswith(".py"):
                        python_files.append(file_path)
                    elif file.endswith((".yaml", ".yml")):
                        yaml_files.append(file_path)
                    elif file.endswith(".md"):
                        markdown_files.append(file_path)

            print("📊 Repository Analysis Results:")
            print(f"   🐍 Python files: {len(python_files)}")
            print(f"   ⚙️  YAML files: {len(yaml_files)}")
            print(f"   📖 Markdown files: {len(markdown_files)}")

            # Verify we have the expected file types for a Python project
            assert len(python_files) > 0, "Should have Python files"
            assert len(yaml_files) > 0, "Should have YAML config files"
            assert len(markdown_files) > 0, "Should have documentation files"

            # Test parsing a specific Python file (our enhanced ingestion plugin)
            ingestion_file = None
            for py_file in python_files:
                if "ingestion_service/plugins/ingestion.py" in py_file:
                    ingestion_file = py_file
                    break

            assert ingestion_file is not None, "Should find the ingestion plugin file"

            # Verify the file contains our enhancements
            with open(ingestion_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for key enhancements we added
            enhancements = [
                "enterprise security patterns",
                "GITHUB_TOKEN",
                "GIT_USERNAME",
                "GIT_PASSWORD",
                "timeout=300",
                "os.chmod(temp_dir, 0o700)",
                "GitCommandError",
                "TimeoutError",
            ]

            found_enhancements = []
            for enhancement in enhancements:
                if enhancement in content:
                    found_enhancements.append(enhancement)

            print(
                f"✅ Found {len(found_enhancements)}/{len(enhancements)} expected enhancements"
            )

            # Should find most of our enhancements
            assert len(found_enhancements) >= 6, (
                f"Expected most enhancements, found: {found_enhancements}"
            )

        finally:
            # Clean up
            if "repo_path" in locals() and os.path.exists(repo_path):
                shutil.rmtree(repo_path, ignore_errors=True)


if __name__ == "__main__":
    # Run the tests directly
    pytest.main([__file__, "-v", "-s"])
