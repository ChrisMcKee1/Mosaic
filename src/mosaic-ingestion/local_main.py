#!/usr/bin/env python3
"""
Mosaic Ingestion Service - Local Development Entry Point
Lightweight version for local development and testing with minimal dependencies
"""

import asyncio
import argparse
import logging
import os
import sys
import tempfile
import shutil
from typing import Dict, Any
from pathlib import Path
from datetime import datetime

# Add the parent directory to the path to import mosaic modules
sys.path.append(str(Path(__file__).parent.parent))

import git  # GitPython

# CRUD-001: Import CommitStateManager for commit state tracking
from utils.commit_state_manager import CommitStateManager

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mosaic_ingestion.log", mode="w"),
    ],
)

# Set specific loggers to avoid spam
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("git").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


class LocalIngestionService:
    """
    Local Development Ingestion Service for basic functionality testing.

    This version:
    - Clones repositories using GitPython
    - Scans files and extracts basic statistics
    - Works without heavy Azure dependencies
    - Provides comprehensive logging for debugging
    - Ideal for local development and testing
    """

    def __init__(self):
        """Initialize the Local Development Ingestion Service."""
        self.supported_extensions = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".c": "c",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".cs": "csharp",
            ".html": "html",
            ".htm": "html",
            ".css": "css",
            ".scss": "css",
            ".sass": "css",
        }

        # Statistics
        self.stats = {
            "files_processed": 0,
            "lines_of_code": 0,
            "languages_detected": set(),
            "entities_found": 0,
        }

        # CRUD-001: Initialize commit state manager for development (optional)
        self.commit_state_manager = None
        try:
            # Try to initialize CommitStateManager if Cosmos config is available
            from agents.base_agent import BaseAgent

            settings = BaseAgent.load_settings()
            self.commit_state_manager = CommitStateManager.from_settings(settings)
            logger.info("✅ CommitStateManager initialized for local development")
        except Exception as e:
            logger.info(
                f"ℹ️  CommitStateManager not available (running in simple mode): {e}"
            )
            self.commit_state_manager = None

    async def ingest_repository(
        self, repository_url: str, branch: str = "main"
    ) -> Dict[str, Any]:
        """
        Ingest a single repository using simplified approach.

        Args:
            repository_url: Git repository URL
            branch: Git branch to process

        Returns:
            Ingestion summary with statistics
        """
        temp_dir = None
        repo = None
        try:
            logger.info(
                f"🚀 Starting repository ingestion: {repository_url} (branch: {branch})"
            )

            # CRUD-001: Check commit state for incremental processing
            last_commit_state = None
            if self.commit_state_manager:
                try:
                    last_commit_state = await self.commit_state_manager.get_last_commit(
                        repository_url, branch
                    )
                    if last_commit_state:
                        logger.info(
                            f"📋 Found previous commit state: {last_commit_state.last_commit_sha[:8]}"
                        )
                    else:
                        logger.info(
                            "📋 No previous commit state found (first-time ingestion)"
                        )
                except Exception as e:
                    logger.warning(f"⚠️  Failed to retrieve commit state: {e}")

            # Step 1: Clone repository
            temp_dir, repo = await self._clone_repository(repository_url, branch)
            logger.info(f"✅ Repository cloned to: {temp_dir}")

            # CRUD-001: Process commits since last state
            commits_processed = 0
            if self.commit_state_manager and repo:
                try:
                    last_commit_sha = (
                        last_commit_state.last_commit_sha if last_commit_state else None
                    )
                    new_commits = self.commit_state_manager.iter_commits_since_last(
                        repo, repository_url, branch, last_commit_sha
                    )
                    commits_processed = len(new_commits)

                    if commits_processed > 0:
                        logger.info(f"📈 Processing {commits_processed} new commits")
                    else:
                        logger.info("✅ Repository is up to date (no new commits)")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to analyze commits: {e}")

            # Step 2: Scan and analyze files
            await self._scan_repository(temp_dir)
            logger.info("✅ Repository scan completed")

            # CRUD-001: Update commit state after successful processing
            current_commit_sha = None
            if self.commit_state_manager and repo:
                try:
                    current_commit_sha = repo.head.commit.hexsha
                    new_count = (
                        last_commit_state.commit_count if last_commit_state else 0
                    ) + commits_processed

                    await self.commit_state_manager.update_commit_state(
                        repository_url=repository_url,
                        branch_name=branch,
                        current_commit_sha=current_commit_sha,
                        commit_count=new_count,
                        processing_status="completed",
                    )
                    logger.info(f"✅ Updated commit state: {current_commit_sha[:8]}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to update commit state: {e}")

            # Step 3: Generate summary
            summary = {
                "repository_url": repository_url,
                "branch": branch,
                "files_processed": self.stats["files_processed"],
                "lines_of_code": self.stats["lines_of_code"],
                "languages_detected": list(self.stats["languages_detected"]),
                "entities_extracted": self.stats["entities_found"],
                "relationships_found": 0,  # Simplified version doesn't extract relationships
                "commits_processed": commits_processed,
                "current_commit": current_commit_sha[:8]
                if current_commit_sha
                else None,
                "last_commit": last_commit_state.last_commit_sha[:8]
                if last_commit_state
                else None,
                "commit_state_tracking": bool(self.commit_state_manager),
                "timestamp": datetime.utcnow().isoformat(),
                "status": "completed",
                "mode": "local_development",
            }

            logger.info("✅ Repository ingestion completed successfully")
            return summary

        except Exception as e:
            logger.error(f"❌ Repository ingestion failed: {e}")
            raise
        finally:
            # Cleanup temporary directory
            if temp_dir and Path(temp_dir).exists():
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.debug(f"🧹 Cleaned up temporary directory: {temp_dir}")
                except Exception as cleanup_error:
                    logger.warning(f"⚠️  Failed to cleanup {temp_dir}: {cleanup_error}")

    async def _clone_repository(
        self, repository_url: str, branch: str
    ) -> tuple[str, git.Repo]:
        """
        Clone repository using GitPython with comprehensive error handling.

        Returns:
            Tuple of (temp_directory_path, git_repo_object)
        """
        temp_dir = None
        try:
            # Create secure temporary directory
            temp_dir = tempfile.mkdtemp(prefix="mosaic_local_", suffix="_clone")
            logger.info(f"📁 Created temporary directory: {temp_dir}")

            # Configure Git authentication if available
            git_env = {}
            auth_methods = []

            if os_env_github_token := os.getenv("GITHUB_TOKEN"):
                auth_methods.append("GITHUB_TOKEN")
                if "github.com" in repository_url:
                    git_env["GIT_HTTP_EXTRAHEADER"] = (
                        f"Authorization: token {os_env_github_token}"
                    )

            git_username = os.getenv("GIT_USERNAME")
            git_password = os.getenv("GIT_PASSWORD")
            if git_username and git_password:
                auth_methods.append("GIT_USERNAME/GIT_PASSWORD")

            if auth_methods:
                logger.info(f"🔐 Using authentication methods: {auth_methods}")

            logger.info(f"📥 Cloning {repository_url} (branch: {branch})...")

            # CRUD-001: Use CommitStateManager for full history clone
            if self.commit_state_manager:
                repo = self.commit_state_manager.clone_with_full_history(
                    repository_url, temp_dir, branch
                )
            else:
                # Execute clone with full history (no depth limitation)
                repo = git.Repo.clone_from(
                    repository_url,
                    temp_dir,
                    branch=branch,
                    # CRUD-001: Removed depth=1 shallow clone to enable commit history access
                    env=git_env if git_env else None,
                )

            # Verify clone success
            if not repo.heads:
                raise git.exc.InvalidGitRepositoryError(
                    "No branches found in cloned repository"
                )

            commit_hash = repo.head.commit.hexsha[:8]
            logger.info(f"✅ Clone successful - Commit: {commit_hash}")

            return temp_dir, repo

        except git.exc.GitCommandError as git_error:
            error_msg = str(git_error).lower()
            if "authentication failed" in error_msg:
                raise git.exc.GitCommandError(
                    f"❌ Authentication failed for {repository_url}. "
                    f"Set GITHUB_TOKEN, GIT_USERNAME/GIT_PASSWORD environment variables."
                ) from git_error
            elif "repository not found" in error_msg:
                raise git.exc.GitCommandError(
                    f"❌ Repository not found: {repository_url}. Check URL and permissions."
                ) from git_error
            elif "network" in error_msg or "timeout" in error_msg:
                raise git.exc.GitCommandError(
                    f"❌ Network timeout cloning {repository_url}. Check connectivity."
                ) from git_error
            else:
                raise

        except Exception as e:
            # Cleanup on failure
            if temp_dir and Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            logger.error(f"❌ Repository clone failed: {e}")
            raise

    async def _scan_repository(self, repo_path: str) -> None:
        """
        Scan repository files and extract basic statistics.
        """
        try:
            repo_root = Path(repo_path)
            logger.info(f"🔍 Scanning repository at: {repo_path}")

            # Reset statistics
            self.stats = {
                "files_processed": 0,
                "lines_of_code": 0,
                "languages_detected": set(),
                "entities_found": 0,
            }

            # Skip directories
            skip_dirs = {
                ".git",
                "node_modules",
                "__pycache__",
                ".pytest_cache",
                "target",
                "dist",
                "build",
                ".venv",
                "venv",
                "vendor",
            }

            # Walk through files
            for file_path in repo_root.rglob("*"):
                if not file_path.is_file():
                    continue

                # Skip files in ignored directories
                if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                    continue

                # Process supported file types
                extension = file_path.suffix.lower()
                if extension not in self.supported_extensions:
                    continue

                language = self.supported_extensions[extension]
                self.stats["languages_detected"].add(language)

                try:
                    await self._process_file(file_path, language)
                except Exception as e:
                    logger.warning(f"⚠️  Failed to process {file_path}: {e}")
                    continue

            logger.info("📊 Scan Results:")
            logger.info(f"   • Files processed: {self.stats['files_processed']}")
            logger.info(f"   • Lines of code: {self.stats['lines_of_code']}")
            logger.info(
                f"   • Languages: {', '.join(self.stats['languages_detected'])}"
            )
            logger.info(f"   • Basic entities found: {self.stats['entities_found']}")

        except Exception as e:
            logger.error(f"❌ Repository scan failed: {e}")
            raise

    async def _process_file(self, file_path: Path, language: str) -> None:
        """Process a single file and extract basic statistics."""
        try:
            # Check file size (skip very large files > 1MB)
            file_size = file_path.stat().st_size
            if file_size > 1024 * 1024:
                logger.debug(f"⏭️  Skipping large file ({file_size} bytes): {file_path}")
                return

            # Read and analyze file
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.debug(f"⚠️  Failed to read {file_path}: {e}")
                return

            # Count lines
            lines = content.split("\n")
            non_empty_lines = len([line for line in lines if line.strip()])

            self.stats["files_processed"] += 1
            self.stats["lines_of_code"] += non_empty_lines

            # Simple entity detection based on common patterns
            entities = self._detect_simple_entities(content, language)
            self.stats["entities_found"] += len(entities)

            if len(entities) > 0:
                logger.debug(
                    f"📝 {file_path.name}: {non_empty_lines} lines, {len(entities)} entities ({language})"
                )

        except Exception as e:
            logger.debug(f"❌ Error processing file {file_path}: {e}")

    def _detect_simple_entities(self, content: str, language: str) -> list:
        """Detect simple entities using basic text patterns."""
        entities = []
        lines = content.split("\n")

        # Simple pattern-based entity detection
        patterns = {
            "python": [
                ("function", r"def \w+\("),
                ("class", r"class \w+[:\(]"),
                ("async_function", r"async def \w+\("),
            ],
            "javascript": [
                ("function", r"function \w+\("),
                ("class", r"class \w+ "),
                ("arrow_function", r"const \w+ = \([^)]*\) =>"),
                ("method", r"\w+\([^)]*\) \{"),
            ],
            "typescript": [
                ("function", r"function \w+\("),
                ("class", r"class \w+ "),
                ("interface", r"interface \w+ "),
                ("type", r"type \w+ ="),
            ],
            "java": [
                ("class", r"public class \w+"),
                ("method", r"public \w+ \w+\("),
                ("private_method", r"private \w+ \w+\("),
            ],
            "go": [
                ("function", r"func \w+\("),
                ("type", r"type \w+ struct"),
                ("method", r"func \([^)]+\) \w+\("),
            ],
            "csharp": [
                ("class", r"public class \w+"),
                ("method", r"public \w+ \w+\("),
                ("property", r"public \w+ \w+ \{"),
            ],
        }

        import re

        language_patterns = patterns.get(language, [])
        for line_num, line in enumerate(lines, 1):
            for entity_type, pattern in language_patterns:
                if re.search(pattern, line):
                    entities.append(
                        {
                            "type": entity_type,
                            "line": line_num,
                            "content": line.strip()[:100],  # First 100 chars
                        }
                    )

        return entities


async def main() -> None:
    """Main entry point for the Local Development Ingestion Service."""
    logger.info("🚀 Starting Local Development Mosaic Ingestion Service...")

    parser = argparse.ArgumentParser(
        description="Mosaic Local Development Ingestion Service"
    )
    parser.add_argument(
        "--repository-url", required=True, help="Git repository URL to ingest"
    )
    parser.add_argument(
        "--branch", default="main", help="Git branch to process (default: main)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    try:
        args = parser.parse_args()
        logger.info(
            f"📋 Arguments: repository={args.repository_url}, branch={args.branch}"
        )

        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("🐛 Debug logging enabled")

        # Create and run service
        logger.info("🏗️  Creating Local Development Ingestion Service...")
        service = LocalIngestionService()

        # Perform ingestion
        logger.info("⚡ Starting repository ingestion...")
        result = await service.ingest_repository(args.repository_url, args.branch)

        # Log final result
        logger.info("\n" + "=" * 60)
        logger.info("📊 INGESTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"🔗 Repository: {result['repository_url']}")
        logger.info(f"🌿 Branch: {result['branch']}")
        logger.info(f"📁 Files processed: {result['files_processed']}")
        logger.info(f"📝 Lines of code: {result['lines_of_code']}")
        logger.info(f"💻 Languages: {', '.join(result['languages_detected'])}")
        logger.info(f"🎯 Entities found: {result['entities_extracted']}")
        logger.info(f"✅ Status: {result['status']}")
        logger.info(f"🔧 Mode: {result['mode']}")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("🛑 Received shutdown signal (Ctrl+C)")
    except ImportError as import_error:
        logger.error(f"❌ Import error - missing dependency: {import_error}")
        logger.error("💡 Try: pip install GitPython python-dotenv")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Local Development Ingestion Service error: {e}")
        logger.exception("🔍 Full error traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
