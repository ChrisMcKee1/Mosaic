#!/usr/bin/env python3
"""
Enhanced Mosaic Ingestion Service - Local Development with Cosmos DB Persistence
This version combines local development simplicity with real Cosmos DB persistence
"""

import asyncio
import argparse
import logging
import os
import sys
import tempfile
import shutil
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime
import json

# Add the parent directory to the path to import mosaic modules
sys.path.append(str(Path(__file__).parent.parent))

import git  # GitPython

# Import Cosmos SDK and authentication
try:
    from azure.cosmos import CosmosClient, exceptions as cosmos_exceptions
    from azure.identity import DefaultAzureCredential

    AZURE_SDK_AVAILABLE = True
except ImportError:
    AZURE_SDK_AVAILABLE = False

# Import our dual-mode manager
from cosmos_mode_manager import CosmosModeManager

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


class EnhancedLocalIngestionService:
    """
    Enhanced Local Development Ingestion Service with Cosmos DB persistence.

    This version:
    - Clones repositories using GitPython
    - Scans files and extracts basic statistics
    - Persists data to Cosmos DB (local or cloud)
    - Creates proper Golden Node entities
    - Works with real OmniRAG infrastructure
    """

    def __init__(self, mode: str = "local"):
        """Initialize the enhanced service."""
        self.mode = mode
        self.cosmos_manager = CosmosModeManager(mode)
        self.commit_state_manager = None
        self.cosmos_client = None
        self.database = None
        self.containers = {}

        # Initialize statistics
        self.stats = {
            "files_processed": 0,
            "entities_created": 0,
            "relationships_created": 0,
            "containers_created": 0,
            "errors": 0,
        }

    async def initialize(self) -> bool:
        """Initialize Cosmos DB connection and containers."""
        if not AZURE_SDK_AVAILABLE:
            logger.error("❌ Azure SDK not available")
            return False

        try:
            # Get Cosmos client
            self.cosmos_client = self.cosmos_manager.get_cosmos_client()
            if not self.cosmos_client:
                logger.error("❌ Failed to create Cosmos client")
                return False

            # Get database
            database_name = self.cosmos_manager.config["database"]
            self.database = self.cosmos_client.get_database_client(database_name)

            # Verify database exists
            try:
                self.database.read()
                logger.info(f"✅ Connected to database: {database_name}")
            except cosmos_exceptions.CosmosResourceNotFoundError:
                logger.warning(f"⚠️ Database '{database_name}' not found")
                return False

            # Initialize containers
            await self._initialize_containers()

            # Initialize commit state manager
            try:
                self.commit_state_manager = CommitStateManager(
                    self.cosmos_manager.config
                )
                logger.info("✅ CommitStateManager initialized")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize CommitStateManager: {e}")

            logger.info(f"✅ Enhanced service initialized (mode: {self.mode})")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize enhanced service: {e}")
            return False

    async def _initialize_containers(self):
        """Initialize Cosmos DB containers for data persistence."""
        container_names = [
            "knowledge",
            "memory",
            "golden_nodes",
            "diagrams",
            "code_entities",
            "code_relationships",
            "repositories",
            "mosaic",
            "context",
        ]

        for container_name in container_names:
            try:
                container = self.database.get_container_client(container_name)
                # Test container exists
                container.read()
                self.containers[container_name] = container
                logger.debug(f"✅ Container '{container_name}' available")
            except cosmos_exceptions.CosmosResourceNotFoundError:
                logger.warning(f"⚠️ Container '{container_name}' not found - skipping")
            except Exception as e:
                logger.warning(f"⚠️ Failed to access container '{container_name}': {e}")

        logger.info(f"📋 Initialized {len(self.containers)} containers")

    async def ingest_repository(
        self, repository_url: str, branch: str = "main"
    ) -> Dict[str, Any]:
        """
        Ingest a single repository with Cosmos DB persistence.

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
                f"🚀 Starting enhanced repository ingestion: {repository_url} (branch: {branch})"
            )

            # Ensure initialization
            if not self.cosmos_client:
                if not await self.initialize():
                    raise RuntimeError("Failed to initialize Cosmos DB connection")

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
                    logger.warning(f"⚠️ Failed to retrieve commit state: {e}")

            # Step 1: Clone repository
            temp_dir, repo = await self._clone_repository(repository_url, branch)
            logger.info(f"✅ Repository cloned to: {temp_dir}")

            # Step 2: Create repository entity
            repository_entity = await self._create_repository_entity(
                repository_url, branch, repo
            )
            if repository_entity:
                await self._persist_entity(repository_entity, "repositories")

            # Step 3: Scan and process files
            file_entities = await self._scan_repository_files(temp_dir, repository_url)

            # Step 4: Create code entities and relationships
            for file_entity in file_entities:
                await self._persist_entity(file_entity, "code_entities")
                self.stats["entities_created"] += 1

            # Step 5: Create inter-file relationships
            relationships = await self._create_file_relationships(file_entities)
            for relationship in relationships:
                await self._persist_entity(relationship, "code_relationships")
                self.stats["relationships_created"] += 1

            # CRUD-001: Update commit state
            if self.commit_state_manager and repo:
                try:
                    current_commit_sha = repo.head.commit.hexsha
                    await self.commit_state_manager.update_commit_state(
                        repository_url, branch, current_commit_sha
                    )
                    logger.info(f"📋 Updated commit state to: {current_commit_sha[:8]}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to update commit state: {e}")

            # Step 6: Create summary
            summary = {
                "repository_url": repository_url,
                "branch": branch,
                "files_processed": self.stats["files_processed"],
                "entities_created": self.stats["entities_created"],
                "relationships_created": self.stats["relationships_created"],
                "containers_available": len(self.containers),
                "last_commit_sha": repo.head.commit.hexsha if repo else None,
                "commit_state_tracking": bool(self.commit_state_manager),
                "timestamp": datetime.utcnow().isoformat(),
                "status": "completed",
                "mode": f"enhanced_{self.mode}",
                "persistence": "cosmos_db",
            }

            logger.info("✅ Enhanced repository ingestion completed successfully")
            logger.info(f"📊 Final stats: {self.stats}")
            return summary

        except Exception as e:
            logger.error(f"❌ Enhanced repository ingestion failed: {e}")
            self.stats["errors"] += 1
            raise
        finally:
            # Cleanup temporary directory
            if temp_dir and Path(temp_dir).exists():
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.debug(f"🧹 Cleaned up temporary directory: {temp_dir}")
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ Failed to cleanup {temp_dir}: {cleanup_error}")

    async def _clone_repository(
        self, repository_url: str, branch: str
    ) -> tuple[str, git.Repo]:
        """Clone repository using GitPython with comprehensive error handling."""
        temp_dir = None
        try:
            # Create secure temporary directory
            temp_dir = tempfile.mkdtemp(prefix="mosaic_enhanced_", suffix="_clone")
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
                logger.info(
                    f"🔐 Using authentication methods: {', '.join(auth_methods)}"
                )

            # Clone repository
            logger.info(f"📥 Cloning repository: {repository_url} (branch: {branch})")
            repo = git.Repo.clone_from(
                repository_url,
                temp_dir,
                branch=branch,
                depth=1,  # Shallow clone for faster processing
                env=git_env if git_env else None,
            )

            logger.info("✅ Repository cloned successfully")
            return temp_dir, repo

        except git.exc.GitError as e:
            logger.error(f"❌ Git error during cloning: {e}")
            if temp_dir and Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error during cloning: {e}")
            if temp_dir and Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    async def _create_repository_entity(
        self, repository_url: str, branch: str, repo: git.Repo
    ) -> Dict[str, Any]:
        """Create a repository entity for Cosmos DB storage."""
        try:
            entity = {
                "id": f"repo_{hash(repository_url)}",
                "type": "repository",
                "repository_url": repository_url,
                "branch": branch,
                "name": repository_url.split("/")[-1]
                if "/" in repository_url
                else repository_url,
                "commit_sha": repo.head.commit.hexsha,
                "commit_message": repo.head.commit.message.strip(),
                "commit_author": repo.head.commit.author.name,
                "commit_date": repo.head.commit.committed_datetime.isoformat(),
                "created_at": datetime.utcnow().isoformat(),
                "ingestion_mode": f"enhanced_{self.mode}",
                "_partitionKey": f"repo_{hash(repository_url)}",
            }

            logger.info(f"📦 Created repository entity: {entity['name']}")
            return entity

        except Exception as e:
            logger.error(f"❌ Failed to create repository entity: {e}")
            return None

    async def _scan_repository_files(
        self, temp_dir: str, repository_url: str
    ) -> List[Dict[str, Any]]:
        """Scan repository files and create basic entities."""
        file_entities = []
        repo_path = Path(temp_dir)

        # File extensions to process
        code_extensions = {
            ".py",
            ".js",
            ".ts",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".hpp",
            ".cs",
            ".go",
            ".rs",
            ".php",
            ".rb",
            ".swift",
            ".kt",
            ".scala",
            ".md",
            ".txt",
            ".json",
            ".yaml",
            ".yml",
            ".xml",
            ".html",
            ".css",
        }

        try:
            for file_path in repo_path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in code_extensions:
                    # Skip hidden files and directories
                    if any(part.startswith(".") for part in file_path.parts):
                        continue

                    try:
                        # Get relative path
                        relative_path = file_path.relative_to(repo_path)

                        # Read file content (with size limit)
                        file_size = file_path.stat().st_size
                        if file_size > 1024 * 1024:  # Skip files > 1MB
                            logger.debug(f"⏭️ Skipping large file: {relative_path}")
                            continue

                        # Read file content
                        content = ""
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                        except UnicodeDecodeError:
                            # Try with different encoding
                            try:
                                with open(file_path, "r", encoding="latin-1") as f:
                                    content = f.read()
                            except Exception:
                                logger.debug(f"⏭️ Skipping binary file: {relative_path}")
                                continue

                        # Create file entity
                        entity = {
                            "id": f"file_{hash(str(relative_path))}_{hash(repository_url)}",
                            "type": "file",
                            "repository_url": repository_url,
                            "file_path": str(relative_path),
                            "file_name": file_path.name,
                            "file_extension": file_path.suffix.lower(),
                            "file_size": file_size,
                            "line_count": content.count("\n") + 1 if content else 0,
                            "content_preview": content[:500] if content else "",
                            "content_hash": hash(content),
                            "language": self._detect_language(file_path.suffix.lower()),
                            "created_at": datetime.utcnow().isoformat(),
                            "ingestion_mode": f"enhanced_{self.mode}",
                            "_partitionKey": f"file_{hash(repository_url)}",
                        }

                        file_entities.append(entity)
                        self.stats["files_processed"] += 1

                        if self.stats["files_processed"] % 50 == 0:
                            logger.info(
                                f"📊 Processed {self.stats['files_processed']} files..."
                            )

                    except Exception as e:
                        logger.warning(f"⚠️ Failed to process file {relative_path}: {e}")
                        self.stats["errors"] += 1
                        continue

            logger.info(f"✅ Scanned {len(file_entities)} files from repository")
            return file_entities

        except Exception as e:
            logger.error(f"❌ Failed to scan repository files: {e}")
            return []

    def _detect_language(self, extension: str) -> str:
        """Detect programming language from file extension."""
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".go": "go",
            ".rs": "rust",
            ".php": "php",
            ".rb": "ruby",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".md": "markdown",
            ".txt": "text",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".xml": "xml",
            ".html": "html",
            ".css": "css",
        }
        return language_map.get(extension, "unknown")

    async def _create_file_relationships(
        self, file_entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create basic relationships between files."""
        relationships = []

        # Group files by directory
        directory_files = {}
        for entity in file_entities:
            file_path = Path(entity["file_path"])
            directory = str(file_path.parent)

            if directory not in directory_files:
                directory_files[directory] = []
            directory_files[directory].append(entity)

        # Create directory relationships
        for directory, files in directory_files.items():
            if len(files) > 1:
                for i, file1 in enumerate(files):
                    for file2 in files[i + 1 :]:
                        relationship = {
                            "id": f"rel_{hash(file1['id'] + file2['id'])}",
                            "type": "relationship",
                            "relationship_type": "same_directory",
                            "source_id": file1["id"],
                            "target_id": file2["id"],
                            "source_type": "file",
                            "target_type": "file",
                            "directory": directory,
                            "created_at": datetime.utcnow().isoformat(),
                            "_partitionKey": f"rel_{hash(file1['repository_url'])}",
                        }
                        relationships.append(relationship)

        # Create language-based relationships
        language_files = {}
        for entity in file_entities:
            language = entity["language"]
            if language not in language_files:
                language_files[language] = []
            language_files[language].append(entity)

        for language, files in language_files.items():
            if len(files) > 1 and language != "unknown":
                # Sample relationships to avoid explosion
                import random

                sample_size = min(10, len(files))
                sampled_files = random.sample(files, sample_size)

                for i, file1 in enumerate(sampled_files):
                    for file2 in sampled_files[i + 1 :]:
                        relationship = {
                            "id": f"rel_{hash(file1['id'] + file2['id'] + 'lang')}",
                            "type": "relationship",
                            "relationship_type": "same_language",
                            "source_id": file1["id"],
                            "target_id": file2["id"],
                            "source_type": "file",
                            "target_type": "file",
                            "language": language,
                            "created_at": datetime.utcnow().isoformat(),
                            "_partitionKey": f"rel_{hash(file1['repository_url'])}",
                        }
                        relationships.append(relationship)

        logger.info(f"✅ Created {len(relationships)} file relationships")
        return relationships

    async def _persist_entity(
        self, entity: Dict[str, Any], container_name: str
    ) -> bool:
        """Persist an entity to Cosmos DB."""
        if container_name not in self.containers:
            logger.warning(
                f"⚠️ Container '{container_name}' not available - skipping persistence"
            )
            return False

        try:
            container = self.containers[container_name]
            container.create_item(body=entity)
            logger.debug(
                f"✅ Persisted entity to '{container_name}': {entity.get('id', 'unknown')}"
            )
            return True

        except cosmos_exceptions.CosmosResourceExistsError:
            # Entity already exists - could update or skip
            logger.debug(
                f"⏭️ Entity already exists in '{container_name}': {entity.get('id', 'unknown')}"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Failed to persist entity to '{container_name}': {e}")
            self.stats["errors"] += 1
            return False


async def main():
    """Main function for enhanced local ingestion."""
    parser = argparse.ArgumentParser(
        description="Enhanced Mosaic Ingestion Service with Cosmos DB persistence"
    )
    parser.add_argument(
        "--repository-url", required=True, help="Git repository URL to ingest"
    )
    parser.add_argument(
        "--branch", default="main", help="Git branch to process (default: main)"
    )
    parser.add_argument(
        "--mode",
        choices=["local", "azure"],
        default="local",
        help="Cosmos DB mode (default: local)",
    )

    args = parser.parse_args()

    # Initialize service
    service = EnhancedLocalIngestionService(mode=args.mode)

    # Initialize Cosmos DB
    if not await service.initialize():
        logger.error("❌ Failed to initialize service")
        sys.exit(1)

    try:
        # Ingest repository
        result = await service.ingest_repository(args.repository_url, args.branch)

        logger.info("✅ Enhanced ingestion completed successfully!")
        logger.info(f"📊 Summary: {json.dumps(result, indent=2)}")

    except Exception as e:
        logger.error(f"❌ Enhanced ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
