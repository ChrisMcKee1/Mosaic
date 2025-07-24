"""
IngestionPlugin for Mosaic Ingestion Service

Implements Phase 1 Foundation: Repository Access, AST Parsing, and Entity Extraction.
This addresses the critical code ingestion pipeline gap identified in the implementation analysis.

ARCHITECTURAL NOTE: This plugin has been moved from the MCP Query Server to maintain
proper architectural separation. This is NOT a Semantic Kernel plugin
since it runs in a standalone Azure Container App Job.

Key Components:
- GitPython-based repository cloning and access
- tree-sitter multi-language AST parsing
- Entity extraction and graph modeling
- Azure Cosmos DB knowledge population using OmniRAG pattern

Based on Context7 research validation of 2025 best practices.
"""

import logging
import os
import shutil
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import asyncio
import hashlib
import random

# GitPython for repository access
import git
import tempfile

# Semantic Kernel for AI-powered descriptions with error handling
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextEmbedding,
)
from semantic_kernel.functions import KernelArguments

# tree-sitter for AST parsing
import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_typescript
import tree_sitter_java
import tree_sitter_go
import tree_sitter_rust
import tree_sitter_c
import tree_sitter_cpp
import tree_sitter_c_sharp
import tree_sitter_html
import tree_sitter_css
from tree_sitter import Language, Parser

# Azure services with Semantic Kernel for AI-powered analysis
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential

# AI analysis and error handling components
from .code_analysis_plugin import CodeAnalysisPlugin
from .ai_error_handler import AIOrchestrationErrorHandler, configure_azure_telemetry
from semantic_kernel.prompt_template import PromptTemplateConfig

# Import from parent mosaic package
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from mosaic.config.settings import MosaicSettings


logger = logging.getLogger(__name__)


class IngestionPlugin:
    """
    Standalone ingestion plugin for the Mosaic Ingestion Service.

    NOTE: This plugin has been moved from the MCP Query Server to maintain
    proper architectural separation. This is NOT a Semantic Kernel plugin
    since it runs in a standalone Azure Container App Job.

    Implements Phase 1 Foundation requirements:
    - Repository access and cloning with GitPython
    - Multi-language AST parsing with tree-sitter
    - Entity extraction and relationship modeling
    - Knowledge graph population using OmniRAG pattern
    """

    def __init__(self, settings: MosaicSettings):
        """Initialize the IngestionPlugin with validated technologies."""
        self.settings = settings
        self.cosmos_client: Optional[CosmosClient] = None
        self.database = None
        self.knowledge_container = None
        self.embedding_service: Optional[AzureTextEmbedding] = None
        self.kernel: Optional[Kernel] = None
        self.chat_service: Optional[AzureChatCompletion] = None

        # Initialize AI enhancement components with error handling
        self._description_cache = {}  # SHA-256 hash -> description cache
        self._processing_semaphore = asyncio.Semaphore(5)  # Limit concurrent API calls
        self._cache_hits = 0
        self._cache_misses = 0
        self._api_calls = 0

        # Initialize Azure monitoring and error handling
        connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
        self.tracer = configure_azure_telemetry(connection_string)
        self.error_handler = AIOrchestrationErrorHandler(connection_string)
        self.code_analysis_plugin: Optional[CodeAnalysisPlugin] = None

        # Initialize tree-sitter languages
        self.languages = self._initialize_languages()
        self.parsers = self._initialize_parsers()

        # Supported file extensions per language
        self.language_extensions = {
            "python": {".py", ".pyi"},
            "javascript": {".js", ".jsx", ".mjs"},
            "typescript": {".ts", ".tsx"},
            "java": {".java"},
            "go": {".go"},
            "rust": {".rs"},
            "c": {".c", ".h"},
            "cpp": {".cpp", ".cc", ".cxx", ".hpp", ".hxx"},
            "csharp": {
                ".cs",
                ".cshtml",
                ".razor",
            },  # C#, Razor Pages, Blazor components
            "html": {".html", ".htm", ".xhtml"},
            "css": {".css", ".scss", ".sass", ".less"},  # CSS and preprocessors
        }

    def _initialize_languages(self) -> Dict[str, Language]:
        """Initialize tree-sitter languages using Context7 validated approach."""
        languages = {}

        try:
            # Initialize each supported language
            languages["python"] = Language(tree_sitter_python.language())
            languages["javascript"] = Language(tree_sitter_javascript.language())
            languages["typescript"] = Language(
                tree_sitter_typescript.language_typescript()
            )
            languages["java"] = Language(tree_sitter_java.language())
            languages["go"] = Language(tree_sitter_go.language())
            languages["rust"] = Language(tree_sitter_rust.language())
            languages["c"] = Language(tree_sitter_c.language())
            languages["cpp"] = Language(tree_sitter_cpp.language())
            languages["csharp"] = Language(tree_sitter_c_sharp.language())
            languages["html"] = Language(tree_sitter_html.language())
            languages["css"] = Language(tree_sitter_css.language())

            logger.info(f"Initialized {len(languages)} tree-sitter languages")

        except Exception as e:
            logger.error(f"Failed to initialize tree-sitter languages: {e}")
            raise

        return languages

    def _initialize_parsers(self) -> Dict[str, Parser]:
        """Initialize tree-sitter parsers for each language."""
        parsers = {}

        for lang_name, language in self.languages.items():
            parser = Parser()
            parser.set_language(language)
            parsers[lang_name] = parser

        return parsers

    async def initialize(self) -> None:
        """Initialize Azure services and connections."""
        try:
            # Initialize Cosmos DB client with managed identity
            await self._initialize_cosmos()

            # Initialize embedding service
            await self._initialize_embedding_service()

            # Initialize Semantic Kernel for AI-powered code analysis
            await self._initialize_semantic_kernel()

            logger.info("IngestionPlugin initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize IngestionPlugin: {e}")
            raise

    async def _initialize_cosmos(self) -> None:
        """Initialize Azure Cosmos DB connection with managed identity."""
        cosmos_config = self.settings.get_cosmos_config()

        # Always use managed identity as validated by Context7 research
        credential = DefaultAzureCredential()
        self.cosmos_client = CosmosClient(cosmos_config["endpoint"], credential)

        # Get database and container references
        self.database = self.cosmos_client.get_database_client(
            cosmos_config["database_name"]
        )
        self.knowledge_container = self.database.get_container_client(
            cosmos_config["container_name"]
        )

    async def _initialize_embedding_service(self) -> None:
        """Initialize Azure embedding service with managed identity."""
        try:
            self.embedding_service = AzureTextEmbedding(
                deployment_name=self.settings.azure_openai_text_embedding_deployment_name,
                endpoint=self.settings.azure_openai_endpoint,
                service_id="ingestion_embedding",
            )
            logger.info("Azure embedding service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure embedding service: {e}")
            raise

    async def _initialize_semantic_kernel(self) -> None:
        """Initialize Semantic Kernel for AI-powered code analysis."""
        try:
            # Initialize kernel with comprehensive error handling
            self.kernel = Kernel()

            # Add error handler filter to kernel for comprehensive AI error handling
            self.kernel.add_filter(self.error_handler)

            # Add Azure OpenAI chat service for code analysis
            self.chat_service = AzureChatCompletion(
                deployment_name=self.settings.azure_openai_chat_deployment_name,
                endpoint=self.settings.azure_openai_endpoint,
                service_id="code_analyzer",
            )
            self.kernel.add_service(self.chat_service)

            # Initialize code analysis plugin with error handling
            self.code_analysis_plugin = CodeAnalysisPlugin(self.settings)
            self.kernel.add_plugin(
                self.code_analysis_plugin, plugin_name="ai_code_analysis"
            )

            # Register code analysis function
            await self._register_code_analysis_functions()

            logger.info("Semantic Kernel initialized for code analysis")

        except Exception as e:
            logger.error(f"Failed to initialize Semantic Kernel: {e}")
            raise

    async def _register_code_analysis_functions(self) -> None:
        """Register enhanced Semantic Kernel functions for comprehensive code analysis."""

        # Enhanced AI analyst prompt template
        ai_analyst_prompt = """
You are a senior software architect analyzing code. Provide a comprehensive analysis of this {{$entity_type}}.

Code Entity: {{$name}}
Language: {{$language}}
File: {{$file_path}}

Code:
```{{$language}}
{{$content}}
```

Please provide a structured analysis in exactly this format:

SUMMARY: [One clear sentence describing the primary purpose and functionality]
COMPLEXITY: [Score from 1-10, where 1=trivial, 5=moderate, 10=very complex]
DEPENDENCIES: [List key dependencies this code relies on, separated by commas]
TAGS: [Relevant keywords/categories for this code, separated by commas]

Guidelines:
- SUMMARY: Focus on what this code accomplishes, not how
- COMPLEXITY: Consider algorithm complexity, number of responsibilities, and maintainability
- DEPENDENCIES: Include libraries, frameworks, other modules, external systems
- TAGS: Include domain concepts, patterns, technologies, functional areas

Be concise but accurate. Each section should be on its own line with the exact label format shown."""

        # Create the enhanced prompt template configuration
        prompt_config = PromptTemplateConfig(
            template=ai_analyst_prompt,
            name="analyze_code_entity",
            description="Generate comprehensive AI analyst insights for code entities",
        )

        # Register the enhanced AI analyst function with the kernel
        self.kernel.add_function(
            plugin_name="CodeAnalysis",
            function_name="analyze_code_entity",
            prompt_template_config=prompt_config,
        )

    async def ingest_repository(
        self, repository_url: str, branch: str = "main", force_full: bool = False
    ) -> Dict[str, Any]:
        """
        Enhanced two-pass repository ingestion with incremental update support.

        MODES:
        - Full Ingestion: Complete analysis of entire repository (first time or with force_full=True)
        - Incremental Update: Only analyze files changed from main branch (much faster)

        PROCESS:
        - PASS 1: Build comprehensive entity map across all files
        - PASS 2: Extract cross-file relationships using the entity map
        - ENHANCEMENT: Upgrade AI from description writer to code analyst
        - AI FALLBACK: Handle unknown file types with Semantic Kernel

        Args:
            repository_url: Git repository URL
            branch: Git branch to process
            force_full: Force full re-ingestion even if repo exists

        Returns:
            Ingestion summary with statistics
        """
        temp_dir = None
        repo = None

        try:
            # Step 1: Determine ingestion mode (full vs incremental)
            logger.info(
                f"Starting enhanced repository ingestion: {repository_url} (branch: {branch})"
            )
            temp_dir = await self._clone_repository(repository_url, branch)
            repo = git.Repo(temp_dir)

            # Check for incremental update possibility
            if not force_full and branch != "main":
                logger.info("Checking for incremental update possibility...")
                changed_files = await self._get_changed_files(repo, branch)

                if changed_files:
                    logger.info(
                        f"INCREMENTAL MODE: Processing {len(changed_files)} changed files"
                    )
                    return await self._process_incremental_update(
                        temp_dir, changed_files, repository_url, branch
                    )
                else:
                    logger.info("No changes detected. Repository is up-to-date.")
                    return {
                        "repository_url": repository_url,
                        "branch": branch,
                        "status": "up-to-date",
                        "files_processed": 0,
                        "mode": "incremental_check",
                        "timestamp": datetime.utcnow().isoformat(),
                    }

            # Step 2: FULL INGESTION MODE
            logger.info("FULL INGESTION MODE: Complete repository analysis")

            # PASS 1 - Build global entity map
            logger.info("PASS 1: Building comprehensive entity map across all files")
            entity_map = await self._first_pass_build_entity_map(temp_dir)

            # PASS 2 - Extract relationships using the entity map
            logger.info("PASS 2: Extracting cross-file relationships using entity map")
            relationships = await self._second_pass_extract_relationships(
                temp_dir, entity_map
            )

            # AI Enhancement - Upgrade entities with AI code analyst
            logger.info("AI ENHANCEMENT: Upgrading entities with AI code analyst")
            enhanced_entities = await self._enhance_entities_with_ai_analyst(
                list(entity_map.values())
            )

            # Generate richer embeddings and populate knowledge base
            logger.info(
                "Populating knowledge base with enhanced entities and relationships"
            )
            await self._populate_knowledge_base(enhanced_entities, relationships)

            # Generate comprehensive summary
            summary = {
                "repository_url": repository_url,
                "branch": branch,
                "entities_extracted": len(enhanced_entities),
                "relationships_found": len(relationships),
                "cross_file_connections": len(
                    [
                        r
                        for r in relationships
                        if r.get("type") == "cross_file_dependency"
                    ]
                ),
                "ai_enhanced_entities": len(
                    [e for e in enhanced_entities if e.get("has_ai_analyst", False)]
                ),
                "ai_analyzed_files": len(
                    [e for e in enhanced_entities if e.get("analyzed_by_ai", False)]
                ),
                "timestamp": datetime.utcnow().isoformat(),
                "status": "completed",
                "mode": "full_ingestion",
                "enhancement_version": "2.0_two_pass_ai_analyst_incremental",
            }

            logger.info(f"Enhanced repository ingestion completed: {summary}")
            return summary

        except Exception as e:
            logger.error(f"Repository ingestion failed: {e}")
            raise
        finally:
            # Cleanup temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    async def _get_changed_files(
        self, repo: git.Repo, current_branch: str
    ) -> list[str]:
        """
        Get list of files changed from main branch using git diff.

        This enables incremental ingestion by only processing files that have
        been modified, added, or deleted compared to the main branch.

        Args:
            repo: GitPython repository object
            current_branch: Current branch name to compare

        Returns:
            List of relative file paths that have changed
        """
        try:
            # Ensure we have the main branch reference
            main_branch = None
            for ref in ["main", "master", "develop"]:
                try:
                    main_branch = repo.heads[ref]
                    break
                except (IndexError, KeyError):
                    continue

            if not main_branch:
                logger.warning(
                    "No main branch found (main/master/develop), forcing full ingestion"
                )
                return []

            current_commit = repo.head.commit
            main_commit = main_branch.commit

            # Get files changed between main and current branch
            diff = main_commit.diff(current_commit)

            changed_files = []
            for item in diff:
                # Handle different types of changes
                if item.a_path:  # Modified or deleted file
                    changed_files.append(item.a_path)
                if item.b_path and item.b_path != item.a_path:  # Renamed file
                    changed_files.append(item.b_path)

            # Filter to only include supported file types
            supported_extensions = set()
            for exts in self.language_extensions.values():
                supported_extensions.update(exts)

            filtered_files = [
                f
                for f in changed_files
                if any(f.endswith(ext) for ext in supported_extensions)
            ]

            logger.info(
                f"Found {len(filtered_files)} changed files with supported extensions"
            )
            return filtered_files

        except Exception as e:
            logger.error(f"Error getting changed files: {e}")
            return []  # Fall back to full ingestion

    async def _process_incremental_update(
        self, repo_path: str, changed_files: list[str], repository_url: str, branch: str
    ) -> dict:
        """
        Process incremental update for only the changed files.

        This is much faster than full ingestion as it only processes files
        that have changed from the main branch.

        Args:
            repo_path: Path to the cloned repository
            changed_files: List of files that have changed
            repository_url: Repository URL for metadata
            branch: Branch name for metadata

        Returns:
            Summary of incremental processing
        """
        try:
            logger.info(f"Processing incremental update for {len(changed_files)} files")

            # Step 1: Delete existing entities for changed files
            await self._delete_entities_by_filepath(changed_files)

            # Step 2: Build entity map for only changed files
            entity_map = {}
            processed_files = 0

            for rel_file_path in changed_files:
                full_file_path = Path(repo_path) / rel_file_path

                if not full_file_path.exists():
                    logger.debug(
                        f"File {rel_file_path} no longer exists (likely deleted)"
                    )
                    continue

                try:
                    # Extract entities from this specific file
                    file_entities = await self._extract_entities_from_file(
                        full_file_path
                    )

                    # Add to entity map with relative paths as keys
                    for entity in file_entities:
                        entity_name = entity.get("name", "unknown")
                        relative_path = self._calculate_relative_path(
                            full_file_path, repo_path
                        )
                        key = f"{relative_path}::{entity_name}"
                        entity_map[key] = entity

                    processed_files += 1
                    logger.debug(f"Processed changed file: {rel_file_path}")

                except Exception as e:
                    logger.warning(
                        f"Error processing changed file {rel_file_path}: {e}"
                    )
                    continue

            # Step 3: Extract relationships for changed files only
            # Note: This is simplified - full cross-file analysis would require
            # loading the complete entity map, but for incremental updates we
            # focus on relationships within the changed files
            relationships = []

            for entity_key, entity in entity_map.items():
                file_path = Path(entity["file_path"])
                try:
                    # Extract relationships within this file and to other files
                    file_relationships = await self._extract_file_relationships(
                        file_path, entity, entity_map
                    )
                    relationships.extend(file_relationships)
                except Exception as e:
                    logger.debug(f"Error extracting relationships for {file_path}: {e}")
                    continue

            # Step 4: AI enhancement for changed entities
            enhanced_entities = await self._enhance_entities_with_ai_analyst(
                list(entity_map.values())
            )

            # Step 5: Update knowledge base
            await self._populate_knowledge_base(enhanced_entities, relationships)

            # Step 6: Generate summary
            summary = {
                "repository_url": repository_url,
                "branch": branch,
                "mode": "incremental_update",
                "files_changed": len(changed_files),
                "files_processed": processed_files,
                "entities_updated": len(enhanced_entities),
                "relationships_updated": len(relationships),
                "ai_enhanced_entities": len(
                    [e for e in enhanced_entities if e.get("has_ai_analyst", False)]
                ),
                "timestamp": datetime.utcnow().isoformat(),
                "status": "completed",
                "enhancement_version": "2.0_incremental_ai_analyst",
            }

            logger.info(f"Incremental update completed: {summary}")
            return summary

        except Exception as e:
            logger.error(f"Incremental update failed: {e}")
            raise

    async def _delete_entities_by_filepath(self, changed_files: list[str]) -> None:
        """
        Delete existing entities for files that have changed.

        This ensures that incremental updates don't create duplicate
        entities when files are modified.

        Args:
            changed_files: List of relative file paths to clean up
        """
        try:
            for file_path in changed_files:
                # Query for entities matching this file path
                query = "SELECT * FROM c WHERE c.file_path CONTAINS @file_path"
                parameters = [{"name": "@file_path", "value": file_path}]

                items = self.knowledge_container.query_items(
                    query=query,
                    parameters=parameters,
                    enable_cross_partition_query=True,
                )

                # Delete each matching entity
                for item in items:
                    try:
                        await self.knowledge_container.delete_item(
                            item=item["id"], partition_key=item.get("type", "unknown")
                        )
                    except Exception as e:
                        logger.debug(
                            f"Error deleting entity {item.get('id', 'unknown')}: {e}"
                        )
                        continue

            logger.info(f"Cleaned up entities for {len(changed_files)} changed files")

        except Exception as e:
            logger.warning(f"Error during entity cleanup: {e}")
            # Don't fail the entire process if cleanup has issues

    async def _extract_file_relationships(
        self, file_path: Path, entity: dict, entity_map: dict
    ) -> list[dict]:
        """
        Extract relationships for a specific file and entity.

        This is a simplified version of relationship extraction focused
        on a single file for incremental updates.

        Args:
            file_path: Path to the file being analyzed
            entity: Entity dictionary from the file
            entity_map: Map of all entities for reference resolution

        Returns:
            List of relationship dictionaries
        """
        relationships = []

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            language = self._detect_language(file_path)

            if language == "python":
                # Extract Python import relationships
                import_lines = [
                    line.strip()
                    for line in content.split("\n")
                    if line.strip().startswith(("from ", "import "))
                ]

                for import_line in import_lines:
                    # Simple import parsing for incremental updates
                    if " import " in import_line:
                        try:
                            target_module = (
                                import_line.split(" import ")[0]
                                .replace("from ", "")
                                .strip()
                            )

                            # Look for matching entity in entity map
                            for key, target_entity in entity_map.items():
                                if target_module in target_entity.get("name", ""):
                                    relationship = {
                                        "id": f"import_{entity['id']}_{target_entity['id']}",
                                        "type": "imports",
                                        "source_id": entity["id"],
                                        "target_id": target_entity["id"],
                                        "source_file": str(file_path),
                                        "target_file": target_entity["file_path"],
                                        "confidence": 0.8,
                                        "timestamp": datetime.utcnow().isoformat(),
                                    }
                                    relationships.append(relationship)
                                    break
                        except Exception as e:
                            logger.debug(
                                f"Error parsing import line '{import_line}': {e}"
                            )
                            continue

            return relationships

        except Exception as e:
            logger.debug(f"Error extracting relationships for {file_path}: {e}")
            return []

    async def _clone_repository(self, repository_url: str, branch: str) -> str:
        """
        Clone repository using GitPython with enterprise security patterns.

        Implements research-backed authentication and error handling for:
        - GitHub/GitLab repositories with token authentication
        - Private repository access with credentials
        - Network timeout and connection error handling
        - Secure temporary directory management with cleanup

        Args:
            repository_url: Repository URL (supports HTTPS, SSH, and modified formats)
            branch: Branch name to clone

        Returns:
            str: Path to cloned repository directory

        Raises:
            git.exc.GitCommandError: Git operation failures
            git.exc.InvalidGitRepositoryError: Invalid repository
            OSError: File system or network errors
            TimeoutError: Network timeout during clone
        """
        temp_dir = None
        try:
            # Create secure temporary directory with restricted permissions
            temp_dir = tempfile.mkdtemp(prefix="mosaic_repo_", suffix="_clone")
            os.chmod(temp_dir, 0o700)  # Owner-only access

            # Configure comprehensive Git authentication
            modified_url = repository_url
            git_env = {}

            # GitHub Token Authentication (GITHUB_TOKEN)
            if os.getenv("GITHUB_TOKEN"):
                if "github.com" in repository_url:
                    # Modify URL to include token for private repos
                    if repository_url.startswith("https://github.com/"):
                        modified_url = repository_url.replace(
                            "https://github.com/",
                            f"https://{os.getenv('GITHUB_TOKEN')}@github.com/",
                        )
                    git_env["GIT_HTTP_EXTRAHEADER"] = (
                        f"Authorization: token {os.getenv('GITHUB_TOKEN')}"
                    )

            # Generic Git Credentials (GIT_USERNAME, GIT_PASSWORD)
            elif os.getenv("GIT_USERNAME") and os.getenv("GIT_PASSWORD"):
                if repository_url.startswith("https://"):
                    # Extract domain and path for credential injection
                    url_parts = repository_url.replace("https://", "").split("/", 1)
                    if len(url_parts) == 2:
                        domain, path = url_parts
                        modified_url = f"https://{os.getenv('GIT_USERNAME')}:{os.getenv('GIT_PASSWORD')}@{domain}/{path}"

            # Legacy Git Token Support (GIT_TOKEN for Azure DevOps/GitLab)
            elif os.getenv("GIT_TOKEN"):
                git_env["GIT_HTTP_EXTRAHEADER"] = (
                    f"Authorization: Basic {os.getenv('GIT_TOKEN')}"
                )

            logger.info(
                f"Cloning repository {repository_url} branch '{branch}' to {temp_dir}"
            )

            # Execute clone with comprehensive error handling and timeout
            try:
                # Shallow clone for performance with timeout protection
                repo = git.Repo.clone_from(
                    modified_url,
                    temp_dir,
                    branch=branch,
                    depth=1,  # Shallow clone for performance
                    env=git_env if git_env else None,
                    timeout=300,  # 5-minute timeout for large repositories
                )

                # Verify clone success
                if not repo.heads:
                    raise git.exc.InvalidGitRepositoryError(
                        "No branches found in cloned repository"
                    )

                logger.info(
                    f"Successfully cloned repository to {temp_dir} (commit: {repo.head.commit.hexsha[:8]})"
                )
                return temp_dir

            except git.exc.GitCommandError as git_error:
                # Enhanced Git-specific error handling
                error_msg = str(git_error).lower()
                if (
                    "authentication failed" in error_msg
                    or "could not read username" in error_msg
                ):
                    raise git.exc.GitCommandError(
                        f"Authentication failed for {repository_url}. "
                        f"Verify GITHUB_TOKEN, GIT_USERNAME/GIT_PASSWORD, or GIT_TOKEN environment variables."
                    ) from git_error
                elif (
                    "repository not found" in error_msg or "does not exist" in error_msg
                ):
                    raise git.exc.GitCommandError(
                        f"Repository not found: {repository_url}. Verify URL and access permissions."
                    ) from git_error
                elif "network" in error_msg or "timeout" in error_msg:
                    raise TimeoutError(
                        f"Network timeout cloning {repository_url}. Check network connectivity."
                    ) from git_error
                else:
                    raise git.exc.GitCommandError(
                        f"Git clone failed for {repository_url}: {git_error}"
                    ) from git_error

            except Exception as unexpected_error:
                # Handle unexpected errors with context
                raise OSError(
                    f"Unexpected error cloning {repository_url}: {unexpected_error}"
                ) from unexpected_error

        except Exception as e:
            # Comprehensive cleanup on any failure
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.debug(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as cleanup_error:
                    logger.warning(
                        f"Failed to cleanup temporary directory {temp_dir}: {cleanup_error}"
                    )

            # Re-raise with enhanced error context
            logger.error(
                f"Repository clone failed for {repository_url} (branch: {branch}): {e}"
            )
            raise

    async def _first_pass_build_entity_map(
        self, repo_path: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        PASS 1: Build comprehensive entity map across all repository files.

        Creates a global lookup table of all classes, functions, and exports
        that can be imported or called by other files. This is the foundation
        for cross-file relationship extraction.

        Args:
            repo_path: Path to cloned repository

        Returns:
            Dict mapping entity names to their full details
            Format: {"entity_name": {"file_path": "...", "node_id": "...", ...}}
        """
        entity_map = {}

        try:
            repo_root = Path(repo_path)
            logger.info(f"PASS 1: Scanning {repo_path} for exportable entities")

            # Walk directory structure
            for file_path in repo_root.rglob("*"):
                if not file_path.is_file():
                    continue

                # Skip common non-code directories
                if any(
                    skip in file_path.parts
                    for skip in [
                        ".git",
                        "node_modules",
                        "__pycache__",
                        ".pytest_cache",
                        "target",
                        "dist",
                        "build",
                        ".venv",
                        "venv",
                    ]
                ):
                    continue

                # Determine language by file extension
                language = self._detect_language(file_path)
                if not language:
                    # AI FALLBACK: Use Semantic Kernel to analyze unknown file types
                    ai_entities = await self._ai_analyze_unknown_file(file_path)
                    if ai_entities:
                        for entity in ai_entities:
                            # Create multiple lookup keys for AI-analyzed entities
                            entity_names = self._generate_entity_lookup_keys(
                                entity, file_path
                            )
                            for name in entity_names:
                                entity_map[name] = entity
                    continue

                # Parse file and extract exportable entities using tree-sitter
                file_entities = await self._extract_exportable_entities(
                    file_path, language
                )

                # Add to global entity map with qualified names
                for entity in file_entities:
                    # Create multiple lookup keys for flexibility
                    entity_names = self._generate_entity_lookup_keys(entity, file_path)

                    for name in entity_names:
                        if name in entity_map:
                            # Handle naming conflicts by preferring public exports
                            existing = entity_map[name]
                            if self._should_prefer_entity(entity, existing):
                                entity_map[name] = entity
                        else:
                            entity_map[name] = entity

            logger.info(
                f"PASS 1 COMPLETE: Built entity map with {len(entity_map)} exportable entities"
            )
            return entity_map

        except Exception as e:
            logger.error(f"PASS 1 FAILED: Error building entity map: {e}")
            raise

    async def _extract_exportable_entities(
        self, file_path: Path, language: str
    ) -> list[Dict[str, Any]]:
        """Extract entities that can be imported/called by other files."""
        exportable_entities = []

        try:
            # Read and parse file content
            content = self._read_file_content(file_path)
            if not content:
                return exportable_entities

            parser = self.parsers.get(language)
            if not parser:
                return exportable_entities

            tree = parser.parse(content)
            if tree.root_node.has_error:
                logger.debug(
                    f"Parsing errors in {file_path}, proceeding with partial tree"
                )

            # Extract exportable entity types per language
            exportable_types = {
                "python": [
                    "function_definition",
                    "async_function_definition",
                    "class_definition",
                    "assignment",  # For module-level variables
                ],
                "javascript": [
                    "function_declaration",
                    "class_declaration",
                    "export_statement",
                    "variable_declaration",
                    "const_declaration",
                    "let_declaration",
                ],
                "typescript": [
                    "function_declaration",
                    "class_declaration",
                    "interface_declaration",
                    "type_alias_declaration",
                    "export_statement",
                    "variable_declaration",
                ],
                "java": [
                    "class_declaration",
                    "interface_declaration",
                    "method_declaration",
                    "constructor_declaration",
                    "enum_declaration",
                ],
                "go": [
                    "function_declaration",
                    "type_declaration",
                    "var_declaration",
                    "const_declaration",
                    "interface_declaration",
                ],
                "rust": [
                    "function_item",
                    "struct_item",
                    "trait_item",
                    "impl_item",
                    "enum_item",
                    "const_item",
                    "static_item",
                    "mod_item",
                ],
                "c": ["function_definition", "struct_specifier", "typedef_definition"],
                "cpp": [
                    "function_definition",
                    "class_specifier",
                    "struct_specifier",
                    "namespace_definition",
                    "template_declaration",
                ],
                "csharp": [
                    "class_declaration",
                    "interface_declaration",
                    "method_declaration",
                    "property_declaration",
                    "constructor_declaration",
                    "namespace_declaration",
                ],
            }

            target_types = exportable_types.get(language, [])

            async def extract_exportable_node(node):
                if node.type in target_types:
                    try:
                        entity_content = content[
                            node.start_byte : node.end_byte
                        ].decode("utf-8", errors="replace")

                        # Skip very large entities
                        if len(entity_content) > 10000:
                            return

                        # Calculate proper relative path from repository root
                        relative_path = self._calculate_relative_path(
                            file_path,
                            str(file_path).split("src")[0]
                            if "src" in str(file_path)
                            else str(Path(file_path).parents[3]),
                        )

                        entity = {
                            "id": f"{file_path.name}_{node.start_point[0]}_{hash(entity_content[:100])}",
                            "name": await self._ai_extract_entity_name(
                                node, language, content, entity_content
                            ),
                            "type": "exportable_entity",
                            "entity_type": node.type,
                            "language": language,
                            "file_path": str(file_path),
                            "relative_path": relative_path,
                            "file_name": file_path.name,
                            "directory": str(file_path.parent),
                            "file_extension": file_path.suffix,
                            "content": entity_content,
                            "start_line": node.start_point[0] + 1,
                            "end_line": node.end_point[0] + 1,
                            "line_count": entity_content.count("\n") + 1,
                            "char_count": len(entity_content),
                            # AI-powered visibility analysis (replaces _is_public_entity and _is_exported_entity)
                            "visibility_analysis": await self._ai_analyze_entity_visibility(
                                node, language, content, entity_content
                            ),
                            "timestamp": datetime.utcnow().isoformat(),
                            # File metadata for better navigation and classification
                            "file_metadata": {
                                "size_bytes": len(content),
                                "encoding": "utf-8",
                                "detected_language": language,
                                "is_test_file": self._is_test_file(file_path),
                                "is_config_file": self._is_config_file(file_path),
                                "is_documentation": self._is_documentation_file(
                                    file_path
                                ),
                            },
                            # AI-powered comprehensive file classification for OmniRAG filtering
                            "classification": await self._ai_classify_file(
                                file_path, content
                            ),
                            "tags": await self._ai_generate_tags(
                                file_path, content, entity_content
                            ),
                        }

                        if entity["name"]:  # Only add if we can extract a name
                            exportable_entities.append(entity)

                    except Exception as e:
                        logger.debug(
                            f"Error extracting entity at {file_path}:{node.start_point[0]}: {e}"
                        )

                # Recursively process children
                for child in node.children:
                    await extract_exportable_node(child)

            await extract_exportable_node(tree.root_node)
            logger.debug(
                f"Extracted {len(exportable_entities)} exportable entities from {file_path}"
            )
            return exportable_entities

        except Exception as e:
            logger.warning(
                f"Error extracting exportable entities from {file_path}: {e}"
            )
            return exportable_entities

    # REMOVED: _extract_entity_name() - replaced with AI-powered _ai_extract_entity_name()

    async def _ai_analyze_entity_visibility(
        self, node, language: str, content: bytes, entity_content: str
    ) -> dict:
        """
        AI-powered comprehensive entity analysis replacing 50+ lines of deterministic logic.

        This single AI function replaces:
        - _is_public_entity() (17 lines of if/else)
        - _is_exported_entity() (18 lines of if/else)
        - Complex language-specific visibility rules

        Returns comprehensive visibility analysis with confidence scores.
        """
        try:
            # AI prompt to analyze entity visibility and export status
            visibility_prompt = """
            Analyze this {{$language}} code entity for visibility and export status.
            
            CODE ENTITY:
            ```{{$language}}
            {{$entity_content}}
            ```
            
            Determine:
            1. Is this entity publicly accessible from outside its module/class?
            2. Is this entity explicitly exported (can be imported by other files)?
            3. What visibility modifiers are used?
            
            Consider language-specific rules:
            - Python: underscores indicate private, __all__ affects exports
            - Java/C#: public/private/protected keywords
            - JavaScript/TypeScript: export statements, private fields
            - Rust: pub keyword, module visibility
            - Go: capitalization for public/private
            
            Return JSON:
            {
              "is_public": true/false,
              "is_exported": true/false,
              "visibility_modifier": "public|private|protected|internal|null",
              "confidence": 0.95
            }
            """

            # Create AI visibility analysis function
            visibility_function = self.kernel.add_function(
                function_name="analyze_entity_visibility",
                plugin_name="code_analysis",
                prompt_template_config=self._create_visibility_analysis_prompt_config(
                    visibility_prompt
                ),
            )

            # Invoke AI analysis
            result = await self.kernel.invoke(
                visibility_function,
                language=language,
                entity_content=entity_content[:300],  # Focus on declaration
            )

            # Parse AI response
            analysis = self._parse_ai_visibility_analysis(str(result))
            return analysis

        except Exception as e:
            logger.debug(f"AI visibility analysis failed: {e}")

            # Fallback to simple heuristics
            entity_content_str = (
                entity_content
                if isinstance(entity_content, str)
                else content[node.start_byte : node.end_byte].decode(
                    "utf-8", errors="replace"
                )
            )
            return {
                "is_public": not entity_content_str.strip().startswith("_"),
                "is_exported": not entity_content_str.strip().startswith("_"),
                "visibility_modifier": None,
                "confidence": 0.3,
                "ai_powered": False,
                "fallback_reason": str(e),
            }

    def _generate_entity_lookup_keys(
        self, entity: Dict[str, Any], file_path: Path
    ) -> list[str]:
        """Generate multiple lookup keys for entity resolution."""
        keys = []
        name = entity.get("name", "")

        if not name or name == "unknown":
            return keys

        # Add basic name
        keys.append(name)

        # Add module.name format for Python
        if entity["language"] == "python":
            module_name = file_path.stem
            keys.append(f"{module_name}.{name}")

            # Add package.module.name if in a package
            if len(file_path.parts) > 1:
                package_parts = file_path.parts[-2:]  # Last 2 parts
                package_name = (
                    package_parts[0] if len(package_parts) > 1 else module_name
                )
                keys.append(f"{package_name}.{name}")

        # Add file_name.entity format for other languages
        elif entity["language"] in ["javascript", "typescript", "java", "csharp"]:
            file_name = file_path.stem
            keys.append(f"{file_name}.{name}")

        # Add qualified names for classes and methods
        if entity["entity_type"] in ["class_definition", "class_declaration"]:
            keys.append(f"class_{name}")
        elif entity["entity_type"] in [
            "function_definition",
            "function_declaration",
            "method_declaration",
        ]:
            keys.append(f"function_{name}")

        return keys

    def _should_prefer_entity(
        self, new_entity: Dict[str, Any], existing_entity: Dict[str, Any]
    ) -> bool:
        """Determine which entity to prefer when there are naming conflicts."""
        # Prefer exported entities over non-exported
        if new_entity.get("is_exported", False) and not existing_entity.get(
            "is_exported", False
        ):
            return True
        if existing_entity.get("is_exported", False) and not new_entity.get(
            "is_exported", False
        ):
            return False

        # Prefer public entities over private
        if new_entity.get("is_public", False) and not existing_entity.get(
            "is_public", False
        ):
            return True
        if existing_entity.get("is_public", False) and not new_entity.get(
            "is_public", False
        ):
            return False

        # Prefer newer entities (by default)
        return True

    def _calculate_relative_path(self, file_path: Path, repo_root: str) -> str:
        """Calculate proper relative path from repository root."""
        try:
            repo_root_path = Path(repo_root)
            if file_path.is_relative_to(repo_root_path):
                return str(file_path.relative_to(repo_root_path))
            else:
                # Fallback to filename if we can't determine relative path
                return file_path.name
        except Exception:
            return file_path.name

    def _is_test_file(self, file_path: Path) -> bool:
        """Determine if file is a test file."""
        path_str = str(file_path).lower()
        name = file_path.name.lower()

        return (
            "test" in path_str
            or name.startswith("test_")
            or name.endswith("_test.py")
            or name.endswith(".test.js")
            or name.endswith(".spec.js")
            or name.endswith(".spec.ts")
            or "spec" in name
            or "__tests__" in path_str
        )

    def _is_config_file(self, file_path: Path) -> bool:
        """Determine if file is a configuration file."""
        name = file_path.name.lower()

        config_patterns = [
            "config",
            "settings",
            "environment",
            "env",
            ".env",
            "docker",
            "makefile",
            "requirements",
            "package.json",
            "tsconfig",
            "webpack",
            "babel",
            "eslint",
            "prettier",
            "pyproject.toml",
            "setup.py",
            "cargo.toml",
            "go.mod",
            "pom.xml",
            "build.gradle",
        ]

        return any(pattern in name for pattern in config_patterns)

    def _is_documentation_file(self, file_path: Path) -> bool:
        """Determine if file is documentation."""
        name = file_path.name.lower()
        extension = file_path.suffix.lower()

        return (
            extension in [".md", ".rst", ".txt", ".adoc"]
            or name in ["readme", "changelog", "license", "contributing", "authors"]
            or "docs" in str(file_path).lower()
        )

    async def _ai_analyze_unknown_file(self, file_path: Path) -> list[Dict[str, Any]]:
        """
        AI FALLBACK: Use Semantic Kernel to analyze file types not supported by tree-sitter.

        This handles files like:
        - Configuration files (YAML, TOML, INI, XML)
        - Documentation (Markdown, reStructuredText)
        - Data files (JSON, CSV)
        - Scripts (Shell, PowerShell, Batch)
        - Templates (Jinja2, Handlebars)
        - And any other file types the AI can understand
        """
        try:
            # Skip binary files and very large files
            if (
                self._is_binary_file(file_path)
                or file_path.stat().st_size > 1024 * 1024
            ):
                return []

            # Read file content
            content = self._read_file_content(file_path)
            if not content:
                return []

            content_str = content.decode("utf-8", errors="replace")

            # Skip empty or very short files
            if len(content_str.strip()) < 20:
                return []

            # Use AI to analyze the file
            ai_analysis = await self._call_ai_file_analyzer(file_path, content_str)

            if ai_analysis:
                # Convert AI analysis to entity format
                entity = {
                    "id": f"ai_{file_path.name}_{hash(content_str[:100])}",
                    "name": file_path.stem,
                    "type": "ai_analyzed_file",
                    "entity_type": ai_analysis.get("file_type", "unknown"),
                    "language": ai_analysis.get("language", "unknown"),
                    "file_path": str(file_path),
                    "relative_path": self._calculate_relative_path(
                        file_path, str(file_path.parents[3])
                    ),
                    "file_name": file_path.name,
                    "directory": str(file_path.parent),
                    "file_extension": file_path.suffix,
                    "content": content_str[:5000],  # Limit content for storage
                    "start_line": 1,
                    "end_line": content_str.count("\n") + 1,
                    "line_count": content_str.count("\n") + 1,
                    "char_count": len(content_str),
                    "is_public": True,  # AI-analyzed files are considered public
                    "is_exported": False,
                    "timestamp": datetime.utcnow().isoformat(),
                    # AI-specific fields
                    "ai_file_type": ai_analysis.get("file_type", "unknown"),
                    "ai_purpose": ai_analysis.get("purpose", ""),
                    "ai_key_elements": ai_analysis.get("key_elements", []),
                    "ai_technology": ai_analysis.get("technology", ""),
                    "analyzed_by_ai": True,
                    "file_metadata": {
                        "size_bytes": len(content),
                        "encoding": "utf-8",
                        "detected_language": ai_analysis.get("language", "unknown"),
                        "is_test_file": self._is_test_file(file_path),
                        "is_config_file": self._is_config_file(file_path),
                        "is_documentation": self._is_documentation_file(file_path),
                        "ai_confidence": ai_analysis.get("confidence", 0.7),
                    },
                }

                return [entity]

            return []

        except Exception as e:
            logger.debug(f"AI analysis failed for {file_path}: {e}")
            return []

    async def _call_ai_file_analyzer(
        self, file_path: Path, content: str
    ) -> Optional[Dict[str, Any]]:
        """Call AI to analyze unknown file types."""
        try:
            # Create specialized prompt for file analysis
            file_analysis_prompt = f"""
You are a senior software architect analyzing a file. Analyze this file and provide structured information about it.

File: {file_path.name}
Extension: {file_path.suffix}
Content (first 2000 chars):
```
{content[:2000]}
```

Please provide analysis in exactly this format:

FILE_TYPE: [The type of file - e.g., configuration, documentation, script, data, template, etc.]
LANGUAGE: [The language/format - e.g., yaml, markdown, json, shell, xml, etc.]
PURPOSE: [One sentence describing what this file does or contains]
KEY_ELEMENTS: [List 3-5 key components, separated by commas]
TECHNOLOGY: [Related technology/framework if applicable, or 'general' if not specific]
CONFIDENCE: [Your confidence in this analysis from 0.1 to 1.0]

Guidelines:
- FILE_TYPE: Choose from: configuration, documentation, script, data, template, test, build, deployment, other
- LANGUAGE: Use the actual language/format name (yaml, json, markdown, shell, etc.)
- PURPOSE: Be specific about what this file accomplishes
- KEY_ELEMENTS: Extract the most important components or sections
- TECHNOLOGY: Name the specific framework/tool this relates to
"""

            # Prepare arguments for Semantic Kernel
            arguments = KernelArguments(
                file_name=file_path.name,
                file_extension=file_path.suffix,
                content=content[:2000],  # Limit content to avoid token limits
            )

            # Call the AI file analyzer
            result = await self.kernel.invoke(
                plugin_name="CodeAnalysis",
                function_name="analyze_code_entity",  # Reuse existing function
                arguments=arguments,
            )

            # Parse the AI response
            analysis_text = str(result).strip()
            return self._parse_ai_file_analysis(analysis_text)

        except Exception as e:
            logger.error(f"AI file analysis call failed: {e}")
            return None

    def _parse_ai_file_analysis(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse AI file analysis response into structured format."""
        try:
            analysis = {
                "file_type": "unknown",
                "language": "unknown",
                "purpose": "",
                "key_elements": [],
                "technology": "general",
                "confidence": 0.7,
            }

            lines = response_text.split("\n")

            for line in lines:
                line = line.strip()

                if line.startswith("FILE_TYPE:"):
                    file_type = line[10:].strip().lower()
                    if file_type:
                        analysis["file_type"] = file_type

                elif line.startswith("LANGUAGE:"):
                    language = line[9:].strip().lower()
                    if language:
                        analysis["language"] = language

                elif line.startswith("PURPOSE:"):
                    purpose = line[8:].strip()
                    if purpose and len(purpose) > 5:
                        analysis["purpose"] = purpose[:200]  # Limit length

                elif line.startswith("KEY_ELEMENTS:"):
                    elements_text = line[13:].strip()
                    if elements_text and elements_text.lower() != "none":
                        elements = [elem.strip() for elem in elements_text.split(",")]
                        analysis["key_elements"] = [
                            e for e in elements if e and len(e) > 1
                        ][:8]

                elif line.startswith("TECHNOLOGY:"):
                    technology = line[11:].strip()
                    if technology and len(technology) > 1:
                        analysis["technology"] = technology[:50]

                elif line.startswith("CONFIDENCE:"):
                    confidence_text = line[11:].strip()
                    try:
                        confidence = float(confidence_text)
                        if 0.0 <= confidence <= 1.0:
                            analysis["confidence"] = confidence
                    except (ValueError, TypeError):
                        pass

            # Validate we got meaningful analysis
            if analysis["file_type"] != "unknown" and analysis["purpose"]:
                return analysis
            else:
                return None

        except Exception as e:
            logger.debug(f"Error parsing AI file analysis response: {e}")
            return None

    async def _second_pass_extract_relationships(
        self, repo_path: str, entity_map: Dict[str, Dict[str, Any]]
    ) -> list[Dict[str, Any]]:
        """
        PASS 2: Extract cross-file relationships using the entity map.

        This is where we solve the disconnected islands problem by finding
        imports, function calls, and other dependencies that connect entities
        across different files in the repository.

        Args:
            repo_path: Path to cloned repository
            entity_map: Global entity map from Pass 1

        Returns:
            List of relationship mappings connecting entities
        """
        relationships = []

        try:
            repo_root = Path(repo_path)
            logger.info(f"PASS 2: Extracting cross-file relationships from {repo_path}")

            # Walk through all files again to find imports and calls
            for file_path in repo_root.rglob("*"):
                if not file_path.is_file():
                    continue

                # Skip non-code directories
                if any(
                    skip in file_path.parts
                    for skip in [
                        ".git",
                        "node_modules",
                        "__pycache__",
                        ".pytest_cache",
                        "target",
                        "dist",
                        "build",
                        ".venv",
                        "venv",
                    ]
                ):
                    continue

                language = self._detect_language(file_path)
                if not language:
                    continue

                # Extract relationships from this file
                file_relationships = await self._extract_file_relationships(
                    file_path, language, entity_map
                )
                relationships.extend(file_relationships)

            logger.info(
                f"PASS 2 COMPLETE: Extracted {len(relationships)} cross-file relationships"
            )
            return relationships

        except Exception as e:
            logger.error(f"PASS 2 FAILED: Error extracting relationships: {e}")
            raise

    async def _extract_file_relationships(
        self, file_path: Path, language: str, entity_map: Dict[str, Dict[str, Any]]
    ) -> list[Dict[str, Any]]:
        """Extract relationships from a single file using the entity map."""
        relationships = []

        try:
            content = self._read_file_content(file_path)
            if not content:
                return relationships

            parser = self.parsers.get(language)
            if not parser:
                return relationships

            tree = parser.parse(content)
            if tree.root_node.has_error:
                logger.debug(
                    f"Parsing errors in {file_path}, proceeding with partial tree"
                )

            # Find import statements and function calls
            import_relationships = await self._extract_import_relationships(
                tree.root_node, file_path, language, content, entity_map
            )
            call_relationships = await self._extract_call_relationships(
                tree.root_node, file_path, language, content, entity_map
            )

            relationships.extend(import_relationships)
            relationships.extend(call_relationships)

            logger.debug(
                f"Extracted {len(relationships)} relationships from {file_path}"
            )
            return relationships

        except Exception as e:
            logger.warning(f"Error extracting relationships from {file_path}: {e}")
            return relationships

    async def _extract_import_relationships(
        self,
        root_node,
        file_path: Path,
        language: str,
        content: bytes,
        entity_map: Dict[str, Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        """Extract import relationships by matching import statements to entity map."""
        import_relationships = []

        # Language-specific import node types
        import_types = {
            "python": ["import_statement", "import_from_statement"],
            "javascript": ["import_statement", "require_call"],
            "typescript": ["import_statement", "require_call"],
            "java": ["import_declaration"],
            "go": ["import_declaration"],
            "rust": ["use_declaration"],
            "c": ["preproc_include"],
            "cpp": ["preproc_include"],
            "csharp": ["using_directive"],
        }

        target_types = import_types.get(language, [])

        async def find_imports(node):
            if node.type in target_types:
                try:
                    import_text = content[node.start_byte : node.end_byte].decode(
                        "utf-8", errors="replace"
                    )
                    imported_names = await self._ai_parse_import_statement(
                        import_text, language
                    )

                    for imported_name in imported_names:
                        # Look up in entity map
                        target_entity = self._resolve_import_target(
                            imported_name, entity_map, language
                        )

                        if target_entity:
                            # Create relationship from this file to the imported entity
                            relationship = {
                                "id": f"{file_path.name}_imports_{target_entity['id']}",
                                "type": "cross_file_dependency",
                                "relationship_type": "imports",
                                "source_file": str(file_path),
                                "target_file": target_entity["file_path"],
                                "source_entity_name": file_path.stem,
                                "target_entity_name": imported_name,
                                "target_entity_id": target_entity["id"],
                                "import_statement": import_text.strip(),
                                "confidence": 0.9,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                            import_relationships.append(relationship)

                except Exception as e:
                    logger.debug(
                        f"Error processing import at {file_path}:{node.start_point[0]}: {e}"
                    )

            # Recursively search children
            for child in node.children:
                find_imports(child)

        find_imports(root_node)
        return import_relationships

    async def _ai_parse_import_statement(
        self, import_text: str, language: str
    ) -> list[str]:
        """
        AI-powered import statement parsing replacing 70+ lines of complex language-specific logic.

        This single AI function replaces deterministic parsing for:
        - Python: import/from statements with complex syntax
        - JavaScript/TypeScript: import/export with destructuring
        - Java: package.Class imports
        - Go: package imports with quotes
        - Rust: use statements with paths
        - C#: using namespace statements
        - And automatically handles new languages
        """
        try:
            # AI prompt for intelligent import parsing
            import_parse_prompt = """
            Extract all imported names from this {{$language}} import/use statement.
            
            STATEMENT: {{$import_text}}
            
            Extract names that become available in the current scope:
            - For aliased imports, return the alias name
            - For destructured imports, return individual names
            - For wildcard imports, return the module/package name
            - Include both short names and qualified names when useful
            
            Language examples:
            - Python "from os import path, environ as env" → ["path", "env"]
            - JS/TS "import { useState, useEffect } from 'react'" → ["useState", "useEffect"]
            - Java "import java.util.List;" → ["List", "java.util.List"]
            - Go "import \"fmt\"" → ["fmt"]
            - Rust "use std::collections::HashMap;" → ["HashMap", "std::collections::HashMap"]
            
            Return ONLY a JSON array: ["name1", "name2"]
            """

            # Create AI import parsing function
            import_function = self.kernel.add_function(
                function_name="parse_import_statement",
                plugin_name="code_analysis",
                prompt_template_config=self._create_import_parse_prompt_config(
                    import_parse_prompt
                ),
            )

            # Invoke AI parsing
            result = await self.kernel.invoke(
                import_function, language=language, import_text=import_text.strip()
            )

            # Parse AI response
            imported_names = self._parse_ai_import_result(str(result))
            return imported_names

        except Exception as e:
            logger.debug(f"AI import parsing failed for '{import_text}': {e}")

            # Simple fallback - extract identifiers
            import re

            words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", import_text)
            keywords = {
                "import",
                "from",
                "as",
                "export",
                "default",
                "const",
                "let",
                "var",
                "use",
                "using",
                "package",
            }
            return [word for word in words if word not in keywords][:5]

    def _resolve_import_target(
        self, imported_name: str, entity_map: Dict[str, Dict[str, Any]], language: str
    ) -> Optional[Dict[str, Any]]:
        """Resolve imported name to actual entity using the entity map."""

        # Direct lookup first
        if imported_name in entity_map:
            return entity_map[imported_name]

        # Try variations
        variations = [
            imported_name,
            f"class_{imported_name}",
            f"function_{imported_name}",
        ]

        # Add language-specific variations
        if language == "python":
            variations.extend(
                [
                    f"{imported_name}.{imported_name}",  # module.class pattern
                    imported_name.split(".")[-1],  # Last part of qualified name
                ]
            )
        elif language in ["javascript", "typescript"]:
            variations.extend(
                [
                    imported_name.replace("./", "").replace(
                        "../", ""
                    ),  # Remove relative paths
                ]
            )

        for variation in variations:
            if variation in entity_map:
                return entity_map[variation]

        return None

    async def _extract_call_relationships(
        self,
        root_node,
        file_path: Path,
        language: str,
        content: bytes,
        entity_map: Dict[str, Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        """Extract function call relationships by matching calls to entity map."""
        call_relationships = []

        # Language-specific call node types
        call_types = {
            "python": ["call", "attribute"],
            "javascript": ["call_expression", "member_expression"],
            "typescript": ["call_expression", "member_expression"],
            "java": ["method_invocation"],
            "go": ["call_expression"],
            "rust": ["call_expression"],
            "c": ["call_expression"],
            "cpp": ["call_expression"],
            "csharp": ["invocation_expression"],
        }

        target_types = call_types.get(language, [])

        def find_calls(node):
            if node.type in target_types:
                try:
                    call_text = content[node.start_byte : node.end_byte].decode(
                        "utf-8", errors="replace"
                    )
                    called_names = self._parse_function_call(call_text, language)

                    for called_name in called_names:
                        # Look up in entity map
                        target_entity = self._resolve_call_target(
                            called_name, entity_map, language
                        )

                        if target_entity and target_entity["file_path"] != str(
                            file_path
                        ):
                            # Create cross-file call relationship
                            relationship = {
                                "id": f"{file_path.name}_calls_{target_entity['id']}",
                                "type": "cross_file_dependency",
                                "relationship_type": "calls",
                                "source_file": str(file_path),
                                "target_file": target_entity["file_path"],
                                "source_entity_name": file_path.stem,
                                "target_entity_name": called_name,
                                "target_entity_id": target_entity["id"],
                                "call_statement": call_text.strip()[
                                    :100
                                ],  # First 100 chars
                                "confidence": 0.7,  # Lower confidence for calls
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                            call_relationships.append(relationship)

                except Exception as e:
                    logger.debug(
                        f"Error processing call at {file_path}:{node.start_point[0]}: {e}"
                    )

            # Recursively search children
            for child in node.children:
                find_calls(child)

        find_calls(root_node)
        return call_relationships

    def _parse_function_call(self, call_text: str, language: str) -> list[str]:
        """Parse function call to extract called function names."""
        called_names = []

        try:
            call_text = call_text.strip()

            # Extract function name from call patterns
            if language == "python":
                # Handle: function(), module.function(), obj.method()
                if "(" in call_text:
                    func_part = call_text.split("(")[0]
                    if "." in func_part:
                        # module.function or obj.method
                        parts = func_part.split(".")
                        called_names.append(parts[-1])  # function name
                        called_names.append(func_part)  # full qualified name
                    else:
                        called_names.append(func_part)

            elif language in ["javascript", "typescript", "java", "csharp"]:
                # Similar pattern: function(), object.method()
                if "(" in call_text:
                    func_part = call_text.split("(")[0]
                    if "." in func_part:
                        parts = func_part.split(".")
                        called_names.append(parts[-1])  # method name
                        called_names.append(func_part)  # full name
                    else:
                        called_names.append(func_part)

            # Additional parsing for other languages can be added here

        except Exception as e:
            logger.debug(f"Error parsing function call '{call_text}': {e}")

        return called_names

    def _resolve_call_target(
        self, called_name: str, entity_map: Dict[str, Dict[str, Any]], language: str
    ) -> Optional[Dict[str, Any]]:
        """Resolve called function name to actual entity using the entity map."""

        # Similar to import resolution but focused on functions
        variations = [
            called_name,
            f"function_{called_name}",
        ]

        # Add language-specific variations
        if language == "python":
            variations.extend(
                [
                    called_name.split(".")[-1],  # Last part if qualified
                ]
            )

        for variation in variations:
            if variation in entity_map:
                return entity_map[variation]

        return None

    async def _enhance_entities_with_ai_analyst(
        self, entities: list[Dict[str, Any]]
    ) -> list[Dict[str, Any]]:
        """
        AI ENHANCEMENT: Upgrade AI role from description writer to code analyst.

        The AI now provides comprehensive analysis including:
        - Concise, one-sentence summary of purpose
        - Primary dependencies (what it relies on)
        - Estimated complexity score (1-10)
        - Relevant keywords/tags for categorization
        - Enhanced text for richer embeddings

        Args:
            entities: List of entities from Pass 1

        Returns:
            List of entities enhanced with AI analyst insights
        """
        if not entities:
            return entities

        logger.info(
            f"AI ENHANCEMENT: Starting AI analyst enhancement for {len(entities)} entities"
        )

        # Filter entities that benefit from AI analyst treatment
        candidates = self._filter_entities_for_ai_analyst(entities)

        if not candidates:
            logger.info("No entities require AI analyst enhancement")
            return entities

        logger.info(f"Processing {len(candidates)} entities for AI analyst enhancement")

        # Process entities with enhanced AI analyst
        enhanced_lookup = await self._process_entities_with_ai_analyst(candidates)

        # Apply enhancements to original entities
        enhanced_entities = []
        for entity in entities:
            entity_id = entity.get("id")
            if entity_id in enhanced_lookup:
                enhanced_entity = entity.copy()
                enhanced_entity.update(enhanced_lookup[entity_id])
                enhanced_entities.append(enhanced_entity)
            else:
                enhanced_entities.append(entity)

        # Log AI analyst performance metrics
        total_enhanced = sum(
            1 for e in enhanced_entities if e.get("has_ai_analyst", False)
        )
        logger.info(
            f"AI ANALYST ENHANCEMENT completed: {total_enhanced} entities enhanced with comprehensive analysis"
        )

        return enhanced_entities

    def _filter_entities_for_ai_analyst(
        self, entities: list[Dict[str, Any]]
    ) -> list[Dict[str, Any]]:
        """Filter entities that should receive AI analyst treatment."""
        candidates = []

        # Entity types that benefit from comprehensive AI analysis
        analyst_worthy_types = {
            "function_definition",
            "async_function_definition",
            "function_declaration",
            "class_definition",
            "class_declaration",
            "interface_declaration",
            "method_declaration",
            "constructor_declaration",
            "struct_item",
            "impl_item",
            "trait_item",
            "enum_item",
            "namespace_definition",
            "type_alias_declaration",
        }

        for entity in entities:
            # Skip if already has AI analyst treatment
            if entity.get("has_ai_analyst", False):
                continue

            # Only process analyst-worthy entity types
            if entity["entity_type"] not in analyst_worthy_types:
                continue

            # Content should be substantial enough for analysis
            content = entity.get("content", "")
            if len(content) < 50 or len(content) > 8000:  # Reasonable limits
                continue

            candidates.append(entity)

        return candidates

    async def _process_entities_with_ai_analyst(
        self, entities: list[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Process entities with AI analyst using batch processing and caching."""
        enhanced_lookup = {}

        # Process in smaller batches for AI analyst (more complex analysis)
        batch_size = 10
        batches = [
            entities[i : i + batch_size] for i in range(0, len(entities), batch_size)
        ]

        logger.info(f"Processing {len(batches)} batches for AI analyst enhancement")

        # Process batches with concurrency control
        batch_tasks = []
        for i, batch in enumerate(batches):
            task = self._process_ai_analyst_batch(batch, batch_id=i)
            batch_tasks.append(task)

        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # Collect results
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.warning(f"AI analyst batch processing failed: {batch_result}")
                continue
            enhanced_lookup.update(batch_result)

        return enhanced_lookup

    async def _process_ai_analyst_batch(
        self, batch: list[Dict[str, Any]], batch_id: int
    ) -> Dict[str, Dict[str, Any]]:
        """Process a batch of entities with AI analyst."""
        async with self._processing_semaphore:
            logger.debug(
                f"Processing AI analyst batch {batch_id} with {len(batch)} entities"
            )
            batch_results = {}

            # Process entities in the batch
            tasks = []
            for entity in batch:
                task = self._enhance_entity_with_ai_analyst(entity)
                tasks.append(task)

            entity_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect successful results
            for entity, result in zip(batch, entity_results):
                if isinstance(result, Exception):
                    logger.warning(
                        f"Failed AI analyst enhancement for entity {entity.get('id', 'unknown')}: {result}"
                    )
                    continue

                if result:
                    batch_results[entity["id"]] = result

            logger.debug(
                f"AI analyst batch {batch_id} completed: {len(batch_results)} entities enhanced"
            )
            return batch_results

    async def _enhance_entity_with_ai_analyst(
        self, entity: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Enhance single entity with comprehensive AI analyst."""
        try:
            # Generate cache key
            content = entity.get("content", "")
            cache_key = f"analyst_{hashlib.sha256(content.encode('utf-8')).hexdigest()}"

            # Check analyst cache
            if cache_key in self._description_cache:
                self._cache_hits += 1
                cached_result = self._description_cache[cache_key]
                return {
                    "ai_summary": cached_result["summary"],
                    "ai_complexity": cached_result["complexity"],
                    "ai_dependencies": cached_result["dependencies"],
                    "ai_tags": cached_result["tags"],
                    "has_ai_analyst": True,
                    "ai_analyst_timestamp": datetime.utcnow().isoformat(),
                    "ai_model_used": "gpt-4.1-mini",
                    "analysis_confidence": cached_result.get("confidence", 0.85),
                    "cache_hit": True,
                }

            # Generate new AI analyst insights
            self._cache_misses += 1
            analysis = await self._generate_ai_analyst_insights(entity)

            if analysis:
                # Cache successful result
                self._description_cache[cache_key] = {
                    "summary": analysis["summary"],
                    "complexity": analysis["complexity"],
                    "dependencies": analysis["dependencies"],
                    "tags": analysis["tags"],
                    "confidence": 0.85,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                return {
                    "ai_summary": analysis["summary"],
                    "ai_complexity": analysis["complexity"],
                    "ai_dependencies": analysis["dependencies"],
                    "ai_tags": analysis["tags"],
                    "has_ai_analyst": True,
                    "ai_analyst_timestamp": datetime.utcnow().isoformat(),
                    "ai_model_used": "gpt-4.1-mini",
                    "analysis_confidence": 0.85,
                    "cache_hit": False,
                }

            return None

        except Exception as e:
            logger.warning(
                f"Failed AI analyst enhancement for entity {entity.get('id', 'unknown')}: {e}"
            )
            return None

    async def _generate_ai_analyst_insights(
        self, entity: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate comprehensive AI analyst insights with retry logic."""
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                self._api_calls += 1
                analysis = await self._call_ai_code_analyst(entity)

                if analysis:
                    return analysis

                logger.debug(
                    f"AI analyst returned empty analysis for entity {entity.get('id', 'unknown')}"
                )
                return None

            except Exception as e:
                if attempt == max_retries:
                    logger.error(
                        f"AI analyst failed after {max_retries + 1} attempts: {e}"
                    )
                    return None

                # Exponential backoff with jitter
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                logger.debug(
                    f"AI analyst call failed (attempt {attempt + 1}), retrying in {delay:.2f}s: {e}"
                )
                await asyncio.sleep(delay)

        return None

    async def _call_ai_code_analyst(
        self, entity: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Call AI to perform comprehensive code analysis."""
        try:
            # Create enhanced analyst prompt
            analyst_prompt = self._create_ai_analyst_prompt(entity)

            # Prepare arguments for Semantic Kernel
            arguments = KernelArguments(
                entity_type=entity["entity_type"],
                name=entity.get("name", "unknown"),
                language=entity["language"],
                content=entity["content"][:3000],  # More content for analyst
                file_path=entity.get("file_path", ""),
            )

            # Call enhanced AI analyst function
            result = await self.kernel.invoke(
                plugin_name="CodeAnalysis",
                function_name="analyze_code_entity",
                arguments=arguments,
            )

            # Parse the comprehensive analysis
            analysis_text = str(result).strip()
            return self._parse_ai_analyst_response(analysis_text)

        except Exception as e:
            logger.error(f"AI analyst call failed: {e}")
            raise

    def _create_ai_analyst_prompt(self, entity: Dict[str, Any]) -> str:
        """Create comprehensive AI analyst prompt."""
        entity_type = entity["entity_type"]

        # Enhanced analyst prompt template
        analyst_prompt = f"""
You are a senior software architect analyzing code. Provide a comprehensive analysis of this {entity_type}.

Code Entity: {entity.get("name", "unknown")}
Language: {entity["language"]}
File: {entity.get("file_path", "")}

Code:
```{entity["language"]}
{entity["content"][:2000]}
```

Please provide a structured analysis in exactly this format:

SUMMARY: [One clear sentence describing the primary purpose and functionality]
COMPLEXITY: [Score from 1-10, where 1=trivial, 5=moderate, 10=very complex]
DEPENDENCIES: [List key dependencies this code relies on, separated by commas]
TAGS: [Relevant keywords/categories for this code, separated by commas]

Guidelines:
- SUMMARY: Focus on what this code accomplishes, not how
- COMPLEXITY: Consider algorithm complexity, number of responsibilities, and maintainability
- DEPENDENCIES: Include libraries, frameworks, other modules, external systems
- TAGS: Include domain concepts, patterns, technologies, functional areas

Be concise but accurate. Each section should be on its own line with the exact label format shown.
"""
        return analyst_prompt

    def _parse_ai_analyst_response(
        self, response_text: str
    ) -> Optional[Dict[str, Any]]:
        """Parse AI analyst response into structured format."""
        try:
            analysis = {
                "summary": "",
                "complexity": 5,  # Default moderate complexity
                "dependencies": [],
                "tags": [],
            }

            lines = response_text.split("\n")

            for line in lines:
                line = line.strip()

                if line.startswith("SUMMARY:"):
                    summary = line[8:].strip()
                    if len(summary) > 10:
                        analysis["summary"] = summary[:200]  # Limit length

                elif line.startswith("COMPLEXITY:"):
                    complexity_text = line[11:].strip()
                    try:
                        # Extract numeric score
                        complexity_score = int(
                            "".join(filter(str.isdigit, complexity_text))
                        )
                        if 1 <= complexity_score <= 10:
                            analysis["complexity"] = complexity_score
                    except (ValueError, TypeError):
                        pass  # Keep default

                elif line.startswith("DEPENDENCIES:"):
                    deps_text = line[13:].strip()
                    if deps_text and deps_text.lower() != "none":
                        deps = [dep.strip() for dep in deps_text.split(",")]
                        analysis["dependencies"] = [
                            d for d in deps if d and len(d) > 1
                        ][:10]  # Limit to 10

                elif line.startswith("TAGS:"):
                    tags_text = line[5:].strip()
                    if tags_text and tags_text.lower() != "none":
                        tags = [tag.strip() for tag in tags_text.split(",")]
                        analysis["tags"] = [t for t in tags if t and len(t) > 1][
                            :15
                        ]  # Limit to 15

            # Validate we got meaningful analysis
            if analysis["summary"] and len(analysis["summary"]) > 10:
                return analysis
            else:
                return None

        except Exception as e:
            logger.debug(f"Error parsing AI analyst response: {e}")
            return None

    async def _parse_repository(self, repo_path: str) -> list[Dict[str, Any]]:
        """
        Parse repository files using tree-sitter multi-language AST parsing.

        Based on Context7 research: Supports 11 languages with fallback strategies
        for unsupported or malformed files.

        Args:
            repo_path: Path to cloned repository

        Returns:
            List of extracted code entities
        """
        entities = []

        try:
            repo_root = Path(repo_path)
            logger.info(f"Parsing repository at {repo_path}")

            # Walk directory structure
            for file_path in repo_root.rglob("*"):
                if not file_path.is_file():
                    continue

                # Skip common non-code directories
                if any(
                    skip in file_path.parts
                    for skip in [
                        ".git",
                        "node_modules",
                        "__pycache__",
                        ".pytest_cache",
                        "target",
                        "dist",
                    ]
                ):
                    continue

                # Determine language by file extension
                language = self._detect_language(file_path)
                if not language:
                    continue

                # Parse file with appropriate tree-sitter parser
                file_entities = await self._parse_file(file_path, language)

                # Generate AI descriptions for significant entities
                enhanced_entities = await self._enhance_entities_with_ai_descriptions(
                    file_entities
                )
                entities.extend(enhanced_entities)

            logger.info(f"Extracted {len(entities)} entities from {repo_path}")
            return entities

        except Exception as e:
            logger.error(f"Failed to parse repository {repo_path}: {e}")
            raise

    def _detect_language(self, file_path: Path) -> Optional[str]:
        """Detect programming language by file extension."""
        extension = file_path.suffix.lower()

        for language, extensions in self.language_extensions.items():
            if extension in extensions:
                return language

        return None

    async def _parse_file(self, file_path: Path, language: str) -> list[Dict[str, Any]]:
        """Parse individual file using tree-sitter with comprehensive error handling."""
        entities = []

        try:
            # Validate file before processing
            if not file_path.exists():
                logger.warning(f"File does not exist: {file_path}")
                return entities

            # Check file size (skip very large files > 1MB)
            file_size = file_path.stat().st_size
            if file_size > 1024 * 1024:  # 1MB limit
                logger.debug(f"Skipping large file ({file_size} bytes): {file_path}")
                return entities

            # Check if file is binary
            if self._is_binary_file(file_path):
                logger.debug(f"Skipping binary file: {file_path}")
                return entities

            # Read file content with encoding detection
            content = self._read_file_content(file_path)
            if not content:
                logger.debug(f"Empty or unreadable file: {file_path}")
                return entities

            # Get appropriate parser
            parser = self.parsers.get(language)
            if not parser:
                logger.warning(f"No parser available for language: {language}")
                return entities

            # Parse with timeout protection
            try:
                tree = parser.parse(content)

                # Check for parsing errors
                if tree.root_node.has_error:
                    logger.debug(
                        f"Parsing errors detected in {file_path}, proceeding with partial tree"
                    )

                # Extract entities based on language
                entities = self._extract_entities_from_tree(
                    tree.root_node, file_path, language, content
                )

                logger.debug(f"Extracted {len(entities)} entities from {file_path}")
                return entities

            except Exception as parse_error:
                logger.warning(
                    f"Tree-sitter parsing failed for {file_path}: {parse_error}"
                )
                return entities

        except PermissionError:
            logger.warning(f"Permission denied reading file: {file_path}")
            return entities
        except UnicodeDecodeError as decode_error:
            logger.warning(f"Encoding error in file {file_path}: {decode_error}")
            return entities
        except OSError as os_error:
            logger.warning(f"OS error reading file {file_path}: {os_error}")
            return entities
        except Exception as e:
            logger.warning(f"Unexpected error parsing file {file_path}: {e}")
            return entities

    def _is_binary_file(self, file_path: Path) -> bool:
        """Check if file is binary by reading first 1024 bytes."""
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(1024)
                if b"\0" in chunk:  # Null bytes indicate binary
                    return True
                # Check for high ratio of non-text characters
                text_chars = sum(
                    1 for byte in chunk if 32 <= byte <= 126 or byte in (9, 10, 13)
                )
                return len(chunk) > 0 and (text_chars / len(chunk)) < 0.75
        except Exception:
            return True  # Assume binary if we can't read it

    def _read_file_content(self, file_path: Path) -> Optional[bytes]:
        """Read file content with encoding detection and error handling."""
        try:
            # Try reading as UTF-8 first
            with open(file_path, "rb") as f:
                content = f.read()

            # Validate it's valid UTF-8 or contains mostly text
            try:
                content.decode("utf-8")
                return content
            except UnicodeDecodeError:
                # Try other common encodings
                for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                    try:
                        decoded = content.decode(encoding)
                        return decoded.encode("utf-8", errors="replace")
                    except UnicodeDecodeError:
                        continue

                # Last resort: replace invalid characters
                return content.decode("utf-8", errors="replace").encode("utf-8")

        except Exception as e:
            logger.debug(f"Failed to read content from {file_path}: {e}")
            return None

    def _extract_entities_from_tree(
        self, node, file_path: Path, language: str, content: bytes
    ) -> list[Dict[str, Any]]:
        """Extract code entities from parsed tree."""
        entities = []

        # Define comprehensive entity types per language (Context7 + research validated)
        entity_types = {
            "python": [
                "function_definition",
                "async_function_definition",
                "class_definition",
                "method_definition",
                "import_statement",
                "import_from_statement",
                "decorated_definition",
            ],
            "javascript": [
                "function_declaration",
                "function_expression",
                "arrow_function",
                "class_declaration",
                "method_definition",
                "import_statement",
                "export_statement",
                "variable_declaration",
            ],
            "typescript": [
                "function_declaration",
                "function_expression",
                "arrow_function",
                "class_declaration",
                "method_definition",
                "interface_declaration",
                "type_alias_declaration",
                "import_statement",
                "export_statement",
            ],
            "java": [
                "method_declaration",
                "class_declaration",
                "interface_declaration",
                "constructor_declaration",
                "import_declaration",
                "package_declaration",
                "enum_declaration",
            ],
            "go": [
                "function_declaration",
                "method_declaration",
                "type_declaration",
                "import_declaration",
                "interface_declaration",
                "var_declaration",
                "const_declaration",
            ],
            "rust": [
                "function_item",
                "struct_item",
                "impl_item",
                "trait_item",
                "use_declaration",
                "mod_item",
                "enum_item",
                "const_item",
            ],
            "c": [
                "function_definition",
                "struct_specifier",
                "union_specifier",
                "preproc_include",
                "preproc_define",
                "typedef_definition",
            ],
            "cpp": [
                "function_definition",
                "class_specifier",
                "struct_specifier",
                "namespace_definition",
                "preproc_include",
                "preproc_define",
                "template_declaration",
            ],
            "csharp": [
                "method_declaration",
                "class_declaration",
                "interface_declaration",
                "using_directive",
                "namespace_declaration",
                "property_declaration",
                "constructor_declaration",
            ],
            "html": [
                "element",
                "doctype",
                "script_element",
                "style_element",
                "comment",
            ],
            "css": [
                "rule_set",
                "at_rule",
                "media_query",
                "keyframe_block",
                "import_statement",
            ],
        }

        target_types = entity_types.get(language, [])

        def traverse_node(node):
            """Recursively traverse tree nodes with error handling."""
            try:
                # Skip ERROR nodes (malformed syntax)
                if node.type == "ERROR":
                    logger.debug(
                        f"Skipping ERROR node at {file_path}:{node.start_point[0]}"
                    )
                    return

                if node.type in target_types:
                    # Extract entity information with comprehensive error handling
                    try:
                        entity_content = content[
                            node.start_byte : node.end_byte
                        ].decode("utf-8", errors="replace")

                        # Skip if content is too large (>50KB per entity)
                        if len(entity_content) > 50000:
                            logger.debug(
                                f"Skipping large entity ({len(entity_content)} chars) at {file_path}:{node.start_point[0]}"
                            )
                            return

                        entity = {
                            "id": f"{file_path.name}_{node.start_point[0]}_{node.start_point[1]}_{hash(entity_content[:100])}",
                            "type": "code_entity",
                            "entity_type": node.type,
                            "language": language,
                            "file_path": str(file_path),
                            "content": entity_content,
                            "start_line": node.start_point[0]
                            + 1,  # 1-based line numbers
                            "end_line": node.end_point[0] + 1,
                            "start_column": node.start_point[1],
                            "end_column": node.end_point[1],
                            "byte_range": [node.start_byte, node.end_byte],
                            "timestamp": datetime.utcnow().isoformat(),
                        }

                        # Extract entity name with enhanced patterns
                        name_info = self._find_name_node(node, language)
                        if name_info:
                            if isinstance(name_info, dict):
                                entity.update(name_info)
                            else:
                                entity["name"] = content[
                                    name_info.start_byte : name_info.end_byte
                                ].decode("utf-8", errors="replace")

                        # Add parent context for nested entities
                        if hasattr(node, "parent") and node.parent:
                            entity["parent_type"] = node.parent.type

                        entities.append(entity)

                    except Exception as entity_error:
                        logger.warning(
                            f"Failed to extract entity at {file_path}:{node.start_point[0]}: {entity_error}"
                        )

                # Recursively process children with error handling
                for child in node.children:
                    traverse_node(child)

            except Exception as traverse_error:
                logger.warning(
                    f"Error traversing node {node.type} at {file_path}:{node.start_point[0]}: {traverse_error}"
                )

        traverse_node(node)
        return entities

    def _find_name_node(self, node, language: str):
        """Find the name node within a declaration node with enhanced patterns."""
        try:
            # Enhanced language-specific name extraction patterns
            name_patterns = {
                "python": {
                    "primary_field": "name",
                    "fallback_types": ["identifier"],
                    "special_cases": {
                        "decorated_definition": lambda n: n.child_by_field_name(
                            "definition"
                        ),
                    },
                },
                "javascript": {
                    "primary_field": "name",
                    "fallback_types": ["identifier", "property_identifier"],
                    "special_cases": {
                        "arrow_function": lambda n: self._extract_arrow_function_name(
                            n
                        ),
                        "variable_declaration": lambda n: self._extract_variable_names(
                            n
                        ),
                    },
                },
                "typescript": {
                    "primary_field": "name",
                    "fallback_types": ["identifier", "type_identifier"],
                    "special_cases": {
                        "arrow_function": lambda n: self._extract_arrow_function_name(
                            n
                        ),
                        "type_alias_declaration": lambda n: n.child_by_field_name(
                            "name"
                        ),
                    },
                },
                "java": {
                    "primary_field": "name",
                    "fallback_types": ["identifier"],
                    "special_cases": {
                        "package_declaration": lambda n: n.child_by_field_name("name"),
                        "import_declaration": lambda n: self._extract_java_import_name(
                            n
                        ),
                    },
                },
                "go": {
                    "primary_field": "name",
                    "fallback_types": ["identifier", "field_identifier"],
                    "special_cases": {
                        "var_declaration": lambda n: self._extract_go_var_names(n),
                        "const_declaration": lambda n: self._extract_go_const_names(n),
                    },
                },
                "rust": {
                    "primary_field": "name",
                    "fallback_types": ["identifier", "type_identifier"],
                    "special_cases": {
                        "use_declaration": lambda n: self._extract_rust_use_name(n),
                        "mod_item": lambda n: n.child_by_field_name("name"),
                    },
                },
                "c": {
                    "primary_field": "declarator",
                    "fallback_types": ["identifier", "function_declarator"],
                    "special_cases": {
                        "preproc_include": lambda n: n.child_by_field_name("path"),
                        "preproc_define": lambda n: n.child_by_field_name("name"),
                    },
                },
                "cpp": {
                    "primary_field": "declarator",
                    "fallback_types": [
                        "identifier",
                        "function_declarator",
                        "qualified_identifier",
                    ],
                    "special_cases": {
                        "namespace_definition": lambda n: n.child_by_field_name("name"),
                        "template_declaration": lambda n: self._extract_cpp_template_name(
                            n
                        ),
                    },
                },
                "csharp": {
                    "primary_field": "name",
                    "fallback_types": ["identifier"],
                    "special_cases": {
                        "namespace_declaration": lambda n: n.child_by_field_name(
                            "name"
                        ),
                        "using_directive": lambda n: n.child_by_field_name("name"),
                    },
                },
                "html": {
                    "primary_field": "tag_name",
                    "fallback_types": ["tag_name"],
                    "special_cases": {
                        "script_element": lambda n: {
                            "name": "script",
                            "type": "html_script",
                        },
                        "style_element": lambda n: {
                            "name": "style",
                            "type": "html_style",
                        },
                    },
                },
                "css": {
                    "primary_field": "selector",
                    "fallback_types": ["tag_name", "class_selector", "id_selector"],
                    "special_cases": {
                        "at_rule": lambda n: n.child_by_field_name("name"),
                        "media_query": lambda n: {
                            "name": "media",
                            "query": str(n)[:100],
                        },
                    },
                },
            }

            pattern = name_patterns.get(language, {})

            # Try special case handlers first
            special_cases = pattern.get("special_cases", {})
            if node.type in special_cases:
                result = special_cases[node.type](node)
                if result:
                    return result

            # Try primary field extraction
            primary_field = pattern.get("primary_field")
            if primary_field and hasattr(node, "child_by_field_name"):
                name_node = node.child_by_field_name(primary_field)
                if name_node:
                    return name_node

            # Fallback to type-based search
            fallback_types = pattern.get("fallback_types", ["identifier"])
            for child in node.children:
                if child.type in fallback_types:
                    return child

            # Last resort: first identifier-like child
            for child in node.children:
                if "identifier" in child.type.lower():
                    return child

            return None

        except Exception as e:
            logger.debug(f"Error finding name node for {node.type} in {language}: {e}")
            return None

    def _extract_arrow_function_name(self, node):
        """Extract name from arrow function assignments."""
        # Look for patterns like: const funcName = () => {}
        parent = getattr(node, "parent", None)
        if parent and parent.type == "variable_declarator":
            return parent.child_by_field_name("name")
        return None

    def _extract_variable_names(self, node):
        """Extract variable names from declarations."""
        # Return first declarator name
        for child in node.children:
            if child.type == "variable_declarator":
                return child.child_by_field_name("name")
        return None

    def _extract_java_import_name(self, node):
        """Extract import name from Java import statement."""
        import_name = node.child_by_field_name("name")
        if import_name:
            return {"name": str(import_name), "import_type": "java_import"}
        return None

    def _extract_go_var_names(self, node):
        """Extract variable names from Go var declarations."""
        # Return first identifier
        for child in node.children:
            if child.type == "identifier":
                return child
        return None

    def _extract_go_const_names(self, node):
        """Extract constant names from Go const declarations."""
        for child in node.children:
            if child.type == "identifier":
                return child
        return None

    def _extract_rust_use_name(self, node):
        """Extract use declaration target from Rust."""
        use_tree = node.child_by_field_name("argument")
        if use_tree:
            return {"name": str(use_tree)[:50], "use_type": "rust_use"}
        return None

    def _extract_cpp_template_name(self, node):
        """Extract template declaration name from C++."""
        for child in node.children:
            if child.type in ["function_definition", "class_specifier"]:
                return self._find_name_node(child, "cpp")
        return None

    # Legacy AI description methods removed - replaced by comprehensive AI analyst:
    # - _enhance_entities_with_ai_descriptions -> _enhance_entities_with_ai_analyst
    # - _filter_entities_for_ai_analysis -> _filter_entities_for_ai_analyst
    # - _process_entities_in_batches -> _process_entities_with_ai_analyst
    # - All related helper methods now provide richer analysis (summary, complexity, dependencies, tags)

    # Legacy methods removed - replaced by enhanced two-pass architecture:
    # - _extract_relationships -> _second_pass_extract_relationships
    # - _analyze_cross_file_dependencies -> comprehensive import/call analysis in Pass 2
    # - _enhance_entities_with_ai_descriptions -> _enhance_entities_with_ai_analyst

    async def _populate_knowledge_base(
        self, entities: list[Dict[str, Any]], relationships: list[Dict[str, Any]]
    ) -> None:
        """
        Populate Azure Cosmos DB knowledge base using OmniRAG pattern.

        Based on Context7 research: Uses vector embeddings with batch processing,
        proper indexing policies, and Azure OpenAI text-embedding-3-small.

        Args:
            entities: Code entities to store
            relationships: Relationships to store
        """
        try:
            logger.info(
                f"Populating knowledge base with {len(entities)} entities and {len(relationships)} relationships"
            )

            # Generate embeddings for entities in batches
            await self._generate_and_store_entity_embeddings(entities)

            # Store relationships
            await self._store_relationships(relationships)

            logger.info("Knowledge base population completed successfully")

        except Exception as e:
            logger.error(f"Failed to populate knowledge base: {e}")
            raise

    async def _generate_and_store_entity_embeddings(
        self, entities: list[Dict[str, Any]]
    ) -> None:
        """Generate enhanced embeddings using AI analyst insights for richer vector representations."""
        batch_size = 100  # Azure OpenAI recommended batch size

        for i in range(0, len(entities), batch_size):
            batch = entities[i : i + batch_size]

            try:
                # Prepare enhanced texts for embedding generation using AI analyst insights
                texts = []
                for entity in batch:
                    # Create rich, composite text using AI analyst insights
                    enhanced_text = self._create_enhanced_embedding_text(entity)
                    texts.append(enhanced_text)

                # Generate embeddings using Azure OpenAI Text Embedding service
                # Based on Context7 research: AzureTextEmbedding.generate_embeddings(texts) returns list of vectors
                embeddings_result = await self.embedding_service.generate_embeddings(
                    texts
                )

                # Store entities with embeddings in Cosmos DB
                documents = []
                for j, entity in enumerate(batch):
                    # Format according to enhanced OmniRAG schema with AI analyst insights
                    doc = {
                        "id": entity["id"],
                        "type": "code_entity",
                        "entity_type": entity["entity_type"],
                        "name": entity.get("name", "unknown"),
                        "language": entity["language"],
                        "file_path": entity["file_path"],
                        "content": entity["content"],
                        # Enhanced AI analyst fields
                        "ai_summary": entity.get("ai_summary", ""),
                        "ai_complexity": entity.get("ai_complexity", 5),
                        "ai_dependencies": entity.get("ai_dependencies", []),
                        "ai_tags": entity.get("ai_tags", []),
                        "has_ai_analyst": entity.get("has_ai_analyst", False),
                        "ai_analyst_timestamp": entity.get("ai_analyst_timestamp", ""),
                        # Legacy support
                        "ai_description": entity.get("ai_description", ""),
                        "has_ai_analysis": entity.get("has_ai_analysis", False),
                        # Enhanced embedding
                        "embedding": embeddings_result[j]
                        if j < len(embeddings_result)
                        else [],
                        "timestamp": entity["timestamp"],
                    }
                    documents.append(doc)

                # Batch upsert to Cosmos DB
                for doc in documents:
                    await self.knowledge_container.upsert_item(doc)

                logger.info(
                    f"Stored batch {i // batch_size + 1} of {(len(entities) + batch_size - 1) // batch_size}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to process entity batch {i}-{i + batch_size}: {e}"
                )
                # Continue with next batch rather than failing completely
                continue

    def _create_enhanced_embedding_text(self, entity: Dict[str, Any]) -> str:
        """
        Create rich, composite text for enhanced embeddings using AI analyst insights.

        This generates much more meaningful vectors by combining:
        - Entity name and type
        - AI-generated summary
        - Complexity score context
        - Dependency information
        - Relevant tags
        - Code preview

        The resulting text provides much better semantic search capabilities.
        """
        try:
            # Start with basic entity information
            name = entity.get("name", "unknown")
            entity_type = entity.get("entity_type", "")
            language = entity.get("language", "")

            # Build enhanced text components
            text_parts = []

            # 1. Entity identification
            text_parts.append(f"{name} {entity_type} in {language}")

            # 2. AI-generated summary (most important for semantic meaning)
            ai_summary = entity.get("ai_summary", "")
            if ai_summary and len(ai_summary) > 10:
                text_parts.append(f"Purpose: {ai_summary}")
            else:
                # Fallback to legacy description
                ai_description = entity.get("ai_description", "")
                if ai_description:
                    text_parts.append(f"Description: {ai_description}")

            # 3. Complexity context (helps with difficulty/skill-level searches)
            ai_complexity = entity.get("ai_complexity", 5)
            if ai_complexity <= 3:
                complexity_label = "simple"
            elif ai_complexity <= 6:
                complexity_label = "moderate"
            else:
                complexity_label = "complex"
            text_parts.append(f"Complexity: {complexity_label} ({ai_complexity}/10)")

            # 4. Dependencies (critical for relationship searches)
            ai_dependencies = entity.get("ai_dependencies", [])
            if ai_dependencies:
                deps_text = ", ".join(ai_dependencies[:5])  # Limit to first 5
                text_parts.append(f"Depends on: {deps_text}")

            # 5. Tags (excellent for categorical/domain searches)
            ai_tags = entity.get("ai_tags", [])
            if ai_tags:
                tags_text = ", ".join(ai_tags[:8])  # Limit to first 8
                text_parts.append(f"Tags: {tags_text}")

            # 6. File context
            file_path = entity.get("file_path", "")
            if file_path:
                file_name = Path(file_path).name
                text_parts.append(f"File: {file_name}")

            # 7. Code preview (smaller since we have AI insights)
            content = entity.get("content", "")
            if content:
                # Use first 200 chars for context, since AI summary provides semantic meaning
                content_preview = content[:200].replace("\n", " ").strip()
                if len(content_preview) > 50:
                    text_parts.append(f"Code: {content_preview}")

            # Combine all parts into rich embedding text
            enhanced_text = " | ".join(text_parts)

            # Ensure reasonable length (embedding models have token limits)
            if len(enhanced_text) > 2000:
                enhanced_text = enhanced_text[:1997] + "..."

            return enhanced_text

        except Exception as e:
            logger.debug(
                f"Error creating enhanced embedding text for entity {entity.get('id', 'unknown')}: {e}"
            )

            # Fallback to basic text
            name = entity.get("name", "unknown")
            entity_type = entity.get("entity_type", "")
            content_preview = entity.get("content", "")[:300]
            return f"{name} {entity_type} {content_preview}"

    async def _store_relationships(self, relationships: list[Dict[str, Any]]) -> None:
        """Store relationship data in Cosmos DB."""
        batch_size = 100

        for i in range(0, len(relationships), batch_size):
            batch = relationships[i : i + batch_size]

            try:
                # Store relationships in Cosmos DB
                for relationship in batch:
                    # Format according to OmniRAG graph pattern
                    doc = {
                        "id": relationship["id"],
                        "type": "relationship",
                        "relationship_type": relationship["type"],
                        "source_id": relationship["source_id"],
                        "target_id": relationship["target_id"],
                        "source_file": relationship["source_file"],
                        "target_file": relationship["target_file"],
                        "timestamp": relationship["timestamp"],
                    }
                    await self.knowledge_container.upsert_item(doc)

                logger.info(f"Stored relationship batch {i // batch_size + 1}")

            except Exception as e:
                logger.error(f"Failed to store relationship batch: {e}")
                continue

    async def _ai_classify_file(self, file_path: Path, content: bytes) -> dict:
        """
        AI-powered comprehensive file classification using Semantic Kernel with structured outputs.

        This replaces deterministic classification logic with intelligent AI analysis that can:
        - Detect architectural patterns (MVC, service layer, DTO, etc.)
        - Identify development tool files (.claude, .cursor, vscode configs)
        - Classify by purpose (service, model, controller, middleware, etc.)
        - Detect frameworks and libraries being used

        Args:
            file_path: Path to the file being classified
            content: Raw file content as bytes

        Returns:
            Comprehensive classification dictionary with structured AI insights
        """
        try:
            # Convert content to text for AI analysis
            content_text = content.decode("utf-8", errors="ignore")

            # Prepare classification prompt with structured output schema
            classification_prompt = """
            Analyze this file and provide comprehensive classification information.
            
            FILE PATH: {{$file_path}}
            FILE CONTENT (first 2000 chars):
            {{$content_preview}}
            
            Provide a detailed classification including:
            1. Primary file type and purpose
            2. Architectural pattern classification (MVC, service layer, etc.)
            3. Development tool classification (AI tools, IDE configs, etc.)
            4. Framework/library identification
            5. Specific role in codebase
            
            Return structured JSON with the exact schema specified.
            """

            # Create structured output schema using Semantic Kernel function
            classification_function = self.kernel.add_function(
                function_name="classify_file",
                plugin_name="file_analysis",
                prompt_template_config=self._create_classification_prompt_config(
                    classification_prompt
                ),
            )

            # Invoke AI classification with structured output
            result = await self.kernel.invoke(
                classification_function,
                file_path=str(file_path),
                content_preview=content_text[:2000],
            )

            # Parse structured output
            classification_data = self._parse_ai_classification(str(result))

            return classification_data

        except Exception as e:
            logger.debug(f"AI classification failed for {file_path}: {e}")

            # Fallback to basic classification
            return {
                "primary_type": "unknown",
                "architectural_pattern": "unknown",
                "development_tool_type": None,
                "framework": None,
                "purpose": "unknown",
                "confidence": 0.1,
                "ai_powered": False,
                "fallback_reason": str(e),
            }

    async def _ai_generate_tags(
        self, file_path: Path, content: bytes, entity_content: str
    ) -> list[str]:
        """
        AI-powered tag generation using Semantic Kernel for OmniRAG filtering.

        This generates intelligent tags that enable powerful searches like:
        - "all services" - finds all service-related files
        - "all tests" - finds all testing files
        - "all AI tools" - finds .claude, .cursor, vscode configs
        - "all DTOs" - finds data transfer objects
        - "all MVC controllers" - finds controller files

        Args:
            file_path: Path to the file
            content: Raw file content as bytes
            entity_content: Specific entity content being analyzed

        Returns:
            List of intelligent tags for OmniRAG filtering
        """
        try:
            content_text = content.decode("utf-8", errors="ignore")

            # AI-powered tag generation prompt
            tag_generation_prompt = """
            Generate comprehensive tags for this code file and entity for semantic search and filtering.
            
            FILE: {{$file_path}}
            ENTITY CONTENT: {{$entity_content}}
            FILE CONTEXT (first 1000 chars): {{$file_context}}
            
            Generate tags in these categories:
            1. ARCHITECTURAL: service, controller, model, dto, middleware, adapter, repository, etc.
            2. PURPOSE: business-logic, data-access, ui-component, configuration, utility, etc.
            3. TECHNOLOGY: framework names, library types, language-specific patterns
            4. DEVELOPMENT: test, mock, fixture, ai-tool, ide-config, build-script, etc.
            5. DOMAIN: specific business domain concepts found in the code
            
            Return ONLY a JSON array of strings, no explanation.
            Example: ["service", "business-logic", "spring-boot", "rest-api", "user-management"]
            """

            # Create tag generation function
            tag_function = self.kernel.add_function(
                function_name="generate_tags",
                plugin_name="file_analysis",
                prompt_template_config=self._create_tag_generation_prompt_config(
                    tag_generation_prompt
                ),
            )

            # Invoke AI tag generation
            result = await self.kernel.invoke(
                tag_function,
                file_path=str(file_path),
                entity_content=entity_content[:1000],
                file_context=content_text[:1000],
            )

            # Parse tag array from AI response
            tags = self._parse_ai_tags(str(result))

            # Add basic deterministic tags as backup
            backup_tags = self._generate_backup_tags(file_path)

            # Combine AI tags with backup, remove duplicates
            all_tags = list(set(tags + backup_tags))

            return all_tags

        except Exception as e:
            logger.debug(f"AI tag generation failed for {file_path}: {e}")

            # Fallback to deterministic tags
            return self._generate_backup_tags(file_path)

    def _create_classification_prompt_config(self, prompt_template: str):
        """Create Semantic Kernel prompt config for file classification with structured output."""
        from semantic_kernel.prompt_template import PromptTemplateConfig, InputVariable
        from semantic_kernel.connectors.ai.open_ai import (
            AzureChatPromptExecutionSettings,
        )

        # Configure structured output for classification
        execution_settings = AzureChatPromptExecutionSettings(
            service_id="azure_openai_chat",
            ai_model_id=self.settings.azure_openai_chat_deployment_name,
            max_tokens=1000,
            temperature=0.1,  # Low temperature for consistent classification
            response_format="json_object",  # Ensure JSON output
        )

        return PromptTemplateConfig(
            template=prompt_template
            + """
            
            Return JSON in this exact format:
            {
              "primary_type": "service|controller|model|dto|test|config|documentation|script|ai_tool|ide_config|build|unknown",
              "architectural_pattern": "mvc|service_layer|repository|adapter|middleware|microservice|monolith|utility|unknown",
              "development_tool_type": "ai_assistant|ide_config|build_tool|testing|linting|formatting|deployment|null",
              "framework": "detected framework name or null",
              "purpose": "business_logic|data_access|ui_component|configuration|testing|tooling|documentation|unknown",
              "confidence": 0.95,
              "ai_powered": true,
              "specialized_tags": ["tag1", "tag2", "tag3"]
            }
            """,
            name="file_classification",
            template_format="semantic-kernel",
            input_variables=[
                InputVariable(
                    name="file_path", description="Full file path", is_required=True
                ),
                InputVariable(
                    name="content_preview",
                    description="File content preview",
                    is_required=True,
                ),
            ],
            execution_settings=execution_settings,
        )

    def _create_tag_generation_prompt_config(self, prompt_template: str):
        """Create Semantic Kernel prompt config for AI tag generation."""
        from semantic_kernel.prompt_template import PromptTemplateConfig, InputVariable
        from semantic_kernel.connectors.ai.open_ai import (
            AzureChatPromptExecutionSettings,
        )

        execution_settings = AzureChatPromptExecutionSettings(
            service_id="azure_openai_chat",
            ai_model_id=self.settings.azure_openai_chat_deployment_name,
            max_tokens=500,
            temperature=0.2,  # Slightly higher for creative tag generation
            response_format="json_object",
        )

        return PromptTemplateConfig(
            template=prompt_template,
            name="tag_generation",
            template_format="semantic-kernel",
            input_variables=[
                InputVariable(
                    name="file_path", description="File path", is_required=True
                ),
                InputVariable(
                    name="entity_content",
                    description="Entity content",
                    is_required=True,
                ),
                InputVariable(
                    name="file_context", description="File context", is_required=True
                ),
            ],
            execution_settings=execution_settings,
        )

    def _parse_ai_classification(self, ai_response: str) -> dict:
        """Parse AI classification response with fallback."""
        try:
            import json

            # Try to parse as JSON
            data = json.loads(ai_response.strip())

            # Validate required fields
            required_fields = [
                "primary_type",
                "architectural_pattern",
                "purpose",
                "confidence",
            ]
            for field in required_fields:
                if field not in data:
                    data[field] = "unknown"

            return data

        except Exception as e:
            logger.debug(f"Failed to parse AI classification: {e}")
            return {
                "primary_type": "unknown",
                "architectural_pattern": "unknown",
                "purpose": "unknown",
                "confidence": 0.1,
                "ai_powered": False,
                "parse_error": str(e),
            }

    def _parse_ai_tags(self, ai_response: str) -> list[str]:
        """Parse AI tag generation response with fallback."""
        try:
            import json

            # Handle different response formats
            cleaned_response = ai_response.strip()

            # Try direct JSON array parsing
            if cleaned_response.startswith("["):
                return json.loads(cleaned_response)

            # Try extracting JSON from larger response
            import re

            json_match = re.search(r"\[.*?\]", cleaned_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

            # Try parsing as lines
            lines = [
                line.strip(" \"',-")
                for line in cleaned_response.split("\n")
                if line.strip()
            ]
            return [line for line in lines if line and not line.startswith("{")][
                :10
            ]  # Limit to 10 tags

        except Exception as e:
            logger.debug(f"Failed to parse AI tags: {e}")
            return []

    def _generate_backup_tags(self, file_path: Path) -> list[str]:
        """Generate basic deterministic tags as backup when AI fails."""
        tags = []

        path_str = str(file_path).lower()
        name = file_path.name.lower()
        extension = file_path.suffix.lower()

        # Programming language tags
        if extension in [".py"]:
            tags.append("python")
        elif extension in [".js", ".jsx"]:
            tags.append("javascript")
        elif extension in [".ts", ".tsx"]:
            tags.append("typescript")
        elif extension in [".java"]:
            tags.append("java")
        elif extension in [".go"]:
            tags.append("go")
        elif extension in [".rs"]:
            tags.append("rust")
        elif extension in [".c", ".h"]:
            tags.append("c")
        elif extension in [".cpp", ".hpp"]:
            tags.append("cpp")
        elif extension in [".cs"]:
            tags.append("csharp")

        # Basic file type tags
        if "test" in path_str or name.startswith("test_"):
            tags.extend(["test", "testing"])
        if "config" in path_str or name in ["config.py", "settings.py"]:
            tags.extend(["config", "configuration"])
        if extension in [".md", ".rst", ".txt"]:
            tags.extend(["documentation", "docs"])
        if ".claude" in path_str:
            tags.extend(["ai-tool", "claude"])
        if ".cursor" in path_str or ".vscode" in path_str:
            tags.extend(["ide-config", "development-tool"])
        if "dockerfile" in name or "docker-compose" in name:
            tags.extend(["docker", "containerization"])
        if name.endswith(".json") or name.endswith(".yaml") or name.endswith(".yml"):
            tags.extend(["config", "data"])

        return tags

    async def _ai_extract_entity_name(
        self, node, language: str, content: bytes, entity_content: str
    ) -> str:
        """
        AI-powered entity name extraction using Semantic Kernel.

        This replaces deterministic name extraction logic with intelligent AI analysis
        that can handle complex cases like:
        - Generic types and type parameters
        - Anonymous functions and lambdas
        - Complex inheritance hierarchies
        - Framework-specific naming patterns
        - Overloaded methods and operators

        Args:
            node: Tree-sitter AST node
            language: Programming language
            content: Raw file content as bytes
            entity_content: Specific entity content

        Returns:
            Intelligent entity name or meaningful identifier
        """
        try:
            # Try deterministic extraction first for performance
            basic_name = self._extract_entity_name(node, language, content)

            # If basic extraction succeeds and looks good, use it
            if basic_name and basic_name != "unknown" and len(basic_name.strip()) > 0:
                return basic_name.strip()

            # Fall back to AI for complex cases
            name_extraction_prompt = """
            Extract the most meaningful identifier/name for this code entity.
            
            LANGUAGE: {{$language}}
            ENTITY TYPE: {{$entity_type}}
            ENTITY CONTENT: {{$entity_content}}
            
            Rules:
            1. Return the primary identifier (class name, function name, variable name, etc.)
            2. For anonymous entities, create a descriptive name based on purpose
            3. For complex generics, simplify to the main type
            4. For overloaded methods, include distinguishing characteristics
            5. Must be a valid identifier (no spaces, special chars except _)
            
            Return ONLY the name, no explanation or quotes.
            """

            # Create AI name extraction function
            name_function = self.kernel.add_function(
                function_name="extract_entity_name",
                plugin_name="code_analysis",
                prompt_template_config=self._create_name_extraction_prompt_config(
                    name_extraction_prompt
                ),
            )

            # Invoke AI name extraction
            result = await self.kernel.invoke(
                name_function,
                language=language,
                entity_type=node.type if hasattr(node, "type") else "unknown",
                entity_content=entity_content[
                    :500
                ],  # Limit content for focused analysis
            )

            # Clean and validate AI result
            ai_name = str(result).strip().strip("\"'`")

            # Validate AI-generated name
            if ai_name and len(ai_name) > 0 and len(ai_name) < 100:
                return ai_name

            # Final fallback
            return (
                basic_name
                if basic_name != "unknown"
                else f"entity_{hash(entity_content[:50]) % 10000}"
            )

        except Exception as e:
            logger.debug(f"AI entity name extraction failed: {e}")
            # Fallback to deterministic method
            return self._extract_entity_name(node, language, content)

    def _create_name_extraction_prompt_config(self, prompt_template: str):
        """Create Semantic Kernel prompt config for AI entity name extraction."""
        from semantic_kernel.prompt_template import PromptTemplateConfig, InputVariable
        from semantic_kernel.connectors.ai.open_ai import (
            AzureChatPromptExecutionSettings,
        )

        execution_settings = AzureChatPromptExecutionSettings(
            service_id="azure_openai_chat",
            ai_model_id=self.settings.azure_openai_chat_deployment_name,
            max_tokens=50,  # Short response for just the name
            temperature=0.1,  # Very low temperature for consistent naming
        )

        return PromptTemplateConfig(
            template=prompt_template,
            name="extract_entity_name",
            template_format="semantic-kernel",
            input_variables=[
                InputVariable(
                    name="language",
                    description="Programming language",
                    is_required=True,
                ),
                InputVariable(
                    name="entity_type",
                    description="Type of code entity",
                    is_required=True,
                ),
                InputVariable(
                    name="entity_content",
                    description="Entity source code",
                    is_required=True,
                ),
            ],
            execution_settings=execution_settings,
        )

    def _create_visibility_analysis_prompt_config(self, prompt_template: str):
        """Create Semantic Kernel prompt config for AI visibility analysis."""
        from semantic_kernel.prompt_template import PromptTemplateConfig, InputVariable
        from semantic_kernel.connectors.ai.open_ai import (
            AzureChatPromptExecutionSettings,
        )

        execution_settings = AzureChatPromptExecutionSettings(
            service_id="azure_openai_chat",
            ai_model_id=self.settings.azure_openai_chat_deployment_name,
            max_tokens=200,  # Short response for visibility analysis
            temperature=0.1,  # Very low temperature for consistent analysis
            response_format="json_object",  # Ensure JSON output
        )

        return PromptTemplateConfig(
            template=prompt_template,
            name="analyze_entity_visibility",
            template_format="semantic-kernel",
            input_variables=[
                InputVariable(
                    name="language",
                    description="Programming language",
                    is_required=True,
                ),
                InputVariable(
                    name="entity_content",
                    description="Entity source code",
                    is_required=True,
                ),
            ],
            execution_settings=execution_settings,
        )

    def _parse_ai_visibility_analysis(self, ai_response: str) -> dict:
        """Parse AI visibility analysis response with fallback."""
        try:
            import json

            data = json.loads(ai_response.strip())

            # Validate and set defaults
            result = {
                "is_public": data.get("is_public", True),
                "is_exported": data.get("is_exported", True),
                "visibility_modifier": data.get("visibility_modifier"),
                "confidence": data.get("confidence", 0.8),
                "ai_powered": True,
            }

            return result

        except Exception as e:
            logger.debug(f"Failed to parse AI visibility analysis: {e}")
            return {
                "is_public": True,
                "is_exported": True,
                "visibility_modifier": None,
                "confidence": 0.3,
                "ai_powered": False,
                "parse_error": str(e),
            }

    def _create_import_parse_prompt_config(self, prompt_template: str):
        """Create Semantic Kernel prompt config for AI import parsing."""
        from semantic_kernel.prompt_template import PromptTemplateConfig, InputVariable
        from semantic_kernel.connectors.ai.open_ai import (
            AzureChatPromptExecutionSettings,
        )

        execution_settings = AzureChatPromptExecutionSettings(
            service_id="azure_openai_chat",
            ai_model_id=self.settings.azure_openai_chat_deployment_name,
            max_tokens=300,  # Moderate response for import names
            temperature=0.1,  # Very low temperature for consistent parsing
            response_format="json_object",  # Ensure JSON array output
        )

        return PromptTemplateConfig(
            template=prompt_template,
            name="parse_import_statement",
            template_format="semantic-kernel",
            input_variables=[
                InputVariable(
                    name="language",
                    description="Programming language",
                    is_required=True,
                ),
                InputVariable(
                    name="import_text",
                    description="Import statement text",
                    is_required=True,
                ),
            ],
            execution_settings=execution_settings,
        )

    def _parse_ai_import_result(self, ai_response: str) -> list[str]:
        """Parse AI import parsing response with fallback."""
        try:
            import json

            # Handle different response formats
            cleaned_response = ai_response.strip()

            # Try direct JSON array parsing
            if cleaned_response.startswith("["):
                return json.loads(cleaned_response)

            # Try extracting JSON from larger response
            import re

            json_match = re.search(r"\[.*?\]", cleaned_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

            # Try parsing as lines if JSON fails
            lines = [
                line.strip(" \"',-")
                for line in cleaned_response.split("\n")
                if line.strip()
            ]
            return [line for line in lines if line and not line.startswith("{")][
                :8
            ]  # Limit to 8 imports

        except Exception as e:
            logger.debug(f"Failed to parse AI import result: {e}")
            return []

    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        logger.info("IngestionPlugin cleanup completed")
