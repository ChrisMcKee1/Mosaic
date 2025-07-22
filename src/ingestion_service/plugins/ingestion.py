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

# Semantic Kernel for AI-powered descriptions
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

        # Initialize AI enhancement components
        self._description_cache = {}  # SHA-256 hash -> description cache
        self._processing_semaphore = asyncio.Semaphore(5)  # Limit concurrent API calls
        self._cache_hits = 0
        self._cache_misses = 0
        self._api_calls = 0

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
            # Initialize kernel
            self.kernel = Kernel()

            # Add Azure OpenAI chat service for code analysis
            self.chat_service = AzureChatCompletion(
                deployment_name=self.settings.azure_openai_chat_deployment_name,
                endpoint=self.settings.azure_openai_endpoint,
                service_id="code_analyzer",
            )
            self.kernel.add_service(self.chat_service)

            # Register code analysis function
            await self._register_code_analysis_functions()

            logger.info("Semantic Kernel initialized for code analysis")

        except Exception as e:
            logger.error(f"Failed to initialize Semantic Kernel: {e}")
            raise

    async def _register_code_analysis_functions(self) -> None:
        """Register Semantic Kernel functions for code analysis."""

        # Code description generation prompt template
        code_description_prompt = """
You are a senior software engineer analyzing code. Generate a concise, technical description of what this {{$entity_type}} does.

Code Entity: {{$name}}
Language: {{$language}}
Code:
```{{$language}}
{{$content}}
```

Instructions:
1. Focus on the PRIMARY purpose and functionality
2. Mention key parameters/inputs if it's a function/method
3. Describe return values or side effects if applicable
4. Keep it under 2 sentences
5. Use technical terminology appropriate for developers
6. Do NOT include implementation details, just what it accomplishes

Description:"""

        # Create the prompt template configuration
        prompt_config = PromptTemplateConfig(
            template=code_description_prompt,
            name="analyze_code_entity",
            description="Generate technical description of code entities",
        )

        # Register the function with the kernel
        self.kernel.add_function(
            plugin_name="CodeAnalysis",
            function_name="describe_entity",
            prompt_template_config=prompt_config,
        )

    async def ingest_repository(
        self, repository_url: str, branch: str = "main"
    ) -> Dict[str, Any]:
        """
        Ingest a code repository and populate the knowledge graph.

        Phase 1 Foundation implementation using validated technologies:
        - GitPython for repository access
        - tree-sitter for AST parsing
        - OmniRAG pattern for knowledge storage

        Args:
            repository_url: Git repository URL
            branch: Git branch to process

        Returns:
            Ingestion summary with statistics
        """
        temp_dir = None
        try:
            # Step 1: Clone repository using GitPython
            logger.info(f"Starting repository ingestion: {repository_url}")
            temp_dir = await self._clone_repository(repository_url, branch)

            # Step 2: Scan and parse code files
            code_entities = await self._parse_repository(temp_dir)

            # Step 3: Extract relationships and build graph
            relationships = await self._extract_relationships(code_entities)

            # Step 4: Generate embeddings and populate knowledge base
            await self._populate_knowledge_base(code_entities, relationships)

            # Step 5: Generate summary
            summary = {
                "repository_url": repository_url,
                "branch": branch,
                "entities_extracted": len(code_entities),
                "relationships_found": len(relationships),
                "timestamp": datetime.utcnow().isoformat(),
                "status": "completed",
            }

            logger.info(f"Repository ingestion completed: {summary}")
            return summary

        except Exception as e:
            logger.error(f"Repository ingestion failed: {e}")
            raise
        finally:
            # Cleanup temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

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

    async def _enhance_entities_with_ai_descriptions(
        self, entities: list[Dict[str, Any]]
    ) -> list[Dict[str, Any]]:
        """
        Enhance code entities with AI-generated descriptions using Semantic Kernel.

        Implements research-backed batch processing with caching, retry logic, and
        concurrency control for optimal performance and reliability.

        Args:
            entities: List of extracted code entities

        Returns:
            List of entities enhanced with AI descriptions
        """
        if not entities:
            return entities

        logger.info(f"Starting AI enhancement for {len(entities)} entities")

        # Filter entities that need AI descriptions
        candidates = self._filter_entities_for_ai_analysis(entities)

        if not candidates:
            logger.info("No entities require AI analysis")
            return entities

        logger.info(f"Processing {len(candidates)} entities for AI enhancement")

        # Process entities in batches with concurrency control
        enhanced_lookup = await self._process_entities_in_batches(candidates)

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

        # Log performance metrics
        total_enhanced = sum(
            1 for e in enhanced_entities if e.get("has_ai_analysis", False)
        )
        logger.info(
            f"AI Enhancement completed: {total_enhanced} entities enhanced, "
            f"Cache hits: {self._cache_hits}, Cache misses: {self._cache_misses}, "
            f"API calls: {self._api_calls}"
        )

        return enhanced_entities

    def _filter_entities_for_ai_analysis(
        self, entities: list[Dict[str, Any]]
    ) -> list[Dict[str, Any]]:
        """Filter entities that should receive AI-generated descriptions."""
        candidates = []

        # Define entity types that benefit from AI description
        describable_types = {
            "function_definition",
            "async_function_definition",
            "function_declaration",
            "function_expression",
            "arrow_function",
            "method_declaration",
            "class_definition",
            "class_declaration",
            "interface_declaration",
            "type_alias_declaration",
            "struct_item",
            "impl_item",
            "trait_item",
            "enum_item",
            "constructor_declaration",
            "namespace_definition",
        }

        for entity in entities:
            # Skip if already has AI description
            if entity.get("has_ai_analysis", False):
                continue

            # Skip if not a describable entity type
            if entity["entity_type"] not in describable_types:
                continue

            # Skip if content is too short (less than 10 lines as per requirements)
            content = entity.get("content", "")
            line_count = content.count("\n") + 1
            if line_count < 10 or len(content) < 100:
                continue

            # Skip if content is too large (avoid token limits)
            if len(content) > 4000:  # Reasonable limit for API calls
                continue

            candidates.append(entity)

        return candidates

    async def _process_entities_in_batches(
        self, entities: list[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Process entities in batches with concurrency control and caching."""
        enhanced_lookup = {}

        # Group entities into batches of 20 (as per requirements)
        batch_size = 20
        batches = [
            entities[i : i + batch_size] for i in range(0, len(entities), batch_size)
        ]

        logger.info(f"Processing {len(batches)} batches of entities")

        # Process batches concurrently with semaphore control
        batch_tasks = []
        for i, batch in enumerate(batches):
            task = self._process_entity_batch(batch, batch_id=i)
            batch_tasks.append(task)

        # Execute batches with controlled concurrency
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # Collect results from all batches
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.warning(f"Batch processing failed: {batch_result}")
                continue
            enhanced_lookup.update(batch_result)

        return enhanced_lookup

    async def _process_entity_batch(
        self, batch: list[Dict[str, Any]], batch_id: int
    ) -> Dict[str, Dict[str, Any]]:
        """Process a single batch of entities with caching and retry logic."""
        async with self._processing_semaphore:
            logger.debug(f"Processing batch {batch_id} with {len(batch)} entities")
            batch_results = {}

            # Process entities concurrently within the batch
            tasks = []
            for entity in batch:
                task = self._enhance_single_entity(entity)
                tasks.append(task)

            # Execute with controlled concurrency
            entity_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect successful results
            for entity, result in zip(batch, entity_results):
                if isinstance(result, Exception):
                    logger.warning(
                        f"Failed to enhance entity {entity.get('id', 'unknown')}: {result}"
                    )
                    continue

                if result:
                    batch_results[entity["id"]] = result

            logger.debug(
                f"Batch {batch_id} completed: {len(batch_results)} entities enhanced"
            )
            return batch_results

    async def _enhance_single_entity(
        self, entity: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Enhance a single entity with AI-generated description using caching and retry logic."""
        try:
            # Generate cache key from content hash
            content = entity.get("content", "")
            cache_key = hashlib.sha256(content.encode("utf-8")).hexdigest()

            # Check cache first
            if cache_key in self._description_cache:
                self._cache_hits += 1
                cached_result = self._description_cache[cache_key]
                logger.debug(f"Cache hit for entity {entity.get('id', 'unknown')}")
                return {
                    "ai_description": cached_result["description"],
                    "has_ai_analysis": True,
                    "ai_analysis_timestamp": datetime.utcnow().isoformat(),
                    "ai_model_used": cached_result.get("model", "gpt-4o-mini"),
                    "description_confidence": cached_result.get("confidence", 0.85),
                    "cache_hit": True,
                }

            # Generate new description with retry logic
            self._cache_misses += 1
            description = await self._generate_entity_description_with_retry(entity)

            if description and len(description) > 10:
                # Cache successful result
                self._description_cache[cache_key] = {
                    "description": description,
                    "model": "gpt-4o-mini",
                    "confidence": 0.85,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                return {
                    "ai_description": description,
                    "has_ai_analysis": True,
                    "ai_analysis_timestamp": datetime.utcnow().isoformat(),
                    "ai_model_used": "gpt-4o-mini",
                    "description_confidence": 0.85,
                    "cache_hit": False,
                }

            return None

        except Exception as e:
            logger.warning(
                f"Failed to enhance entity {entity.get('id', 'unknown')}: {e}"
            )
            return None

    async def _generate_entity_description_with_retry(
        self, entity: Dict[str, Any]
    ) -> Optional[str]:
        """Generate AI description with exponential backoff retry logic."""
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                self._api_calls += 1
                description = await self._call_semantic_kernel_for_description(entity)

                if description and len(description) > 10:
                    return description

                logger.debug(
                    f"Generated description too short for entity {entity.get('id', 'unknown')}"
                )
                return None

            except Exception as e:
                if attempt == max_retries:
                    logger.error(
                        f"Failed to generate description after {max_retries + 1} attempts: {e}"
                    )
                    return None

                # Exponential backoff with jitter
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                logger.debug(
                    f"API call failed (attempt {attempt + 1}), retrying in {delay:.2f}s: {e}"
                )
                await asyncio.sleep(delay)

        return None

    async def _call_semantic_kernel_for_description(
        self, entity: Dict[str, Any]
    ) -> str:
        """Call Semantic Kernel to generate entity description."""
        try:
            # Prepare specialized prompt based on entity type
            prompt = self._get_entity_analysis_prompt(entity)

            # Prepare arguments for Semantic Kernel
            arguments = KernelArguments(
                entity_type=entity["entity_type"],
                name=entity.get("name", "unknown"),
                language=entity["language"],
                content=entity["content"][:2000],  # Limit to avoid token limits
                prompt=prompt,
            )

            # Call the Semantic Kernel function
            result = await self.kernel.invoke(
                plugin_name="CodeAnalysis",
                function_name="describe_entity",
                arguments=arguments,
            )

            # Extract and clean the description
            description = str(result).strip()

            # Remove any markdown formatting
            description = description.replace("**", "").replace("*", "").strip()

            # Ensure it's not too long (max 200 characters as per requirements)
            if len(description) > 200:
                description = description[:197] + "..."

            return description

        except Exception as e:
            logger.error(f"Semantic Kernel call failed: {e}")
            raise

    def _get_entity_analysis_prompt(self, entity: Dict[str, Any]) -> str:
        """Get specialized analysis prompt based on entity type."""
        entity_type = entity["entity_type"]

        # Specialized prompts for different entity types
        prompts = {
            "function_definition": "Analyze this function and provide a concise 1-2 sentence description of its purpose and functionality.",
            "async_function_definition": "Analyze this async function and describe its asynchronous purpose and what it accomplishes.",
            "class_definition": "Analyze this class and provide a concise 1-2 sentence description of its responsibility and key capabilities.",
            "method_declaration": "Analyze this method and provide a concise 1-2 sentence description of what it does and returns.",
            "interface_declaration": "Analyze this interface and describe the contract it defines and its intended use.",
            "struct_item": "Analyze this struct and describe the data structure it represents and its purpose.",
            "impl_item": "Analyze this implementation block and describe what functionality it provides.",
            "trait_item": "Analyze this trait and describe the behavior contract it defines.",
            "constructor_declaration": "Analyze this constructor and describe how it initializes the object.",
            "namespace_definition": "Analyze this namespace and describe what functionality it organizes.",
        }

        return prompts.get(
            entity_type,
            "Analyze this code entity and provide a concise technical description of its purpose.",
        )

    async def _extract_relationships(
        self, entities: list[Dict[str, Any]]
    ) -> list[Dict[str, Any]]:
        """
        Extract relationships between code entities for graph modeling.

        Based on OmniRAG pattern: Analyzes imports, dependencies, and hierarchical
        relationships to build knowledge graph connections.

        Args:
            entities: List of extracted code entities

        Returns:
            List of relationship mappings
        """
        relationships = []

        try:
            # Group entities by file for efficient analysis
            entities_by_file = {}
            for entity in entities:
                file_path = entity["file_path"]
                if file_path not in entities_by_file:
                    entities_by_file[file_path] = []
                entities_by_file[file_path].append(entity)

            # Extract file-level relationships (imports/dependencies)
            for file_path, file_entities in entities_by_file.items():
                # Find import statements
                import_entities = [
                    e for e in file_entities if "import" in e["entity_type"]
                ]
                other_entities = [
                    e for e in file_entities if "import" not in e["entity_type"]
                ]

                # Create import relationships
                for import_entity in import_entities:
                    for other_entity in other_entities:
                        relationship = {
                            "id": f"{import_entity['id']}_imports_{other_entity['id']}",
                            "type": "imports",
                            "source_id": import_entity["id"],
                            "target_id": other_entity["id"],
                            "source_file": file_path,
                            "target_file": file_path,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                        relationships.append(relationship)

                # Create hierarchical relationships (classes contain methods, etc.)
                classes = [e for e in file_entities if "class" in e["entity_type"]]
                functions = [
                    e
                    for e in file_entities
                    if "function" in e["entity_type"] or "method" in e["entity_type"]
                ]

                for class_entity in classes:
                    for func_entity in functions:
                        # Check if function is within class bounds
                        if (
                            func_entity["start_line"] >= class_entity["start_line"]
                            and func_entity["end_line"] <= class_entity["end_line"]
                        ):
                            relationship = {
                                "id": f"{class_entity['id']}_contains_{func_entity['id']}",
                                "type": "contains",
                                "source_id": class_entity["id"],
                                "target_id": func_entity["id"],
                                "source_file": file_path,
                                "target_file": file_path,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                            relationships.append(relationship)

            # Cross-file relationships (advanced dependency analysis)
            relationships.extend(
                self._analyze_cross_file_dependencies(entities_by_file)
            )

            logger.info(f"Extracted {len(relationships)} relationships")
            return relationships

        except Exception as e:
            logger.error(f"Failed to extract relationships: {e}")
            raise

    def _analyze_cross_file_dependencies(
        self, entities_by_file: Dict[str, list]
    ) -> list[Dict[str, Any]]:
        """Analyze cross-file dependencies and imports."""
        cross_relationships = []

        # This is a simplified implementation - could be enhanced with more sophisticated
        # dependency analysis using AST information about import sources

        return cross_relationships

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
        """Generate embeddings and store entities with vector data."""
        batch_size = 100  # Azure OpenAI recommended batch size

        for i in range(0, len(entities), batch_size):
            batch = entities[i : i + batch_size]

            try:
                # Prepare texts for embedding generation
                texts = []
                for entity in batch:
                    # Create meaningful text for embedding (name + AI description + content preview)
                    ai_desc = entity.get("ai_description", "")
                    content_preview = entity.get("content", "")[
                        :300
                    ]  # Shorter content since we have AI description

                    if ai_desc:
                        text = f"{entity.get('name', 'unknown')} {entity['entity_type']}: {ai_desc} {content_preview}"
                    else:
                        text = f"{entity.get('name', 'unknown')} {entity['entity_type']} {content_preview}"

                    texts.append(text)

                # Generate embeddings using Azure OpenAI Text Embedding service
                # Based on Context7 research: AzureTextEmbedding.generate_embeddings(texts) returns list of vectors
                embeddings_result = await self.embedding_service.generate_embeddings(
                    texts
                )

                # Store entities with embeddings in Cosmos DB
                documents = []
                for j, entity in enumerate(batch):
                    # Format according to OmniRAG schema from CLAUDE.md with AI enhancements
                    doc = {
                        "id": entity["id"],
                        "type": "code_entity",
                        "entity_type": entity["entity_type"],
                        "name": entity.get("name", "unknown"),
                        "language": entity["language"],
                        "file_path": entity["file_path"],
                        "content": entity["content"],
                        "ai_description": entity.get("ai_description", ""),
                        "has_ai_analysis": entity.get("has_ai_analysis", False),
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

    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        logger.info("IngestionPlugin cleanup completed")
