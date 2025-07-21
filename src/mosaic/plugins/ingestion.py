"""
IngestionPlugin for Mosaic MCP Tool

Implements Phase 1 Foundation: Repository Access, AST Parsing, and Entity Extraction.
This addresses the critical code ingestion pipeline gap identified in the implementation analysis.

Key Components:
- GitPython-based repository cloning and access
- tree-sitter multi-language AST parsing
- Entity extraction and graph modeling
- Azure Cosmos DB knowledge population using OmniRAG pattern

Based on Context7 research validation of 2025 best practices.
"""

import logging
import os
import tempfile
import shutil
from typing import List, Dict, Any, Optional, Generator
from pathlib import Path
from datetime import datetime
import hashlib

# GitPython for repository access
from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError

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
from tree_sitter import Language, Parser, Node

# Semantic Kernel integration
from semantic_kernel.plugin_definition import sk_function, sk_function_context_parameter

# Azure services
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding

from ..config.settings import MosaicSettings
from ..models.base import CodeEntity, EntityType


logger = logging.getLogger(__name__)


class IngestionPlugin:
    """
    Semantic Kernel plugin for code ingestion and knowledge graph population.

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
        self.embedding_service = AzureTextEmbedding(
            deployment_name=self.settings.azure_openai_text_embedding_deployment_name,
            endpoint=self.settings.azure_openai_endpoint,
            service_id="ingestion_embedding",
        )

    @sk_function(
        description="Ingest code repository using GitPython and populate knowledge graph",
        name="ingest_repository",
    )
    @sk_function_context_parameter(
        name="repository_url", description="Git repository URL to ingest", type_="str"
    )
    @sk_function_context_parameter(
        name="branch", description="Git branch to checkout (default: main)", type_="str"
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
        """Clone repository using GitPython with Context7 validated patterns."""
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="mosaic_ingestion_")

            # Clone repository using GitPython
            logger.info(f"Cloning repository to: {temp_dir}")

            # Use shallow clone for efficiency (Context7 best practice)
            repo = Repo.clone_from(
                repository_url,
                temp_dir,
                branch=branch,
                depth=1,  # Shallow clone
                single_branch=True,
            )

            logger.info(
                f"Repository cloned successfully. HEAD: {repo.head.commit.hexsha}"
            )
            return temp_dir

        except (GitCommandError, InvalidGitRepositoryError) as e:
            logger.error(f"Git operation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Repository cloning failed: {e}")
            raise

    async def _parse_repository(self, repo_path: str) -> List[CodeEntity]:
        """Parse repository files using tree-sitter multi-language AST parsing."""
        code_entities = []

        try:
            repo_path_obj = Path(repo_path)

            # Walk through all files
            for file_path in self._get_code_files(repo_path_obj):
                try:
                    # Determine language from file extension
                    language = self._detect_language(file_path)
                    if not language:
                        continue

                    # Parse file using appropriate tree-sitter parser
                    entities = await self._parse_file(file_path, language)
                    code_entities.extend(entities)

                except Exception as e:
                    logger.warning(f"Failed to parse {file_path}: {e}")
                    continue

            logger.info(f"Parsed {len(code_entities)} entities from repository")
            return code_entities

        except Exception as e:
            logger.error(f"Repository parsing failed: {e}")
            raise

    def _get_code_files(self, repo_path: Path) -> Generator[Path, None, None]:
        """Get all code files from repository using efficient traversal."""
        # Skip common non-code directories
        skip_dirs = {
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            "build",
            "dist",
            ".pytest_cache",
            ".mypy_cache",
            "target",
            ".next",
            ".nuxt",
            "coverage",
        }

        for file_path in repo_path.rglob("*"):
            if file_path.is_file():
                # Skip if in excluded directory
                if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                    continue

                # Check if it's a code file
                detected_lang = self._detect_language(file_path)
                if detected_lang:
                    yield file_path

    def _detect_language(self, file_path: Path) -> Optional[str]:
        """Detect programming language from file extension."""
        extension = file_path.suffix.lower()

        # Special handling for Razor/Blazor files which contain mixed content
        if extension in {".cshtml", ".razor"}:
            # These files contain C# code embedded in HTML-like markup
            return "csharp"

        # Check standard extensions
        for language, extensions in self.language_extensions.items():
            if extension in extensions:
                return language

        return None

    async def _parse_file(self, file_path: Path, language: str) -> List[CodeEntity]:
        """Parse individual file using tree-sitter AST parsing."""
        entities = []

        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Parse using tree-sitter
            parser = self.parsers[language]
            tree = parser.parse(bytes(content, "utf8"))

            # Extract entities from AST
            entities = self._extract_entities_from_ast(
                tree.root_node, file_path, language, content
            )

        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            raise

        return entities

    def _extract_entities_from_ast(
        self, node: Node, file_path: Path, language: str, content: str
    ) -> List[CodeEntity]:
        """Extract code entities from AST using language-specific patterns."""
        entities = []

        # Language-specific entity extraction
        if language == "python":
            entities.extend(self._extract_python_entities(node, file_path, content))
        elif language in ["javascript", "typescript"]:
            entities.extend(self._extract_js_ts_entities(node, file_path, content))
        elif language == "java":
            entities.extend(self._extract_java_entities(node, file_path, content))
        elif language == "go":
            entities.extend(self._extract_go_entities(node, file_path, content))
        elif language == "rust":
            entities.extend(self._extract_rust_entities(node, file_path, content))
        elif language in ["c", "cpp"]:
            entities.extend(self._extract_c_cpp_entities(node, file_path, content))
        elif language == "csharp":
            entities.extend(self._extract_csharp_entities(node, file_path, content))
        elif language == "html":
            entities.extend(self._extract_html_entities(node, file_path, content))
        elif language == "css":
            entities.extend(self._extract_css_entities(node, file_path, content))

        return entities

    def _extract_python_entities(
        self, node: Node, file_path: Path, content: str
    ) -> List[CodeEntity]:
        """Extract Python-specific entities using tree-sitter queries."""
        entities = []

        # Recursive traversal for Python entities
        for child in node.children:
            if child.type == "function_definition":
                entity = self._create_function_entity(
                    child, file_path, "python", content
                )
                entities.append(entity)
            elif child.type == "class_definition":
                entity = self._create_class_entity(child, file_path, "python", content)
                entities.append(entity)
            elif (
                child.type == "import_statement"
                or child.type == "import_from_statement"
            ):
                entity = self._create_import_entity(child, file_path, "python", content)
                entities.append(entity)

            # Recursively process children
            entities.extend(self._extract_python_entities(child, file_path, content))

        return entities

    def _extract_js_ts_entities(
        self, node: Node, file_path: Path, content: str
    ) -> List[CodeEntity]:
        """Extract JavaScript/TypeScript entities."""
        entities = []

        for child in node.children:
            if child.type in ["function_declaration", "method_definition"]:
                entity = self._create_function_entity(
                    child, file_path, "javascript", content
                )
                entities.append(entity)
            elif child.type == "class_declaration":
                entity = self._create_class_entity(
                    child, file_path, "javascript", content
                )
                entities.append(entity)
            elif child.type in ["import_statement", "import_declaration"]:
                entity = self._create_import_entity(
                    child, file_path, "javascript", content
                )
                entities.append(entity)

            entities.extend(self._extract_js_ts_entities(child, file_path, content))

        return entities

    def _extract_java_entities(
        self, node: Node, file_path: Path, content: str
    ) -> List[CodeEntity]:
        """Extract Java entities."""
        entities = []

        for child in node.children:
            if child.type == "method_declaration":
                entity = self._create_function_entity(child, file_path, "java", content)
                entities.append(entity)
            elif child.type == "class_declaration":
                entity = self._create_class_entity(child, file_path, "java", content)
                entities.append(entity)
            elif child.type == "import_declaration":
                entity = self._create_import_entity(child, file_path, "java", content)
                entities.append(entity)

            entities.extend(self._extract_java_entities(child, file_path, content))

        return entities

    def _extract_go_entities(
        self, node: Node, file_path: Path, content: str
    ) -> List[CodeEntity]:
        """Extract Go entities."""
        entities = []

        for child in node.children:
            if child.type == "function_declaration":
                entity = self._create_function_entity(child, file_path, "go", content)
                entities.append(entity)
            elif child.type in ["type_declaration", "struct_type"]:
                entity = self._create_class_entity(child, file_path, "go", content)
                entities.append(entity)
            elif child.type == "import_declaration":
                entity = self._create_import_entity(child, file_path, "go", content)
                entities.append(entity)

            entities.extend(self._extract_go_entities(child, file_path, content))

        return entities

    def _extract_rust_entities(
        self, node: Node, file_path: Path, content: str
    ) -> List[CodeEntity]:
        """Extract Rust entities."""
        entities = []

        for child in node.children:
            if child.type in ["function_item", "impl_item"]:
                entity = self._create_function_entity(child, file_path, "rust", content)
                entities.append(entity)
            elif child.type in ["struct_item", "enum_item"]:
                entity = self._create_class_entity(child, file_path, "rust", content)
                entities.append(entity)
            elif child.type == "use_declaration":
                entity = self._create_import_entity(child, file_path, "rust", content)
                entities.append(entity)

            entities.extend(self._extract_rust_entities(child, file_path, content))

        return entities

    def _extract_c_cpp_entities(
        self, node: Node, file_path: Path, content: str
    ) -> List[CodeEntity]:
        """Extract C/C++ entities."""
        entities = []

        for child in node.children:
            if child.type == "function_definition":
                entity = self._create_function_entity(child, file_path, "c", content)
                entities.append(entity)
            elif child.type in ["struct_specifier", "class_specifier"]:
                entity = self._create_class_entity(child, file_path, "c", content)
                entities.append(entity)
            elif child.type == "preproc_include":
                entity = self._create_import_entity(child, file_path, "c", content)
                entities.append(entity)

            entities.extend(self._extract_c_cpp_entities(child, file_path, content))

        return entities

    def _extract_csharp_entities(
        self, node: Node, file_path: Path, content: str
    ) -> List[CodeEntity]:
        """Extract C# entities."""
        entities = []

        for child in node.children:
            if child.type == "method_declaration":
                entity = self._create_function_entity(
                    child, file_path, "csharp", content
                )
                entities.append(entity)
            elif child.type in [
                "class_declaration",
                "struct_declaration",
                "interface_declaration",
            ]:
                entity = self._create_class_entity(child, file_path, "csharp", content)
                entities.append(entity)
            elif child.type == "using_directive":
                entity = self._create_import_entity(child, file_path, "csharp", content)
                entities.append(entity)
            elif child.type == "namespace_declaration":
                # Process namespace contents
                entities.extend(
                    self._extract_csharp_entities(child, file_path, content)
                )

            # Recursively process other children
            entities.extend(self._extract_csharp_entities(child, file_path, content))

        return entities

    def _extract_html_entities(
        self, node: Node, file_path: Path, content: str
    ) -> List[CodeEntity]:
        """Extract HTML entities (elements, scripts, styles)."""
        entities = []

        for child in node.children:
            if child.type == "element":
                # Extract HTML elements (div, component tags, etc.)
                entity = self._create_html_element_entity(child, file_path, content)
                entities.append(entity)
            elif child.type == "script_element":
                # Extract embedded JavaScript
                entity = self._create_html_script_entity(child, file_path, content)
                entities.append(entity)
            elif child.type == "style_element":
                # Extract embedded CSS
                entity = self._create_html_style_entity(child, file_path, content)
                entities.append(entity)

            # Recursively process children
            entities.extend(self._extract_html_entities(child, file_path, content))

        return entities

    def _extract_css_entities(
        self, node: Node, file_path: Path, content: str
    ) -> List[CodeEntity]:
        """Extract CSS entities (rules, selectors, properties)."""
        entities = []

        for child in node.children:
            if child.type == "rule_set":
                # CSS rules (selectors + declarations)
                entity = self._create_css_rule_entity(child, file_path, content)
                entities.append(entity)
            elif child.type == "at_rule":
                # CSS at-rules (@media, @import, @keyframes, etc.)
                entity = self._create_css_at_rule_entity(child, file_path, content)
                entities.append(entity)
            elif child.type == "import_statement":
                # CSS imports
                entity = self._create_import_entity(child, file_path, "css", content)
                entities.append(entity)

            # Recursively process children
            entities.extend(self._extract_css_entities(child, file_path, content))

        return entities

    def _create_html_element_entity(
        self, node: Node, file_path: Path, content: str
    ) -> CodeEntity:
        """Create HTML element entity."""
        # Extract tag name
        tag_name = self._extract_html_tag_name(node, content)

        start_byte = node.start_byte
        end_byte = node.end_byte
        source_code = content[start_byte:end_byte]

        entity_id = self._generate_entity_id(str(file_path), tag_name, "html_element")

        return CodeEntity(
            id=entity_id,
            name=tag_name,
            type=EntityType.MODULE,  # Using MODULE for HTML elements
            language="html",
            file_path=str(file_path),
            source_code=source_code[:500],  # Limit HTML content
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            metadata={
                "ast_type": node.type,
                "tag_name": tag_name,
                "byte_range": [start_byte, end_byte],
            },
        )

    def _create_html_script_entity(
        self, node: Node, file_path: Path, content: str
    ) -> CodeEntity:
        """Create HTML script element entity."""
        start_byte = node.start_byte
        end_byte = node.end_byte
        source_code = content[start_byte:end_byte]

        entity_id = self._generate_entity_id(str(file_path), "script", "html_script")

        return CodeEntity(
            id=entity_id,
            name="<script>",
            type=EntityType.FUNCTION,  # Scripts contain executable code
            language="html",
            file_path=str(file_path),
            source_code=source_code[:500],
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            metadata={
                "ast_type": node.type,
                "element_type": "script",
                "byte_range": [start_byte, end_byte],
            },
        )

    def _create_html_style_entity(
        self, node: Node, file_path: Path, content: str
    ) -> CodeEntity:
        """Create HTML style element entity."""
        start_byte = node.start_byte
        end_byte = node.end_byte
        source_code = content[start_byte:end_byte]

        entity_id = self._generate_entity_id(str(file_path), "style", "html_style")

        return CodeEntity(
            id=entity_id,
            name="<style>",
            type=EntityType.MODULE,  # Styles define presentation rules
            language="html",
            file_path=str(file_path),
            source_code=source_code[:500],
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            metadata={
                "ast_type": node.type,
                "element_type": "style",
                "byte_range": [start_byte, end_byte],
            },
        )

    def _create_css_rule_entity(
        self, node: Node, file_path: Path, content: str
    ) -> CodeEntity:
        """Create CSS rule entity."""
        # Extract selector
        selector = self._extract_css_selector(node, content)

        start_byte = node.start_byte
        end_byte = node.end_byte
        source_code = content[start_byte:end_byte]

        entity_id = self._generate_entity_id(str(file_path), selector, "css_rule")

        return CodeEntity(
            id=entity_id,
            name=selector,
            type=EntityType.FUNCTION,  # CSS rules define styling behavior
            language="css",
            file_path=str(file_path),
            source_code=source_code,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            metadata={
                "ast_type": node.type,
                "selector": selector,
                "byte_range": [start_byte, end_byte],
            },
        )

    def _create_css_at_rule_entity(
        self, node: Node, file_path: Path, content: str
    ) -> CodeEntity:
        """Create CSS at-rule entity."""
        # Extract at-rule name
        at_rule_name = self._extract_css_at_rule_name(node, content)

        start_byte = node.start_byte
        end_byte = node.end_byte
        source_code = content[start_byte:end_byte]

        entity_id = self._generate_entity_id(
            str(file_path), at_rule_name, "css_at_rule"
        )

        return CodeEntity(
            id=entity_id,
            name=at_rule_name,
            type=EntityType.IMPORT,  # At-rules are directives
            language="css",
            file_path=str(file_path),
            source_code=source_code,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            metadata={
                "ast_type": node.type,
                "at_rule": at_rule_name,
                "byte_range": [start_byte, end_byte],
            },
        )

    def _extract_html_tag_name(self, node: Node, content: str) -> str:
        """Extract HTML tag name from element node."""
        # Look for start_tag child
        for child in node.children:
            if child.type == "start_tag":
                # Get tag name from start tag
                for tag_child in child.children:
                    if tag_child.type == "tag_name":
                        return tag_child.text.decode("utf-8")

        # Fallback to extracting from content
        start_text = content[node.start_byte : node.start_byte + 50]
        if "<" in start_text:
            try:
                tag = start_text.split("<")[1].split()[0].split(">")[0]
                return tag
            except (IndexError, AttributeError):
                pass

        return "unknown"

    def _extract_css_selector(self, node: Node, content: str) -> str:
        """Extract CSS selector from rule node."""
        # Look for selectors child
        for child in node.children:
            if child.type == "selectors":
                selector_text = content[child.start_byte : child.end_byte]
                return selector_text.strip()

        # Fallback to first part of rule
        rule_text = content[node.start_byte : node.end_byte]
        if "{" in rule_text:
            selector = rule_text.split("{")[0].strip()
            return selector[:100]  # Limit length

        return "unknown"

    def _extract_css_at_rule_name(self, node: Node, content: str) -> str:
        """Extract CSS at-rule name."""
        at_rule_text = content[node.start_byte : node.end_byte]

        # Extract @rule-name
        if at_rule_text.startswith("@"):
            parts = at_rule_text.split()
            if parts:
                return parts[0]  # @media, @import, etc.

        return "@unknown"

    def _create_function_entity(
        self, node: Node, file_path: Path, language: str, content: str
    ) -> CodeEntity:
        """Create function entity from AST node."""
        # Extract function name
        name = self._extract_node_name(node, language)

        # Get source code
        start_byte = node.start_byte
        end_byte = node.end_byte
        source_code = content[start_byte:end_byte]

        # Create unique ID
        entity_id = self._generate_entity_id(str(file_path), name, "function")

        return CodeEntity(
            id=entity_id,
            name=name,
            type="function",
            language=language,
            file_path=str(file_path),
            source_code=source_code,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            metadata={"ast_type": node.type, "byte_range": [start_byte, end_byte]},
        )

    def _create_class_entity(
        self, node: Node, file_path: Path, language: str, content: str
    ) -> CodeEntity:
        """Create class entity from AST node."""
        name = self._extract_node_name(node, language)

        start_byte = node.start_byte
        end_byte = node.end_byte
        source_code = content[start_byte:end_byte]

        entity_id = self._generate_entity_id(str(file_path), name, "class")

        return CodeEntity(
            id=entity_id,
            name=name,
            type="class",
            language=language,
            file_path=str(file_path),
            source_code=source_code,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            metadata={"ast_type": node.type, "byte_range": [start_byte, end_byte]},
        )

    def _create_import_entity(
        self, node: Node, file_path: Path, language: str, content: str
    ) -> CodeEntity:
        """Create import entity from AST node."""
        name = self._extract_import_name(node, language, content)

        start_byte = node.start_byte
        end_byte = node.end_byte
        source_code = content[start_byte:end_byte]

        entity_id = self._generate_entity_id(str(file_path), name, "import")

        return CodeEntity(
            id=entity_id,
            name=name,
            type="import",
            language=language,
            file_path=str(file_path),
            source_code=source_code,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            metadata={"ast_type": node.type, "byte_range": [start_byte, end_byte]},
        )

    def _extract_node_name(self, node: Node, language: str) -> str:
        """Extract name from AST node based on language."""
        # Look for identifier nodes in children
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")

        # Fallback to node text if no identifier found
        return node.text.decode("utf-8")[:100]  # Limit length

    def _extract_import_name(self, node: Node, language: str, content: str) -> str:
        """Extract import name from import statement."""
        # Get the full import statement text
        import_text = content[node.start_byte : node.end_byte]

        # Simple extraction - can be enhanced per language
        import_text = import_text.strip().replace("\n", " ")

        return import_text[:200]  # Limit length

    def _generate_entity_id(self, file_path: str, name: str, entity_type: str) -> str:
        """Generate unique entity ID."""
        content = f"{file_path}:{entity_type}:{name}"
        return hashlib.md5(content.encode()).hexdigest()

    async def _extract_relationships(
        self, code_entities: List[CodeEntity]
    ) -> List[Dict[str, Any]]:
        """Extract relationships between code entities."""
        relationships = []

        # Build lookup dictionaries for efficient relationship detection
        entities_by_file = {}
        entities_by_name = {}

        for entity in code_entities:
            # Group by file
            if entity.file_path not in entities_by_file:
                entities_by_file[entity.file_path] = []
            entities_by_file[entity.file_path].append(entity)

            # Index by name for cross-references
            entities_by_name[entity.name] = entity

        # Extract file-level relationships
        for file_path, entities in entities_by_file.items():
            for entity in entities:
                if entity.type == "import":
                    # Create import relationships
                    for other_entity in entities:
                        if (
                            other_entity.type in ["function", "class"]
                            and other_entity != entity
                        ):
                            relationships.append(
                                {
                                    "source_id": entity.id,
                                    "target_id": other_entity.id,
                                    "relationship_type": "imports",
                                    "metadata": {
                                        "file_path": file_path,
                                        "import_statement": entity.source_code,
                                    },
                                }
                            )

        logger.info(f"Extracted {len(relationships)} relationships")
        return relationships

    async def _populate_knowledge_base(
        self, code_entities: List[CodeEntity], relationships: List[Dict[str, Any]]
    ) -> None:
        """Populate Cosmos DB knowledge base using OmniRAG pattern."""
        try:
            # Process entities in batches for efficiency
            batch_size = 10

            for i in range(0, len(code_entities), batch_size):
                batch = code_entities[i : i + batch_size]
                await self._process_entity_batch(batch)

            logger.info(f"Populated knowledge base with {len(code_entities)} entities")

        except Exception as e:
            logger.error(f"Knowledge base population failed: {e}")
            raise

    async def _process_entity_batch(self, entities: List[CodeEntity]) -> None:
        """Process a batch of entities with embeddings and storage."""
        for entity in entities:
            try:
                # Generate embedding for entity
                embedding = await self._generate_embedding(entity.source_code)

                # Create document for storage
                document = {
                    "id": entity.id,
                    "type": "code_entity",
                    "entity_type": entity.type,
                    "name": entity.name,
                    "language": entity.language,
                    "file_path": entity.file_path,
                    "content": entity.source_code,
                    "start_line": entity.start_line,
                    "end_line": entity.end_line,
                    "embedding": embedding,
                    "metadata": entity.metadata,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                # Upsert to Cosmos DB
                self.knowledge_container.upsert_item(document)

            except Exception as e:
                logger.warning(f"Failed to process entity {entity.id}: {e}")
                continue

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Azure embedding service."""
        try:
            result = await self.embedding_service.generate_embeddings([text])
            return result[0]

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    async def get_status(self) -> Dict[str, Any]:
        """Get plugin status information."""
        return {
            "status": "active",
            "supported_languages": list(self.languages.keys()),
            "supported_extensions": {
                lang: list(exts) for lang, exts in self.language_extensions.items()
            },
            "total_languages": len(self.languages),
            "cosmos_connected": self.cosmos_client is not None,
            "embedding_service_connected": self.embedding_service is not None,
        }

    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        logger.info("IngestionPlugin cleanup completed")
