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

# GitPython for repository access
import git
import tempfile

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
from semantic_kernel.connectors.ai.open_ai import (
    AzureTextEmbedding,
    AzureChatCompletion,
)
from semantic_kernel import Kernel
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.functions import KernelArguments

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
        self.embedding_service = AzureTextEmbedding(
            deployment_name=self.settings.azure_openai_text_embedding_deployment_name,
            endpoint=self.settings.azure_openai_endpoint,
            service_id="ingestion_embedding",
        )

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
        Clone repository using GitPython with Azure-validated security patterns.

        Based on Context7 research: Uses secure token authentication and shallow cloning
        for optimal performance.

        Args:
            repository_url: Git repository URL
            branch: Branch to clone

        Returns:
            Path to cloned repository
        """
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="mosaic_repo_")

            # Configure Git authentication for Azure DevOps/GitHub
            # Use environment variables for secure token-based auth
            env_vars = {}
            if os.getenv("GIT_TOKEN"):
                # For Azure DevOps Personal Access Tokens
                env_vars["GIT_HTTP_EXTRAHEADER"] = (
                    f"AUTHORIZATION: basic {os.getenv('GIT_TOKEN')}"
                )

            logger.info(
                f"Cloning repository {repository_url} branch {branch} to {temp_dir}"
            )

            # Shallow clone for performance (Context7 validated approach)
            git.Repo.clone_from(
                repository_url,
                temp_dir,
                branch=branch,
                depth=1,  # Shallow clone for performance
                env=env_vars or None,
            )
            logger.info(f"Successfully cloned repository to {temp_dir}")
            return temp_dir

        except Exception as e:
            logger.error(f"Failed to clone repository {repository_url}: {e}")
            # Cleanup on failure
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
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
        """Parse individual file using tree-sitter."""
        entities = []

        try:
            # Read file content
            with open(file_path, "rb") as f:
                content = f.read()

            # Parse with appropriate parser
            parser = self.parsers.get(language)
            if not parser:
                logger.warning(f"No parser available for language: {language}")
                return entities

            tree = parser.parse(content)

            # Extract entities based on language
            entities = self._extract_entities_from_tree(
                tree.root_node, file_path, language, content
            )

            return entities

        except Exception as e:
            logger.warning(f"Failed to parse file {file_path}: {e}")
            return entities

    def _extract_entities_from_tree(
        self, node, file_path: Path, language: str, content: bytes
    ) -> list[Dict[str, Any]]:
        """Extract code entities from parsed tree."""
        entities = []

        # Define entity types per language (Context7 validated patterns)
        entity_types = {
            "python": [
                "function_definition",
                "class_definition",
                "import_statement",
                "import_from_statement",
            ],
            "javascript": [
                "function_declaration",
                "function_expression",
                "class_declaration",
                "import_statement",
            ],
            "typescript": [
                "function_declaration",
                "function_expression",
                "class_declaration",
                "import_statement",
            ],
            "java": [
                "method_declaration",
                "class_declaration",
                "interface_declaration",
                "import_declaration",
            ],
            "go": [
                "function_declaration",
                "method_declaration",
                "type_declaration",
                "import_declaration",
            ],
            "rust": ["function_item", "struct_item", "impl_item", "use_declaration"],
            "c": ["function_definition", "struct_specifier", "preproc_include"],
            "cpp": [
                "function_definition",
                "class_specifier",
                "struct_specifier",
                "preproc_include",
            ],
            "csharp": [
                "method_declaration",
                "class_declaration",
                "interface_declaration",
                "using_directive",
            ],
            "html": ["element", "doctype"],
            "css": ["rule_set", "at_rule"],
        }

        target_types = entity_types.get(language, [])

        def traverse_node(node):
            """Recursively traverse tree nodes."""
            if node.type in target_types:
                # Extract entity information
                entity = {
                    "id": f"{file_path.name}_{node.start_point[0]}_{node.start_point[1]}",
                    "type": "code_entity",
                    "entity_type": node.type,
                    "language": language,
                    "file_path": str(file_path),
                    "content": content[node.start_byte : node.end_byte].decode(
                        "utf-8", errors="ignore"
                    ),
                    "start_line": node.start_point[0],
                    "end_line": node.end_point[0],
                    "timestamp": datetime.utcnow().isoformat(),
                }

                # Extract entity name if possible
                name_node = self._find_name_node(node, language)
                if name_node:
                    entity["name"] = content[
                        name_node.start_byte : name_node.end_byte
                    ].decode("utf-8", errors="ignore")

                entities.append(entity)

            # Recursively process children
            for child in node.children:
                traverse_node(child)

        traverse_node(node)
        return entities

    def _find_name_node(self, node, language: str):
        """Find the name node within a declaration node."""
        # Language-specific name extraction patterns
        name_fields = {
            "python": "name",
            "javascript": "name",
            "typescript": "name",
            "java": "name",
            "go": "name",
            "rust": "name",
            "c": "declarator",
            "cpp": "declarator",
            "csharp": "name",
        }

        name_field = name_fields.get(language)
        if name_field and hasattr(node, "child_by_field_name"):
            return node.child_by_field_name(name_field)

        # Fallback: look for identifier nodes
        for child in node.children:
            if child.type == "identifier":
                return child

        return None

    async def _enhance_entities_with_ai_descriptions(
        self, entities: list[Dict[str, Any]]
    ) -> list[Dict[str, Any]]:
        """
        Enhance code entities with AI-generated descriptions using Semantic Kernel.

        Generates technical descriptions for functions, classes, and other significant
        code entities to improve searchability and understanding.

        Args:
            entities: List of extracted code entities

        Returns:
            List of entities enhanced with AI descriptions
        """
        enhanced_entities = []

        # Define entity types that benefit from AI description
        describable_types = [
            "function_definition",
            "function_declaration",
            "function_expression",
            "method_declaration",
            "class_definition",
            "class_declaration",
            "interface_declaration",
            "struct_item",
            "impl_item",
        ]

        for entity in entities:
            # Skip if not a describable entity type
            if entity["entity_type"] not in describable_types:
                enhanced_entities.append(entity)
                continue

            # Skip if content is too short/trivial
            if len(entity.get("content", "")) < 50:
                enhanced_entities.append(entity)
                continue

            try:
                # Generate AI description using Semantic Kernel
                description = await self._generate_entity_description(entity)

                # Add description to entity
                enhanced_entity = entity.copy()
                enhanced_entity["ai_description"] = description
                enhanced_entity["has_ai_analysis"] = True

                enhanced_entities.append(enhanced_entity)

            except Exception as e:
                logger.warning(
                    f"Failed to generate AI description for entity {entity.get('id', 'unknown')}: {e}"
                )
                # Include original entity without description rather than failing completely
                enhanced_entities.append(entity)

        logger.info(
            f"Enhanced {sum(1 for e in enhanced_entities if e.get('has_ai_analysis', False))} entities with AI descriptions"
        )
        return enhanced_entities

    async def _generate_entity_description(self, entity: Dict[str, Any]) -> str:
        """Generate AI description for a single code entity."""
        try:
            # Prepare arguments for Semantic Kernel function
            arguments = KernelArguments(
                entity_type=entity["entity_type"],
                name=entity.get("name", "unknown"),
                language=entity["language"],
                content=entity["content"][:1000],  # Limit content to avoid token limits
            )

            # Call the Semantic Kernel function
            result = await self.kernel.invoke(
                plugin_name="CodeAnalysis",
                function_name="describe_entity",
                arguments=arguments,
            )

            # Extract and clean the description
            description = str(result).strip()

            # Fallback if description is empty or too generic
            if not description or len(description) < 10:
                description = f"A {entity['entity_type']} named {entity.get('name', 'unknown')} in {entity['language']}"

            return description

        except Exception as e:
            logger.error(f"Failed to generate description via Semantic Kernel: {e}")
            # Return a basic fallback description
            return f"A {entity['entity_type']} named {entity.get('name', 'unknown')} in {entity['language']}"

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

                # Generate embeddings using Azure OpenAI
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
