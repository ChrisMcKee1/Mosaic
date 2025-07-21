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

# Azure services (no Semantic Kernel - standalone service)
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding

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

    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        logger.info("IngestionPlugin cleanup completed")
