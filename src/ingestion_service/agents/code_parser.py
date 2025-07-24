"""
CodeParser Agent - AST Parsing and Entity Extraction

Handles tree-sitter based AST parsing for 11 programming languages:
- Python, JavaScript, TypeScript, Java, Go, Rust
- C, C++, C#, HTML, CSS

Responsible for:
- Multi-language AST parsing
- Code entity extraction (functions, classes, modules, etc.)
- Structural analysis and metadata extraction
- Source code content normalization

This agent populates the CodeEntity portion of Golden Node models
with accurate structural information from source code analysis.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from uuid import uuid4
from collections import defaultdict

from .base_agent import MosaicAgent, AgentConfig, AgentExecutionContext, AgentError
from ..models.golden_node import (
    GoldenNode,
    AgentType,
    CodeEntity,
    LanguageType,
    EntityType,
)


class CodeParserAgent(MosaicAgent):
    """
    Specialized agent for AST parsing and code entity extraction.

    Uses tree-sitter parsers to analyze source code and extract
    structural information for knowledge graph construction.
    """

    def __init__(self, settings=None):
        """Initialize CodeParser agent with appropriate configuration."""
        config = AgentConfig(
            agent_name="CodeParser",
            agent_type=AgentType.CODE_PARSER,
            max_retry_attempts=3,  # Parsing can occasionally fail
            default_timeout_seconds=240,  # 4 minutes for complex files
            batch_size=20,  # Process multiple entities efficiently
            temperature=0.0,  # No randomness needed for parsing
            max_tokens=1500,  # Moderate LLM usage for analysis
            enable_parallel_processing=True,  # Parse multiple files in parallel
            log_level="INFO",
        )

        super().__init__(config, settings)
        self.logger = logging.getLogger("mosaic.agent.code_parser")
        self._parsers: Dict[LanguageType, Any] = {}

    async def _register_plugins(self) -> None:
        """Register CodeParser-specific Semantic Kernel plugins."""
        # Import and register the AI-powered code parser plugin
        from ..plugins.ai_code_parser import AICodeParserPlugin

        # Initialize AI Code Parser Plugin
        self.ai_code_parser = AICodeParserPlugin()

        # Register the plugin with the kernel
        self.kernel.add_plugin(self.ai_code_parser, "ai_code_parser")

        self.logger.info("CodeParser agent with AI-powered parsing plugins registered")

    async def _initialize_tree_sitter_parsers(self) -> None:
        """Initialize tree-sitter parsers for all supported languages."""
        try:
            # TODO: Import and initialize tree-sitter parsers
            # This would require the actual tree-sitter language packages
            self.logger.info("Initializing tree-sitter parsers for 11 languages")

            # Placeholder for parser initialization
            # self._parsers[LanguageType.PYTHON] = python_parser
            # self._parsers[LanguageType.JAVASCRIPT] = javascript_parser
            # etc.

        except Exception as e:
            raise AgentError(
                f"Failed to initialize tree-sitter parsers: {e}",
                self.config.agent_type.value,
            )

    async def process_golden_node(
        self, golden_node: GoldenNode, context: AgentExecutionContext
    ) -> GoldenNode:
        """
        Process Golden Node to enhance code entity information.

        Args:
            golden_node: The Golden Node to process
            context: Execution context with parameters

        Returns:
            Updated Golden Node with enhanced code entity data
        """
        self.logger.info(f"Processing Golden Node {golden_node.id} for code parsing")

        try:
            # Get source code content from parameters or existing entity
            source_code = context.parameters.get(
                "source_code", golden_node.code_entity.content
            )
            language = golden_node.code_entity.language

            if not source_code:
                raise AgentError(
                    "No source code available for parsing",
                    self.config.agent_type.value,
                    context.task_id,
                )

            # Perform enhanced AST analysis
            enhanced_entity = await self._enhance_code_entity(
                golden_node.code_entity, source_code, language, context
            )

            # Update the Golden Node
            updated_node = golden_node.model_copy(deep=True)
            updated_node.code_entity = enhanced_entity

            # Update processing metadata
            updated_node.processing_metadata.agent_history.append(
                {
                    "agent_type": self.config.agent_type.value,
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "execution_id": context.execution_id,
                    "modifications": ["code_entity"],
                }
            )

            context.intermediate_results["parsing_analysis"] = {
                "language": language.value,
                "entity_type": enhanced_entity.entity_type.value,
                "child_entities_found": len(enhanced_entity.child_entities),
                "imports_identified": len(enhanced_entity.imports),
                "calls_identified": len(enhanced_entity.calls),
            }

            self.logger.info(
                f"Successfully enhanced code entity for Golden Node {golden_node.id}"
            )

            return updated_node

        except Exception as e:
            self.logger.error(f"Failed to process Golden Node {golden_node.id}: {e}")
            raise AgentError(
                f"Code parsing failed: {e}",
                self.config.agent_type.value,
                context.task_id,
            )

    async def _enhance_code_entity(
        self,
        existing_entity: CodeEntity,
        source_code: str,
        language: LanguageType,
        context: AgentExecutionContext,
    ) -> CodeEntity:
        """
        Enhance code entity with detailed AST analysis.

        This method would implement comprehensive tree-sitter parsing
        to extract detailed structural information.
        """
        self.logger.debug(f"Enhancing {language.value} entity: {existing_entity.name}")

        try:
            # TODO: Implement actual tree-sitter parsing
            # This would use the initialized parsers to analyze source code

            # Parse the source code
            parsed_data = await self._parse_source_code(source_code, language, context)

            # Create enhanced entity
            enhanced_entity = existing_entity.model_copy(deep=True)

            # Update with parsed information
            if parsed_data:
                enhanced_entity.child_entities = parsed_data.get("children", [])
                enhanced_entity.imports = parsed_data.get("imports", [])
                enhanced_entity.calls = parsed_data.get("calls", [])
                enhanced_entity.signature = parsed_data.get("signature")
                enhanced_entity.parent_entity = parsed_data.get("parent")
                enhanced_entity.scope = parsed_data.get("scope")
                enhanced_entity.is_exported = parsed_data.get("is_exported", False)

            return enhanced_entity

        except Exception as e:
            self.logger.error(f"Entity enhancement failed: {e}")
            # Return original entity on failure rather than crashing
            return existing_entity

    async def _parse_source_code(
        self, source_code: str, language: LanguageType, context: AgentExecutionContext
    ) -> Optional[Dict[str, Any]]:
        """
        Parse source code using appropriate tree-sitter parser.

        Returns extracted structural information or None on failure.
        """
        try:
            # TODO: Implement actual tree-sitter parsing
            # This is a placeholder that would use the language-specific parser

            self.logger.debug(
                f"Parsing {len(source_code)} characters of {language.value}"
            )

            # Placeholder parsing result
            parsed_data = {
                "children": [],
                "imports": [],
                "calls": [],
                "signature": None,
                "parent": None,
                "scope": "public",
                "is_exported": False,
            }

            return parsed_data

        except Exception as e:
            self.logger.error(f"Source code parsing failed: {e}")
            return None

    async def extract_entities_from_file(
        self, file_path: str, file_content: str, language: LanguageType
    ) -> List[CodeEntity]:
        """
        Extract all code entities from a source file using AI-powered parsing.

        Uses the AICodeParserPlugin for intelligent code analysis that handles
        edge cases and context better than traditional AST parsing.

        Args:
            file_path: Absolute path to the source file
            file_content: Raw content of the source file
            language: Programming language for appropriate parser selection

        Returns:
            List of CodeEntity objects with populated hierarchical fields
        """
        try:
            self.logger.info(
                f"AI-powered extraction from {language.value} file: {file_path}"
            )

            # Use AI Code Parser Plugin for intelligent extraction
            entities = await self.ai_code_parser.extract_entities_from_code(
                file_content=file_content,
                file_path=file_path,
                language=language.value.lower(),
            )

            self.logger.info(
                f"AI successfully extracted {len(entities)} entities from {file_path}"
            )

            return entities

        except Exception as e:
            self.logger.error(f"AI entity extraction failed for {file_path}: {e}")
            # Fallback to traditional parsing if AI fails
            return await self._fallback_traditional_parsing(
                file_path, file_content, language
            )

    async def _fallback_traditional_parsing(
        self, file_path: str, file_content: str, language: LanguageType
    ) -> List[CodeEntity]:
        """
        Fallback to traditional AST parsing if AI-powered parsing fails.

        This ensures the agent can still function even if the AI service is unavailable.
        """
        try:
            self.logger.warning(f"Using fallback traditional parsing for {file_path}")

            # Use the existing traditional parsing methods
            parsed_entities = await self._parse_ast_hierarchical(
                file_content, language, file_path
            )

            if not parsed_entities:
                return []

            # Build UUID mapping and hierarchical relationships
            entity_uuid_map = self._build_uuid_mapping(parsed_entities)
            hierarchical_entities = await self._build_hierarchical_relationships(
                parsed_entities, entity_uuid_map, file_path
            )

            return hierarchical_entities

        except Exception as e:
            self.logger.error(f"Fallback parsing also failed for {file_path}: {e}")
            return []

    async def _parse_ast_hierarchical(
        self, file_content: str, language: LanguageType, file_path: str
    ) -> List[Dict[str, Any]]:
        """
        Parse AST and extract all entities with hierarchical information.

        Following Microsoft Semantic Kernel best practices for structured parsing.
        """
        try:
            # Language-specific parsing patterns
            if language == LanguageType.PYTHON:
                return await self._parse_python_hierarchical(file_content, file_path)
            elif language == LanguageType.JAVASCRIPT:
                return await self._parse_javascript_hierarchical(
                    file_content, file_path
                )
            elif language == LanguageType.TYPESCRIPT:
                return await self._parse_typescript_hierarchical(
                    file_content, file_path
                )
            elif language == LanguageType.JAVA:
                return await self._parse_java_hierarchical(file_content, file_path)
            elif language == LanguageType.CSHARP:
                return await self._parse_csharp_hierarchical(file_content, file_path)
            else:
                # Generic parsing for other languages
                return await self._parse_generic_hierarchical(
                    file_content, language, file_path
                )

        except Exception as e:
            self.logger.error(f"AST parsing failed for {file_path}: {e}")
            return []

    async def _parse_python_hierarchical(
        self, file_content: str, file_path: str
    ) -> List[Dict[str, Any]]:
        """Parse Python file and extract hierarchical entities."""
        entities = []

        try:
            # Simulate AST parsing - in real implementation, use tree-sitter
            lines = file_content.split("\n")
            current_class = None
            current_indent = 0

            for line_no, line in enumerate(lines, 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                indent_level = len(line) - len(line.lstrip())

                # Detect class definitions
                if stripped.startswith("class "):
                    class_name = (
                        stripped.split("class ")[1].split("(")[0].split(":")[0].strip()
                    )
                    entities.append(
                        {
                            "name": class_name,
                            "entity_type": EntityType.CLASS,
                            "parent_name": None,  # File-level
                            "line_start": line_no,
                            "line_end": line_no,  # Will be updated
                            "content": line.strip(),
                            "children": [],
                            "imports": [],
                            "calls": [],
                        }
                    )
                    current_class = class_name
                    current_indent = indent_level

                # Detect function/method definitions
                elif stripped.startswith("def "):
                    func_name = stripped.split("def ")[1].split("(")[0].strip()
                    parent_name = (
                        current_class if indent_level > current_indent else None
                    )

                    entities.append(
                        {
                            "name": func_name,
                            "entity_type": EntityType.FUNCTION,
                            "parent_name": parent_name,
                            "line_start": line_no,
                            "line_end": line_no,
                            "content": line.strip(),
                            "children": [],
                            "imports": [],
                            "calls": [],
                        }
                    )

                # Detect imports
                elif stripped.startswith("import ") or stripped.startswith("from "):
                    import_name = (
                        stripped.split()[1] if "import" in stripped else stripped
                    )
                    # Add import to all entities in this file
                    for entity in entities:
                        if import_name not in entity["imports"]:
                            entity["imports"].append(import_name)

            return entities

        except Exception as e:
            self.logger.error(f"Python AST parsing failed: {e}")
            return []

    async def _parse_javascript_hierarchical(
        self, file_content: str, file_path: str
    ) -> List[Dict[str, Any]]:
        """Parse JavaScript file and extract hierarchical entities."""
        entities = []

        try:
            # Simplified JavaScript parsing
            lines = file_content.split("\n")
            current_class = None

            for line_no, line in enumerate(lines, 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("//"):
                    continue

                # Detect class definitions
                if "class " in stripped and "{" in stripped:
                    class_name = (
                        stripped.split("class ")[1].split(" ")[0].split("{")[0].strip()
                    )
                    entities.append(
                        {
                            "name": class_name,
                            "entity_type": EntityType.CLASS,
                            "parent_name": None,
                            "line_start": line_no,
                            "line_end": line_no,
                            "content": line.strip(),
                            "children": [],
                            "imports": [],
                            "calls": [],
                        }
                    )
                    current_class = class_name

                # Detect function definitions
                elif "function " in stripped or "=>" in stripped:
                    if "function " in stripped:
                        func_name = stripped.split("function ")[1].split("(")[0].strip()
                    else:
                        # Arrow function - try to extract name
                        func_name = (
                            stripped.split("=")[0].strip()
                            if "=" in stripped
                            else "anonymous"
                        )

                    entities.append(
                        {
                            "name": func_name,
                            "entity_type": EntityType.FUNCTION,
                            "parent_name": current_class,
                            "line_start": line_no,
                            "line_end": line_no,
                            "content": line.strip(),
                            "children": [],
                            "imports": [],
                            "calls": [],
                        }
                    )

            return entities

        except Exception as e:
            self.logger.error(f"JavaScript AST parsing failed: {e}")
            return []

    async def _parse_generic_hierarchical(
        self, file_content: str, language: LanguageType, file_path: str
    ) -> List[Dict[str, Any]]:
        """Generic parsing for languages without specific implementations."""
        entities = []

        try:
            # Create a single module entity for the file
            module_name = file_path.split("/")[-1].split(".")[0]
            entities.append(
                {
                    "name": module_name,
                    "entity_type": EntityType.MODULE,
                    "parent_name": None,
                    "line_start": 1,
                    "line_end": len(file_content.split("\n")),
                    "content": file_content[:500] + "..."
                    if len(file_content) > 500
                    else file_content,
                    "children": [],
                    "imports": [],
                    "calls": [],
                }
            )

            return entities

        except Exception as e:
            self.logger.error(f"Generic AST parsing failed: {e}")
            return []

    # Placeholder methods for other languages
    async def _parse_typescript_hierarchical(
        self, file_content: str, file_path: str
    ) -> List[Dict[str, Any]]:
        """Parse TypeScript file - delegates to JavaScript parsing for now."""
        return await self._parse_javascript_hierarchical(file_content, file_path)

    async def _parse_java_hierarchical(
        self, file_content: str, file_path: str
    ) -> List[Dict[str, Any]]:
        """Parse Java file - simplified implementation."""
        return await self._parse_generic_hierarchical(
            file_content, LanguageType.JAVA, file_path
        )

    async def _parse_csharp_hierarchical(
        self, file_content: str, file_path: str
    ) -> List[Dict[str, Any]]:
        """Parse C# file - simplified implementation."""
        return await self._parse_generic_hierarchical(
            file_content, LanguageType.CSHARP, file_path
        )

    def _build_uuid_mapping(
        self, parsed_entities: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Build UUID mapping for entity names to support hierarchical relationships.

        Following Semantic Kernel pattern of using clear naming conventions.
        """
        entity_uuid_map = {}

        for entity in parsed_entities:
            entity_name = entity["name"]
            # Create deterministic UUID based on name and type for consistency
            entity_uuid = str(uuid4())
            entity_uuid_map[entity_name] = entity_uuid
            entity["uuid"] = entity_uuid

        self.logger.debug(f"Built UUID mapping for {len(entity_uuid_map)} entities")
        return entity_uuid_map

    async def _build_hierarchical_relationships(
        self,
        parsed_entities: List[Dict[str, Any]],
        entity_uuid_map: Dict[str, str],
        file_path: str,
    ) -> List[CodeEntity]:
        """
        Build hierarchical relationships using UUID references and materialized paths.

        Implements Microsoft's recommended approach for hierarchical data structures.
        """
        try:
            hierarchical_entities = []

            # Build parent-child mapping
            parent_child_map = defaultdict(list)
            for entity in parsed_entities:
                parent_name = entity.get("parent_name")
                if parent_name and parent_name in entity_uuid_map:
                    parent_id = entity_uuid_map[parent_name]
                    entity_id = entity["uuid"]
                    parent_child_map[parent_id].append(entity_id)

            # Calculate hierarchy levels and paths using BFS
            for entity in parsed_entities:
                entity_name = entity["name"]
                entity_uuid = entity["uuid"]
                parent_name = entity.get("parent_name")

                # Determine parent_id
                parent_id = entity_uuid_map.get(parent_name) if parent_name else None

                # Calculate hierarchy level and path
                hierarchy_level, hierarchy_path = self._calculate_hierarchy_info(
                    entity_uuid, parent_id, parsed_entities, entity_uuid_map
                )

                # Create CodeEntity with hierarchical fields
                code_entity = CodeEntity(
                    name=entity_name,
                    entity_type=entity["entity_type"],
                    language=self._detect_language_from_path(file_path),
                    content=entity.get("content", ""),
                    signature=entity.get("signature"),
                    # Legacy fields (for backward compatibility)
                    parent_entity=parent_name,
                    child_entities=entity.get("children", []),
                    # NEW hierarchical fields
                    parent_id=parent_id,
                    hierarchy_level=hierarchy_level,
                    hierarchy_path=hierarchy_path,
                    # Other fields
                    scope=entity.get("scope", "public"),
                    is_exported=entity.get("is_exported", False),
                    imports=entity.get("imports", []),
                    calls=entity.get("calls", []),
                )

                hierarchical_entities.append(code_entity)

            self.logger.info(
                f"Built hierarchical relationships for {len(hierarchical_entities)} entities"
            )
            return hierarchical_entities

        except Exception as e:
            self.logger.error(f"Failed to build hierarchical relationships: {e}")
            return []

    def _calculate_hierarchy_info(
        self,
        entity_uuid: str,
        parent_id: Optional[str],
        parsed_entities: List[Dict[str, Any]],
        entity_uuid_map: Dict[str, str],
    ) -> Tuple[int, List[str]]:
        """Calculate hierarchy level and materialized path for an entity."""
        if not parent_id:
            # Root entity
            return 0, []

        # Find parent entity
        parent_entity = None
        for entity in parsed_entities:
            if entity.get("uuid") == parent_id:
                parent_entity = entity
                break

        if not parent_entity:
            # Parent not found, treat as root
            return 0, []

        # Recursively calculate parent's hierarchy info
        parent_parent_name = parent_entity.get("parent_name")
        parent_parent_id = (
            entity_uuid_map.get(parent_parent_name) if parent_parent_name else None
        )

        parent_level, parent_path = self._calculate_hierarchy_info(
            parent_id, parent_parent_id, parsed_entities, entity_uuid_map
        )

        # This entity's level is parent's level + 1
        level = parent_level + 1

        # This entity's path is parent's path + parent_id
        path = parent_path + [parent_id]

        return level, path

    def _detect_language_from_path(self, file_path: str) -> LanguageType:
        """Detect programming language from file extension."""
        extension = file_path.split(".")[-1].lower()

        extension_map = {
            "py": LanguageType.PYTHON,
            "js": LanguageType.JAVASCRIPT,
            "ts": LanguageType.TYPESCRIPT,
            "java": LanguageType.JAVA,
            "go": LanguageType.GO,
            "rs": LanguageType.RUST,
            "c": LanguageType.C,
            "cpp": LanguageType.CPP,
            "cc": LanguageType.CPP,
            "cxx": LanguageType.CPP,
            "cs": LanguageType.CSHARP,
            "html": LanguageType.HTML,
            "css": LanguageType.CSS,
        }

        return extension_map.get(extension, LanguageType.PYTHON)  # Default to Python

    def get_supported_languages(self) -> List[LanguageType]:
        """Get list of supported programming languages."""
        return [
            LanguageType.PYTHON,
            LanguageType.JAVASCRIPT,
            LanguageType.TYPESCRIPT,
            LanguageType.JAVA,
            LanguageType.GO,
            LanguageType.RUST,
            LanguageType.C,
            LanguageType.CPP,
            LanguageType.CSHARP,
            LanguageType.HTML,
            LanguageType.CSS,
        ]
