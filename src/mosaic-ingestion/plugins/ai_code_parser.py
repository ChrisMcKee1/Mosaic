"""
AI-Powered Code Parser Plugin using Semantic Kernel Prompt Functions

Leverages Microsoft's Semantic Kernel prompt functions and structured outputs
to parse code using AI instead of deterministic AST parsing. This approach
is more flexible, handles edge cases better, and can understand context
that traditional parsers miss.

Based on Microsoft's 2025 best practices for Semantic Kernel Python:
- Uses Pydantic models for structured outputs
- Implements prompt functions with JSON schema validation
- Follows type-safe development patterns
- Integrates with Azure OpenAI for enterprise reliability
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import uuid4
from pydantic import BaseModel, Field
import os

from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.contents import ChatHistory

from ..models.golden_node import CodeEntity, EntityType, LanguageType
from ..rdf.triple_generator import generate_triples_for_entities
from rdflib import Graph


# Pydantic models for structured outputs following Microsoft's 2025 patterns
class CodeEntityStructured(BaseModel):
    """Structured output model for AI-parsed code entities."""

    name: str = Field(..., description="The name/identifier of the code entity")
    entity_type: str = Field(
        ...,
        description="Type of entity: function, class, module, interface, struct, enum, variable, constant",
    )
    parent_name: Optional[str] = Field(
        None, description="Name of the parent entity (if nested)"
    )
    content: str = Field(..., description="The actual code content for this entity")
    signature: Optional[str] = Field(
        None, description="Function signature, class declaration, etc."
    )
    scope: str = Field(
        "public", description="Visibility scope: public, private, protected, internal"
    )
    is_exported: bool = Field(
        False, description="Whether this entity is exported/public"
    )
    line_start: int = Field(..., description="Starting line number (1-based)")
    line_end: int = Field(..., description="Ending line number (1-based)")
    imports: List[str] = Field(
        default_factory=list,
        description="List of imports/dependencies this entity uses",
    )
    calls: List[str] = Field(
        default_factory=list,
        description="List of functions/methods called by this entity",
    )
    description: Optional[str] = Field(
        None, description="AI-generated description of what this entity does"
    )


class HierarchicalCodeAnalysis(BaseModel):
    """Complete hierarchical analysis of a source code file."""

    file_language: str = Field(..., description="Programming language detected")
    entities: List[CodeEntityStructured] = Field(
        ..., description="All code entities found in hierarchical order"
    )
    file_level_imports: List[str] = Field(
        default_factory=list, description="File-level imports and dependencies"
    )
    module_description: Optional[str] = Field(
        None, description="AI-generated description of the overall module/file purpose"
    )


class RelationshipAnalysis(BaseModel):
    """AI analysis of relationships between code entities."""

    relationship_type: str = Field(
        ...,
        description="Type: contains, calls, imports, inherits, implements, depends_on, similar_to",
    )
    source_entity: str = Field(..., description="Name of the source entity")
    target_entity: str = Field(..., description="Name of the target entity")
    strength: float = Field(
        ..., description="Relationship strength from 0.0 to 1.0", ge=0.0, le=1.0
    )
    description: str = Field(
        ..., description="Natural language description of the relationship"
    )


class AICodeParserPlugin:
    """
    AI-powered code parser using Semantic Kernel prompt functions.

    This plugin creates its own isolated chat completion service instances
    for each function call, using GPT-4o-mini for cost-effective code analysis.
    Each method handles its own AI interaction with zero chat history.
    """

    def __init__(
        self,
        azure_openai_endpoint: Optional[str] = None,
        azure_openai_key: Optional[str] = None,
    ):
        """Initialize the AI code parser with Azure OpenAI credentials."""
        self.logger = logging.getLogger("mosaic.ai_code_parser")

        # Configure Azure OpenAI settings for GPT-4.1-mini
        self.azure_openai_endpoint = azure_openai_endpoint or os.getenv(
            "AZURE_OPENAI_ENDPOINT"
        )
        self.azure_openai_key = azure_openai_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment_name = os.getenv(
            "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_MINI", "gpt-4.1-mini"
        )

        if not self.azure_openai_endpoint:
            raise ValueError(
                "Azure OpenAI endpoint must be provided via constructor or AZURE_OPENAI_ENDPOINT environment variable"
            )

        self.logger.info(
            f"AI Code Parser initialized with deployment: {self.deployment_name}"
        )

    def _create_isolated_chat_service(self) -> AzureChatCompletion:
        """Create an isolated chat completion service instance for this function call."""
        return AzureChatCompletion(
            deployment_name=self.deployment_name,
            endpoint=self.azure_openai_endpoint,
            api_key=self.azure_openai_key,
            service_id="ai_code_parser_isolated",
        )

    @kernel_function(
        name="parse_code_hierarchical",
        description="Parse source code and extract hierarchical entities using AI analysis",
    )
    async def parse_code_hierarchical(
        self, file_content: str, file_path: str, language: str = "auto-detect"
    ) -> HierarchicalCodeAnalysis:
        """
        AI-powered hierarchical code parsing using isolated chat completion service.

        Creates its own chat service instance with zero chat history,
        uses GPT-4o-mini for cost-effective parsing, and returns structured output.
        """
        try:
            self.logger.info(f"AI parsing {language} file: {file_path}")

            # Create isolated chat completion service for this function call
            chat_service = self._create_isolated_chat_service()

            # Create parsing prompt with injected code content
            parsing_prompt = self._create_hierarchical_parsing_prompt(
                file_content, file_path, language
            )

            # Configure execution settings for structured output with GPT-4.1-mini
            execution_settings = OpenAIChatPromptExecutionSettings(
                response_format=HierarchicalCodeAnalysis,  # Pydantic model for structured output
                temperature=0.1,  # Low temperature for consistent parsing
                max_tokens=4000,  # Sufficient for complex files
                service_id="ai_code_parser_isolated",
            )

            # Create fresh chat history with zero previous context
            chat_history = ChatHistory()
            chat_history.add_system_message(
                "You are an expert code analyst. Parse the provided source code and "
                "extract all entities in hierarchical order with their relationships. "
                "Be precise and comprehensive in your analysis. Return valid JSON matching the schema."
            )
            chat_history.add_user_message(parsing_prompt)

            # Execute AI parsing with isolated service
            response = await chat_service.get_chat_message_content(
                chat_history=chat_history, settings=execution_settings
            )

            # Validate and parse structured output
            parsed_analysis = HierarchicalCodeAnalysis.model_validate_json(
                response.content
            )

            self.logger.info(
                f"AI successfully parsed {len(parsed_analysis.entities)} entities from {file_path}"
            )
            return parsed_analysis

        except Exception as e:
            self.logger.error(f"AI code parsing failed for {file_path}: {e}")
            # Return empty analysis on failure to allow processing to continue
            return HierarchicalCodeAnalysis(
                file_language=language,
                entities=[],
                file_level_imports=[],
                module_description=f"Parsing failed: {str(e)}",
            )

    @kernel_function(
        name="analyze_code_relationships",
        description="Analyze relationships between code entities using AI understanding",
    )
    async def analyze_code_relationships(
        self,
        entities: List[CodeEntityStructured],
        file_content: str,
        cross_file_context: Optional[Dict[str, Any]] = None,
    ) -> List[RelationshipAnalysis]:
        """
        AI-powered relationship analysis using isolated chat completion service.

        Creates its own service instance, injects entity data into prompts,
        and returns structured relationship analysis using GPT-4.1-mini.
        """
        try:
            if not entities:
                return []

            self.logger.info(
                f"AI analyzing relationships between {len(entities)} entities"
            )

            # Create isolated chat completion service for this function call
            chat_service = self._create_isolated_chat_service()

            # Create relationship analysis prompt with injected entity data
            relationship_prompt = self._create_relationship_analysis_prompt(
                entities, file_content, cross_file_context
            )

            # Configure execution settings for structured output
            execution_settings = OpenAIChatPromptExecutionSettings(
                response_format=List[
                    RelationshipAnalysis
                ],  # List of relationship objects
                temperature=0.2,  # Slightly higher for relationship inference
                max_tokens=3000,
                service_id="ai_code_parser_isolated",
            )

            # Create fresh chat history with zero previous context
            chat_history = ChatHistory()
            chat_history.add_system_message(
                "You are an expert software architect. Analyze the provided code entities "
                "and identify all meaningful relationships between them. Consider both "
                "explicit relationships (calls, inheritance) and implicit ones (semantic similarity, "
                "functional cohesion). Provide relationship strength scores based on coupling. "
                "Return valid JSON array matching the schema."
            )
            chat_history.add_user_message(relationship_prompt)

            # Execute AI relationship analysis with isolated service
            response = await chat_service.get_chat_message_content(
                chat_history=chat_history, settings=execution_settings
            )

            # Parse structured output into list of RelationshipAnalysis objects
            relationships_data = response.content
            if isinstance(relationships_data, str):
                import json

                relationships_data = json.loads(relationships_data)

            relationships = [
                RelationshipAnalysis.model_validate(r) for r in relationships_data
            ]

            self.logger.info(f"AI identified {len(relationships)} relationships")
            return relationships

        except Exception as e:
            self.logger.error(f"AI relationship analysis failed: {e}")
            return []

    @kernel_function(
        name="extract_entities_from_code",
        description="Extract code entities from source code - simplified interface for agents",
    )
    async def extract_entities_from_code(
        self, file_content: str, file_path: str, language: str = "auto-detect"
    ) -> List[CodeEntity]:
        """
        Simplified interface for agents to extract code entities.

        This method handles the complete flow:
        1. AI parsing with isolated chat service
        2. Structured output validation
        3. Conversion to GoldenNode CodeEntity objects
        4. Hierarchical relationship building

        Returns ready-to-use CodeEntity objects with hierarchical fields populated.
        """
        try:
            # Step 1: AI-powered parsing with isolated service
            parsed_analysis = await self.parse_code_hierarchical(
                file_content=file_content, file_path=file_path, language=language
            )

            # Step 2: Convert to GoldenNode format with hierarchical relationships
            golden_node_entities = await self.convert_to_golden_node_entities(
                parsed_analysis=parsed_analysis, file_path=file_path
            )

            self.logger.info(
                f"Successfully extracted {len(golden_node_entities)} entities from {file_path}"
            )
            return golden_node_entities

        except Exception as e:
            self.logger.error(f"Entity extraction failed for {file_path}: {e}")
            return []

    @kernel_function(
        name="convert_to_golden_node_entities",
        description="Convert AI-parsed entities to GoldenNode CodeEntity objects with hierarchical relationships",
    )
    async def convert_to_golden_node_entities(
        self, parsed_analysis: HierarchicalCodeAnalysis, file_path: str
    ) -> List[CodeEntity]:
        """
        Convert AI-parsed structured output to GoldenNode CodeEntity objects.

        Builds hierarchical relationships with UUID references and materialized paths
        following the updated GoldenNode model requirements.
        """
        try:
            self.logger.info(
                f"Converting {len(parsed_analysis.entities)} AI-parsed entities to GoldenNode format"
            )

            # Build UUID mapping for hierarchical relationships
            entity_uuid_map = {}
            for entity in parsed_analysis.entities:
                entity_uuid_map[entity.name] = str(uuid4())

            # Convert entities with hierarchical relationships
            golden_node_entities = []

            for entity in parsed_analysis.entities:
                # Map entity types
                entity_type = self._map_entity_type(entity.entity_type)
                language = self._map_language_type(parsed_analysis.file_language)

                # Determine parent relationships
                parent_id = (
                    entity_uuid_map.get(entity.parent_name)
                    if entity.parent_name
                    else None
                )

                # Calculate hierarchy level and path
                hierarchy_level, hierarchy_path = self._calculate_ai_hierarchy_info(
                    entity, parsed_analysis.entities, entity_uuid_map
                )

                # Create GoldenNode CodeEntity with hierarchical fields
                code_entity = CodeEntity(
                    name=entity.name,
                    entity_type=entity_type,
                    language=language,
                    content=entity.content,
                    signature=entity.signature,
                    # Legacy fields (backward compatibility)
                    parent_entity=entity.parent_name,
                    child_entities=[],  # Will be populated by relationships
                    # NEW hierarchical fields from AI analysis
                    parent_id=parent_id,
                    hierarchy_level=hierarchy_level,
                    hierarchy_path=hierarchy_path,
                    # Other fields from AI analysis
                    scope=entity.scope,
                    is_exported=entity.is_exported,
                    imports=entity.imports,
                    calls=entity.calls,
                )

                golden_node_entities.append(code_entity)

            self.logger.info(
                f"Successfully converted to {len(golden_node_entities)} GoldenNode entities"
            )
            return golden_node_entities

        except Exception as e:
            self.logger.error(f"Entity conversion failed: {e}")
            return []

    def _create_hierarchical_parsing_prompt(
        self, file_content: str, file_path: str, language: str
    ) -> str:
        """Create a comprehensive prompt for AI code parsing."""
        return f"""
Analyze this {language} source code file and extract ALL code entities in hierarchical order.

File: {file_path}
Language: {language}

Source Code:
```{language}
{file_content}
```

Instructions:
1. Identify ALL code entities: functions, classes, modules, interfaces, structs, enums, variables, constants
2. Determine the hierarchical parent-child relationships (e.g., methods belong to classes)
3. Extract imports, dependencies, and function calls
4. Provide accurate line numbers for each entity
5. Generate brief descriptions of what each entity does
6. Detect the programming language if "auto-detect" was specified
7. Identify file-level imports and overall module purpose

Focus on:
- Complete hierarchical structure (file → class → method relationships)
- Accurate entity boundaries and content
- All dependencies and relationships
- Proper scope and visibility analysis

Return a structured analysis following the specified JSON schema.
"""

    def _create_relationship_analysis_prompt(
        self,
        entities: List[CodeEntityStructured],
        file_content: str,
        cross_file_context: Optional[Dict[str, Any]],
    ) -> str:
        """Create a prompt for AI relationship analysis."""
        entities_summary = "\n".join(
            [
                f"- {e.name} ({e.entity_type}): {e.description or 'No description'}"
                for e in entities
            ]
        )

        cross_file_info = ""
        if cross_file_context:
            cross_file_info = f"""
Cross-file context available:
- Related files: {cross_file_context.get("related_files", [])}
- External dependencies: {cross_file_context.get("external_deps", [])}
"""

        return f"""
Analyze relationships between these code entities:

Entities in this file:
{entities_summary}

{cross_file_info}

Source code context:
```
{file_content[:2000]}{"..." if len(file_content) > 2000 else ""}
```

Identify ALL meaningful relationships:
1. **Hierarchical**: parent-child containment (class contains method)
2. **Functional**: function calls and method invocations
3. **Structural**: inheritance, interface implementation
4. **Dependency**: imports and references
5. **Semantic**: entities with similar purpose or domain

For each relationship, provide:
- Precise relationship type
- Source and target entity names
- Strength score (0.0-1.0) based on coupling strength
- Clear description of the relationship

Focus on both explicit code relationships and implicit semantic connections.
"""

    def _map_entity_type(self, ai_entity_type: str) -> EntityType:
        """Map AI-detected entity type to GoldenNode EntityType enum."""
        type_mapping = {
            "function": EntityType.FUNCTION,
            "class": EntityType.CLASS,
            "module": EntityType.MODULE,
            "interface": EntityType.INTERFACE,
            "struct": EntityType.STRUCT,
            "enum": EntityType.ENUM,
            "variable": EntityType.VARIABLE,
            "constant": EntityType.CONSTANT,
            "import": EntityType.IMPORT,
            "html_element": EntityType.HTML_ELEMENT,
            "css_rule": EntityType.CSS_RULE,
            "comment": EntityType.COMMENT,
            "docstring": EntityType.DOCSTRING,
        }

        return type_mapping.get(ai_entity_type.lower(), EntityType.OTHER)

    def _map_language_type(self, ai_language: str) -> LanguageType:
        """Map AI-detected language to GoldenNode LanguageType enum."""
        language_mapping = {
            "python": LanguageType.PYTHON,
            "javascript": LanguageType.JAVASCRIPT,
            "typescript": LanguageType.TYPESCRIPT,
            "java": LanguageType.JAVA,
            "go": LanguageType.GO,
            "rust": LanguageType.RUST,
            "c": LanguageType.C,
            "cpp": LanguageType.CPP,
            "c++": LanguageType.CPP,
            "csharp": LanguageType.CSHARP,
            "c#": LanguageType.CSHARP,
            "html": LanguageType.HTML,
            "css": LanguageType.CSS,
        }

        return language_mapping.get(ai_language.lower(), LanguageType.PYTHON)

    def _calculate_ai_hierarchy_info(
        self,
        entity: CodeEntityStructured,
        all_entities: List[CodeEntityStructured],
        entity_uuid_map: Dict[str, str],
    ) -> tuple[int, List[str]]:
        """Calculate hierarchy level and materialized path from AI analysis."""
        if not entity.parent_name:
            return 0, []

        # Find parent entity
        parent_entity = None
        for e in all_entities:
            if e.name == entity.parent_name:
                parent_entity = e
                break

        if not parent_entity:
            return 0, []

        # Recursively calculate parent's hierarchy
        parent_level, parent_path = self._calculate_ai_hierarchy_info(
            parent_entity, all_entities, entity_uuid_map
        )

        # This entity's level and path
        level = parent_level + 1
        parent_uuid = entity_uuid_map.get(entity.parent_name)
        path = parent_path + ([parent_uuid] if parent_uuid else [])

        return level, path

    @kernel_function(
        name="extract_entities_with_rdf_triples",
        description="Extract code entities and generate RDF triples - complete AI+RDF pipeline",
    )
    async def extract_entities_with_rdf_triples(
        self,
        file_content: str,
        file_path: str,
        language: str = "auto-detect",
        base_namespace: str = "http://mosaic.ai/graph#",
        validate_rdf: bool = True,
    ) -> Dict[str, Any]:
        """
        Complete pipeline: AI parsing + RDF triple generation.

        This method provides the full OmniRAG pipeline integration:
        1. AI-powered code parsing with isolated chat service
        2. CodeEntity object generation
        3. RDF triple generation using defined ontologies
        4. Validation against ontology schemas

        Args:
            file_content: Source code content to parse
            file_path: File path for context and URI generation
            language: Programming language (auto-detect if not specified)
            base_namespace: Base namespace for RDF URIs
            validate_rdf: Whether to validate RDF triples against ontologies

        Returns:
            Dictionary with both CodeEntity objects and RDF Graph:
            {
                "entities": List[CodeEntity],
                "rdf_graph": Graph,
                "statistics": {
                    "entities_count": int,
                    "triples_count": int,
                    "relationships_count": int
                }
            }
        """
        try:
            self.logger.info(f"Full AI+RDF pipeline for {file_path}")

            # Step 1: AI-powered entity extraction
            entities = await self.extract_entities_from_code(
                file_content=file_content, file_path=file_path, language=language
            )

            if not entities:
                self.logger.warning(f"No entities extracted from {file_path}")
                return {
                    "entities": [],
                    "rdf_graph": Graph(),
                    "statistics": {
                        "entities_count": 0,
                        "triples_count": 0,
                        "relationships_count": 0,
                    },
                }

            # Step 2: RDF triple generation
            rdf_graph = await self.generate_rdf_triples_for_entities(
                entities=entities,
                file_path=file_path,
                base_namespace=base_namespace,
                validate=validate_rdf,
            )

            # Step 3: Compile statistics
            statistics = {
                "entities_count": len(entities),
                "triples_count": len(rdf_graph),
                "relationships_count": self._count_relationships_in_entities(entities),
            }

            self.logger.info(
                f"AI+RDF pipeline completed: {statistics['entities_count']} entities, "
                f"{statistics['triples_count']} triples, {statistics['relationships_count']} relationships"
            )

            return {
                "entities": entities,
                "rdf_graph": rdf_graph,
                "statistics": statistics,
            }

        except Exception as e:
            self.logger.error(f"AI+RDF pipeline failed for {file_path}: {e}")
            # Return empty results to allow processing to continue
            return {
                "entities": [],
                "rdf_graph": Graph(),
                "statistics": {
                    "entities_count": 0,
                    "triples_count": 0,
                    "relationships_count": 0,
                },
            }

    @kernel_function(
        name="generate_rdf_triples_for_entities",
        description="Generate RDF triples from CodeEntity objects using ontologies",
    )
    async def generate_rdf_triples_for_entities(
        self,
        entities: List[CodeEntity],
        file_path: str,
        base_namespace: str = "http://mosaic.ai/graph#",
        validate: bool = True,
    ) -> Graph:
        """
        Generate RDF triples from CodeEntity objects.

        This method uses the TripleGenerator to convert parsed entities
        into semantic RDF representation using the defined ontologies.

        Args:
            entities: List of CodeEntity objects to convert
            file_path: Source file path for URI generation context
            base_namespace: Base namespace for RDF URIs
            validate: Whether to validate against ontology schemas

        Returns:
            RDF Graph containing generated triples
        """
        try:
            if not entities:
                return Graph()

            self.logger.info(
                f"Generating RDF triples for {len(entities)} entities from {file_path}"
            )

            # Use the TripleGenerator to create RDF representation
            rdf_graph = generate_triples_for_entities(
                entities=entities,
                file_path=file_path,
                base_namespace=base_namespace,
                validate=validate,
            )

            self.logger.info(f"Generated {len(rdf_graph)} RDF triples for {file_path}")
            return rdf_graph

        except Exception as e:
            self.logger.error(f"RDF triple generation failed for {file_path}: {e}")
            # Return empty graph to allow processing to continue
            return Graph()

    def _count_relationships_in_entities(self, entities: List[CodeEntity]) -> int:
        """Count total relationships across all entities."""
        relationship_count = 0

        for entity in entities:
            # Count parent relationships
            if entity.parent_entity:
                relationship_count += 1

            # Count function calls
            if hasattr(entity, "calls") and entity.calls:
                relationship_count += len(entity.calls)

            # Count imports
            if hasattr(entity, "imports") and entity.imports:
                relationship_count += len(entity.imports)

        return relationship_count
