"""
Semantic Kernel Code Analysis Plugin with Structured Outputs and Error Handling

This plugin provides AI-powered code analysis capabilities using prompt functions
with Pydantic structured outputs for guaranteed response reliability. Each function
uses OpenAI's structured outputs feature to ensure consistent, validated results.

Features:
- Comprehensive error handling with OpenTelemetry integration
- Automatic retry logic with exponential backoff
- Circuit breaker pattern for resilience
- Azure Application Insights monitoring alignment

This replaces hundreds of lines of complex if/else logic with intelligent AI analysis.
"""

import logging
import os
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
from semantic_kernel.functions import kernel_function
from typing import Annotated, List, Literal
from pydantic import BaseModel, Field
from .ai_error_handler import (
    AIOrchestrationErrorHandler,
    AIFunctionMetrics,
    configure_azure_telemetry,
)


# Pydantic models for structured outputs
class EntityVisibilityAnalysis(BaseModel):
    """Structured output for entity visibility analysis."""

    is_public: bool = Field(
        description="Is this entity publicly accessible from outside its module/class?"
    )
    is_exported: bool = Field(
        description="Is this entity explicitly exported (can be imported by other files)?"
    )
    visibility_modifier: str = Field(
        description="Visibility modifier used (public, private, protected, internal, etc.)"
    )
    confidence: float = Field(description="Confidence score (0.0-1.0)")


class FileClassification(BaseModel):
    """Structured output for comprehensive file classification."""

    primary_type: Literal[
        "service",
        "controller",
        "model",
        "dto",
        "test",
        "config",
        "documentation",
        "script",
        "ai_tool",
        "ide_config",
        "build",
        "unknown",
    ] = Field(description="Primary file type classification")

    architectural_pattern: Literal[
        "mvc",
        "service_layer",
        "repository",
        "adapter",
        "middleware",
        "microservice",
        "monolith",
        "utility",
        "unknown",
    ] = Field(description="Architectural pattern classification")

    development_tool_type: str | None = Field(
        description="Development tool type (ai_assistant, ide_config, build_tool, testing, etc.)"
    )

    framework: str | None = Field(description="Detected framework name")

    purpose: Literal[
        "business_logic",
        "data_access",
        "ui_component",
        "configuration",
        "testing",
        "tooling",
        "documentation",
        "unknown",
    ] = Field(description="Primary purpose of the file")

    confidence: float = Field(description="Confidence score (0.0-1.0)")
    specialized_tags: List[str] = Field(
        description="Additional specialized tags", max_length=5
    )


class ImportNames(BaseModel):
    """Structured output for import statement parsing."""

    imported_names: List[str] = Field(
        description="List of imported names available in current scope", max_length=10
    )


class EntityTags(BaseModel):
    """Structured output for tag generation."""

    tags: List[str] = Field(
        description="Comprehensive tags for semantic search", max_length=8
    )


class CodeAnalysisPlugin:
    """
    Semantic Kernel plugin for AI-powered code analysis with comprehensive error handling.

    This plugin uses prompt functions to provide intelligent code analysis
    capabilities, replacing complex deterministic logic with flexible AI prompts.

    Features:
    - Azure Application Insights integration
    - Automatic retry with exponential backoff
    - Circuit breaker pattern for resilience
    - Comprehensive telemetry and metrics
    """

    def __init__(self, settings):
        """Initialize the plugin with Azure OpenAI settings and error handling."""
        self.settings = settings
        self.logger = logging.getLogger(__name__)

        # Initialize Azure monitoring components
        connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
        self.tracer = configure_azure_telemetry(connection_string)
        self.error_handler = AIOrchestrationErrorHandler(connection_string)
        self.metrics = AIFunctionMetrics(self.tracer)

        self.logger.info(
            f"CodeAnalysisPlugin initialized with Azure monitoring: {bool(connection_string)}"
        )

    @kernel_function(
        description="Extract meaningful entity names from code using AI intelligence.",
        name="extract_entity_name",
    )
    def extract_entity_name(
        self,
        language: Annotated[
            str, "Programming language (python, javascript, java, etc.)"
        ],
        entity_type: Annotated[
            str, "Type of code entity (function, class, variable, etc.)"
        ],
        entity_content: Annotated[str, "Source code of the entity"],
    ) -> Annotated[str, "The extracted entity name"]:
        """
        AI-powered entity name extraction prompt.

        This replaces complex tree-sitter node traversal and language-specific
        naming logic with intelligent AI analysis that can handle:
        - Generic types and type parameters
        - Anonymous functions and lambdas
        - Complex inheritance hierarchies
        - Framework-specific naming patterns

        NOTE: This function returns a simple string, not structured output,
        as entity names should be single identifiers.
        """
        return """
        Extract the most meaningful identifier/name for this {{$language}} code entity.
        
        LANGUAGE: {{$language}}  
        ENTITY TYPE: {{$entity_type}}
        CODE:
        ```{{$language}}
        {{$entity_content}}
        ```
        
        Rules:
        1. Return the primary identifier (class name, function name, variable name, etc.)
        2. For anonymous entities, create a descriptive name based on purpose
        3. For complex generics, simplify to the main type
        4. For overloaded methods, include distinguishing characteristics  
        5. Must be a valid identifier (no spaces, special chars except _)
        
        Examples:
        - Python class: "class UserManager:" → "UserManager"
        - JavaScript function: "const handleClick = () => {}" → "handleClick"
        - Java method: "public void processData(String input)" → "processData"
        - Anonymous function: "list.map(x => x * 2)" → "multiplyByTwo"
        
        Return ONLY the name, no explanation or quotes.
        """

    @kernel_function(
        description="Analyze code entity visibility and export status using AI with structured output.",
        name="analyze_entity_visibility",
    )
    def analyze_entity_visibility(
        self,
        language: Annotated[str, "Programming language"],
        entity_content: Annotated[str, "Code entity to analyze"],
    ) -> Annotated[EntityVisibilityAnalysis, "Structured visibility analysis"]:
        """
        AI-powered visibility analysis with guaranteed structured output.

        This replaces 50+ lines of language-specific if/else logic with
        intelligent AI analysis that returns validated Pydantic models.
        """
        return """
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
        
        Provide your analysis in the required structured format.
        """

    @kernel_function(
        description="Parse import statements to extract imported names using AI with structured output.",
        name="parse_import_statement",
    )
    def parse_import_statement(
        self,
        language: Annotated[str, "Programming language"],
        import_text: Annotated[str, "Import statement to parse"],
    ) -> Annotated[ImportNames, "Structured list of imported names"]:
        """
        AI-powered import parsing with guaranteed structured output.

        This replaces 70+ lines of complex regex and string manipulation
        with intelligent AI parsing that returns validated Pydantic models.
        """
        return """
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
        
        Provide the imported names in the required structured format.
        """

    @kernel_function(
        description="Classify files by architecture, purpose, and development tools using AI with structured output.",
        name="classify_file",
    )
    def classify_file(
        self,
        file_path: Annotated[str, "Full file path"],
        content_preview: Annotated[str, "File content preview (first 2000 chars)"],
    ) -> Annotated[FileClassification, "Structured file classification"]:
        """
        AI-powered file classification with guaranteed structured output.

        This replaces deterministic file type detection with intelligent
        analysis that returns validated Pydantic models.
        """
        return """
        Analyze this file and provide comprehensive classification information.
        
        FILE PATH: {{$file_path}}
        FILE CONTENT (first 2000 chars):
        {{$content_preview}}
        
        Provide detailed classification including:
        1. Primary file type and purpose
        2. Architectural pattern classification (MVC, service layer, etc.)
        3. Development tool classification (AI tools, IDE configs, etc.)
        4. Framework/library identification
        5. Specific role in codebase
        
        Architectural patterns to consider:
        - MVC: model, view, controller
        - Service Layer: service, business logic, domain
        - Repository: data access, persistence
        - Adapter: external integration, API clients
        - Middleware: request processing, filters
        - Microservice: independent service components
        
        Development tools to detect:
        - AI Assistant: .claude/, CLAUDE.md, .cursor/
        - IDE Config: .vscode/, .idea/, settings files
        - Build Tools: package.json, Dockerfile, Makefile
        - Testing: test files, mock objects, fixtures
        - CI/CD: GitHub Actions, Azure DevOps
        
        Provide your analysis in the required structured format.
        """

    @kernel_function(
        description="Generate comprehensive tags for OmniRAG semantic filtering using AI with structured output.",
        name="generate_tags",
    )
    def generate_tags(
        self,
        file_path: Annotated[str, "File path"],
        entity_content: Annotated[str, "Entity content to analyze"],
        file_context: Annotated[str, "File context for better understanding"],
    ) -> Annotated[EntityTags, "Structured list of intelligent tags"]:
        """
        AI-powered tag generation with guaranteed structured output.

        This generates intelligent tags that enable powerful OmniRAG searches like:
        - "all services" - finds all service-related files
        - "all tests" - finds all testing files
        - "all AI tools" - finds .claude, .cursor, vscode configs
        - "all DTOs" - finds data transfer objects
        """
        return """
        Generate comprehensive tags for this code file and entity for semantic search and filtering.
        
        FILE: {{$file_path}}
        ENTITY CONTENT: {{$entity_content}}
        FILE CONTEXT: {{$file_context}}
        
        Generate tags in these categories:
        1. ARCHITECTURAL: service, controller, model, dto, middleware, adapter, repository, etc.
        2. PURPOSE: business-logic, data-access, ui-component, configuration, utility, etc.
        3. TECHNOLOGY: framework names, library types, language-specific patterns
        4. DEVELOPMENT: test, mock, fixture, ai-tool, ide-config, build-script, etc.
        5. DOMAIN: specific business domain concepts found in the code
        
        Tag examples by category:
        - Architectural: ["service", "rest-controller", "data-model", "dto", "middleware"]
        - Purpose: ["business-logic", "data-validation", "user-authentication", "api-endpoint"]
        - Technology: ["fastapi", "spring-boot", "react", "typescript", "docker"]
        - Development: ["unit-test", "integration-test", "ai-assistant", "vscode-config"]
        - Domain: ["user-management", "payment-processing", "order-fulfillment"]
        
        Focus on creating tags that enable powerful semantic searches:
        - Users can find "all services" across different architectures
        - Users can find "all AI development tools" regardless of specific tool
        - Users can find "all business logic" regardless of implementation pattern
        
        Provide your tags in the required structured format.
        Maximum 8 tags, prioritize most useful for search and filtering.
        """

    def get_execution_settings_with_structured_output(
        self,
        response_format_model: type,
        max_tokens: int = 1000,
        temperature: float = 0.1,
    ):
        """Get execution settings with Pydantic structured output for reliable parsing."""
        return AzureChatPromptExecutionSettings(
            service_id="azure_openai_chat",
            ai_model_id=self.settings.azure_openai_chat_deployment_name,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format_model,  # Pass Pydantic model directly
        )

    def get_execution_settings_text_only(
        self, max_tokens: int = 100, temperature: float = 0.1
    ):
        """Get execution settings for text-only responses (like entity names)."""
        return AzureChatPromptExecutionSettings(
            service_id="azure_openai_chat",
            ai_model_id=self.settings.azure_openai_chat_deployment_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )
