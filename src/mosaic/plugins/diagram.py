"""
DiagramPlugin for Mosaic MCP Tool

Implements FR-12 (Mermaid Generation) and FR-13 (Mermaid as Context Resource)
using Azure OpenAI GPT model as Semantic Function.

This plugin provides:
- Natural language to Mermaid diagram generation
- Diagram storage and retrieval via MCP interface
- Multiple diagram types (flowchart, sequence, class, etc.)
"""

import logging
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
import asyncio

from semantic_kernel import Kernel
from semantic_kernel.plugin_definition import sk_function, sk_function_context_parameter
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.functions import KernelFunction
from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from azure.identity import DefaultAzureCredential

from ..config.settings import MosaicSettings
from ..models.base import DiagramResponse


logger = logging.getLogger(__name__)


class DiagramPlugin:
    """
    Semantic Kernel plugin for Mermaid diagram generation and management.
    
    Uses Azure OpenAI GPT model to convert natural language descriptions
    into Mermaid diagram syntax, with storage for future reference.
    """
    
    def __init__(self, settings: MosaicSettings, kernel: Kernel):
        """Initialize the DiagramPlugin."""
        self.settings = settings
        self.kernel = kernel
        self.credential = DefaultAzureCredential()
        
        # Storage for diagrams
        self.cosmos_client: Optional[CosmosClient] = None
        self.diagram_container = None
        
        # Mermaid generation function
        self.mermaid_generator: Optional[KernelFunction] = None
        
    async def initialize(self) -> None:
        """Initialize the DiagramPlugin."""
        try:
            # Initialize Cosmos DB for diagram storage
            await self._initialize_cosmos()
            
            # Create Mermaid generation semantic function
            await self._create_mermaid_generator()
            
            logger.info("DiagramPlugin initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize DiagramPlugin: {e}")
            raise
    
    async def _initialize_cosmos(self) -> None:
        """Initialize Azure Cosmos DB connection for diagram storage."""
        cosmos_config = self.settings.get_cosmos_config()
        
        self.cosmos_client = CosmosClient(
            cosmos_config["endpoint"],
            self.credential
        )
        
        database = self.cosmos_client.get_database_client(
            cosmos_config["database_name"]
        )
        
        # Use a dedicated container for diagrams
        self.diagram_container = database.get_container_client("diagrams")
    
    async def _create_mermaid_generator(self) -> None:
        """Create semantic function for Mermaid generation using Azure OpenAI."""
        
        # Comprehensive prompt template for Mermaid generation
        mermaid_prompt = """
You are an expert at creating Mermaid diagrams from natural language descriptions.

Given a description, generate clean, well-structured Mermaid syntax that accurately represents the described system, process, or relationship.

IMPORTANT GUIDELINES:
1. Use appropriate Mermaid diagram types:
   - flowchart TD/LR for processes and workflows
   - sequenceDiagram for interactions and communications
   - classDiagram for object relationships and data models
   - erDiagram for database schemas
   - gitgraph for version control flows
   - journey for user journeys
   - gantt for project timelines

2. Keep node IDs simple and descriptive (no spaces, use underscores)
3. Use clear, concise labels for nodes and connections
4. Include appropriate styling where beneficial
5. Follow Mermaid best practices for readability

DESCRIPTION: {{$description}}

DIAGRAM TYPE PREFERENCE: {{$diagram_type}}

Generate ONLY the Mermaid syntax - no explanations or markdown code blocks:
"""
        
        # Create prompt template configuration
        prompt_config = PromptTemplateConfig(
            template=mermaid_prompt,
            name="generate_mermaid",
            description="Generate Mermaid diagram syntax from natural language",
            input_variables=[
                {"name": "description", "description": "Natural language description", "is_required": True},
                {"name": "diagram_type", "description": "Preferred diagram type", "is_required": False}
            ]
        )
        
        # Create the semantic function
        self.mermaid_generator = self.kernel.create_function_from_prompt(
            function_name="generate_mermaid",
            plugin_name="diagram",
            prompt_template_config=prompt_config
        )
    
    @sk_function(
        description="Generate Mermaid diagram syntax from natural language description",
        name="generate"
    )
    @sk_function_context_parameter(
        name="description",
        description="Natural language description of the diagram",
        type_="str"
    )
    async def generate(self, description: str) -> str:
        """
        Generate Mermaid diagram syntax from natural language description (FR-12).
        
        Args:
            description: Natural language description of the diagram
            
        Returns:
            Generated Mermaid diagram syntax
        """
        try:
            # Determine diagram type from description
            diagram_type = self._determine_diagram_type(description)
            
            # Generate Mermaid syntax using semantic function
            result = await self.mermaid_generator.invoke(
                self.kernel,
                description=description,
                diagram_type=diagram_type
            )
            
            # Clean up the generated syntax
            mermaid_syntax = self._clean_mermaid_syntax(str(result))
            
            # Generate diagram ID for storage
            diagram_id = self._generate_diagram_id(description, mermaid_syntax)
            
            # Store the diagram for future reference (FR-13)
            await self._store_diagram(diagram_id, description, mermaid_syntax, diagram_type)
            
            logger.info(f"Generated Mermaid diagram: {diagram_id}")
            
            return mermaid_syntax
            
        except Exception as e:
            logger.error(f"Mermaid generation failed: {e}")
            # Return a simple fallback diagram
            return self._create_fallback_diagram(description)
    
    async def get_stored_diagram(self, diagram_id: str) -> str:
        """
        Retrieve stored Mermaid diagram by ID (FR-13).
        
        Args:
            diagram_id: Unique diagram identifier
            
        Returns:
            Stored Mermaid diagram syntax
        """
        try:
            item = await asyncio.to_thread(
                self.diagram_container.read_item,
                item=diagram_id,
                partition_key=diagram_id
            )
            
            return item["mermaid_syntax"]
            
        except CosmosResourceNotFoundError:
            logger.warning(f"Diagram not found: {diagram_id}")
            return f"graph TD\n    A[Diagram {diagram_id} not found]"
        except Exception as e:
            logger.error(f"Failed to retrieve diagram {diagram_id}: {e}")
            raise
    
    async def list_diagrams(self, limit: int = 20) -> list:
        """List stored diagrams with metadata."""
        try:
            query = "SELECT c.id, c.description, c.diagram_type, c.timestamp FROM c ORDER BY c.timestamp DESC OFFSET 0 LIMIT @limit"
            parameters = [{"name": "@limit", "value": limit}]
            
            items = list(await asyncio.to_thread(
                lambda: list(self.diagram_container.query_items(
                    query=query,
                    parameters=parameters,
                    enable_cross_partition_query=True
                ))
            ))
            
            return items
            
        except Exception as e:
            logger.error(f"Failed to list diagrams: {e}")
            return []
    
    def _determine_diagram_type(self, description: str) -> str:
        """Determine appropriate Mermaid diagram type from description."""
        description_lower = description.lower()
        
        # Keywords that suggest specific diagram types
        type_indicators = {
            "flowchart": [
                "process", "workflow", "flow", "step", "procedure", 
                "algorithm", "decision", "if", "then", "loop"
            ],
            "sequenceDiagram": [
                "interaction", "communication", "sequence", "message",
                "api", "request", "response", "call", "exchange"
            ],
            "classDiagram": [
                "class", "object", "inheritance", "extends", "implements",
                "relationship", "association", "composition", "aggregation"
            ],
            "erDiagram": [
                "database", "table", "entity", "relationship", "schema",
                "foreign key", "primary key", "one-to-many", "many-to-many"
            ],
            "gitgraph": [
                "git", "branch", "merge", "commit", "version control",
                "feature branch", "release", "hotfix"
            ],
            "journey": [
                "user journey", "customer journey", "experience", "touchpoint",
                "user flow", "persona", "interaction path"
            ],
            "gantt": [
                "timeline", "schedule", "project", "milestone", "task",
                "duration", "deadline", "phases", "deliverable"
            ]
        }
        
        # Score each diagram type
        type_scores = {}
        for diagram_type, keywords in type_indicators.items():
            score = sum(1 for keyword in keywords if keyword in description_lower)
            if score > 0:
                type_scores[diagram_type] = score
        
        # Return the highest scoring type, or default to flowchart
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        
        return "flowchart TD"
    
    def _clean_mermaid_syntax(self, raw_syntax: str) -> str:
        """Clean and validate generated Mermaid syntax."""
        # Remove markdown code blocks if present
        syntax = raw_syntax.strip()
        if syntax.startswith("```mermaid"):
            syntax = syntax[10:]
        if syntax.startswith("```"):
            syntax = syntax[3:]
        if syntax.endswith("```"):
            syntax = syntax[:-3]
        
        # Remove any leading/trailing whitespace
        syntax = syntax.strip()
        
        # Basic validation - ensure it starts with a valid diagram type
        valid_starts = [
            "flowchart", "graph", "sequenceDiagram", "classDiagram",
            "erDiagram", "journey", "gantt", "gitgraph", "pie",
            "quadrantChart", "requirement", "mindmap"
        ]
        
        if not any(syntax.startswith(start) for start in valid_starts):
            # If it doesn't start with a valid diagram type, assume flowchart
            syntax = f"flowchart TD\n{syntax}"
        
        return syntax
    
    def _create_fallback_diagram(self, description: str) -> str:
        """Create a simple fallback diagram when generation fails."""
        return f"""flowchart TD
    A[Input: {description[:30]}...] --> B[Processing]
    B --> C[Output: Diagram]
    C --> D[Review Result]
    
    style A fill:#e1f5fe
    style D fill:#f3e5f5"""
    
    def _generate_diagram_id(self, description: str, mermaid_syntax: str) -> str:
        """Generate unique diagram ID."""
        data = f"{description}:{mermaid_syntax}:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    async def _store_diagram(
        self, 
        diagram_id: str, 
        description: str, 
        mermaid_syntax: str, 
        diagram_type: str
    ) -> None:
        """Store diagram in Cosmos DB for future reference."""
        try:
            item = {
                "id": diagram_id,
                "description": description,
                "mermaid_syntax": mermaid_syntax,
                "diagram_type": diagram_type,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    "source": "mermaid_generator",
                    "version": "1.0"
                }
            }
            
            await asyncio.to_thread(
                self.diagram_container.create_item,
                body=item
            )
            
        except Exception as e:
            logger.warning(f"Failed to store diagram {diagram_id}: {e}")
            # Don't fail the generation if storage fails
    
    async def get_status(self) -> Dict[str, Any]:
        """Get plugin status information."""
        cosmos_connected = False
        generator_ready = False
        
        try:
            if self.diagram_container:
                # Test query
                list(self.diagram_container.query_items(
                    query="SELECT TOP 1 c.id FROM c",
                    enable_cross_partition_query=True
                ))
                cosmos_connected = True
        except:
            pass
        
        generator_ready = self.mermaid_generator is not None
        
        return {
            "status": "active",
            "cosmos_connected": cosmos_connected,
            "generator_ready": generator_ready,
            "supported_diagram_types": [
                "flowchart", "sequenceDiagram", "classDiagram", 
                "erDiagram", "journey", "gantt", "gitgraph"
            ]
        }
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        # Cosmos client cleanup is handled automatically
        self.mermaid_generator = None
        logger.info("DiagramPlugin cleanup completed")