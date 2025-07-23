"""
AI Agent Framework for Mosaic Ingestion Service

Implements the 5-agent architecture for distributed code ingestion processing:

1. GitSleuth Agent - Repository access and git operations
2. CodeParser Agent - AST parsing and entity extraction
3. GraphArchitect Agent - Relationship mapping and dependency analysis
4. DocuWriter Agent - AI-powered documentation and enrichment
5. GraphAuditor Agent - Quality assurance and validation

Based on Microsoft Docs MCP research:
- Semantic Kernel Python agents and plugins integration
- Azure OpenAI service orchestration patterns
- Function calling and structured outputs
- Agent coordination and task management

All agents inherit from the base MosaicAgent class which provides:
- Semantic Kernel integration
- Azure OpenAI connectivity
- Golden Node model compatibility
- Error handling and retry logic
- Telemetry and logging
"""

from .base_agent import (
    MosaicAgent,
    AgentConfig,
    AgentExecutionContext,
    AgentError,
    AgentTimeout,
)

from .git_sleuth import GitSleuthAgent
from .code_parser import CodeParserAgent
from .graph_architect import GraphArchitectAgent
from .docu_writer import DocuWriterAgent
from .graph_auditor import GraphAuditorAgent

__all__ = [
    # Base classes
    "MosaicAgent",
    "AgentConfig",
    "AgentExecutionContext",
    "AgentError",
    "AgentTimeout",
    # Specialized agents
    "GitSleuthAgent",
    "CodeParserAgent",
    "GraphArchitectAgent",
    "DocuWriterAgent",
    "GraphAuditorAgent",
]
