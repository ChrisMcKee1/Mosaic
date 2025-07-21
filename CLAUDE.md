# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Mosaic MCP Tool - Project Memory & Development Guidelines

## Project Overview

The Mosaic MCP Tool is a standardized, high-performance Model Context Protocol (MCP) Tool that provides a unified, multi-layered framework for advanced context engineering. It serves as a centralized "brain" that AI applications can connect to for knowledge retrieval, dependency analysis, memory management, and context refinement.

## Core Architecture Requirements

### Mandatory Technology Stack
- **Core Framework**: Python Semantic Kernel (FR-2) - ALL functionality MUST be implemented as Semantic Kernel Plugins
- **MCP Framework**: FastMCP Python library for protocol compliance and Streamable HTTP transport
- **LLM & Embedding Models**: Azure OpenAI Service exclusively
- **Hosting**: Azure Container Apps (Consumption Plan for POC)
- **Communication**: FastMCP with Streamable HTTP transport for FR-3 compliance
- **Unified Data Backend**: Azure Cosmos DB for NoSQL (OmniRAG pattern) - vector search, graph, and memory
- **Short-Term Memory**: Azure Cache for Redis (Basic C0 tier)
- **Memory Consolidation**: Azure Functions (Consumption Plan, Timer Trigger)
- **Semantic Reranking**: Azure Machine Learning Endpoint hosting cross-encoder/ms-marco-MiniLM-L-12-v2
- **Authentication**: Microsoft Entra ID (OAuth 2.1) for MCP Authorization

### Required Azure POC SKUs (Simplified Architecture)
- Azure Container Apps: Consumption Plan (serverless, pay-per-use)
- Azure Cosmos DB for NoSQL: Free Tier (1000 RU/s, 25 GB, unified backend for vector search, graph, and memory)
- Azure Cache for Redis: Basic (C0) Tier (250MB cache)
- Azure Machine Learning: Pay-as-you-go (Standard_DS2_v2 compute)
- Azure Functions: Consumption Plan (serverless)
- Azure OpenAI Service: Pay-as-you-go
- Microsoft Entra ID: Free tier (OAuth 2.1 authentication)

## Functional Requirements Compliance

### FR-1: MCP Server Implementation
- MUST implement MCP-compliant server exposing tools and resources
- MUST handle MCP protocol messaging correctly
- MUST provide standardized communication channel for AI components

### FR-2: Semantic Kernel Integration
- ALL functionality MUST be implemented as Semantic Kernel Plugins
- MUST promote modularity and reusability
- MUST allow easy composition and extension of context engineering capabilities

### FR-3: Streamable HTTP Communication
- MUST use Streamable HTTP transport protocol for all MCP client communication
- MUST enable real-time, non-blocking context streaming
- MUST prevent blocking I/O during context retrieval
- MUST comply with MCP specification 2025-03-26

### FR-4: Azure Native Deployment
- MUST deploy on Azure using specified POC SKUs
- MUST ensure scalability, reliability, and performance
- MUST leverage managed Azure services exclusively

### FR-5: Hybrid Search (RetrievalPlugin) - OmniRAG Pattern
- MUST orchestrate Vector Search and Keyword Search using unified Azure Cosmos DB backend
- MUST implement OmniRAG pattern for consolidated data operations
- MUST capture both semantic and lexical relevance in single service

### FR-6: Graph-Based Code Analysis (RetrievalPlugin) - OmniRAG Pattern
- MUST model graph relationships directly within NoSQL documents using embedded JSON arrays
- MUST use dependency_ids and developers arrays for relationship analysis
- MUST provide query functions for structural understanding using NoSQL document operations
- MUST follow Microsoft's OmniRAG pattern eliminating need for separate Gremlin API

### FR-7: Candidate Aggregation (RetrievalPlugin)
- MUST aggregate and de-duplicate results from multiple retrieval methods
- MUST create unified candidate pool for refinement stage

### FR-8: Semantic Reranking (RefinementPlugin)
- MUST use cross-encoder/ms-marco-MiniLM-L-12-v2 model
- MUST deploy to Azure Machine Learning Endpoint
- MUST implement httpx API calls for reranking
- MUST address "lost in the middle" problem

### FR-9: Unified Memory Interface (MemoryPlugin)
- MUST provide simple interface: save_memory, retrieve_memory, clear_memory
- MUST abstract complexity of multi-layered memory system
- MUST solve AI "amnesia" problem

### FR-10: Multi-Layered Storage (MemoryPlugin) - OmniRAG Pattern
- MUST support hybrid storage model
- MUST use Redis for short-term memory (conversational state)
- MUST use unified Cosmos DB backend for long-term memory (persistent knowledge)
- MUST implement HybridMemory class leveraging OmniRAG architecture

### FR-11: LLM-Powered Consolidation (MemoryPlugin)
- MUST implement asynchronous background function
- MUST extract, consolidate, and update long-term memory
- MUST prevent memory overload and context poisoning
- MUST use Azure Functions with timer trigger

### FR-12: Mermaid Generation (DiagramPlugin)
- MUST generate Mermaid diagram syntax from natural language
- MUST use Azure OpenAI GPT model as Semantic Function
- MUST lower barrier to architectural documentation

### FR-13: Mermaid as Context Resource (DiagramPlugin)
- MUST store and retrieve Mermaid diagrams via MCP interface
- MUST create machine-readable "source of truth"
- MUST enable both human and AI reference

### FR-14: Secure MCP Endpoint (NEW)
- MUST implement OAuth 2.1 authentication using MCP Authorization specification
- MUST use Microsoft Entra ID as identity provider
- MUST secure all MCP server endpoints from unauthorized access
- MUST be production-ready security implementation

## MCP Interface Requirements (TDD Section 6.0)

MUST implement these exact function signatures via FastMCP framework:
- `mosaic.retrieval.hybrid_search(query: str) -> List[Document]` (unified Cosmos DB backend)
- `mosaic.retrieval.query_code_graph(library_id: str, relationship_type: str) -> List[LibraryNode]` (OmniRAG embedded graph relationships)
- `mosaic.refinement.rerank(query: str, documents: List[Document]) -> List[Document]` (cross-encoder/ms-marco-MiniLM-L-12-v2)
- `mosaic.memory.save(session_id: str, content: str, type: str)` (OmniRAG pattern)
- `mosaic.memory.retrieve(session_id: str, query: str, limit: int) -> List[MemoryEntry]` (OmniRAG pattern)
- `mosaic.diagram.generate(description: str) -> str`

## Data Models (TDD Section 5.0)

### Cosmos DB Memory Schema (NoSQL API)
MUST implement exact schema:
```json
{
  "id": "unique_memory_id",
  "sessionId": "user_or_agent_session_id",
  "type": "episodic | semantic | procedural",
  "content": "The user confirmed that the 'auth-service' should be written in Go.",
  "embedding": [0.012, "...", -0.045],
  "importanceScore": 0.85,
  "timestamp": "2025-07-09T12:05:00Z",
  "metadata": {
    "source": "conversation_summary",
    "tool_id": "mosaic.memory.save",
    "conversation_turn": 5
  }
}
```

### Graph Relationship Schema (OmniRAG Pattern)
MUST implement embedded JSON relationships in NoSQL documents:
```json
{
  "id": "pypi_flask",
  "libtype": "pypi", 
  "libname": "flask",
  "developers": ["contact@palletsprojects.com"],
  "dependency_ids": ["pypi_werkzeug", "pypi_jinja2"],
  "used_by_lib": ["pypi_flask_sqlalchemy"],
  "embedding": [0.012, "...", -0.045]
}
```

## Development Workflow

### Azure CLI and azd CLI Requirements
- MUST use Azure Developer CLI (azd) for all deployment operations
- MUST use `azd up` for complete environment provisioning
- MUST use `azd deploy` for application deployment
- MUST use `az cli` for individual service management
- MUST create Bicep templates for all infrastructure

### Testing Requirements
- MUST run tests before any deployment
- MUST validate all FR requirements in tests
- MUST test MCP protocol compliance
- MUST test Azure service integration

### Code Standards
- MUST use Python type hints everywhere
- MUST implement comprehensive error handling
- MUST add structured logging throughout
- MUST follow Semantic Kernel plugin patterns
- MUST validate all inputs and outputs

## Project Structure Requirements

MUST follow this exact structure:
```
src/mosaic/
├── server/          # FastMCP server (FR-1, FR-3, FR-14)
│   ├── main.py      # FastMCP application with Streamable HTTP
│   ├── kernel.py    # Semantic Kernel management
│   └── auth.py      # OAuth 2.1 authentication utilities
├── plugins/         # Semantic Kernel plugins (FR-2)
│   ├── retrieval.py # RetrievalPlugin (FR-5, FR-6, FR-7) - OmniRAG
│   ├── refinement.py # RefinementPlugin (FR-8) - cross-encoder/ms-marco-MiniLM-L-12-v2
│   ├── memory.py    # MemoryPlugin (FR-9, FR-10, FR-11) - OmniRAG
│   └── diagram.py   # DiagramPlugin (FR-12, FR-13)
├── models/          # Data models and schemas
├── utils/           # Utility functions
└── config/          # Configuration management
```

## Infrastructure Requirements

### Bicep Templates (infra/ directory) - Simplified Architecture
MUST create:
- `main.bicep` - Main template orchestrator
- `container-apps.bicep` - Container Apps (Consumption Plan)
- `cosmos-db.bicep` - Cosmos DB (Free tier, unified backend for vector search, NoSQL)
- `redis.bicep` - Redis (Basic C0 tier)
- `ml-workspace.bicep` - ML workspace + endpoint (cross-encoder/ms-marco-MiniLM-L-12-v2)
- `functions.bicep` - Functions (Consumption Plan)
- `openai.bicep` - OpenAI service connection
- `entra-id.bicep` - Microsoft Entra ID OAuth 2.1 configuration

### Azure Functions
MUST create `functions/memory-consolidator/` for FR-11 compliance

## Security and Configuration

### Environment Variables (Managed Identity + OAuth 2.1)
MUST configure for unified architecture:
- `AZURE_OPENAI_ENDPOINT` (managed identity auth)
- `AZURE_COSMOS_DB_ENDPOINT` (managed identity auth, unified backend)
- `AZURE_REDIS_ENDPOINT` (managed identity auth)
- `AZURE_ML_ENDPOINT_URL` (cross-encoder model)
- `AZURE_TENANT_ID` (OAuth 2.1 configuration)
- `AZURE_CLIENT_ID` (OAuth 2.1 configuration)

### Git Configuration
MUST include in `.gitignore`:
- `conport/` (MCP tools directory)
- `.claude/settings.local.json`
- `.env` and environment files
- Azure deployment artifacts

## Development Commands

### Core Development Commands

**Testing and Quality Assurance:**
```bash
# Run all tests
pytest

# Run tests by category
pytest -m unit                    # Unit tests only
pytest -m integration            # Integration tests only  
pytest -m "not azure"           # Skip Azure-dependent tests
pytest -m "not slow"            # Skip slow tests

# Code quality checks
black src/ tests/                # Code formatting
isort src/ tests/                # Import sorting
mypy src/                        # Type checking
bandit -r src/                   # Security scanning
pre-commit run --all-files       # Run all pre-commit hooks

# Install pre-commit hooks
pre-commit install
```

**Package Management:**
```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install project in development mode
pip install -e .

# Install specific dependency groups
pip install -e ".[dev]"          # Development dependencies
pip install -e ".[test]"         # Testing dependencies
pip install -e ".[docs]"         # Documentation dependencies
```

**Azure Deployment (Primary Workflow):**
```bash
# Initial setup
az login
azd auth login

# Full environment provisioning and deployment
azd up

# Application deployment only
azd deploy

# Infrastructure management
azd provision                    # Infrastructure only
azd down                        # Tear down environment

# Model deployment (semantic reranking)
az ml model deploy              # Deploy cross-encoder/ms-marco-MiniLM-L-12-v2
```

**Local Development:**
```bash
# Run MCP server locally
python -m mosaic.server.main

# Run with specific environment
ENVIRONMENT=development python -m mosaic.server.main

# Start server with debug logging
python -m mosaic.server.main --log-level debug
```

## Success Criteria

The project is successful when:
1. All FR-1 through FR-14 requirements are implemented (including OAuth 2.1)
2. All MCP interface functions work correctly via FastMCP framework
3. Simplified Azure architecture is properly configured (unified Cosmos DB backend)
4. Complete `azd up` workflow succeeds with OmniRAG pattern
5. MCP protocol compliance is validated (Streamable HTTP + OAuth 2.1)
6. All tests pass consistently
7. cross-encoder/ms-marco-MiniLM-L-12-v2 model is operational

## Problem Statement Reminders

This tool solves:
- **Context is More Than a Prompt**: Sophisticated multi-source retrieval
- **AI "Amnesia"**: Persistent, multi-layered memory system
- **Dependency Blindness**: Graph-based code analysis
- **Contextual Noise**: Semantic reranking and refinement
- **Lack of Interoperability**: Standardized MCP protocol

## Key Architecture Patterns

**Plugin-Based Architecture:** All functionality MUST be implemented as Semantic Kernel Plugins. The system uses a plugin-based architecture where each major capability (retrieval, refinement, memory, diagrams) is implemented as a separate plugin that can be composed together.

**OmniRAG Unified Backend:** The project uses Microsoft's OmniRAG pattern with Azure Cosmos DB as the unified backend for vector search, graph relationships (embedded as JSON arrays), and memory storage. This eliminates the need for separate vector databases or graph databases.

**Multi-Layered Memory System:** 
- **Short-term**: Azure Redis Cache for conversational state
- **Long-term**: Azure Cosmos DB for persistent knowledge with LLM-powered consolidation
- **Consolidation**: Azure Functions with timer trigger for background memory processing

**FastMCP Protocol Implementation:** Uses FastMCP Python library for MCP protocol compliance with Streamable HTTP transport, enabling real-time context streaming to AI applications.

**Cross-Encoder Semantic Reranking:** Deploys cross-encoder/ms-marco-MiniLM-L-12-v2 model on Azure ML Endpoint to address the "lost in the middle" problem in retrieval systems.

## Current Implementation Status

**CRITICAL IMPLEMENTATION GAP:** The system lacks the fundamental code ingestion pipeline required to populate the knowledge graph with actual codebase data. While query, memory, and refinement capabilities exist, the system cannot currently ingest and analyze codebases.

**Completed:** Infrastructure, deployment pipeline, plugin architecture, memory system
**Missing:** Code ingestion, real-time updates, AI integration capabilities

See `docs/IMPLEMENTATION_ROADMAP.md` for detailed implementation plan and `docs/CODE_INGESTION_ANALYSIS.md` for gap analysis.

## Key Success Metrics

- MCP protocol compliance (FR-1)
- Semantic Kernel plugin architecture (FR-2) 
- Real-time SSE communication (FR-3)
- Azure native deployment (FR-4)
- All functional requirements (FR-5 through FR-14)
- Complete end-to-end workflow with `azd up`
- 80% minimum test coverage requirement