# Mosaic MCP Tool - Project Memory & Development Guidelines

## Project Overview

The Mosaic MCP Tool is a standardized, high-performance Model Context Protocol (MCP) Tool that provides a unified, multi-layered framework for advanced context engineering. It serves as a centralized "brain" that AI applications can connect to for knowledge retrieval, dependency analysis, memory management, and context refinement.

## Core Architecture Requirements

### Mandatory Technology Stack
- **Core Framework**: Python Semantic Kernel (FR-2) - ALL functionality MUST be implemented as Semantic Kernel Plugins
- **LLM & Embedding Models**: Azure OpenAI Service exclusively
- **Hosting**: Azure Container Apps (Consumption Plan for POC)
- **Communication**: FastAPI with Server-Sent Events (SSE) for FR-3 compliance
- **Hybrid Search**: Azure AI Search (Free F tier for POC)
- **Graph Database**: Azure Cosmos DB (Gremlin API, Free tier)
- **Long-Term Memory**: Azure Cosmos DB (NoSQL API, Free tier)
- **Short-Term Memory**: Azure Cache for Redis (Basic C0 tier)
- **Memory Consolidation**: Azure Functions (Consumption Plan, Timer Trigger)
- **Semantic Reranking**: Azure Machine Learning Endpoint hosting cross-encoder/ms-marco-MiniLM-L-12-v2

### Required Azure POC SKUs (TDD Section 8.0)
- Azure Container Apps: Consumption Plan (serverless, pay-per-use)
- Azure AI Search: Free (F) Tier (up to 3 indexes, 50 MB, supports vector search)
- Azure Cosmos DB: Free Tier (1000 RU/s, 25 GB, both NoSQL and Gremlin APIs)
- Azure Cache for Redis: Basic (C0) Tier (250MB cache)
- Azure Machine Learning: Pay-as-you-go (Standard_DS2_v2 compute)
- Azure Functions: Consumption Plan (serverless)
- Azure OpenAI Service: Pay-as-you-go

## Functional Requirements Compliance

### FR-1: MCP Server Implementation
- MUST implement MCP-compliant server exposing tools and resources
- MUST handle MCP protocol messaging correctly
- MUST provide standardized communication channel for AI components

### FR-2: Semantic Kernel Integration
- ALL functionality MUST be implemented as Semantic Kernel Plugins
- MUST promote modularity and reusability
- MUST allow easy composition and extension of context engineering capabilities

### FR-3: Asynchronous SSE Communication
- MUST use Server-Sent Events (SSE) for all MCP client communication
- MUST enable real-time, non-blocking context streaming
- MUST prevent blocking I/O during context retrieval

### FR-4: Azure Native Deployment
- MUST deploy on Azure using specified POC SKUs
- MUST ensure scalability, reliability, and performance
- MUST leverage managed Azure services exclusively

### FR-5: Hybrid Search (RetrievalPlugin)
- MUST orchestrate Vector Search and Keyword Search in parallel
- MUST use AzureAISearchMemoryStore Semantic Kernel connector
- MUST capture both semantic and lexical relevance

### FR-6: Graph-Based Code Analysis (RetrievalPlugin)
- MUST ingest codebases and parse into dependency graphs
- MUST use gremlinpython library for Cosmos DB Gremlin API
- MUST provide query functions for structural understanding

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

### FR-10: Multi-Layered Storage (MemoryPlugin)
- MUST support hybrid storage model
- MUST use Redis for short-term memory (conversational state)
- MUST use Cosmos DB for long-term memory (persistent knowledge)
- MUST implement HybridMemory class

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

## MCP Interface Requirements (TDD Section 6.0)

MUST implement these exact function signatures:
- `mosaic.retrieval.hybrid_search(query: str) -> List[Document]`
- `mosaic.retrieval.query_code_graph(gremlin_query: str) -> List[GraphNode]`
- `mosaic.refinement.rerank(query: str, documents: List[Document]) -> List[Document]`
- `mosaic.memory.save(session_id: str, content: str, type: str)`
- `mosaic.memory.retrieve(session_id: str, query: str, limit: int) -> List[MemoryEntry]`
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

### Graph Database Schema (Gremlin API)
MUST implement:
- **Node Types**: File, Function, Class, Variable, Module
- **Edge Types**: IMPORTS, CALLS, INHERITS_FROM, CONTAINS

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
├── server/          # FastAPI MCP server (FR-1, FR-3)
│   ├── main.py      # FastAPI application with SSE
│   ├── kernel.py    # Semantic Kernel management
│   └── mcp_utils.py # Custom MCP protocol utilities
├── plugins/         # Semantic Kernel plugins (FR-2)
│   ├── retrieval.py # RetrievalPlugin (FR-5, FR-6, FR-7)
│   ├── refinement.py # RefinementPlugin (FR-8)
│   ├── memory.py    # MemoryPlugin (FR-9, FR-10, FR-11)
│   └── diagram.py   # DiagramPlugin (FR-12, FR-13)
├── models/          # Data models and schemas
├── utils/           # Utility functions
└── config/          # Configuration management
```

## Infrastructure Requirements

### Bicep Templates (infra/ directory)
MUST create:
- `main.bicep` - Main template orchestrator
- `container-apps.bicep` - Container Apps (Consumption Plan)
- `ai-search.bicep` - AI Search (Free F tier)
- `cosmos-db.bicep` - Cosmos DB (Free tier, NoSQL + Gremlin)
- `redis.bicep` - Redis (Basic C0 tier)
- `ml-workspace.bicep` - ML workspace + endpoint
- `functions.bicep` - Functions (Consumption Plan)
- `openai.bicep` - OpenAI service connection

### Azure Functions
MUST create `functions/memory-consolidator/` for FR-11 compliance

## Security and Configuration

### Environment Variables
MUST configure for all Azure services:
- `AZURE_OPENAI_API_KEY`
- `AZURE_SEARCH_SERVICE_NAME`
- `AZURE_COSMOS_DB_CONNECTION_STRING`
- `AZURE_REDIS_CONNECTION_STRING`
- `AZURE_ML_ENDPOINT_URL`

### Git Configuration
MUST include in `.gitignore`:
- `conport/` (MCP tools directory)
- `.claude/settings.local.json`
- `.env` and environment files
- Azure deployment artifacts

## Development Commands

### Required Commands
- `azd up` - Complete environment provisioning
- `azd deploy` - Application deployment
- `az ml model deploy` - ML model deployment
- `gh repo create Mosaic --private` - Repository creation

### Testing Commands
- Run full test suite before any deployment
- Validate MCP protocol compliance
- Test all Azure service connections

## Success Criteria

The project is successful when:
1. All FR-1 through FR-13 requirements are implemented
2. All MCP interface functions work correctly
3. All Azure POC SKUs are properly configured
4. Complete `azd up` workflow succeeds
5. MCP protocol compliance is validated
6. All tests pass consistently

## Problem Statement Reminders

This tool solves:
- **Context is More Than a Prompt**: Sophisticated multi-source retrieval
- **AI "Amnesia"**: Persistent, multi-layered memory system
- **Dependency Blindness**: Graph-based code analysis
- **Contextual Noise**: Semantic reranking and refinement
- **Lack of Interoperability**: Standardized MCP protocol

## Key Success Metrics

- MCP protocol compliance (FR-1)
- Semantic Kernel plugin architecture (FR-2)
- Real-time SSE communication (FR-3)
- Azure native deployment (FR-4)
- All functional requirements (FR-5 through FR-13)
- Complete end-to-end workflow with `azd up`