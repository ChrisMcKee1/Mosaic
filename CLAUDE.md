# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Mosaic MCP Tool is a standardized, high-performance Model Context Protocol (MCP) server that provides advanced context engineering for AI applications. It serves as a centralized "brain" for knowledge retrieval, dependency analysis, memory management, and context refinement.

## Critical Architecture: Two-Service Separation

**ARCHITECTURAL UPDATE (2025-01-21)**: The system has been separated into two distinct services to solve performance and scaling issues:

### Service 1: Query Server (Real-time MCP)
- **Location**: `src/mosaic-mcp/` old name just `mosaic`
- **Purpose**: Real-time MCP request handling with FastMCP framework
- **Resources**: 0.25 CPU, 0.5Gi memory (lightweight, always-on)
- **Components**: RetrievalPlugin, RefinementPlugin, MemoryPlugin, DiagramPlugin

### Service 2: Mosaic Ingestion Service (Heavy Processing)
- **Location**: `src/mosaic-ingestion/`
- **Purpose**: Repository ingestion and knowledge graph population
- **Resources**: 2.0 CPU, 4Gi memory (on-demand processing)
- **Components**: GitPython, tree-sitter AST parsing, entity extraction

### Shared Backend
- **OmniRAG Pattern**: Unified Azure Cosmos DB for vector search, graph relationships, and memory storage
- **Authentication**: Microsoft Entra ID OAuth 2.1 for production security

## Technology Stack (Mandatory)

### Core Framework
- **Python Semantic Kernel**: ALL functionality MUST be implemented as Semantic Kernel plugins
- **FastMCP**: Python library for MCP protocol compliance with Streamable HTTP transport
- **Azure OpenAI Service**: Exclusively for LLM and embedding models
- **Azure Container Apps**: Consumption Plan hosting

### Data Backend (OmniRAG Pattern)
- **Azure Cosmos DB for NoSQL**: Unified backend for vector search, graph, and memory
- **Azure Cache for Redis**: Short-term memory (Basic C0 tier)
- **Azure Machine Learning**: cross-encoder/ms-marco-MiniLM-L-12-v2 for semantic reranking
- **Azure Functions**: Memory consolidation with timer triggers

## Required MCP Interface Functions

The system MUST implement these exact function signatures:

```python
# Query Server (Real-time)
mosaic.retrieval.hybrid_search(query: str) -> List[Document]
mosaic.retrieval.query_code_graph(library_id: str, relationship_type: str) -> List[LibraryNode]
mosaic.refinement.rerank(query: str, documents: List[Document]) -> List[Document]
mosaic.memory.save(session_id: str, content: str, type: str)
mosaic.memory.retrieve(session_id: str, query: str, limit: int) -> List[MemoryEntry]
mosaic.diagram.generate(description: str) -> str

# Mosaic Ingestion Service (Background)
python -m mosaic_ingestion.main --repository-url <url> --branch <branch>
az containerapp job start --name mosaic-ingestion-job-dev --resource-group rg-dev
```

## Development Commands

### Deployment
```bash
# Complete two-service deployment
azd up

# Deploy individual services
azd deploy query-server
azd deploy ingestion-service

# Login and authentication
az login && azd auth login
```

### Code Quality
```bash
# Format and lint (REQUIRED before commits)
ruff check . --fix
ruff format .

# Type checking
mypy src/

# Run tests (if pytest available)
pytest --tb=short

# Git workflow
git add .
git commit -m "descriptive message"
```

### Manual Repository Ingestion
```bash
# Trigger ingestion job manually
az containerapp job start \
  --name mosaic-ingestion-job-dev \
  --resource-group rg-dev \
  --args "--repository-url https://github.com/user/repo --branch main"
```

### Local Development
```bash
# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install GitPython pydantic pydantic-settings python-dotenv

# Run Query Server locally (requires Azure credentials)
python -m mosaic_mcp.server.main

# Run Local Development Ingestion Service (minimal dependencies)
source venv/bin/activate
python3 -m src.mosaic_ingestion.local_main --repository-url https://github.com/user/repo --branch main --debug

# Run Full Mosaic Ingestion Service (requires complete Azure setup)
python -m mosaic_ingestion.main --repository-url <url> --branch main
```

## Data Models and Schemas

### Cosmos DB Memory Schema (EXACT FORMAT)
```json
{
  "id": "unique_memory_id",
  "sessionId": "user_or_agent_session_id",
  "type": "episodic | semantic | procedural",
  "content": "The user confirmed that the 'auth-service' should be written in Go.",
  "embedding": [0.012, "...", -0.045],
  "importanceScore": 0.85,
  "timestamp": "2025-01-21T12:05:00Z",
  "metadata": {
    "source": "conversation_summary",
    "tool_id": "mosaic.memory.save",
    "conversation_turn": 5
  }
}
```

### Code Entity Schema (Ingestion Service)
```json
{
  "id": "entity_md5_hash",
  "type": "code_entity",
  "entity_type": "function | class | module | import",
  "name": "function_name",
  "language": "python | javascript | java | go | rust | c | cpp | csharp | html | css",
  "file_path": "/absolute/path/to/file.py",
  "content": "def example_function():\n    pass",
  "embedding": [0.012, "...", -0.045],
  "timestamp": "2025-01-21T12:05:00Z"
}
```

### Graph Relationships (OmniRAG Pattern)
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

## Architecture Implementation Requirements

### Functional Requirements Compliance (FR-1 through FR-15)
All code must comply with these functional requirements:
- **FR-1**: MCP Server Implementation with Streamable HTTP
- **FR-2**: Semantic Kernel Integration (ALL functionality as plugins)
- **FR-3**: Streamable HTTP Communication (prevent blocking I/O)
- **FR-4**: Azure Native Deployment
- **FR-5**: Hybrid Search (OmniRAG unified Cosmos DB backend)
- **FR-6**: Graph-Based Code Analysis (embedded JSON relationships)
- **FR-7**: Candidate Aggregation
- **FR-8**: Semantic Reranking (cross-encoder/ms-marco-MiniLM-L-12-v2)
- **FR-9**: Unified Memory Interface
- **FR-10**: Multi-Layered Storage (OmniRAG pattern)
- **FR-11**: LLM-Powered Consolidation
- **FR-12**: Mermaid Generation
- **FR-13**: Mermaid as Context Resource
- **FR-14**: Secure MCP Endpoint (OAuth 2.1)
- **FR-15**: Repository Ingestion (11 languages with tree-sitter)

### Authentication and Security
- **Microsoft Entra ID**: OAuth 2.1 authentication for MCP endpoints
- **Managed Identity**: All Azure service connections use DefaultAzureCredential
- **No Secrets**: No connection strings or API keys in code

### Environment Variables (Required)
```bash
AZURE_OPENAI_ENDPOINT         # Azure OpenAI service endpoint
AZURE_COSMOS_DB_ENDPOINT      # Cosmos DB endpoint (unified backend)
AZURE_REDIS_ENDPOINT          # Redis cache endpoint
AZURE_ML_ENDPOINT_URL         # ML endpoint for semantic reranking
AZURE_TENANT_ID               # OAuth 2.1 tenant ID
AZURE_CLIENT_ID               # OAuth 2.1 client ID
```

## Code Standards (Mandatory)

### Python Development
- **Type Hints**: Required everywhere
- **Async/Await**: All I/O operations must be async
- **Error Handling**: Comprehensive try-catch blocks
- **Logging**: Structured logging with proper levels
- **Testing**: Unit tests for all plugins and MCP compliance

### Plugin Architecture
- **Semantic Kernel Plugins**: All functionality as SK plugins
- **Modular Design**: Each plugin focuses on specific FR requirements
- **OmniRAG Integration**: Use unified Cosmos DB backend (not separate services)
- **FastMCP Framework**: All MCP tools exposed via FastMCP

## Repository Ingestion Capabilities

The ingestion service supports **11 programming languages** with tree-sitter AST parsing:
- Python, JavaScript, TypeScript, Java, Go, Rust
- C, C++, C# (including Razor/Blazor)
- HTML, CSS (including preprocessors)

### Language-Specific Features
- **AST Parsing**: Complete syntax tree analysis for entity extraction
- **Entity Types**: Functions, classes, modules, imports, HTML elements, CSS rules
- **Relationship Modeling**: Cross-file dependencies and import analysis
- **Embedding Generation**: Azure OpenAI embeddings for semantic search

## Performance and Scalability

### Query Server Optimization
- **Response Time**: <100ms for hybrid search operations
- **Throughput**: 100+ concurrent MCP requests
- **Memory Efficiency**: Plugin caching and connection pooling
- **Auto-scaling**: 1-3 replicas based on load

### Ingestion Service Optimization
- **Repository Processing**: Large repositories (1GB+) supported
- **Parallel Processing**: Concurrent entity extraction
- **Batch Operations**: Efficient Cosmos DB population
- **Resource Isolation**: Heavy operations don't impact query performance

## Important File Locations

### Core Implementation
- `src/mosaic-mcp/server/main.py`: FastMCP server entry point
- `src/mosaic-mcp/plugins/`: Query Server plugins (retrieval, refinement, memory, diagram)
- `src/mosaic-ingestion/main.py`: Mosaic Ingestion service entry point
- `src/mosaic-ingestion/plugins/ingestion.py`: Repository processing logic

### Infrastructure and Deployment
- `infra/main.bicep`: Main infrastructure orchestration
- `infra/query-server.bicep`: Query Server Container App
- `infra/ingestion-service.bicep`: Ingestion Service Container Job
- `azure.yaml`: Azure Developer CLI configuration

### Documentation
- `docs/TDD_UNIFIED.md`: Complete technical design document
- `ARCHITECTURE_LOG.md`: Architectural decision log
- `.cursorrules`: Development guidelines and requirements

## Success Criteria

The system is ready for production when:
1. All FR-1 through FR-15 requirements are implemented
2. Both services deploy successfully with `azd up`
3. MCP protocol compliance validated with Streamable HTTP
4. Query Server responds <100ms for hybrid search
5. Ingestion Service processes repositories with 11-language support
6. OAuth 2.1 authentication working with Microsoft Entra ID
7. OmniRAG pattern operational with unified Cosmos DB backend

## Current Status (2025-01-21)

### ✅ Completed (Production Ready)
- Two-service architectural separation
- Query Server with FastMCP and all plugins
- Ingestion Service with 11-language AST parsing
- Azure infrastructure templates (Bicep)
- Azure Developer CLI configuration
- Unified TDD documentation

### 🔄 Remaining (Low Priority)
- Integration testing between services
- Performance benchmarking under load
- Production deployment validation

The critical architectural work is complete and the system is ready for Azure deployment with proper separation of concerns between real-time queries and heavy ingestion operations.