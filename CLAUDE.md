# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Mosaic MCP Tool is a standardized, high-performance Model Context Protocol (MCP) server that provides advanced context engineering for AI applications. It serves as a centralized "brain" for knowledge retrieval, dependency analysis, memory management, and context refinement.

## Critical Architecture: Three-Service Separation

**ARCHITECTURAL UPDATE (2025-01-21)**: The system has been separated into three distinct services to solve performance and scaling issues:

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

### Service 3: Mosaic UI (Interactive Web Interface)

- **Location**: `src/mosaic-ui/`
- **Purpose**: Interactive Streamlit web application for knowledge graph exploration
- **Resources**: 0.5 CPU, 1Gi memory (web interface)
- **Components**: D3.js/Pyvis/Plotly graph visualizations, OmniRAG integration, chat interface

### Shared Backend

- **OmniRAG Pattern**: Unified Azure Cosmos DB for vector search, graph relationships, and memory storage
- **Authentication**: Microsoft Entra ID OAuth 2.1 for production security

## Technology Stack (Mandatory)

### Core Framework

- **Python Semantic Kernel**: ALL functionality MUST be implemented as Semantic Kernel plugins
- **FastMCP**: Python library for MCP protocol compliance with Streamable HTTP transport
- **Streamlit**: Interactive web application framework for UI service
- **Azure OpenAI Service**: Exclusively for LLM and embedding models
- **Azure Container Apps**: Consumption Plan hosting

### Data Backend (OmniRAG Pattern)

- **Azure Cosmos DB for NoSQL**: Unified backend for vector search, graph, and memory
- **Azure Cache for Redis**: Short-term memory (Basic C0 tier)
- **Azure Machine Learning**: cross-encoder/ms-marco-MiniLM-L-12-v2 for semantic reranking
- **Azure Functions**: Memory consolidation with timer triggers

### OmniRAG Enhancement Dependencies

- **RDFLib**: RDF triple store and SPARQL query processing
- **NetworkX**: Graph relationship modeling and traversal
- **Transformers**: Intent detection and query classification
- **SPARQLWrapper**: SPARQL endpoint integration

### UI Service Dependencies

- **Streamlit**: Core web application framework
- **D3.js**: Advanced interactive graph visualizations
- **Pyvis**: Network graph visualizations with physics simulation
- **Plotly**: Statistical and analytical graph representations
- **Pandas/NumPy**: Data processing for graph analytics

## Required MCP Interface Functions

The system MUST implement these exact function signatures:

```python
# Query Server (Real-time) - Core Functions
mosaic.retrieval.hybrid_search(query: str) -> List[Document]
mosaic.retrieval.query_code_graph(library_id: str, relationship_type: str) -> List[LibraryNode]
mosaic.refinement.rerank(query: str, documents: List[Document]) -> List[Document]
mosaic.memory.save(session_id: str, content: str, type: str)
mosaic.memory.retrieve(session_id: str, query: str, limit: int) -> List[MemoryEntry]
mosaic.diagram.generate(description: str) -> str

# Query Server - OmniRAG Enhanced Functions
mosaic.retrieval.hierarchical_vector_search(query: str, parent_id: str, entity_type: str) -> List[Dict]
mosaic.retrieval.get_entity_children(entity_id: str, max_depth: int) -> List[Dict]
mosaic.retrieval.get_entity_parents(entity_id: str, max_depth: int) -> List[Dict]
mosaic.retrieval.get_entity_siblings(entity_id: str) -> List[Dict]

# Mosaic Ingestion Service (Background)
python -m mosaic_ingestion.main --repository-url <url> --branch <branch>
az containerapp job start --name mosaic-ingestion-job-dev --resource-group rg-dev

# UI Service Functions
streamlit run src/mosaic-ui/app.py
# OmniRAG UI integration for query routing and graph filtering
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

# Run tests with different scopes
pytest --tb=short                    # All tests
pytest -m unit                       # Unit tests only
pytest -m integration                # Integration tests only
pytest -m "not azure"                # Skip Azure service tests
pytest tests/test_specific_file.py   # Run specific test file

# Run tests with coverage
pytest --cov=src/mosaic-mcp --cov-report=html --cov-report=term-missing

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

# Install using pyproject.toml (recommended)
pip install -e ".[dev,test]"

# Alternative: Install using requirements files
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-ingestion.txt

# Run Query Server locally (requires Azure credentials)
python -m mosaic_mcp.server.main

# Run using CLI script (if installed with pip install -e .)
mosaic-server

# Ingestion Service Options:

# 1. Local Development Ingestion (minimal dependencies, GitPython only)
python -m src.mosaic_ingestion.local_main --repository-url https://github.com/user/repo --branch main --debug

# 2. Full Mosaic Ingestion Service (requires complete Azure setup with Semantic Kernel)
python -m src.mosaic_ingestion.main --repository-url <url> --branch main

# 3. Test specific ingestion components
python -m src.mosaic_ingestion.plugins.graph_builder  # Test graph building
python src/mosaic_ingestion/validate_graph_builder.py  # Validate graph builder
python src/mosaic_ingestion/standalone_sparql_validation.py  # Test SPARQL

# Test OmniRAG RDF components (development)
python -m src.mosaic_ingestion.rdf.test_ontology_manager
python -m src.mosaic_ingestion.rdf.test_triple_generator

# Run UI Service locally
cd src/mosaic-ui
streamlit run app.py

# Run UI tests with coverage
cd src/mosaic-ui
python3 run_tests.py                    # All tests with coverage
python3 run_tests.py --unit             # Unit tests only
python3 run_tests.py --integration      # Integration tests only
python3 run_tests.py --performance      # Performance benchmarks
python3 run_tests.py --verbose          # Detailed output

# Development workflow with hot reload
# Note: Configure your IDE to run the server with auto-reload for development
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

### Code Entity Schema (Enhanced Ingestion Service)

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
  "timestamp": "2025-01-21T12:05:00Z",
  "rdf_triples": [
    {
      "subject": "file://path/file.py#example_function",
      "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
      "object": "http://mosaic.ai/code#Function"
    }
  ],
  "relationships": [
    {
      "type": "calls",
      "target": "entity_id_2",
      "metadata": {"line_number": 42}
    }
  ]
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

### UI Service Data Structure

```json
{
  "entities": [
    {
      "id": "unique_identifier",
      "name": "display_name", 
      "category": "server|plugin|service|test",
      "lines": 285,
      "language": "python",
      "description": "entity_description"
    }
  ],
  "relationships": [
    {
      "source": "source_entity_id",
      "target": "target_entity_id", 
      "type": "imports|calls|uses|depends_on",
      "strength": 0.8
    }
  ]
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
# Core Azure Services
AZURE_OPENAI_ENDPOINT         # Azure OpenAI service endpoint
AZURE_COSMOS_DB_ENDPOINT      # Cosmos DB endpoint (unified backend)
AZURE_REDIS_ENDPOINT          # Redis cache endpoint
AZURE_ML_ENDPOINT_URL         # ML endpoint for semantic reranking
AZURE_TENANT_ID               # OAuth 2.1 tenant ID
AZURE_CLIENT_ID               # OAuth 2.1 client ID

# UI Service Configuration (Optional)
STREAMLIT_SERVER_PORT=8501    # Streamlit server port
STREAMLIT_SERVER_ADDRESS=0.0.0.0  # Server address
PYTHONPATH=.                  # Python path for imports
```

## Code Standards (Mandatory)

### Python Development

- **Type Hints**: Required everywhere (enforced by mypy)
- **Async/Await**: All I/O operations must be async
- **Error Handling**: Comprehensive try-catch blocks with structured logging
- **Logging**: Use structlog for structured logging with proper levels
- **Testing**: Unit tests for all plugins with pytest, 80%+ coverage required
- **Code Formatting**: Use ruff for linting and formatting (replaces black/flake8)
- **Import Sorting**: Configured in pyproject.toml with isort profile "black"

### Plugin Architecture

- **Semantic Kernel Plugins**: All functionality as SK plugins
- **Modular Design**: Each plugin focuses on specific FR requirements  
- **OmniRAG Integration**: Use unified Cosmos DB backend (not separate services)
- **FastMCP Framework**: All MCP tools exposed via FastMCP
- **Dependency Injection**: Use proper DI patterns for Azure service clients

### Testing Standards

```bash
# Test markers available (configured in pyproject.toml):
pytest -m unit           # Fast unit tests
pytest -m integration    # Integration tests (slower)
pytest -m azure          # Tests requiring Azure services
pytest -m "not slow"     # Skip slow tests
```

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

#### Query Server (MCP)
- `src/mosaic-mcp/server/main.py`: FastMCP server entry point
- `src/mosaic-mcp/server/kernel.py`: Semantic Kernel management
- `src/mosaic-mcp/server/auth.py`: OAuth 2.1 authentication utilities
- `src/mosaic-mcp/plugins/`: Query Server plugins (retrieval, refinement, memory, diagram)

#### UI Service (Streamlit)
- `src/mosaic-ui/app.py`: Main Streamlit application (1859 lines)
- `src/mosaic-ui/plugins/omnirag_plugin.py`: OmniRAG integration for UI (479 lines)
- `src/mosaic-ui/run_tests.py`: Test runner script with coverage reporting
- `src/mosaic-ui/tests/`: Comprehensive test suite (6 files, 4400+ lines)

### Ingestion Service Implementation

- `src/mosaic-ingestion/main.py`: Full ingestion service with Semantic Kernel orchestration
- `src/mosaic-ingestion/local_main.py`: Local development version (minimal dependencies)
- `src/mosaic-ingestion/orchestrator.py`: AI agent orchestration system
- `src/mosaic-ingestion/plugins/ingestion.py`: Repository processing logic
- `src/mosaic-ingestion/plugins/graph_builder.py`: Knowledge graph construction
- `src/mosaic-ingestion/plugins/ai_code_parser.py`: AI-powered code analysis
- `src/mosaic-ingestion/models/golden_node.py`: Core data model for entities

### OmniRAG Components

#### Ingestion Service RDF Infrastructure
- `src/mosaic-ingestion/rdf/`: RDF processing and ontology management
- `src/mosaic-ingestion/rdf/ontology_manager.py`: OWL ontology management
- `src/mosaic-ingestion/rdf/triple_generator.py`: RDF triple generation
- `src/mosaic-ingestion/rdf/sparql_builder.py`: Dynamic SPARQL query construction
- `src/mosaic-ingestion/ontologies/`: Domain-specific OWL ontologies (code_base.owl, python.owl)

#### Query Server OmniRAG
- `src/mosaic-mcp/plugins/vector_search.py`: Hierarchical vector search capabilities
- `src/mosaic-mcp/plugins/graph_data_service.py`: Graph relationship queries
- `src/mosaic-mcp/plugins/omnirag_orchestrator.py`: Main OmniRAG coordination
- `src/mosaic-mcp/plugins/enhanced_omnirag.py`: Advanced OmniRAG features
- `src/mosaic-mcp/plugins/nl2sparql_translator.py`: Natural language to SPARQL conversion
- `src/mosaic-mcp/plugins/sparql_query_executor.py`: SPARQL execution engine

### Configuration and Dependencies

- `pyproject.toml`: Python project configuration, dependencies, and tool settings
- `requirements.txt`: Base runtime dependencies
- `requirements-dev.txt`: Development dependencies
- `requirements-ingestion.txt`: Ingestion service specific dependencies
- `environment-variables.template`: Template for required environment variables

### Infrastructure and Deployment

- `infra/main.bicep`: Main infrastructure orchestration
- `infra/query-server.bicep`: Query Server Container App
- `infra/ingestion-service.bicep`: Ingestion Service Container Job
- `azure.yaml`: Azure Developer CLI configuration

### Testing

- `tests/`: Main test files with unit, integration, and Azure service tests
- `src/mosaic-mcp/tests/`: MCP service specific tests (6 files, 4200+ lines)
- `src/mosaic-ingestion/tests/`: Ingestion service specific tests (6 files, 4300+ lines)
- `src/mosaic-ui/tests/`: UI service specific tests (6 files, 4400+ lines)
- `src/mosaic-mcp/pytest.ini`: Pytest configuration for MCP service
- `pytest` configuration in `pyproject.toml`: Main test configuration

### Documentation and Guidelines

- `docs/TDD_UNIFIED.md`: Complete technical design document
- `docs/omnirag-implementation/`: OmniRAG transformation documentation
- `ARCHITECTURE_LOG.md`: Architectural decision log
- `.cursorrules`: Cursor AI development guidelines and requirements
- `.github/copilot-instructions.md`: GitHub Copilot specific instructions

## Success Criteria

The system is ready for production when:

1. All FR-1 through FR-15 requirements are implemented
2. Both services deploy successfully with `azd up`
3. MCP protocol compliance validated with Streamable HTTP
4. Query Server responds <100ms for hybrid search
5. Ingestion Service processes repositories with 11-language support
6. OAuth 2.1 authentication working with Microsoft Entra ID
7. OmniRAG pattern operational with unified Cosmos DB backend

## Current Status (2025-01-26)

### ✅ Completed (Production Ready)

- Three-service architectural separation
- Query Server with FastMCP and all plugins
- Ingestion Service with 11-language AST parsing
- UI Service with interactive graph visualizations and OmniRAG integration
- Azure infrastructure templates (Bicep)
- Azure Developer CLI configuration
- Unified TDD documentation
- Memory plugin with multi-layered storage (Redis + Cosmos DB)
- Hierarchical vector search capabilities in RetrievalPlugin
- Comprehensive test suites for all three services (70%+ coverage)

### 🔄 OmniRAG Transformation (In Progress)

- RDF infrastructure components (`src/mosaic-ingestion/rdf/`)
- Ontology definitions (`src/mosaic-ingestion/ontologies/`)
- Graph-based relationship modeling
- Intent detection and query routing
- Multi-source context orchestration

### 📋 Memory Tools Available

- **Current Memory Graph**: Empty (use MCP `read-graph` tool to verify)
- **Memory Operations**: Save, retrieve, clear with importance scoring
- **Storage Layers**: Redis (short-term) + Cosmos DB (long-term)
- **Search Capabilities**: Vector similarity + text matching hybrid search

### 📈 Next Priority: OmniRAG Implementation

Follow the transformation guide in `docs/omnirag-implementation/` to evolve from Basic RAG to OmniRAG pattern with RDF triples, SPARQL queries, and intelligent query routing.

The critical architectural work is complete and the system is ready for Azure deployment with proper separation of concerns between real-time queries, heavy ingestion operations, and interactive web-based visualization.

## Common Development Workflows

### Adding New MCP Tools

1. Create the tool function in appropriate plugin (e.g., `src/mosaic-mcp/plugins/retrieval.py`)
2. Implement Semantic Kernel plugin pattern with proper type hints
3. Add MCP tool registration in server main module
4. Write unit tests with appropriate markers (`@pytest.mark.unit`)
5. Update CLAUDE.md if the tool changes core functionality

### Debugging MCP Protocol Issues

```bash
# Enable debug logging for MCP communication
export MOSAIC_LOG_LEVEL=DEBUG
python -m mosaic_mcp.server.main

# Test MCP tools directly using the CLI
# (Implementation depends on your MCP client setup)

# Check FastMCP server health
curl http://localhost:8000/health
```

### Working with Cosmos DB and OmniRAG

```bash
# Validate Cosmos DB connection
python -c "from mosaic_mcp.plugins.vector_search import VectorSearchPlugin; VectorSearchPlugin().test_connection()"

# Test RDF triple generation
python -m src.mosaic_ingestion.rdf.test_triple_generator

# Debug SPARQL queries
python -m src.mosaic_mcp.plugins.sparql_query_executor --debug

# Test ingestion components individually
python src/mosaic_ingestion/validate_direct.py  # Direct validation
python src/mosaic_ingestion/validate_sparql.py  # SPARQL validation
```

### Understanding Ingestion Service Modes

**Local Development Mode** (`local_main.py`):
- Uses only GitPython for repository cloning
- Basic file scanning and statistics
- No Azure dependencies required
- Ideal for testing and development

**Full Production Mode** (`main.py`):
- Semantic Kernel orchestration with AI agents
- Complete AST parsing with tree-sitter
- Azure Cosmos DB integration
- OmniRAG pattern implementation

### Working with UI Service and Graph Visualizations

```bash
# Run Streamlit app with specific configuration
cd src/mosaic-ui
streamlit run app.py --server.port 8501 --logger.level debug

# Test graph visualization components individually
python3 -m pytest tests/test_graph_visualizations.py::TestD3GraphVisualization -v

# Test OmniRAG integration in UI
python3 -m pytest tests/test_omnirag_plugin.py::TestOmniRAGPlugin::test_strategy_determination -v

# Performance testing for UI components
python3 -m pytest tests/test_performance_ui.py::TestGraphRenderingPerformance -v

# Test session state management
python3 -m pytest tests/test_streamlit_integration.py::TestSessionState -v
```

### UI Service Architecture Key Points

**Session State Management**: Uses Streamlit's session state for persistence of graph data, chat history, and selected nodes.

**Graph Visualization Stack**: Three distinct approaches:
- **D3.js**: Advanced interactive graphs with zoom, pan, and selection
- **Pyvis**: Network graphs with physics simulation
- **Plotly**: Statistical and analytical representations

**OmniRAG Integration**: Query routing logic determines optimal strategy:
- Database RAG for direct entity queries
- Vector RAG for semantic similarity
- Graph RAG for relationship traversal

### Performance Monitoring

- Use `structlog` with appropriate log levels for debugging
- Monitor Azure Container Apps metrics through Azure Portal
- Set up Application Insights for distributed tracing
- Use Redis cache monitoring for memory layer performance
- Monitor Streamlit app performance with built-in metrics (`st.experimental_memo` for caching)
