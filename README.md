# Mosaic MCP Tool

> Advanced Context Engineering for AI Applications

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Azure](https://img.shields.io/badge/Azure-Native-blue)](https://azure.microsoft.com/)

## Overview

The Mosaic MCP Tool is a standardized, high-performance Model Context Protocol (MCP) Tool that provides a unified, multi-layered framework for advanced context engineering. It serves as a centralized "brain" that AI applications can connect to for knowledge retrieval, dependency analysis, memory management, and context refinement.

## Features

- **🚀 MCP Server Implementation**: Full MCP protocol compliance with Streamable HTTP transport
- **🔐 OAuth 2.1 Security**: Microsoft Entra ID authentication for production-ready security
- **🧠 Semantic Kernel Integration**: Modular plugin-based architecture
- **☁️ Azure Native**: Deployed on Azure with simplified, cost-optimized architecture
- **🔍 Unified Data Backend**: OmniRAG pattern with Azure Cosmos DB (serverless) for vector search, graph analysis, and memory
- **📊 Semantic Reranking**: cross-encoder/ms-marco-MiniLM-L-12-v2 model on Azure ML Endpoint for context refinement
- **📈 Mermaid Diagrams**: AI-powered architectural documentation
- **⚡ FastMCP Framework**: Industry-standard MCP server implementation

## Architecture

The Mosaic MCP Tool is built with:

- **Core**: Python Semantic Kernel with FastMCP framework
- **Hosting**: Azure Container Apps (Consumption Plan)
- **Unified Backend**: Azure Cosmos DB for NoSQL (serverless, vector search, embedded graph relationships, memory)
- **Short-term Memory**: Azure Cache for Redis (Basic C0)
- **AI Models**: Azure OpenAI Service (GPT-4o 2024-11-20, text-embedding-3-small)
- **ML**: Azure Machine Learning (cross-encoder/ms-marco-MiniLM-L-12-v2)
- **Functions**: Azure Functions (memory consolidation, consumption plan)
- **Authentication**: Microsoft Entra ID (OAuth 2.1) + Managed Identity
- **DevOps**: Azure Developer CLI (azd) with Bicep templates

## Quick Start

### Prerequisites

- Python 3.10+
- Azure CLI
- Azure Developer CLI (azd)
- Azure subscription with appropriate permissions

### Installation

```bash
# Clone the repository
git clone https://github.com/ChrisMcKee1/Mosaic.git
cd Mosaic

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### Deployment

```bash
# Login to Azure
az login
azd auth login

# Deploy infrastructure and application
azd up
```

## Project Structure

```
src/mosaic/
├── server/          # FastMCP server implementation
│   ├── main.py      # FastMCP application with Streamable HTTP
│   ├── kernel.py    # Semantic Kernel management
│   └── auth.py      # OAuth 2.1 authentication utilities
├── plugins/         # Semantic Kernel plugins
│   ├── retrieval.py # RetrievalPlugin (unified Cosmos DB backend)
│   ├── refinement.py # RefinementPlugin (cross-encoder reranking)
│   ├── memory.py    # MemoryPlugin (OmniRAG pattern storage)
│   └── diagram.py   # DiagramPlugin (Mermaid generation)
├── models/          # Data models and schemas
├── utils/           # Utility functions
└── config/          # Configuration management
```

## MCP Interface

The tool exposes these MCP functions:

- `mosaic.retrieval.hybrid_search(query: str) -> List[Document]`
- `mosaic.retrieval.query_code_graph(library_id: str, relationship_type: str) -> List[LibraryNode]`
- `mosaic.refinement.rerank(query: str, documents: List[Document]) -> List[Document]`
- `mosaic.memory.save(session_id: str, content: str, type: str)`
- `mosaic.memory.retrieve(session_id: str, query: str, limit: int) -> List[MemoryEntry]`
- `mosaic.diagram.generate(description: str) -> str`

## Development

### Running Tests

```bash
pytest
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Security scanning
bandit -r src/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Configuration

See [CLAUDE.md](./CLAUDE.md) for comprehensive development guidelines and requirements.

## 🚨 **IMPLEMENTATION STATUS**

**CRITICAL NOTICE:** The Mosaic MCP Tool has a **critical implementation gap** that blocks AI-assisted development capabilities. While the system includes sophisticated querying, memory management, and context refinement, it **lacks the fundamental code ingestion pipeline** required to populate the knowledge graph with actual codebase data.

### **Current Status**
- ✅ **Infrastructure & Deployment** - Complete Azure architecture with `azd up`
- ✅ **Query & Retrieval** - Hybrid search and graph traversal capabilities  
- ✅ **Memory & Context** - Multi-layered storage with consolidation
- ✅ **Refinement & Diagrams** - Semantic reranking and Mermaid generation
- ❌ **Code Ingestion** - **MISSING** - Repository access, parsing, and graph construction
- ❌ **Real-time Updates** - **MISSING** - File monitoring and incremental updates
- ❌ **AI Integration** - **MISSING** - Generated code insertion and correlation

### **Implementation Documents**
- **[Implementation Roadmap](docs/IMPLEMENTATION_ROADMAP.md)** - Comprehensive 12-week implementation plan
- **[Code Ingestion Analysis](docs/CODE_INGESTION_ANALYSIS.md)** - Executive summary of the critical gap
- **[Technical Design Document](docs/TDD_UNIFIED.md)** - Unified system design and architecture specification

**Next Action:** Begin Phase 1 implementation with Context7 MCP tool research for technology validation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Please read [CLAUDE.md](./CLAUDE.md) for development guidelines and contribution instructions.

## Support

For questions and support, please open an issue in the [GitHub repository](https://github.com/ChrisMcKee1/Mosaic/issues).