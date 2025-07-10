# Mosaic MCP Tool

> Advanced Context Engineering for AI Applications

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Azure](https://img.shields.io/badge/Azure-Native-blue)](https://azure.microsoft.com/)

## Overview

The Mosaic MCP Tool is a standardized, high-performance Model Context Protocol (MCP) Tool that provides a unified, multi-layered framework for advanced context engineering. It serves as a centralized "brain" that AI applications can connect to for knowledge retrieval, dependency analysis, memory management, and context refinement.

## Features

- **🚀 MCP Server Implementation**: Full MCP protocol compliance with Server-Sent Events (SSE)
- **🧠 Semantic Kernel Integration**: Modular plugin-based architecture
- **☁️ Azure Native**: Deployed on Azure with POC-optimized SKUs
- **🔍 Hybrid Search**: Vector and keyword search with Azure AI Search
- **🕸️ Graph Code Analysis**: Dependency analysis with Azure Cosmos DB Gremlin API
- **🧠 Multi-Layered Memory**: Redis + Cosmos DB persistent agent memory
- **📊 Semantic Reranking**: Cross-encoder model for context refinement
- **📈 Mermaid Diagrams**: AI-powered architectural documentation

## Architecture

The Mosaic MCP Tool is built with:

- **Core**: Python Semantic Kernel with FastAPI
- **Hosting**: Azure Container Apps (Consumption Plan)
- **Search**: Azure AI Search (Free tier)
- **Graph DB**: Azure Cosmos DB (Gremlin API)
- **Memory**: Redis (short-term) + Cosmos DB (long-term)
- **ML**: Azure Machine Learning (cross-encoder reranking)
- **Functions**: Azure Functions (memory consolidation)

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
├── server/          # FastAPI MCP server
│   ├── main.py      # FastAPI application with SSE
│   ├── kernel.py    # Semantic Kernel management
│   └── mcp_utils.py # Custom MCP protocol utilities
├── plugins/         # Semantic Kernel plugins
│   ├── retrieval.py # RetrievalPlugin (hybrid search, graph analysis)
│   ├── refinement.py # RefinementPlugin (semantic reranking)
│   ├── memory.py    # MemoryPlugin (multi-layered storage)
│   └── diagram.py   # DiagramPlugin (Mermaid generation)
├── models/          # Data models and schemas
├── utils/           # Utility functions
└── config/          # Configuration management
```

## MCP Interface

The tool exposes these MCP functions:

- `mosaic.retrieval.hybrid_search(query: str) -> List[Document]`
- `mosaic.retrieval.query_code_graph(gremlin_query: str) -> List[GraphNode]`
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Please read [CLAUDE.md](./CLAUDE.md) for development guidelines and contribution instructions.

## Support

For questions and support, please open an issue in the [GitHub repository](https://github.com/ChrisMcKee1/Mosaic/issues).