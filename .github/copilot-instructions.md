# Mosaic MCP Tool Development Guidelines

## Project Context

You are working on the **Mosaic MCP Tool** - a comprehensive Model Context Protocol (MCP) server for enterprise-grade AI-driven software development workflows. This project integrates multiple Azure services and provides intelligent context management.

## Technology Stack

- **Primary Language**: Python 3.12+
- **Framework**: FastAPI for HTTP servers, direct MCP for protocol implementation
- **AI/ML**: Azure OpenAI Service, Semantic Kernel
- **Storage**: Azure Cosmos DB (with vector search), Azure Redis Cache
- **Infrastructure**: Azure Container Apps, Azure Resource Manager (ARM/Bicep)
- **Development**: Git workflows, GitHub Actions, ConPort for task management

## Environment Variables

```
AZURE_RESOURCE_GROUP=rg-mosaic-dev
AZURE_LOCATION=eastus2
AZURE_ENV_NAME=mosaic-dev
AZURE_OPENAI_SERVICE_NAME=mosaic-openai-mosaic-dev
AZURE_COSMOS_DB_ACCOUNT_NAME=mosaic-cosmos-mosaic-dev
AZURE_REDIS_CACHE_NAME=mosaic-redis-mosaic-dev
AZURE_ML_WORKSPACE_NAME=mosaic-ml-mosaic-dev
MOSAIC_ENVIRONMENT=development
```

## Core Development Principles

### Code Quality Standards

- Always use type hints and docstrings for Python functions
- Follow PEP 8 style guidelines (enforced by ruff)
- Implement comprehensive error handling with structured logging
- Use async/await patterns for I/O operations
- Write unit tests with pytest for all business logic

### Architecture Patterns

- Use dependency injection for Azure service clients
- Implement the Repository pattern for data access layers
- Apply SOLID principles, especially Single Responsibility
- Use Factory patterns for creating MCP tools and plugins
- Implement proper separation between FastAPI routes and business logic

### MCP Development Guidelines

- All MCP tools must implement proper JSON-RPC 2.0 protocol
- Use structured responses with consistent error handling
- Implement proper resource management and cleanup
- Follow MCP specification for tool discovery and capabilities
- Use semantic versioning for tool and server versions

### Azure Integration Best Practices

- Use Azure Identity SDK for authentication across all services
- Implement proper retry logic with exponential backoff
- Use Azure Application Insights for distributed tracing
- Configure proper health checks for Container Apps
- Use Azure Key Vault for sensitive configuration

### Memory MCP Integration

- Always use Memory MCP for task management and progress tracking
- Log architectural decisions using `create_entities` with entityType "decision"
- Link related items using `create_relations`
- Update task progress using `add_observations`
- Create detailed implementation logs for complex features
- Store reusable patterns as entities with entityType "pattern"

### Git and GitHub Workflow

- Use conventional commit messages (feat:, fix:, docs:, etc.)
- Create feature branches with descriptive names (feature/FR-X-description)
- Include comprehensive PR descriptions with testing notes
- Always run pre-commit validation (ruff check, ruff format, pytest)
- Reference related Memory MCP entity IDs in commits and PRs

## Response Style Guidelines

- Provide step-by-step implementation plans before coding
- Include relevant code examples and configuration snippets
- Reference official documentation when suggesting solutions
- Explain the reasoning behind architectural decisions
- Offer alternative approaches when applicable
- Always consider Azure cost implications for resource decisions

## Security and Compliance

- Never expose sensitive keys or connection strings in code
- Use Azure Key Vault references in configuration
- Implement proper RBAC for Azure resources
- Follow least-privilege principles for service permissions
- Include security scanning in CI/CD pipelines

## Testing Strategy

- Write unit tests for business logic (target 80%+ coverage)
- Create integration tests for Azure service interactions
- Use pytest fixtures for common test setup
- Mock external dependencies in unit tests
- Include performance tests for critical paths
