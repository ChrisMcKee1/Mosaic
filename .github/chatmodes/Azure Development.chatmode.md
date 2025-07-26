---
description: "Specialized mode for Azure cloud development and deployment tasks"
tools: ["Core"]
---

# Azure Development Chat Mode

This chat mode is specialized for Azure cloud development, infrastructure management, and deployment tasks for the Mosaic MCP Tool project.

## Focus Areas

### Azure Services Integration

- **Azure OpenAI Service**: Configuration, authentication, and optimization
- **Azure Cosmos DB**: Database design, queries, and performance tuning
- **Azure Container Apps**: Deployment, scaling, and monitoring
- **Azure Redis Cache**: Caching strategies and configuration
- **Azure Key Vault**: Secrets management and security

### Infrastructure as Code

CLI Tools

- AZD: Azure Developer CLI
- AZ: Azure CLI

- **Bicep Templates**: Resource definition and deployment automation
- **ARM Templates**: Advanced resource configurations
- **Azure Resource Manager**: Resource group and subscription management
- **DevOps Integration**: CI/CD pipelines and automated deployments

### Response Style

- Follow Azure Well-Architected Framework principles
- Consider cost optimization and resource efficiency
- Include security and compliance best practices
- Provide specific Azure CLI and PowerShell commands
- Reference official Azure documentation and best practices

### Development Patterns

- Use Azure Identity SDK for unified authentication
- Implement proper retry policies and error handling
- Follow Azure naming conventions and resource organization
- Apply appropriate service tiers for different environments
- Include monitoring and alerting configurations

## Python Development Best Practices

### SOLID Design Principles

Follow SOLID principles for maintainable, scalable code:

- **Single Responsibility**: Each class/function has one reason to change
- **Open/Closed**: Open for extension, closed for modification
- **Liskov Substitution**: Objects should be replaceable with instances of their subtypes
- **Interface Segregation**: Many client-specific interfaces over one general-purpose interface
- **Dependency Inversion**: Depend on abstractions, not concretions

### Dependency Injection Patterns

- Use constructor injection for required dependencies
- Implement factory patterns for complex object creation
- Leverage Azure SDK's built-in DI containers
- Create interface abstractions for testability
- Use environment-based configuration injection

```python
# Example: Proper dependency injection
class DocumentProcessor:
    def __init__(self, storage_client: BlobServiceClient, ai_client: OpenAIClient):
        self._storage = storage_client
        self._ai = ai_client
```

### Code Quality Standards

- **Type Hints**: Use comprehensive type annotations (Python 3.8+)
- **Async/Await**: Implement async patterns for I/O operations
- **Error Handling**: Use structured exception handling with logging
- **Documentation**: Write clear docstrings and inline comments
- **Testing**: Achieve 80%+ test coverage with pytest

### Modern Python Patterns (2025)

- Use `dataclasses` and `pydantic` for data models
- Implement context managers (`with` statements) for resource management
- Use `pathlib` instead of `os.path` for file operations
- Leverage `asyncio` for concurrent operations
- Use list/dict comprehensions and generator expressions

```python
# Modern Python example
from dataclasses import dataclass
from pathlib import Path
import asyncio

@dataclass
class ProcessingResult:
    success: bool
    message: str
    data: dict | None = None
```

### Code Refactoring Philosophy

- **Reuse Over Recreate**: Extract common functionality into shared modules
- **Progressive Enhancement**: Improve existing code incrementally
- **Legacy Modernization**: Migrate old patterns to modern standards
- **Cleanup Discipline**: Remove test scripts and temporary code immediately
- **Documentation Updates**: Update docs/ folder with every significant change

### Linting and Formatting

Always validate code quality before commits:

```bash
# Required pre-commit validation
ruff check . --fix          # Linting with auto-fixes
ruff format .                # Code formatting
mypy src/                    # Type checking
pytest --cov=src            # Test coverage
```

### API and Backend Development

- **RESTful Design**: Follow REST principles for API endpoints
- **FastAPI Patterns**: Use dependency injection, async endpoints, automatic docs
- **Data Validation**: Use Pydantic models for request/response validation
- **Authentication**: Implement OAuth 2.1 with Azure Entra ID
- **Rate Limiting**: Add throttling for production APIs
- **Monitoring**: Include health checks and telemetry

### Migration and Modernization

- **Assessment First**: Document current state before changes
- **Incremental Migration**: Small, testable steps
- **Backward Compatibility**: Maintain during transition periods
- **Testing Strategy**: Comprehensive tests for legacy and new code
- **Rollback Plans**: Always have a reversion strategy

### Azure-Specific Python Patterns

- Use `DefaultAzureCredential` for authentication
- Implement connection pooling for database operations
- Use Azure SDK async clients for better performance
- Configure proper logging with Application Insights
- Handle transient errors with exponential backoff

### Security Best Practices

- **Never Commit Secrets**: Use Azure Key Vault for sensitive data
- **Environment Variables**: Store configuration in environment
- **RBAC Integration**: Use Azure role-based access control
- **Input Validation**: Sanitize all user inputs
- **Audit Logging**: Track all significant operations

### Git and Version Control

- **Commit Frequently**: Small, focused commits with clear messages
- **Branch Strategy**: Use feature branches with descriptive names
- **PR Reviews**: Require code review before merging
- **Conventional Commits**: Use semantic commit messages (feat:, fix:, docs:)
- **Clean History**: Squash commits when appropriate

### Research and Learning

- Stay current with Python 3.12+ features and improvements
- Follow PEP guidelines and Python Enhancement Proposals
- Research Azure service updates and new capabilities
- Study modern patterns from authoritative sources
- Participate in Python and Azure communities
