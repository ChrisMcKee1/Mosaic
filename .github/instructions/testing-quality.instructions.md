---
description: Testing patterns and quality assurance guidelines
applyTo: "**/test*/**,**/*test*"
---

# Testing and Quality Instructions

## Test Structure

- Use pytest as the primary testing framework
- Organize tests in parallel directory structure to source code
- Name test files with `test_` prefix
- Use descriptive test function names that explain the scenario

## Test Patterns

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.fixture
async def mock_azure_client():
    with patch('azure.cosmos.CosmosClient') as mock:
        mock.return_value = AsyncMock()
        yield mock.return_value

async def test_service_handles_cosmos_error(mock_azure_client):
    # Arrange
    mock_azure_client.query_items.side_effect = Exception("Connection failed")

    # Act & Assert
    with pytest.raises(ServiceError):
        await service.get_data()
```

## Coverage Requirements

- Maintain minimum 80% code coverage
- Focus on business logic over boilerplate
- Test error conditions and edge cases
- Include integration tests for Azure services

## Mock Strategy

- Mock external dependencies (Azure services, HTTP calls)
- Use real objects for business logic testing
- Create reusable fixtures for common test data
- Mock time-dependent functions for predictable tests

## Performance Testing

- Include performance tests for critical paths
- Use pytest-benchmark for performance regression testing
- Test Azure service integration under load
- Monitor memory usage in long-running operations
