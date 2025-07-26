---
description: Specialized Semantic Kernel plugin implementation workflow
mode: agent
---

# Semantic Kernel Plugin Implementation Workflow

Specialized workflow for implementing Semantic Kernel plugins with proper research, integration, and documentation.

## Input Parameters

- Plugin Name: ${input:plugin_name:Enter the plugin name (e.g., MemoryPlugin)}
- Feature Requirements: ${input:requirements:Describe the plugin requirements or feature IDs}

## Workflow Steps

### 1. Semantic Kernel Research Phase

- **SK Documentation**: Research latest Semantic Kernel patterns and best practices
- **Plugin Architecture**: Study official SK plugin examples and patterns
- **Integration Patterns**: Research SK integration with Azure services
- **Version Compatibility**: Verify SK version compatibility with current project

### 2. Plugin Design Planning

- **Interface Definition**: Define plugin interface following SK conventions
- **Function Definitions**: Plan plugin functions with proper annotations
- **Dependency Analysis**: Identify required Azure services and dependencies
- **Testing Strategy**: Plan unit and integration testing approach

### 3. Implementation Structure

```python
from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function
from semantic_kernel.core_plugins import BasePlugin

class ${plugin_name}(BasePlugin):
    @kernel_function(
        description="Function description",
        name="function_name"
    )
    async def function_name(self, param: str) -> str:
        # Implementation
        pass
```

### 4. Azure Service Integration

- **Authentication**: Implement Azure Identity integration
- **Service Clients**: Configure required Azure service clients
- **Error Handling**: Implement proper retry and error handling
- **Logging**: Add structured logging for debugging

### 5. Plugin Registration

- **Kernel Setup**: Register plugin with Semantic Kernel
- **Configuration**: Set up plugin configuration management
- **Testing**: Create comprehensive test suite
- **Documentation**: Document plugin usage and examples

### 6. Integration Testing

- **Unit Tests**: Test individual plugin functions
- **Integration Tests**: Test with real Azure services
- **End-to-End Tests**: Test complete workflow scenarios
- **Performance Tests**: Validate performance characteristics

### 7. Documentation and Logging

- **API Documentation**: Document plugin functions and parameters
- **Usage Examples**: Provide clear usage examples
- **ConPort Updates**: Log implementation decisions and link to requirements
- **README Updates**: Update project documentation
