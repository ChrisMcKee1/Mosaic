# Semantic Kernel Workflow

Specialized workflow for implementing Semantic Kernel plugins and functionality with proper validation and tracking.

## Usage
Use `/semantic-kernel-workflow` when you need to:
- Implement any Semantic Kernel plugin
- Create semantic functions or native functions
- Integrate Azure services with Semantic Kernel
- Validate SK plugin architecture compliance

## Chained Workflow Steps

### 1. Semantic Kernel Research and Validation
```
Use mcp__context7__resolve-library-id for "semantic-kernel"
Use mcp__context7__get-library-docs to get current SK documentation
Use WebSearch to validate:
- Latest Semantic Kernel Python version
- Recent breaking changes or updates
- Azure integration best practices
- Community patterns and examples
```

### 2. Sequential Thinking Plugin Design
```
Use mcp__sequential-thinking__sequentialthinking to:
- Analyze plugin requirements against FR-2 mandate
- Design plugin interface and dependencies
- Plan Azure service integration approach
- Consider error handling and logging patterns
```

### 3. ConPort Plugin Planning
```
Use mcp__conport__log_decision for plugin architecture:
- Summary: plugin design chosen
- Rationale: why this approach aligns with FR-2
- Implementation details: specific SK patterns to use

Use mcp__conport__log_progress for plugin development:
- Status: "IN_PROGRESS"  
- Description: specific plugin being implemented
- Link to relevant functional requirements
```

### 4. Desktop Commander Implementation
```
Use mcp__desktop-commander tools for plugin creation:
- read_file: examine existing plugin patterns
- write_file: create plugin class structure
- edit_block: add semantic/native functions
- search_code: find similar implementations
```

### 5. Plugin Structure Validation
```
Required plugin structure per FR-2:
- Inherit from SK base plugin classes
- Implement proper function decorators
- Include comprehensive error handling
- Add structured logging throughout
- Validate inputs and outputs
```

### 6. Azure Service Integration
```
Use mcp__desktop-commander tools to implement:
- Azure OpenAI Service connector configuration
- Azure AI Search memory store setup
- Azure Cosmos DB custom memory store
- Azure Cache for Redis connector
- Proper authentication and security
```

### 7. Testing and Validation
```
Use mcp__desktop-commander__execute_command for:
- pytest plugin tests
- Integration testing with Azure services
- Performance validation
- Error scenario testing
```

### 8. ConPort Documentation and Tracking
```
Use mcp__conport__log_system_pattern for plugin pattern:
- Name: plugin type (e.g., "RetrievalPlugin")
- Description: implementation approach and best practices
- Tags: for future plugin development

Use mcp__conport__log_custom_data for plugin details:
- Category: "SemanticKernel"
- Key: plugin name
- Value: configuration details, dependencies, usage notes

Use mcp__conport__update_progress:
- Status: "DONE"
- Description: plugin implementation completed with test results
```

### 9. Plugin Registration and Integration
```
Use mcp__sequential-thinking__sequentialthinking to:
- Plan plugin registration with kernel
- Validate plugin compatibility with other plugins
- Test end-to-end workflows
- Document usage examples
```

### 10. Linking and Documentation
```
Use mcp__conport__link_conport_items to connect:
- Plugin implementation to functional requirements
- Plugin patterns to system architecture
- Plugin progress to overall project status
```

## Plugin-Specific Guidelines

### RetrievalPlugin (FR-5, FR-6, FR-7)
- Implement hybrid_search using AzureAISearchMemoryStore
- Create query_code_graph with Cosmos NoSQL for Azure
- Add aggregate_candidates utility function

### RefinementPlugin (FR-8)  
- Implement rerank function with httpx Azure ML calls
- Handle cross-encoder model integration
- Add proper retry logic and error handling

### MemoryPlugin (FR-9, FR-10, FR-11)
- Create HybridMemory class abstracting storage
- Implement CosmosDBMemoryStore custom class
- Add RedisMemoryStore for short-term memory

### DiagramPlugin (FR-12, FR-13)
- Implement GenerateMermaid as Semantic Function
- Add Mermaid validation and storage
- Create diagram retrieval mechanisms

## Example Usage

"Implement the RetrievalPlugin for Semantic Kernel following FR-5, FR-6, and FR-7 requirements with proper Azure AI Search integration and comprehensive testing."

## Expected Outputs
- Fully implemented and tested SK plugin
- Proper Azure service integration
- Comprehensive documentation in conport
- Plugin patterns documented for reuse
- Integration ready for MCP server