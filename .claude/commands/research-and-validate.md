# Research and Validate Workflow

Comprehensive research workflow that chains multiple MCP tools for thorough validation before any code implementation.

## Usage
Use `/research-and-validate` when you need to:
- Research any technology, library, or implementation approach
- Validate current best practices and up-to-date information
- Make informed decisions before writing code
- Ensure we're using the most current documentation

## Chained Workflow Steps

### 1. Sequential Thinking Planning
```
Use mcp__sequential-thinking__sequentialthinking to:
- Break down the research question into components
- Identify what we need to validate
- Plan the research approach
- Consider alternative approaches
```

### 2. Context7 Primary Research
```
Use mcp__context7__resolve-library-id to find relevant libraries/docs
Use mcp__context7__get-library-docs to get comprehensive documentation
Focus on: official docs, API references, best practices
```

### 3. Web Search Validation
```
Use WebSearch to validate Context7 findings:
- Search for "LIBRARY_NAME 2025 best practices"
- Look for recent updates, deprecations, or changes
- Find community discussions and real-world usage
- Verify version compatibility
```

### 4. Sequential Thinking Analysis
```
Use mcp__sequential-thinking__sequentialthinking to:
- Compare Context7 docs with web search findings
- Identify any discrepancies or outdated information
- Synthesize the most current and accurate approach
- Generate specific implementation recommendations
```

### 5. ConPort Memory Storage
```
Use mcp__conport__log_custom_data for research findings:
- Category: "Research"
- Key: technology/library name
- Value: validated current best practices and recommendations

Use mcp__conport__log_decision for chosen approach:
- Summary: chosen technology/approach
- Rationale: why this approach based on research
- Implementation details: specific steps to follow
```

### 6. Link to Project Context
```
Use mcp__conport__link_conport_items to connect:
- Research findings to relevant functional requirements
- Decisions to system patterns
- Implementation details to progress tracking
```

## Example Usage

"Research and validate the current best practices for Azure OpenAI Service integration with Python Semantic Kernel, including any recent API changes, recommended authentication methods, and deployment patterns for 2025."

## Expected Outputs
- Comprehensive research stored in conport
- Validated current best practices
- Implementation decisions with full rationale
- Linked project context for future reference
- Ready-to-implement, up-to-date approach