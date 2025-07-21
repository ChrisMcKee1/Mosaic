# Debug and Solve Workflow

Comprehensive debugging workflow that uses all MCP tools to systematically diagnose and solve problems.

## Usage
Use `/debug-and-solve` when you need to:
- Debug failing tests or code issues
- Solve complex implementation problems
- Investigate unexpected behavior
- Troubleshoot deployment or configuration issues

## Chained Workflow Steps

### 1. Problem Analysis with Sequential Thinking
```
Use mcp__sequential-thinking__sequentialthinking to:
- Clearly define the problem/issue
- Identify what we expect vs. what's happening
- Break down potential causes
- Plan systematic debugging approach
```

### 2. ConPort Context Retrieval
```
Use mcp__conport__search_decisions_fts to find related decisions
Use mcp__conport__search_custom_data_value_fts to find related research
Use mcp__conport__get_linked_items to find connected components
Use mcp__conport__get_progress to check recent implementation changes
```

### 3. Desktop Commander Investigation
```
Use mcp__desktop-commander tools for investigation:
- read_file: examine relevant code files
- search_code: find related patterns or similar implementations
- execute_command: run diagnostic commands
- read_output: monitor running processes
- list_directory: check file structure
```

### 4. Research Current Solutions
```
Use mcp__context7__resolve-library-id to find relevant documentation
Use mcp__context7__get-library-docs to get troubleshooting guides
Use WebSearch to find:
- Recent bug reports or issues
- Community solutions
- Stack Overflow discussions
- GitHub issue threads
```

### 5. Sequential Thinking Solution Analysis
```
Use mcp__sequential-thinking__sequentialthinking to:
- Analyze all gathered information
- Compare different solution approaches
- Identify the most likely root cause
- Plan the fix implementation
- Consider potential side effects
```

### 6. Implementation and Testing
```
Use mcp__desktop-commander tools for fixes:
- edit_block: make targeted code changes
- write_file: create new test cases
- execute_command: run tests and validation
- read_output: monitor results
```

### 7. ConPort Memory Updates
```
Use mcp__conport__log_decision for solution chosen:
- Summary: problem and solution
- Rationale: why this solution
- Implementation details: specific changes made

Use mcp__conport__log_system_pattern for debugging pattern:
- Name: problem type (e.g., "Azure authentication issue")
- Description: diagnostic steps and solution
- Tags: for future reference

Use mcp__conport__log_custom_data for troubleshooting notes:
- Category: "Debugging"
- Key: problem identifier
- Value: detailed notes and solution steps
```

### 8. Progress and Learning Updates
```
Use mcp__conport__update_progress for related tasks:
- Update status based on resolution
- Add notes about what was learned

Use mcp__conport__link_conport_items to connect:
- Debug solution to system patterns
- Problem to prevention measures
- Solution to implementation progress
```

## Example Usage

"Debug the Azure OpenAI Service connection timeout issue in the Semantic Kernel integration, investigating authentication, network, and configuration problems."

## Expected Outputs
- Root cause identification and fix
- Comprehensive debugging notes in conport
- Reusable debugging patterns documented
- Prevention measures identified
- Project memory updated with solution knowledge