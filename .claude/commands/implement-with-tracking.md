# Implement with Tracking Workflow

Comprehensive implementation workflow that ensures proper tracking, validation, and memory storage throughout the development process.

## Usage
Use `/implement-with-tracking` when you need to:
- Implement any code feature or component
- Ensure proper progress tracking and decision logging
- Validate implementation against requirements
- Maintain comprehensive project memory

## Chained Workflow Steps

### 1. Project State Assessment
```
First, review existing progress and context:
Use mcp__conport__get_recent_activity_summary to:
- Check what work has been done recently
- Identify any in-progress tasks

Use mcp__conport__get_progress to:
- Review current progress entries
- Find any incomplete tasks
- Determine where we left off

Use mcp__conport__get_decisions to:
- Review recent architectural decisions
- Understand implementation choices made
- Identify any patterns established

Use mcp__conport__get_product_context and mcp__conport__get_active_context to:
- Understand current project state
- Review any context updates since last work
- Identify current focus areas
```

### 2. Pre-Implementation Planning
```
Use mcp__sequential-thinking__sequentialthinking to:
- Analyze the implementation requirements against current state
- Determine if continuing existing work or starting new tasks
- Break down into specific tasks (if new) or continue existing tasks
- Identify potential challenges and solutions
- Plan the implementation approach based on existing decisions
```

### 3. ConPort Progress Initialization/Continuation
```
Use mcp__conport__log_progress for main task:
- Status: "IN_PROGRESS"
- Description: clear description of what's being implemented
- Link to relevant decisions or requirements

Use mcp__conport__log_progress for subtasks:
- Status: "TODO"
- Parent_id: main task ID
- Description: specific implementation steps
```

### 3. Research Validation (if needed)
```
If implementing new technology or unsure about approach:
- Use /research-and-validate workflow
- Ensure we have current best practices
- Log any new decisions or patterns discovered
```

### 4. Implementation with Desktop Commander
```
Use mcp__desktop-commander tools for actual implementation:
- read_file: examine existing code
- write_file: create new code (use chunking for large files)
- edit_block: make surgical changes
- search_code: find relevant patterns
- execute_command: run tests and validation
```

### 5. Continuous Progress Updates
```
As each subtask completes:
Use mcp__conport__update_progress:
- Update status to "DONE"
- Add any implementation notes or discoveries

Log any new patterns discovered:
Use mcp__conport__log_system_pattern:
- Name: pattern name
- Description: implementation details
- Link to related code
```

### 6. Implementation Validation
```
Use mcp__sequential-thinking__sequentialthinking to:
- Review implementation against requirements
- Identify any gaps or improvements needed
- Validate it meets functional requirements (FR-1 through FR-13)
- Plan testing approach
```

### 7. ConPort Final Updates
```
Use mcp__conport__update_progress for main task:
- Status: "DONE"
- Description: updated with final implementation notes

Use mcp__conport__log_decision for any architectural choices:
- Summary: implementation approach chosen
- Rationale: why this approach
- Implementation details: how it was implemented

Link implementation to project context:
Use mcp__conport__link_conport_items to connect:
- Implementation to functional requirements
- Code patterns to system patterns
- Progress to decisions made
```

## Example Usage

"Implement the FastAPI MCP server with SSE support for FR-1 and FR-3 requirements, ensuring proper progress tracking and decision logging throughout the process."

## Expected Outputs
- Fully implemented and tested code
- Complete progress tracking in conport
- All decisions and patterns documented
- Proper linking between implementation and requirements
- Ready for integration with other components