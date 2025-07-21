# Project Status Workflow

Comprehensive project status review using all MCP tools to provide complete project visibility and planning.

## Usage
Use `/project-status` when you need to:
- Get comprehensive project overview
- Plan next development phases
- Review progress and decisions
- Identify blockers and priorities

## Chained Workflow Steps

### 1. ConPort Comprehensive Review
```
Use mcp__conport__get_product_context for project overview
Use mcp__conport__get_active_context for current focus
Use mcp__conport__get_recent_activity_summary for recent changes
Use mcp__conport__get_progress with status filters:
- DONE: completed tasks
- IN_PROGRESS: current work
- TODO: pending tasks
```

### 2. Sequential Thinking Analysis
```
Use mcp__sequential-thinking__sequentialthinking to:
- Analyze overall project health
- Identify completion percentage for each FR requirement
- Assess current velocity and blockers
- Plan next priority tasks
- Consider risk factors and mitigation
```

### 3. Technical Status Review
```
Use mcp__conport__get_decisions to review recent choices
Use mcp__conport__get_system_patterns to check documented patterns
Use mcp__conport__search_custom_data_value_fts to find implementation notes
Use mcp__conport__get_linked_items to understand dependencies
```

### 4. Desktop Commander File System Check
```
Use mcp__desktop-commander tools for technical validation:
- list_directory: verify project structure
- search_files: check for missing files
- read_file: validate key configuration files
- get_file_info: check recent modifications
```

### 5. Current Technology Validation
```
Use mcp__context7__resolve-library-id for key dependencies
Use mcp__context7__get-library-docs to check for updates
Use WebSearch to verify:
- Azure service status and updates
- Python Semantic Kernel updates
- MCP protocol developments
- Relevant security updates
```

### 6. Functional Requirements Assessment
```
Use mcp__sequential-thinking__sequentialthinking to evaluate:
- FR-1 through FR-13 completion status
- Implementation quality and compliance
- Testing coverage for each requirement
- Integration readiness
```

### 7. ConPort Status Documentation
```
Use mcp__conport__update_active_context with current status
Use mcp__conport__log_custom_data for status metrics:
- Category: "ProjectMetrics"
- Key: current date
- Value: completion percentages, velocity, blockers

Use mcp__conport__log_progress for next priorities:
- Status: "TODO"
- Description: identified next tasks
- Priority based on analysis
```

### 8. Strategic Planning
```
Use mcp__sequential-thinking__sequentialthinking for:
- Resource allocation recommendations
- Timeline estimation for remaining work
- Risk assessment and mitigation plans
- Success criteria validation
```

### 9. Stakeholder Communication Prep
```
Use mcp__conport__export_conport_to_markdown for status reports
Generate summary including:
- Completed milestones
- Current progress
- Upcoming deliverables
- Identified risks and mitigation
```

## Output Format

### Executive Summary
- Overall completion percentage
- Current sprint/phase status
- Key achievements this period
- Critical blockers requiring attention

### Technical Status
- Functional requirements completion (FR-1 through FR-13)
- Code quality and test coverage
- Infrastructure readiness
- Security and compliance status

### Next Steps
- Prioritized task list
- Resource requirements
- Timeline estimates
- Risk mitigation actions

### Metrics and Trends
- Development velocity
- Issue resolution rate
- Code quality trends
- Deployment frequency

## Example Usage

"Provide comprehensive project status for Mosaic MCP Tool including all functional requirements progress, current blockers, and next priority tasks with timeline estimates."

## Expected Outputs
- Complete project dashboard view
- Actionable next steps with priorities
- Risk assessment with mitigation plans
- Stakeholder-ready status report
- Updated project context in conport