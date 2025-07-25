---
description: Create new Memory MCP tasks from research proposals
mode: agent
---

# Create Task from Planning

Convert research proposals and planning decisions into actionable Memory MCP task entities.

## Input

Task Information: ${input:task_info:Provide task title, description, or reference to Memory MCP decision entity}

## Workflow Steps

### 1. Gather Task Requirements

- **Source Analysis**: If referencing a Memory MCP decision, use `open_nodes` to retrieve the full proposal details
- **Requirements Extraction**: Extract clear requirements and acceptance criteria
- **Priority Assessment**: Determine appropriate priority level
- **Dependency Identification**: Identify any prerequisite tasks or dependencies

### 2. Task Structure Definition

- **Title**: Create clear, actionable task title
- **Description**: Provide detailed task description with context
- **Acceptance Criteria**: Define specific, testable acceptance criteria
- **Effort Estimation**: Estimate complexity and time requirements

### 3. Memory MCP Task Creation

- **Create Task**: Use Memory MCP to create the new task entity:

```
create_entities --entities=[{
    "name": "${task_id}",
    "entityType": "task",
    "observations": [
        "Title: ${task_title}",
        "Description: ${detailed_description}",
        "Acceptance Criteria: ${criteria_list}",
        "Priority: ${priority_level}",
        "Status: TODO",
        "Estimated Effort: ${effort_estimate}"
    ]
}]
```

- **Link Dependencies**: Link to prerequisite tasks or related decisions using `create_relations`
- **Assign Resources**: Add resource assignment observations if known

### 4. Integration and Linking

- **Decision Linking**: Link to original research decisions using `create_relations`
- **Architecture Linking**: Connect to relevant architectural decisions
- **Documentation**: Ensure all context is properly linked and accessible
- **Notification**: Notify relevant stakeholders of new task creation

### 5. Validation and Planning

- **Review Completeness**: Verify task has all necessary information
- **Priority Validation**: Confirm priority aligns with project goals
- **Resource Planning**: Ensure resources are available for task execution
- **Timeline Integration**: Integrate into project timeline and milestones

## Task Creation Checklist

- [ ] Clear, actionable title
- [ ] Detailed description with context
- [ ] Specific acceptance criteria
- [ ] Appropriate priority level
- [ ] Dependencies identified and linked
- [ ] Related decisions linked
- [ ] Resources assigned if known
- [ ] Timeline considerations noted

## ConPort Task Structure

```json
{
  "title": "Clear, actionable task title",
  "description": "Detailed description with context and background",
  "acceptance_criteria": [
    "Specific, testable criterion 1",
    "Specific, testable criterion 2",
    "Specific, testable criterion 3"
  ],
  "priority": "HIGH|MEDIUM|LOW|CRITICAL",
  "tags": ["feature", "azure", "mcp"],
  "dependencies": ["FR-001", "FR-002"],
  "estimated_effort": "1-3 days"
}
```
