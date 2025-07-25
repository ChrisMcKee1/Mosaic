---
description: Task-driven implementation with integrated research
mode: agent
---

# Task-Driven Implementation Workflow

Execute a specific development task from Memory MCP with comprehensive research, guided implementation against acceptance criteria, and detailed logging.

## Input Parameters

- Task ID: ${input:task_id:Enter the task ID (e.g., FR-6-1)}
- Description: ${input:description:Brief description of what you're implementing}

## Workflow Steps

### 1. Fetch Comprehensive Task Context from Memory MCP

- **Get Task Details**: Use Memory MCP `open_nodes --names=["${task_id}"]` to retrieve task entity with observations
- **Get Related Items**: Use `read_graph` to find all entities related to the task through relations

### 2. Comprehensive Research & Validation

- **Formulate Research Questions**: Analyze task description to identify core technologies requiring validation
- **Primary Documentation Research**: Retrieve official documentation and API references
- **Direct Source Retrieval**: If documentation references specific articles, fetch the full content
- **Community Validation**: Search for real-world context and community sentiment with queries like:
  - "${technology} vs ${alternative} reddit"
  - "${technology} common issues stack overflow"
  - "${technology} tutorial 2025"
- **Synthesize Findings**: Analyze gathered information and formulate final recommendation
- **Log Research**: Use Memory MCP to create decision entity and link to task:

```
create_entities --entities=[{
    "name": "research_${task_id}_${timestamp}",
    "entityType": "decision",
    "observations": ["Research findings: ${recommendation}", "Technologies validated: ${tech_list}"]
}]

create_relations --relations=[{
    "from": "${task_id}",
    "to": "research_${task_id}_${timestamp}",
    "relationType": "has_research"
}]
```

### 3. Pre-Implementation Planning

- **Create Checklist**: Parse acceptance criteria from task observations and create markdown checklist
- **Review Constraints**: Review all linked architectural decisions from Memory MCP
- **Formulate Plan**: Generate step-by-step implementation plan based on goal, checklist, and constraints

### 4. Implementation

- Perform file system and code modifications as outlined in the confirmed implementation plan
- Write/edit code and run tests using terminal commands

### 5. Continuous Progress Updates

- Update progress as each step completes:
```
add_observations --observations=[{
    "entityName": "${task_id}",
    "contents": ["Progress: Step ${step_number} completed", "Implementation status: ${status}"]
}]
```

- Log reusable code patterns:
```
create_entities --entities=[{
    "name": "pattern_${pattern_name}",
    "entityType": "pattern",
    "observations": ["Description: ${pattern_description}", "Code example: ${code}", "Use case: ${use_case}"]
}]
```

### 6. Final Validation & Memory MCP Updates

- **Validate Against Checklist**: Review implemented code against acceptance criteria
- **Mark Complete**: Update task status:
```
add_observations --observations=[{
    "entityName": "${task_id}",
    "contents": ["Status: DONE", "Completed: ${completion_date}", "All acceptance criteria met"]
}]
```

- **Log Final Decision**: Create implementation summary entity
- **Link Everything**: Create relations between completed task, final decision, and new system patterns
