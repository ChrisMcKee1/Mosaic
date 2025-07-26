---
description: Create new Memory MCP tasks from research proposals
mode: agent
---

# Create Task from Planning

Convert research proposals and planning decisions into actionable Memory MCP task entities.

## Input

Task Information: ${input:task_info:Provide task title, description, or reference to Memory MCP decision entity}

## Workflow Steps

### 1. Intelligent Source Analysis and Context Discovery

**Decision Tree for Source Discovery:**

- **If given specific Memory MCP decision entity**: Use `open_nodes --names=["${entity_name}"]` to get the exact decision entity
- **If given vague reference** (e.g., "based on research", "from planning"):
  1. First try `search_nodes --query="${user_reference}"` to find matches
  2. If search returns empty results: Use `read_graph` to discover all available decisions/research
  3. Filter results by entity type and recency to identify the appropriate source
  4. Guide user to confirm which decision/research they want to base the task on

**Context Analysis Strategy:**

- **For known sources**: Use `search_nodes --query="${source_entity}"` to get the entity PLUS related context and relationships
- **Relationship Mapping**: Analyze relationships to understand:
  - Research decisions that inform this task ("builds_on" relations)
  - Architectural constraints that apply ("constrained_by" relations)
  - Related patterns that can be reused ("implements" or "uses" relations)
  - Dependencies on other tasks or components ("depends_on" relations)

**Fallback Strategies:**

- **If no specific source is provided**: Use `read_graph` to show available research decisions and guide user selection
- **If source search fails**: Search for related research by topic or technology area
- **If context is insufficient**: Search for broader project context using entity types and relationships

**Requirements Extraction**: Extract clear requirements and acceptance criteria, enriched by relationship context
**Priority Assessment**: Determine appropriate priority based on relationships to critical decisions and project goals
**Integration Analysis**: Use relationships to identify integration points and affected system components

### 2. Task Structure Definition

- **Title**: Create clear, actionable task title
- **Description**: Provide detailed task description with context
- **Acceptance Criteria**: Define specific, testable acceptance criteria
- **Effort Estimation**: Estimate complexity and time requirements

### 3. Relationship-Aware Memory MCP Task Creation

- **Create Task Entity**: Use Memory MCP to create the new task entity with enriched context:

```bash
create_entities --entities=[{
    "name": "${task_id}",
    "entityType": "task",
    "observations": [
        "Title: ${task_title}",
        "Description: ${detailed_description}",
        "Acceptance Criteria: ${criteria_list}",
        "Priority: ${priority_level}",
        "Status: TODO",
        "Estimated Effort: ${effort_estimate}",
        "Integration Points: ${integration_summary}",
        "Related Patterns: ${pattern_references}",
        "Constraints: ${constraint_summary}"
    ]
}]

# Create comprehensive relationships
create_relations --relations=[{
    "from": "${task_id}",
    "to": "${source_decision_entity}",
    "relationType": "implements"
}, {
    "from": "${task_id}",
    "to": "${prerequisite_task}",
    "relationType": "depends_on"
}, {
    "from": "${task_id}",
    "to": "${architectural_constraint}",
    "relationType": "constrained_by"
}, {
    "from": "${task_id}",
    "to": "${reusable_pattern}",
    "relationType": "uses"
}]
```

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
