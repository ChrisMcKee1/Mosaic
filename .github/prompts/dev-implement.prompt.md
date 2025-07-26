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

### 1. Comprehensive Task Context Discovery with Relationship Validation

**Step 1a: Task Entity Retrieval**

- **If given specific task ID**: Use `open_nodes --names=["${task_id}"]` to get the exact task entity
- **If given vague reference**: Use `search_nodes --query="${user_reference}"` to find potential matches
- **If no results**: Use `read_graph` to discover all available tasks and guide user selection

**Step 1b: Complete Relationship Context Retrieval**
Once you have the specific task ID, use `search_nodes` for optimal relationship retrieval:

```bash
# Primary: Use search_nodes with specific task ID (performs better with exact keys)
search_nodes --query="${task_id}"

# This should return:
- The target task entity with complete details
- ALL relationships connected to this task
- All connected entities (dependencies, research, patterns, constraints)
- Related tasks and their current statuses
```

**Step 1b-Fallback: Complete Graph Retrieval if Needed**
If `search_nodes` doesn't return complete relationship context, use `read_graph` as fallback:

```bash
# Fallback: Get complete graph only if search_nodes is insufficient
read_graph

# Extract from the full graph:
- Target task entity with ALL its relationships
- All connected entities that may have been missed
- Complete relationship network for comprehensive context
```

**Step 1c: Context Validation and Completeness Check**
Validate we have complete context before proceeding:

- ✅ **Task Entity**: Core task with all observations and current status
- ✅ **Dependency Relationships**: All "depends_on", "enables", "blocks" relationships
- ✅ **Research Context**: All "has_research", "builds_on" decision entities
- ✅ **Implementation Context**: All "implements", "uses", "creates" pattern relationships
- ✅ **Constraint Context**: All "constrained_by", "requires" architectural decisions
- ✅ **Integration Context**: All related system components and their relationships

**Step 1d: Relationship Analysis Strategy**
From the complete relationship network, analyze:

- **Prerequisites**: What must be completed before this task (depends_on chains)
- **Blockers**: What might prevent this task from proceeding
- **Research Foundation**: Existing research and decisions that inform implementation
- **Reusable Patterns**: Proven solutions from related implementations
- **Integration Points**: How this task connects to broader system architecture
- **Status Dependencies**: Current completion status of prerequisite tasks

### 2. Comprehensive Research & Validation

- **Leverage Existing Research**: From step 1, check if "has_research" relationships exist for the task or related components
- **Build on Previous Decisions**: Use relationships to find related "decision" entities that inform the current implementation
- **Identify Knowledge Gaps**: Compare task requirements against existing research to find what still needs investigation
- **Multi-Source Research**: For any gaps identified:
  - **Primary Documentation Research**: Retrieve official documentation and API references
  - **Direct Source Retrieval**: If documentation references specific articles, fetch the full content
  - **Community Validation**: Search for real-world context and community sentiment
- **Connect Research to Context**: Link new research findings to existing decision entities using relationships
- **Log Comprehensive Research**: Use Memory MCP to create decision entity and establish rich relationships:

```bash
# Create research decision
create_entities --entities=[{
    "name": "research_${task_id}_${timestamp}",
    "entityType": "decision",
    "observations": ["Research findings: ${recommendation}", "Technologies validated: ${tech_list}", "Builds on: ${existing_research_summary}"]
}]

# Link to task and related decisions
create_relations --relations=[{
    "from": "${task_id}",
    "to": "research_${task_id}_${timestamp}",
    "relationType": "has_research"
}, {
    "from": "research_${task_id}_${timestamp}",
    "to": "related_decision_entity",
    "relationType": "builds_on"
}]
```

### 3. Pre-Implementation Planning

- **Create Dependency-Aware Checklist**: Parse acceptance criteria from task observations and create markdown checklist
- **Review Connected Constraints**: From relationships found in step 1, review all linked architectural decisions and their constraints
- **Identify Reusable Patterns**: Use "implements" and "uses" relationships to find proven patterns from related tasks
- **Assess Integration Points**: Use relationships to understand how this task connects to other system components
- **Formulate Context-Aware Plan**: Generate step-by-step implementation plan based on goal, dependencies, constraints, and reusable patterns

### 4. Implementation Execution

- **Implement with Context**: Execute plan from step 3, consistently referencing connected decisions and patterns from relationships
- **Apply Known Patterns**: Utilize proven patterns identified in step 3 through "implements" and "uses" relationships
- **Honor All Constraints**: Implement within boundaries set by linked architectural decisions
- **Track Decision Points**: For any new technical decisions made during implementation:
  1. Create decision entity using `create_entities` with entityType "decision"
  2. Link to current task using `create_relations`
  3. Include reasoning, alternatives considered, and impact assessment in observations
- **Integration Testing**: Test integrations at points identified in step 3 relationship analysis
- **Perform Implementation**: Write/edit code and run tests using terminal commands

### 5. Continuous Progress Updates

- Update progress as each step completes:

```bash
add_observations --observations=[{
    "entityName": "${task_id}",
    "contents": ["Progress: Step ${step_number} completed", "Implementation status: ${status}"]
}]
```

- Log reusable code patterns:

```bash
create_entities --entities=[{
    "name": "pattern_${pattern_name}",
    "entityType": "pattern",
    "observations": ["Description: ${pattern_description}", "Code example: ${code}", "Use case: ${use_case}"]
}]
```

#### Important note to. Always create relationships. To any entities or observations or other items used inside of memory. create_relations --relations=

### 6. Final Validation & Memory MCP Updates

- **Validate Against Checklist**: Review implemented code against acceptance criteria
- **Mark Complete**: Update task status:

```bash
add_observations --observations=[{
    "entityName": "${task_id}",
    "contents": ["Status: DONE", "Completed: ${completion_date}", "All acceptance criteria met"]
}]
```

#### Important note to. Always create relationships. create_relations --relations=

- **Log Final Decision**: Create implementation summary entity
- **Link Everything**: Create relations between completed task, final decision, and new system patterns
