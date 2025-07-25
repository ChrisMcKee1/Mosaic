# Task Creation Workflow with Memory MCP

Creates well-structured development tasks in Memory MCP knowledge graph with comprehensive context, relationships, and acceptance criteria. Converts research findings and decisions into actionable work items.

## Usage

`/planning-create-task <task description or research reference>`

## Arguments

- `$ARGUMENTS`: Task description or reference to research/decision entity

## Chained Workflow

### 1. Query Memory MCP for Context

**Search for Related Research and Decisions:**

```typescript
// If $ARGUMENTS references existing research/decision
if (isEntityReference($ARGUMENTS)) {
  // Get specific entity context
  await open_nodes([$ARGUMENTS]);
} else {
  // Search for related entities by keywords
  const keywords = extractKeywords($ARGUMENTS);
  await search_nodes(`entityType:research AND (${keywords.join(" OR ")})`);
  await search_nodes(`entityType:decision AND (${keywords.join(" OR ")})`);
}

// Find related patterns and existing tasks
await search_nodes(`entityType:pattern AND (${keywords.join(" OR ")})`);
await search_nodes(`entityType:task AND (${keywords.join(" OR ")})`);
```

### 2. Analyze Task Requirements

Use `sequential-thinking` to analyze:

- Task scope and objectives
- Technical requirements and constraints
- Dependencies on other tasks or decisions
- Acceptance criteria and success metrics
- Priority and effort estimation

**Query Memory MCP for Dependencies:**

```typescript
// Search for blocking or prerequisite tasks
await search_nodes("entityType:task AND Status:IN_PROGRESS");
await search_nodes("entityType:milestone AND Status:ACTIVE");

// Find related architectural decisions
await search_nodes(`entityType:decision AND type:architecture`);
```

### 3. Create Task Entity with Rich Context

```typescript
// Generate unique task identifier
const taskId = generateTaskId($ARGUMENTS); // e.g., "FR-12-semantic-search"

// Create comprehensive task entity
await create_entities([
  {
    name: taskId,
    entityType: "task",
    observations: [
      `Title: ${task_title}`,
      `Description: ${detailed_description}`,
      `Acceptance criteria: ${acceptance_criteria}`,
      `Priority: ${priority_level}`,
      `Estimated effort: ${effort_estimate}`,
      `Created: ${new Date().toISOString()}`,
      "Status: TODO",
      `Type: ${task_type}`, // feature, bug, research, infrastructure
      `Category: ${task_category}`, // frontend, backend, devops, etc.
    ],
  },
]);
```

### 4. Create Task Relationships

**Link to Source Research/Decisions:**

```typescript
// Link to originating research or decision
if (sourceEntity) {
  await create_relations([
    {
      from: taskId,
      to: sourceEntity.name,
      relationType: "based_on",
    },
  ]);
}

// Link to related patterns
for (const pattern of relatedPatterns) {
  await create_relations([
    {
      from: taskId,
      to: pattern.name,
      relationType: "uses",
    },
  ]);
}
```

**Establish Dependencies:**

```typescript
// Create dependency relationships
for (const dependency of identifiedDependencies) {
  await create_relations([
    {
      from: taskId,
      to: dependency.name,
      relationType: "depends_on",
    },
  ]);
}

// Link to current project milestone
await create_relations([
  {
    from: taskId,
    to: "current-milestone",
    relationType: "contributes_to",
  },
]);
```

### 5. Create Sub-tasks if Needed

**Break Down Complex Tasks:**

```typescript
// If task is large, create sub-tasks
if (isComplexTask(task_scope)) {
  const subtasks = breakDownTask(task_scope);

  for (const [index, subtask] of subtasks.entries()) {
    const subtaskId = `${taskId}-${index + 1}`;

    await create_entities([
      {
        name: subtaskId,
        entityType: "task",
        observations: [
          `Title: ${subtask.title}`,
          `Description: ${subtask.description}`,
          `Acceptance criteria: ${subtask.criteria}`,
          `Priority: ${subtask.priority}`,
          "Status: TODO",
          `Parent task: ${taskId}`,
          `Sequence: ${index + 1}`,
        ],
      },
    ]);

    // Link subtask to parent
    await create_relations([
      {
        from: subtaskId,
        to: taskId,
        relationType: "subtask_of",
      },
    ]);
  }
}
```

### 6. Create Implementation Guide

**Generate Implementation Context Entity:**

```typescript
await create_entities([
  {
    name: `guide-${taskId}`,
    entityType: "pattern",
    observations: [
      "Implementation guidance for task",
      `Technical approach: ${technical_approach}`,
      `Recommended tools: ${recommended_tools}`,
      `Implementation steps: ${implementation_steps}`,
      `Testing strategy: ${testing_strategy}`,
      `Risk considerations: ${risk_considerations}`,
    ],
  },
]);

// Link guide to task
await create_relations([
  {
    from: `guide-${taskId}`,
    to: taskId,
    relationType: "guides",
  },
]);
```

### 7. Establish Success Metrics

**Create Success Criteria Entity:**

```typescript
await create_entities([
  {
    name: `success-criteria-${taskId}`,
    entityType: "decision",
    observations: [
      "Task completion criteria and validation",
      `Functional requirements: ${functional_requirements}`,
      `Quality gates: ${quality_gates}`,
      `Performance criteria: ${performance_criteria}`,
      `Review requirements: ${review_requirements}`,
      `Documentation requirements: ${doc_requirements}`,
    ],
  },
]);

// Link success criteria to task
await create_relations([
  {
    from: `success-criteria-${taskId}`,
    to: taskId,
    relationType: "defines_success_for",
  },
]);
```

### 8. Link to Team and Session Context

**Connect to Current Development Context:**

```typescript
// Link to current development session
const activeSession = await search_nodes(
  "entityType:session AND Status:ACTIVE"
);
if (activeSession.length > 0) {
  await create_relations([
    {
      from: taskId,
      to: activeSession[0].name,
      relationType: "created_in",
    },
  ]);
}

// Create team assignment (if specified)
if (assignedTeamMember) {
  await create_relations([
    {
      from: taskId,
      to: assignedTeamMember,
      relationType: "assigned_to",
    },
  ]);
}
```

### 9. Priority and Backlog Management

**Update Project Backlog:**

```typescript
// Search for project backlog and priority ordering
await search_nodes("entityType:milestone AND type:backlog");

// Update backlog priority based on task priority
await add_observations([
  {
    entityName: "project-backlog",
    contents: [
      `New task added: ${taskId}`,
      `Priority: ${priority_level}`,
      `Added: ${new Date().toISOString()}`,
      `Dependencies: ${dependency_count}`,
      `Estimated effort: ${effort_estimate}`,
    ],
  },
]);

// Create priority relationship
await create_relations([
  {
    from: taskId,
    to: "project-backlog",
    relationType: "prioritized_in",
  },
]);
```

### 10. Knowledge Graph Enhancement

**Update Project Intelligence:**

```typescript
// Update project metrics
await add_observations([
  {
    entityName: "project-metrics",
    contents: [
      `Task created: ${new Date().toISOString()}`,
      `Total tasks: ${total_task_count}`,
      `Backlog size: ${backlog_size}`,
      `Planning velocity: ${planning_velocity}`,
    ],
  },
]);

// Create planning pattern
await create_entities([
  {
    name: `planning-pattern-${task_category}`,
    entityType: "pattern",
    observations: [
      `Planning approach for ${task_category} tasks`,
      `Requirements gathering: ${requirements_approach}`,
      `Estimation method: ${estimation_method}`,
      `Success criteria template: ${criteria_template}`,
    ],
  },
]);

// Link planning pattern
await create_relations([
  {
    from: `planning-pattern-${task_category}`,
    to: taskId,
    relationType: "applied_to",
  },
]);
```

### 11. Output Task Summary

Present comprehensive task summary to user:

```markdown
## Task Created: ${taskId}

**Title:** ${task_title}
**Priority:** ${priority_level}
**Status:** TODO
**Estimated Effort:** ${effort_estimate}

### Description

${detailed_description}

### Acceptance Criteria

${acceptance_criteria}

### Dependencies

${dependency_list}

### Implementation Guide

${implementation_approach}

### Next Steps

Use `/dev-implement ${taskId}` to begin implementation
Use `/planning-start-day` to prioritize in daily planning

### Memory MCP Entities Created

- Task: `${taskId}`
- Implementation Guide: `guide-${taskId}`
- Success Criteria: `success-criteria-${taskId}`
- Related Patterns: ${pattern_count}
```

This workflow creates a comprehensive task ecosystem in Memory MCP that provides rich context for implementation and tracks relationships across the entire project knowledge graph.
