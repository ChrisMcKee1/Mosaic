# Task-Driven Implementation Workflow (with Integrated Research)

Executes a specific development task using Memory MCP knowledge graph. This comprehensive workflow begins by fetching the task's full context from Memory MCP, proceeds through mandatory research and validation, guides implementation against acceptance criteria, and concludes with detailed knowledge graph updates and relationship mapping.

## Usage

`/dev-implement <task_id> <brief description of what you're doing>`

Example: `/dev-implement FR-6-1 Implement the GitPython cloning logic`

## Chained Workflow

### 1. Fetch Comprehensive Task Context from Memory MCP

**Identify Task:** The first argument is the task ID (e.g., "FR-6-1").

```typescript
// Get task details and full context
await open_nodes([task_id]);

// Search for related decisions, research, and patterns
await search_nodes(`relates_to:${task_id} OR depends_on:${task_id}`);

// Get linked architectural decisions
await search_nodes(`entityType:decision AND relates_to:${task_id}`);

// Find relevant patterns and research
await search_nodes(`entityType:pattern OR entityType:research`);
```

### 2. Comprehensive Research & Validation

**Formulate Research Questions:** Use `sequential-thinking` to analyze the task from Memory MCP and identify core technologies requiring validation.

**Primary Documentation Research (Context7):** Use **`mcp_context72_get-library-docs`** tool to retrieve official documentation and API references.

**Community Validation (WebSearch):** Use **`fetch_webpage`** tool for real-world context:

- "`<topic>` vs `<alternative>` reddit"
- "`<topic>` common issues stack overflow"
- "`<topic>` tutorial 2025"

**Memory MCP Research Storage:**

```typescript
// Create research entity for this implementation
await create_entities([
  {
    name: `research-${task_id}-${Date.now()}`,
    entityType: "research",
    observations: [
      "Technology validation findings",
      "Best practices discovered",
      "Community insights gathered",
      `Research date: ${new Date().toISOString()}`,
    ],
  },
]);

// Link research to task
await create_relations([
  {
    from: `research-${task_id}-${Date.now()}`,
    to: task_id,
    relationType: "validates",
  },
]);
```

**Synthesize and Log Findings:** Use `sequential-thinking` to analyze all information and formulate recommendations:

```typescript
// Create decision entity with recommendations
await create_entities([
  {
    name: `decision-${task_id}-approach`,
    entityType: "decision",
    observations: [
      "Implementation approach decision",
      "Based on comprehensive research",
      "Includes technology choices and rationale",
      `Decision date: ${new Date().toISOString()}`,
    ],
  },
]);

// Link decision to task and research
await create_relations([
  { from: `decision-${task_id}-approach`, to: task_id, relationType: "guides" },
  {
    from: `decision-${task_id}-approach`,
    to: `research-${task_id}-${Date.now()}`,
    relationType: "based_on",
  },
]);
```

### 3. Pre-Implementation Planning

**Create Implementation Checklist:** Use `sequential-thinking` to parse acceptance criteria from Memory MCP task observations and create specific markdown checklist.

**Review Constraints:** Query Memory MCP for architectural constraints:

```typescript
// Search for architectural decisions and patterns
await search_nodes(
  "entityType:decision AND (architecture OR constraint OR pattern)"
);

// Get related system patterns
await search_nodes(`entityType:pattern AND relates_to:${task_id}`);
```

**Formulate Implementation Plan:** Generate step-by-step plan based on Memory MCP context and present for confirmation.

### 4. Implementation with Progress Tracking

Use appropriate tools (`run_in_terminal`, `create_file`, `replace_string_in_file`) to perform implementation.

**Continuous Progress Updates in Memory MCP:**

```typescript
// Update task progress observations
await add_observations([
  {
    entityName: task_id,
    contents: [
      `Implementation step completed: ${step_description}`,
      `Progress: ${percentage}%`,
      `Timestamp: ${new Date().toISOString()}`,
    ],
  },
]);

// Create pattern entities for reusable code
await create_entities([
  {
    name: `pattern-${pattern_name}`,
    entityType: "pattern",
    observations: [
      "Reusable implementation pattern",
      "Code structure and approach",
      "Usage guidelines and examples",
    ],
  },
]);

// Link patterns to implementation
await create_relations([
  {
    from: `pattern-${pattern_name}`,
    to: task_id,
    relationType: "implemented_in",
  },
]);
```

### 5. Final Validation & Memory MCP Updates

**Validate Against Checklist:** Use `sequential-thinking` to review implemented code against acceptance criteria.

**Complete Task in Memory MCP:**

```typescript
// Mark task as complete
await add_observations([
  {
    entityName: task_id,
    contents: [
      "Status: COMPLETED",
      `Completion date: ${new Date().toISOString()}`,
      "All acceptance criteria met",
      "Implementation validated and tested",
    ],
  },
]);

// Create final implementation summary
await create_entities([
  {
    name: `implementation-${task_id}-summary`,
    entityType: "decision",
    observations: [
      "Final implementation approach used",
      "Key decisions and trade-offs made",
      "Lessons learned and improvements",
      "Testing and validation completed",
    ],
  },
]);

// Link everything together
await create_relations([
  {
    from: `implementation-${task_id}-summary`,
    to: task_id,
    relationType: "summarizes",
  },
  { from: task_id, to: "current-milestone", relationType: "contributes_to" },
]);
```

### 6. Knowledge Graph Enhancement

**Update Related Entities:** Link completed implementation to broader project context:

```typescript
// Search for related features and milestones
await search_nodes("entityType:feature OR entityType:milestone");

// Create relationships to project goals
await create_relations([
  { from: task_id, to: "project-goals", relationType: "advances" },
]);

// Update team knowledge base
await add_observations([
  {
    entityName: "team-knowledge",
    contents: [
      `Implementation pattern: ${pattern_description}`,
      `Technology insight: ${technology_learning}`,
      `Best practice: ${best_practice_discovered}`,
    ],
  },
]);
```

This workflow creates a comprehensive knowledge graph that captures not just what was implemented, but why, how, and how it connects to the broader project ecosystem.
