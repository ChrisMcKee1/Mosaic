# Daily Priority Briefing & Task Start Workflow

Prepares the development session by syncing the repo and identifying the highest-priority tasks from Memory MCP knowledge graph. Creates a new development session entity and formally starts the selected task with comprehensive relationship tracking.

## Usage

`/planning-start-day`

## Chained Workflow

### 1. Create Daily Session Entity

Use Memory MCP to create today's development session:

```typescript
// Create today's session entity
await create_entities([
  {
    name: `session-${new Date().toISOString().split("T")[0]}`,
    entityType: "session",
    observations: [
      `Started: ${new Date().toLocaleTimeString()}`,
      `Developer: ${process.env.USER || "developer"}`,
      "Status: ACTIVE",
    ],
  },
]);
```

### 2. Sync Local Environment

Use `run_in_terminal` to run:

- `git checkout main`
- `git pull`

### 3. Query Memory MCP for High-Priority Tasks

Use Memory MCP to find critical path work:

```typescript
// Search for high-priority tasks
await search_nodes("entityType:task priority:CRITICAL status:TODO");

// If no critical tasks, search for high priority
await search_nodes("entityType:task priority:HIGH status:TODO");
```

### 4. Analyze Task Context and Dependencies

Use `sequential-thinking` to analyze the prioritized tasks from Memory MCP and their relationships:

```typescript
// Get detailed context for each task
await open_nodes([task_names]);

// Search for related decisions and patterns
await search_nodes("relates_to:" + task_name);
```

### 5. Present Prioritized Tasks to User

Based on the Memory MCP knowledge graph analysis, present tasks with:

- Task priority and status
- Related decisions and research
- Dependency relationships
- Estimated complexity based on observations

Ask: "Based on the critical path and dependencies, which task should we start today? Please provide the task ID (e.g., 'FR-006-1')."

### 6. Formally Start the Selected Task

Once the user provides the task ID:

```typescript
// Update task status and link to session
await add_observations([
  {
    entityName: task_id,
    contents: [
      "Status: IN_PROGRESS",
      `Started: ${new Date().toISOString()}`,
      `Session: session-${new Date().toISOString().split("T")[0]}`,
    ],
  },
]);

// Create relationship between session and active task
await create_relations([
  {
    from: `session-${new Date().toISOString().split("T")[0]}`,
    to: task_id,
    relationType: "focuses_on",
  },
]);

// Search for and link related entities
await search_nodes(`relates_to:${task_id}`);
// Link relevant patterns, decisions, and research
```

### 7. Establish Development Context

Create comprehensive context by linking the active task to:

- Related architectural decisions
- Relevant code patterns
- Previous research findings
- Dependent tasks and milestones

```typescript
// Create development context entity
await create_entities([
  {
    name: `context-${task_id}`,
    entityType: "pattern",
    observations: [
      "Development context for active task",
      "Includes related decisions and patterns",
      "Auto-updated during development",
    ],
  },
]);

// Link context to task and session
await create_relations([
  { from: `context-${task_id}`, to: task_id, relationType: "supports" },
  {
    from: `session-${new Date().toISOString().split("T")[0]}`,
    to: `context-${task_id}`,
    relationType: "uses",
  },
]);
```

This workflow establishes a rich, session-aware development context that tracks not just what you're working on, but how it connects to the broader project knowledge graph.
