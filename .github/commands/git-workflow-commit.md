# Git Commit Workflow with Memory MCP

Stages all changes and creates a validated git commit with comprehensive Memory MCP knowledge graph updates. Tracks commit relationships to tasks, features, and patterns while maintaining development history.

## Usage

`/git-commit-workflow <commit message>`

## Arguments

- `$ARGUMENTS`: The commit message (required)

## Chained Workflow

### 1. Pre-Commit Validation

Use `run_in_terminal` to run the full local validation suite:

- `ruff check . --fix`
- `ruff format .`
- `pytest`

If any step fails, stop and report the error.

### 2. Analyze Changes and Memory MCP Context

Use `run_in_terminal` to review changes:

- `git status`
- `git diff --staged`

Query Memory MCP for commit context:

```typescript
// Search for active tasks and features
await search_nodes("Status:IN_PROGRESS OR Status:ACTIVE");

// Get current session context
await search_nodes(`entityType:session AND Status:ACTIVE`);
```

### 3. Create Commit Entity in Memory MCP

```typescript
// Create commit entity with metadata
await create_entities([
  {
    name: `commit-${Date.now()}`,
    entityType: "decision",
    observations: [
      `Commit message: ${commit_message}`,
      `Timestamp: ${new Date().toISOString()}`,
      `Branch: ${current_branch}`,
      `Files changed: ${changed_files.length}`,
      "Type: code_change",
    ],
  },
]);
```

### 4. Link Commit to Active Work

```typescript
// Find active tasks from Memory MCP
const activeTasks = await search_nodes(
  "entityType:task AND Status:IN_PROGRESS"
);

// Link commit to active tasks
for (const task of activeTasks) {
  await create_relations([
    {
      from: `commit-${Date.now()}`,
      to: task.name,
      relationType: "implements",
    },
  ]);
}

// Link to current development session
const activeSession = await search_nodes(
  "entityType:session AND Status:ACTIVE"
);
if (activeSession.length > 0) {
  await create_relations([
    {
      from: `commit-${Date.now()}`,
      to: activeSession[0].name,
      relationType: "part_of",
    },
  ]);
}
```

### 5. Update Task Progress

```typescript
// Update progress on related tasks
for (const task of activeTasks) {
  await add_observations([
    {
      entityName: task.name,
      contents: [
        `Code committed: ${commit_message}`,
        `Commit timestamp: ${new Date().toISOString()}`,
        "Progress: Implementation committed to repository",
      ],
    },
  ]);
}
```

### 6. Extract and Store Code Patterns

Analyze the commit for reusable patterns:

```typescript
// Create pattern entities for significant code changes
await create_entities([
  {
    name: `pattern-${pattern_identifier}`,
    entityType: "pattern",
    observations: [
      `Pattern from commit: ${commit_message}`,
      "Reusable code structure or approach",
      `Implementation details: ${pattern_details}`,
      `Usage context: ${usage_context}`,
    ],
  },
]);

// Link pattern to commit and tasks
await create_relations([
  {
    from: `pattern-${pattern_identifier}`,
    to: `commit-${Date.now()}`,
    relationType: "introduced_in",
  },
  {
    from: `pattern-${pattern_identifier}`,
    to: task.name,
    relationType: "supports",
  },
]);
```

### 7. Execute Git Commit

Use `run_in_terminal` to stage and commit:

- `git add .`
- `git commit -m "$ARGUMENTS"`

### 8. Post-Commit Memory MCP Updates

```typescript
// Mark commit as completed
await add_observations([
  {
    entityName: `commit-${Date.now()}`,
    contents: [
      "Status: COMMITTED",
      `Git hash: ${git_commit_hash}`,
      "Successfully pushed to repository",
    ],
  },
]);

// Update development momentum tracking
await add_observations([
  {
    entityName: "development-metrics",
    contents: [
      `Commit completed: ${new Date().toISOString()}`,
      `Commits today: ${commits_today_count}`,
      `Development velocity: ${velocity_metric}`,
    ],
  },
]);
```

### 9. Knowledge Graph Enhancement

```typescript
// Search for related commits and patterns
await search_nodes(`entityType:decision AND type:code_change`);

// Create relationships to similar work
await create_relations([
  {
    from: `commit-${Date.now()}`,
    to: "codebase-evolution",
    relationType: "contributes_to",
  },
]);

// Update team learning
await add_observations([
  {
    entityName: "team-knowledge",
    contents: [
      `Implementation approach: ${approach_description}`,
      `Code quality insight: ${quality_insight}`,
      `Development pattern: ${development_pattern}`,
    ],
  },
]);
```

This workflow creates a comprehensive audit trail in Memory MCP that tracks not just what code was committed, but how it relates to project goals, team learning, and development patterns.
