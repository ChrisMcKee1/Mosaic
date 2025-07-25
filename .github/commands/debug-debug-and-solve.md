# Debug and Solve Workflow with Memory MCP

Comprehensive debugging workflow that uses Memory MCP knowledge graph to systematically diagnose and solve problems while building institutional knowledge for future troubleshooting.

## Usage

`/debug-and-solve <problem description>`

Use when you need to:

- Debug failing tests or code issues
- Solve complex implementation problems
- Investigate unexpected behavior
- Troubleshoot deployment or configuration issues

## Chained Workflow Steps

### 1. Create Problem Entity in Memory MCP

```typescript
// Create problem entity with initial context
await create_entities([
  {
    name: `problem-${problem_slug}-${Date.now()}`,
    entityType: "issue",
    observations: [
      `Problem description: ${$ARGUMENTS}`,
      `Discovered: ${new Date().toISOString()}`,
      "Status: INVESTIGATING",
      `Severity: ${estimated_severity}`,
      "Type: debugging_investigation",
    ],
  },
]);

// Link to current development session
const activeSession = await search_nodes(
  "entityType:session AND Status:ACTIVE"
);
if (activeSession.length > 0) {
  await create_relations([
    {
      from: `problem-${problem_slug}-${Date.now()}`,
      to: activeSession[0].name,
      relationType: "encountered_in",
    },
  ]);
}
```

### 2. Problem Analysis with Sequential Thinking and Memory MCP Context

Use `sequential-thinking` to:

- Clearly define the problem/issue
- Identify expected vs actual behavior
- Break down potential causes
- Plan systematic debugging approach

**Query Memory MCP for Similar Historical Issues:**

```typescript
// Search for similar problems and solutions
await search_nodes(`entityType:issue AND (${problem_keywords.join(" OR ")})`);

// Find related debugging patterns
await search_nodes(`entityType:pattern AND type:debugging`);

// Get related decisions that might be relevant
await search_nodes(
  `entityType:decision AND (${technology_keywords.join(" OR ")})`
);
```

### 3. Contextual Investigation

**Query Memory MCP for Related Context:**

```typescript
// Find related tasks and recent changes
await search_nodes("entityType:task AND Status:IN_PROGRESS");
await search_nodes("entityType:decision AND type:code_change");

// Search for related patterns and configurations
await search_nodes(
  `entityType:pattern AND (configuration OR setup OR deployment)`
);
```

**Update Problem Entity with Context:**

```typescript
await add_observations([
  {
    entityName: `problem-${problem_slug}-${Date.now()}`,
    contents: [
      `Related historical issues: ${historical_issues_count}`,
      `Recent relevant changes: ${recent_changes}`,
      `System context: ${system_context}`,
      `Investigation approach: ${planned_approach}`,
    ],
  },
]);
```

### 4. Investigation Tools Execution

Use `run_in_terminal`, `read_file`, and `file_search` for investigation:

- Examine relevant code files
- Run diagnostic commands
- Check logs and configurations
- Verify system state

**Document Investigation Steps:**

```typescript
// Create investigation entity
await create_entities([
  {
    name: `investigation-${problem_slug}`,
    entityType: "pattern",
    observations: [
      "Debugging investigation steps",
      `Files examined: ${files_examined}`,
      `Commands executed: ${commands_run}`,
      `Findings: ${investigation_findings}`,
    ],
  },
]);

// Link investigation to problem
await create_relations([
  {
    from: `investigation-${problem_slug}`,
    to: `problem-${problem_slug}-${Date.now()}`,
    relationType: "investigates",
  },
]);
```

### 5. Research Current Solutions

**External Research:**

- Use `mcp_context72_get-library-docs` for troubleshooting guides
- Use `fetch_webpage` for:
  - Recent bug reports or issues
  - Community solutions
  - Stack Overflow discussions
  - GitHub issue threads

**Store Research Findings:**

```typescript
await add_observations([
  {
    entityName: `investigation-${problem_slug}`,
    contents: [
      `External research findings: ${research_findings}`,
      `Community solutions found: ${community_solutions}`,
      `Documentation insights: ${doc_insights}`,
      `Similar issue reports: ${similar_issues}`,
    ],
  },
]);
```

### 6. Root Cause Analysis

Use `sequential-thinking` to:

- Analyze all gathered information
- Compare different solution approaches
- Identify the most likely root cause
- Plan the fix implementation
- Consider potential side effects

**Create Root Cause Entity:**

```typescript
await create_entities([
  {
    name: `root-cause-${problem_slug}`,
    entityType: "decision",
    observations: [
      `Root cause identified: ${root_cause}`,
      `Analysis approach: ${analysis_method}`,
      `Evidence supporting diagnosis: ${evidence}`,
      `Confidence level: ${confidence_level}`,
    ],
  },
]);

// Link root cause to problem and investigation
await create_relations([
  {
    from: `root-cause-${problem_slug}`,
    to: `problem-${problem_slug}-${Date.now()}`,
    relationType: "explains",
  },
  {
    from: `root-cause-${problem_slug}`,
    to: `investigation-${problem_slug}`,
    relationType: "based_on",
  },
]);
```

### 7. Solution Implementation and Testing

Use appropriate tools for fixes:

- `replace_string_in_file`: Make targeted code changes
- `create_file`: Create new test cases
- `run_in_terminal`: Run tests and validation

**Document Solution Process:**

```typescript
// Create solution entity
await create_entities([
  {
    name: `solution-${problem_slug}`,
    entityType: "decision",
    observations: [
      `Solution approach: ${solution_approach}`,
      `Implementation steps: ${implementation_steps}`,
      `Testing strategy: ${testing_strategy}`,
      `Validation results: ${validation_results}`,
    ],
  },
]);

// Create comprehensive relationship network
await create_relations([
  {
    from: `solution-${problem_slug}`,
    to: `root-cause-${problem_slug}`,
    relationType: "addresses",
  },
  {
    from: `solution-${problem_slug}`,
    to: `problem-${problem_slug}-${Date.now()}`,
    relationType: "solves",
  },
]);
```

### 8. Knowledge Pattern Creation

**Create Reusable Debugging Pattern:**

```typescript
await create_entities([
  {
    name: `debug-pattern-${pattern_type}`,
    entityType: "pattern",
    observations: [
      `Problem type: ${problem_category}`,
      `Diagnostic approach: ${diagnostic_steps}`,
      `Common causes: ${common_causes}`,
      `Solution strategies: ${solution_strategies}`,
      `Prevention measures: ${prevention_steps}`,
    ],
  },
]);

// Link pattern to problem and solution
await create_relations([
  {
    from: `debug-pattern-${pattern_type}`,
    to: `problem-${problem_slug}-${Date.now()}`,
    relationType: "generalizes",
  },
  {
    from: `debug-pattern-${pattern_type}`,
    to: `solution-${problem_slug}`,
    relationType: "includes",
  },
]);
```

### 9. Final Resolution and Learning

**Mark Problem as Resolved:**

```typescript
await add_observations([
  {
    entityName: `problem-${problem_slug}-${Date.now()}`,
    contents: [
      "Status: RESOLVED",
      `Resolution time: ${new Date().toISOString()}`,
      `Solution applied: solution-${problem_slug}`,
      `Pattern created: debug-pattern-${pattern_type}`,
      `Time to resolution: ${resolution_time_minutes} minutes`,
    ],
  },
]);

// Update team knowledge base
await add_observations([
  {
    entityName: "team-knowledge",
    contents: [
      `Problem solved: ${problem_summary}`,
      `Root cause: ${root_cause_summary}`,
      `Solution: ${solution_summary}`,
      `Debugging insight: ${debugging_insight}`,
      `Prevention strategy: ${prevention_strategy}`,
    ],
  },
]);
```

### 10. Knowledge Graph Enhancement

```typescript
// Link to related system areas for future reference
await create_relations([
  {
    from: `debug-pattern-${pattern_type}`,
    to: "troubleshooting-guide",
    relationType: "contributes_to",
  },
  {
    from: `solution-${problem_slug}`,
    to: "system-reliability",
    relationType: "improves",
  },
]);

// Create metrics for debugging effectiveness
await add_observations([
  {
    entityName: "debugging-metrics",
    contents: [
      `Problem resolved: ${new Date().toISOString()}`,
      `Resolution efficiency: ${efficiency_score}`,
      `Knowledge reuse potential: ${reuse_potential}`,
      `Team learning value: ${learning_value}`,
    ],
  },
]);
```

## Expected Outputs

- Root cause identification and fix implementation
- Comprehensive debugging knowledge in Memory MCP
- Reusable debugging patterns documented
- Prevention measures identified and stored
- Enhanced team troubleshooting capabilities

This workflow builds institutional debugging knowledge that improves team problem-solving capacity over time.
