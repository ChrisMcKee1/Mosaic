# GitHub Pull Request Workflow with Memory MCP

Comprehensive GitHub PR creation workflow with Memory MCP validation, testing, and knowledge graph documentation. Tracks PR relationships to tasks, features, and team collaboration patterns.

## Usage
`/github-pr-workflow <PR title>`

## Arguments
- `$ARGUMENTS`: The pull request title (required)

## Chained Workflow Steps

### 1. Create PR Entity in Memory MCP

```typescript
// Create PR entity with metadata
await create_entities([
  {
    name: `pr-${pr_slug}-${Date.now()}`,
    entityType: "milestone",
    observations: [
      `PR title: ${$ARGUMENTS}`,
      `Created: ${new Date().toISOString()}`,
      "Status: IN_PROGRESS",
      "Type: pull_request",
      `Branch: ${current_branch}`
    ]
  }
]);

// Link to current development session
const activeSession = await search_nodes("entityType:session AND Status:ACTIVE");
if (activeSession.length > 0) {
  await create_relations([
    {
      from: `pr-${pr_slug}-${Date.now()}`,
      to: activeSession[0].name,
      relationType: "created_in"
    }
  ]);
}
```

### 2. PR Planning with Memory MCP Context

Use `sequential-thinking` to:
- Analyze changes to be included in PR
- Plan PR description and scope
- Identify reviewers and testing requirements
- Consider deployment and integration impacts

**Query Memory MCP for PR Context:**

```typescript
// Find related tasks and features
await search_nodes("entityType:task AND Status:COMPLETED");
await search_nodes("entityType:feature AND Status:IMPLEMENTED");

// Search for related decisions and patterns
await search_nodes(`entityType:decision AND (${feature_keywords.join(' OR ')})`);

// Find recent commits and changes
await search_nodes("entityType:decision AND type:code_change");
```

### 3. Link PR to Completed Work

```typescript
// Find tasks completed in this branch/session
const completedTasks = await search_nodes("entityType:task AND Status:COMPLETED");

// Link PR to completed tasks
for (const task of completedTasks) {
  await create_relations([
    {
      from: `pr-${pr_slug}-${Date.now()}`,
      to: task.name,
      relationType: "includes"
    }
  ]);
}

// Link to implemented features
const implementedFeatures = await search_nodes("entityType:feature AND Status:IMPLEMENTED");
for (const feature of implementedFeatures) {
  await create_relations([
    {
      from: `pr-${pr_slug}-${Date.now()}`,
      to: feature.name,
      relationType: "delivers"
    }
  ]);
}
```

### 4. Git Status and Branch Validation

Use `run_in_terminal` for validation:
- `git status` (check working tree)
- `git branch` (verify current branch)  
- `git log main..HEAD --oneline` (review commits)
- `git diff main...HEAD` (review full changes)

**Document Git Context in Memory MCP:**

```typescript
await add_observations([
  {
    entityName: `pr-${pr_slug}-${Date.now()}`,
    contents: [
      `Files changed: ${files_changed.length}`,
      `Commits included: ${commits_count}`,
      `Lines added: ${lines_added}`,
      `Lines removed: ${lines_removed}`,
      `Commit range: ${commit_range}`
    ]
  }
]);
```

### 5. Testing and Validation

Use `run_in_terminal` for comprehensive testing:
- Full test suite execution
- Code quality checks (ruff, mypy)
- Security scans
- Build verification

**Create Testing Results Entity:**

```typescript
await create_entities([
  {
    name: `testing-${pr_slug}`,
    entityType: "decision",
    observations: [
      `Test suite results: ${test_results}`,
      `Code quality score: ${quality_score}`,
      `Security scan results: ${security_results}`,
      `Build status: ${build_status}`,
      `All validations passed: ${all_passed}`
    ]
  }
]);

// Link testing to PR
await create_relations([
  {
    from: `testing-${pr_slug}`,
    to: `pr-${pr_slug}-${Date.now()}`,
    relationType: "validates"
  }
]);
```

### 6. Generate PR Description from Memory MCP

Use Memory MCP context to auto-generate comprehensive PR description:

```typescript
// Gather all related context
const prContext = await open_nodes([
  `pr-${pr_slug}-${Date.now()}`,
  ...completedTasks.map(t => t.name),
  ...implementedFeatures.map(f => f.name)
]);

// Search for related decisions and patterns
const relatedDecisions = await search_nodes(`entityType:decision AND relates_to:${feature_keywords.join(' OR ')}`);
```

Generate PR description including:
- Summary of completed tasks
- Features implemented
- Technical decisions made
- Testing validation results
- Related patterns and documentation

### 7. GitHub CLI PR Creation

Use `run_in_terminal` to create PR:
```bash
gh pr create --title "$ARGUMENTS" --body "$(cat pr_description.md)" --reviewer @team
```

**Update Memory MCP with PR Details:**

```typescript
await add_observations([
  {
    entityName: `pr-${pr_slug}-${Date.now()}`,
    contents: [
      "Status: CREATED",
      `GitHub PR URL: ${pr_url}`,
      `PR number: ${pr_number}`,
      `Reviewers assigned: ${reviewers}`,
      `Created in GitHub: ${new Date().toISOString()}`
    ]
  }
]);
```

### 8. Team Collaboration Tracking

**Create Review Process Entity:**

```typescript
await create_entities([
  {
    name: `review-process-${pr_slug}`,
    entityType: "pattern",
    observations: [
      "PR review collaboration pattern",
      `Reviewers: ${reviewer_list}`,
      `Review criteria: ${review_criteria}`,
      `Approval requirements: ${approval_requirements}`
    ]
  }
]);

// Link review process to PR and team
await create_relations([
  { from: `review-process-${pr_slug}`, to: `pr-${pr_slug}-${Date.now()}`, relationType: "governs" },
  { from: `review-process-${pr_slug}`, to: "team-collaboration", relationType: "part_of" }
]);
```

### 9. Deployment Readiness Assessment

**Query Memory MCP for Deployment Context:**

```typescript
// Check deployment dependencies
await search_nodes("entityType:milestone AND type:deployment");
await search_nodes("entityType:pattern AND (infrastructure OR deployment)");

// Create deployment readiness entity
await create_entities([
  {
    name: `deployment-readiness-${pr_slug}`,
    entityType: "decision",
    observations: [
      `Deployment impact assessment: ${deployment_impact}`,
      `Infrastructure requirements: ${infrastructure_needs}`,
      `Migration requirements: ${migration_needs}`,
      `Rollback strategy: ${rollback_strategy}`
    ]
  }
]);
```

### 10. Knowledge Graph Enhancement

**Update Project Progress:**

```typescript
// Link PR to project milestones
await create_relations([
  { from: `pr-${pr_slug}-${Date.now()}`, to: "current-milestone", relationType: "contributes_to" },
  { from: `pr-${pr_slug}-${Date.now()}`, to: "project-goals", relationType: "advances" }
]);

// Update team collaboration metrics
await add_observations([
  {
    entityName: "team-metrics",
    contents: [
      `PR created: ${new Date().toISOString()}`,
      `Development velocity: ${velocity_metric}`,
      `Collaboration pattern: ${collaboration_pattern}`,
      `Code review engagement: ${review_engagement}`
    ]
  }
]);

// Create code integration pattern
await create_entities([
  {
    name: `integration-pattern-${pattern_type}`,
    entityType: "pattern",
    observations: [
      `Integration approach: ${integration_approach}`,
      `Merge strategy: ${merge_strategy}`,
      `Quality gates: ${quality_gates}`,
      `Team workflow: ${team_workflow}`
    ]
  }
]);
```

### 11. Post-Creation Monitoring Setup

**Create PR Monitoring Entity:**

```typescript
await create_entities([
  {
    name: `pr-monitoring-${pr_slug}`,
    entityType: "pattern",
    observations: [
      "PR lifecycle monitoring",
      `Review timeline expectations: ${review_timeline}`,
      `Merge criteria: ${merge_criteria}`,
      `Follow-up actions: ${followup_actions}`
    ]
  }
]);

// Link monitoring to PR
await create_relations([
  {
    from: `pr-monitoring-${pr_slug}`,
    to: `pr-${pr_slug}-${Date.now()}`,
    relationType: "monitors"
  }
]);
```

## Expected Outputs

- GitHub PR created with comprehensive description
- Full Memory MCP documentation of PR context and relationships
- Team collaboration patterns tracked and enhanced
- Deployment readiness assessment completed
- Project progress and velocity metrics updated
- Code integration knowledge preserved for future PRs

This workflow creates a comprehensive knowledge graph around the PR creation process, enabling better team coordination and project management insights.
