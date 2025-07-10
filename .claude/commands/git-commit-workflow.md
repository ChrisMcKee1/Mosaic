# Git Commit Workflow

Comprehensive git commit workflow with proper validation, testing, and progress tracking.

## Usage
`/git-commit-workflow <commit message>`

## Arguments
- `$ARGUMENTS`: The commit message to use (required)

## Chained Workflow Steps

### 1. Pre-Commit Analysis
```
Use mcp__sequential-thinking__sequentialthinking to:
- Analyze current changes for commit scope
- Validate commit message follows project conventions
- Identify any potential issues or missing files
- Plan commit strategy (single vs. multiple commits)
```

### 2. ConPort Progress Update
```
Use mcp__conport__log_progress for commit tracking:
- Status: "IN_PROGRESS"
- Description: "Preparing commit: $ARGUMENTS"
- Link to related implementation tasks
```

### 3. Desktop Commander Git Operations
```
Use mcp__desktop-commander__execute_command for:
- git status (check current state)
- git diff (review changes)
- git add . (stage changes)
- git diff --cached (review staged changes)
```

### 4. Pre-Commit Validation
```
Use mcp__desktop-commander__execute_command for:
- Running linters (npm run lint, ruff, etc.)
- Running tests (pytest, npm test)
- Type checking (mypy, tsc)
- Security scanning if configured
```

### 5. Commit Decision Logging
```
Use mcp__conport__log_decision for commit rationale:
- Summary: "Commit: $ARGUMENTS"
- Rationale: what changes are being committed and why
- Implementation details: specific files and modifications
```

### 6. Execute Commit
```
Use mcp__desktop-commander__execute_command for:
- git commit -m "$ARGUMENTS 

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
- git log --oneline -1 (verify commit)
```

### 7. Post-Commit Updates
```
Use mcp__conport__update_progress:
- Status: "DONE"
- Description: "Committed: $ARGUMENTS with hash [commit-hash]"

Use mcp__conport__log_custom_data for commit tracking:
- Category: "GitHistory"
- Key: current date + commit hash
- Value: commit details, files changed, test results
```

### 8. Sequential Thinking Review
```
Use mcp__sequential-thinking__sequentialthinking to:
- Review commit quality and completeness
- Identify any follow-up tasks needed
- Plan next development steps
- Consider push timing and branch strategy
```

## Example Usage
```
/git-commit-workflow "Implement FastAPI MCP server with SSE support for FR-1 and FR-3"
```

## Expected Outputs
- Clean, tested commit with proper message format
- All pre-commit checks passed
- Commit tracked in conport with full context
- Ready for push or PR creation
- Follow-up tasks identified if needed