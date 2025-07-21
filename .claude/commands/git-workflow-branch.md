# Git Branch Workflow

Comprehensive git branch management workflow with proper tracking and validation.

## Usage
`/git-branch-workflow <branch-name>`

## Arguments
- `$ARGUMENTS`: The new branch name to create (required)

## Chained Workflow Steps

### 1. Branch Planning with Sequential Thinking
```
Use mcp__sequential-thinking__sequentialthinking to:
- Validate branch naming follows project conventions
- Plan branch purpose and scope
- Consider integration strategy and timeline
- Identify dependencies and potential conflicts
```

### 2. ConPort Branch Tracking
```
Use mcp__conport__log_progress for branch creation:
- Status: "IN_PROGRESS"
- Description: "Creating branch: $ARGUMENTS"
- Link to related functional requirements or features

Use mcp__conport__log_decision for branch strategy:
- Summary: "Branch $ARGUMENTS created for [purpose]"
- Rationale: why this branch is needed
- Implementation details: planned changes and scope
```

### 3. Desktop Commander Git Operations
```
Use mcp__desktop-commander__execute_command for:
- git status (ensure clean working tree)
- git fetch origin (get latest remote changes)
- git checkout main (switch to main branch)
- git pull origin main (update main branch)
```

### 4. Branch Creation and Setup
```
Use mcp__desktop-commander__execute_command for:
- git checkout -b $ARGUMENTS (create and switch to new branch)
- git push -u origin $ARGUMENTS (set upstream tracking)
- git branch -vv (verify branch setup)
```

### 5. Branch Documentation
```
Use mcp__conport__log_custom_data for branch tracking:
- Category: "GitBranches"
- Key: branch name
- Value: creation date, purpose, related FRs, planned timeline

Use mcp__conport__link_conport_items to connect:
- Branch to functional requirements
- Branch to progress items
- Branch to implementation decisions
```

### 6. Development Environment Setup
```
Use mcp__sequential-thinking__sequentialthinking to:
- Consider environment-specific setup needed
- Plan development workflow for this branch
- Identify testing requirements
- Consider integration checkpoints
```

### 7. Initial Branch Commit (Optional)
```
If branch needs initial setup files:
Use mcp__desktop-commander__execute_command for:
- Creating branch-specific documentation
- Initial configuration files
- git add and git commit for setup
```

### 8. Branch Status Update
```
Use mcp__conport__update_progress:
- Status: "DONE"
- Description: "Branch $ARGUMENTS created and ready for development"

Use mcp__conport__log_system_pattern for branch pattern:
- Name: "Feature Branch Creation"
- Description: standardized branch creation process
- Tags: for future branch workflows
```

## Branch Naming Conventions

### Feature Branches
- `feature/[FR-number]-[short-description]`
- Example: `feature/FR-5-hybrid-search-implementation`

### Bug Fix Branches
- `bugfix/[issue-number]-[short-description]`
- Example: `bugfix/42-azure-auth-timeout`

### Infrastructure Branches
- `infra/[component]-[change]`
- Example: `infra/azure-bicep-templates`

### Research Branches
- `research/[topic]-[investigation]`
- Example: `research/semantic-kernel-memory-patterns`

## Example Usage
```
/git-branch-workflow "feature/FR-8-semantic-reranking"
```

## Expected Outputs
- New branch created with proper upstream tracking
- Branch documented in ConPort with full context
- Development environment ready
- Clear branch purpose and scope defined
- Ready for focused development work