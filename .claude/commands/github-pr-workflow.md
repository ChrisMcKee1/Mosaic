# GitHub Pull Request Workflow

Comprehensive GitHub PR creation workflow with validation, testing, and proper documentation.

## Usage
`/github-pr-workflow <PR title>`

## Arguments
- `$ARGUMENTS`: The pull request title (required)

## Chained Workflow Steps

### 1. PR Planning with Sequential Thinking
```
Use mcp__sequential-thinking__sequentialthinking to:
- Analyze changes to be included in PR
- Plan PR description and scope
- Identify reviewers and testing requirements
- Consider deployment and integration impacts
```

### 2. ConPort PR Preparation
```
Use mcp__conport__get_progress to check related tasks
Use mcp__conport__get_decisions to find relevant decisions
Use mcp__conport__log_progress for PR tracking:
- Status: "IN_PROGRESS"
- Description: "Creating PR: $ARGUMENTS"
```

### 3. Git Status and Branch Validation
```
Use mcp__desktop-commander__execute_command for:
- git status (check working tree)
- git branch (verify current branch)
- git log main..HEAD --oneline (review commits)
- git diff main...HEAD (review full changes)
```

### 4. Testing and Validation
```
Use mcp__desktop-commander__execute_command for:
- Running full test suite
- Code quality checks
- Security scans
- Build verification
```

### 5. GitHub CLI PR Creation
```
Use mcp__desktop-commander__execute_command for:
- git push -u origin current-branch (if needed)
- gh pr create --title "$ARGUMENTS" --body "$(cat <<'EOF'
## Summary
[Generated from ConPort context]

## Changes
[Auto-generated from git log and conport decisions]

## Test Plan
[Auto-generated test checklist]

## Related Issues
[Links to related conport progress items]

🤖 Generated with [Claude Code](https://claude.ai/code)
EOF
)"
```

### 6. PR Documentation Enhancement
```
Use mcp__conport__log_custom_data for PR details:
- Category: "PullRequests"
- Key: PR number or URL
- Value: PR details, related commits, test results

Use mcp__conport__link_conport_items to connect:
- PR to completed progress items
- PR to implementation decisions
- PR to functional requirements
```

### 7. Automated PR Body Generation
```
Use mcp__sequential-thinking__sequentialthinking to:
- Generate comprehensive PR description
- List all changes and their rationale
- Create test plan checklist
- Identify potential risks and mitigation
```

### 8. Post-PR Actions
```
Use mcp__desktop-commander__execute_command for:
- gh pr view (verify PR creation)
- gh pr checks (monitor CI status)

Use mcp__conport__update_progress:
- Status: "DONE"
- Description: "PR created: [PR URL] for $ARGUMENTS"
```

### 9. Review and Notification Setup
```
Use mcp__desktop-commander__execute_command for:
- gh pr edit --add-reviewer [team-members]
- gh pr edit --add-label "ready-for-review"
- gh pr comment --body "Ready for review! See ConPort for full context."
```

## Advanced Features

### Auto-Generated PR Body Template
```markdown
## Summary
Brief description of changes and their purpose.

## Changes Made
- List of specific changes from ConPort decisions
- Modified files and their purpose
- New features or bug fixes implemented

## Functional Requirements
- [ ] FR-X: Requirement met/validated
- [ ] Tests added/updated
- [ ] Documentation updated

## Test Plan
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Azure deployment tested (if applicable)

## Related Work
- ConPort Progress Items: [links]
- Related Decisions: [links]
- System Patterns: [links]

## Deployment Notes
- Any special deployment considerations
- Environment variable changes
- Infrastructure updates needed

🤖 Generated with [Claude Code](https://claude.ai/code)
```

## Example Usage
```
/github-pr-workflow "Implement RetrievalPlugin with hybrid search and graph analysis"
```

## Expected Outputs
- Well-documented GitHub PR with comprehensive description
- All tests passing and validation complete
- PR linked to ConPort project context
- Ready for team review with proper notifications
- Full audit trail in ConPort for future reference