---
description: Comprehensive GitHub Pull Request creation workflow
mode: agent
---

# GitHub PR Workflow

Complete Pull Request creation with documentation, testing, and tracking.

## Input

PR Title: ${input:pr_title:Enter the PR title (e.g., Add semantic reranking functionality)}

## Workflow Steps

### 1. Pre-PR Planning and Validation

- **Branch Verification**: Ensure current branch has all committed changes
- **Testing Complete**: Verify all tests pass and coverage requirements are met
- **Code Quality**: Confirm linting and formatting standards are satisfied
- **Documentation**: Ensure code changes are properly documented

### 2. PR Creation and Description

- **Title Formatting**: Use clear, descriptive title following project conventions
- **Description Template**: Include:
  - **Summary**: What changes were made and why
  - **Testing**: How changes were tested
  - **Related Issues**: Link to related Memory MCP tasks or GitHub issues
  - **Breaking Changes**: Any breaking changes or migration notes
  - **Screenshots**: Visual changes or new features

### 3. Technical Review Preparation

- **Self Review**: Review all changes in diff view
- **Code Comments**: Add comments explaining complex logic
- **Test Coverage**: Ensure adequate test coverage for new functionality
- **Documentation Updates**: Update README, API docs, or other documentation

### 4. Memory MCP Integration

- **Link to Tasks**: Connect PR to related Memory MCP task entities using `create_relations`
- **Update Progress**: Mark related tasks as ready for review using `add_observations`
- **Log Decision**: Document implementation decisions in Memory MCP:

```
create_entities --entities=[{
    "name": "pr_${pr_number}_decision",
    "entityType": "decision",
    "observations": ["Implementation approach: ${approach}", "Technical decisions: ${decisions}"]
}]
```

- **Link Items**: Create relations between PR, tasks, and decisions

### 5. Review and Collaboration

- **Reviewer Assignment**: Assign appropriate reviewers
- **Draft vs Ready**: Mark as draft if not ready for review
- **CI/CD Validation**: Ensure all automated checks pass
- **Feedback Integration**: Be prepared to address review feedback

### 6. Merge and Cleanup

- **Final Validation**: Ensure all requirements are met before merge
- **Merge Strategy**: Use appropriate merge strategy (squash, merge, rebase)
- **Branch Cleanup**: Delete feature branch after successful merge
- **Memory MCP Updates**: Mark related tasks as completed using `add_observations`

## PR Description Template

```markdown
## Summary

Brief description of changes and motivation.

## Changes Made

- List of key changes
- New features added
- Bug fixes included

## Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance testing (if applicable)

## Related Items

- Memory MCP Task: [Task ID]
- Related Issues: [Issue numbers]

## Breaking Changes

- None / List any breaking changes

## Screenshots

[Add screenshots for UI changes]
```

## Quality Gates

- [ ] All tests pass
- [ ] Code coverage maintained
- [ ] Documentation updated
- [ ] Memory MCP tasks linked
- [ ] No merge conflicts
- [ ] CI/CD checks pass
