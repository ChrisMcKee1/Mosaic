---
description: Comprehensive GitHub issue creation with context
mode: agent
---

# GitHub Issue Workflow

Create comprehensive GitHub issues with proper context, research, and tracking.

## Input

Issue Title: ${input:issue_title:Enter the issue title (e.g., Azure Cosmos DB integration timeout)}

## Workflow Steps

### 1. Issue Analysis and Context Gathering

- **Problem Definition**: Clearly define the issue, bug, or feature request
- **Impact Assessment**: Determine severity and impact on users/system
- **Environment Context**: Gather environment details (dev/staging/prod)
- **Reproduction Steps**: Document steps to reproduce if it's a bug

### 2. Research and Investigation

- **Similar Issues**: Search existing issues for duplicates or related problems
- **Documentation Review**: Check if issue is covered in documentation
- **Known Workarounds**: Identify any existing workarounds or partial solutions
- **Root Cause Analysis**: Investigate potential root causes

### 3. Issue Documentation

- **Clear Title**: Use descriptive, searchable title
- **Detailed Description**: Provide comprehensive issue description
- **Labels Assignment**: Apply appropriate labels (bug, enhancement, documentation, etc.)
- **Priority Setting**: Assign priority based on impact and urgency
- **Milestone Assignment**: Link to relevant project milestones

### 4. Technical Details

- **Error Messages**: Include full error messages and stack traces
- **System Information**: Provide relevant system/environment details
- **Code Snippets**: Include relevant code snippets or configuration
- **Screenshots/Logs**: Attach visual evidence or log files

### 5. ConPort Integration

- **Create ConPort Task**: Create corresponding task in ConPort if needed
- **Link Items**: Link GitHub issue to ConPort tasks and decisions
- **Priority Alignment**: Ensure priority aligns with ConPort task priorities
- **Context Sharing**: Share relevant context between systems

### 6. Follow-up and Tracking

- **Assignee**: Assign to appropriate team member if known
- **Watchers**: Add relevant stakeholders as watchers
- **Progress Tracking**: Monitor issue progress and updates
- **Resolution Documentation**: Document resolution when issue is closed

## Issue Template Structure

```markdown
## Description

Clear description of the issue or feature request.

## Environment

- OS: [Operating System]
- Python Version: [Version]
- Azure Services: [Relevant services]
- Package Versions: [Relevant package versions]

## Steps to Reproduce (for bugs)

1. Step 1
2. Step 2
3. Step 3

## Expected Behavior

What should happen.

## Actual Behavior

What actually happens.

## Error Messages
```

[Error messages and stack traces]

```

## Additional Context
Any additional context, screenshots, or logs.

## Proposed Solution (if applicable)
Suggested approach to resolve the issue.
```

## Label Guidelines

- **bug**: Something isn't working
- **enhancement**: New feature or request
- **documentation**: Improvements or additions to documentation
- **priority/high**: High priority item
- **priority/medium**: Medium priority item
- **priority/low**: Low priority item
- **good first issue**: Good for newcomers
- **help wanted**: Extra attention is needed
